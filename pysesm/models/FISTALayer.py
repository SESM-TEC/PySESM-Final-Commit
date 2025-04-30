'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica

FISTA Layer Class

Provides a Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) implementation
for finding the sparse vector h that chooses words in a dictionary to build
a surrogate function.

Authors: The SESM Team 

License: 
'''

import logging
import torch
from typing import Callable, Iterator, Optional
from pysesm.models.BaseISTALayer import BaseISTALayer

class FISTALayer(BaseISTALayer):
    """
    A custom PyTorch module implementing FISTA (Fast Iterative Shrinkage-Thresholding Algorithm).
    
    FISTA is an accelerated version of the ISTA algorithm that incorporates a momentum term
    to improve convergence speed. This algorithm achieves a convergence rate of O(1/k²) compared
    to the O(1/k) of classical ISTA.

    Attributes:
        n_functions (int): Number of basis functions or components in the sparse vector.
        alpha (float): Learning rate for parameter updates.
        lambd (float): Regularization parameter controlling the strength of shrinkage operations.
        criterion (torch.nn.Module): Loss function used for training (default: Mean Squared Error).
        optimizer (torch.optim.Optimizer): Optimizer builder (default: None (uses SGD)).
        h (torch.nn.Parameter): Sparse vector maintained and updated by the layer.
        y_prev (torch.Tensor): Previous value of h used for momentum calculation.
        t (float): Momentum parameter.
        losses (list): List storing the computed losses during training.
        evaluation_func (callable): Function to compute predictions from input and parameters.
    """

    def __init__(
            self,
            n_functions: int,
            alpha: float,
            lambd: float,
            evaluation_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            logger: logging.Logger,
            debug: bool = False,
            criterion=None,
            optimizer: Callable[[Iterator[torch.nn.Parameter], float], torch.optim.Optimizer] = None,
            parameter_hook: Optional[Callable[[dict], None]] = None,
            device=None,
            restart_every: int = 0  # 0 means no restart
    ):
        """
        Initializes the FISTALayer with the specified hyperparameters and components.
        Args:
            n_functions (int): Number of basis functions or components in the sparse vector.
            alpha (float): Learning rate for parameter updates.
            lambd (float): Regularization parameter for shrinkage operations.
            evaluation_func (callable): Function to compute predictions.
            logger (logging.Logger): Logger for recording training information.
            debug (bool, optional): Whether to enable debug mode. Defaults to False.
            criterion (torch.nn.Module, optional): Loss function. Defaults to MSELoss.
            optimizer (callable, optional): Optimizer builder. Defaults to SGD.
            parameter_hook (callable, optional): Hook function for parameter inspection.
            device: Device to run computations on (CPU/GPU).
            restart_every (int, optional): Restart momentum every N iterations. Defaults to 0 (no restart).
        """
        super(FISTALayer, self).__init__()
        self.h = None
        self.y_prev = None  # Auxiliary variable for the FISTA algorithm
        self.n_functions = n_functions
        self.alpha = alpha
        self.lambd = lambd
        self.evaluation_func = evaluation_func
        self.losses = []
        self.device = device
        self.logger = logger
        self.debug = debug
        self.parameter_hook = parameter_hook
        self.restart_every = restart_every
        self.iter_count = 0
        self.t = torch.tensor(1.0, device=self.device)  # Instead of self.t = 1.0
        self.setup()

        if criterion is None:
            self.criterion = torch.nn.MSELoss()
        else:
            self.criterion = criterion

        if optimizer is None:
            self.optimizer = torch.optim.SGD(self.parameters(), lr=alpha, weight_decay=0)
        else:
            self.optimizer = optimizer(self.parameters(), lr=alpha)

        # Threshold for custom regularization
        self.threshold = 11
        self.penalty_weight = 0  # Deactivated for now

    def setup(self, h: torch.Tensor = None) -> None:
        """
        Initializes the sparse vector h and the auxiliary variable y for FISTA.

        Args:
            h (torch.Tensor, optional): Initial value for the sparse vector.
        """
        if h is not None:
            # Ensure h is 2D
            if h.dim() != 2:
                h = h.reshape(-1, 1)  # Default to column vector if reshaping needed
            self.h = torch.nn.Parameter(h.to(self.device), requires_grad=True)
        else:
            # Initialize h as 2D
            self.h = torch.nn.Parameter(
                torch.rand(self.n_functions, 1).to(self.device), 
                requires_grad=True
            )
            self.h.data /= self.h.data.sum()
        
        # Initialize the auxiliary variable y for FISTA
        self.y_prev = self.h.clone().detach()

    def get_custom_regularization(self) -> float:
        """
        Computes the custom regularization term to penalize parameters exceeding the defined threshold.

        Returns:
            torch.Tensor: The computed regularization penalty.
        """
        params = torch.cat([p.view(-1) for p in self.parameters()])
        penalty = torch.relu(torch.abs(params) - self.threshold)
        return self.penalty_weight * torch.sum(penalty ** 2)

    def shrinkage(self, tensor=None) -> torch.Tensor:
        """
        Applies soft thresholding to promote sparsity.
        
        Args:
            tensor (torch.Tensor, optional): Tensor to apply shrinkage to. If None, uses self.h.
            
        Returns:
            torch.Tensor: Result after applying soft thresholding.
        """
        if tensor is None:
            tensor = self.h
            
        return torch.where(
            torch.abs(tensor) < self.lambd,
            torch.zeros_like(tensor, device=self.device),
            tensor - self.lambd * torch.sign(tensor)
        )

    def forward(self, y_true, dictionary, log_losses=True):
        """
        Performs a forward pass.

        Args:
            y_true (torch.Tensor): Ground truth or target vector.
            dictionary (torch.Tensor): Input dictionary for predictions.
            log_losses (bool): Whether to log the computed losses.

        Returns:
            torch.Tensor: Computed loss.
        """
        # Ensure all tensors are on the right device
        y_true = y_true.to(self.device)
        dictionary = dictionary.to(self.device)

        # Combine the words in the dictionary using current parameter
        y_pred = self.evaluation_func(dictionary, self.h)
        assert y_pred.shape == y_true.shape, f"Shape mismatch: y_pred {y_pred.shape} != y_true {y_true.shape}"
        
        loss = self.criterion(y_true, y_pred)
        total_loss = loss  # + self.get_custom_regularization() if used

        if log_losses:
            self.losses.append(total_loss.item())

        return total_loss

    def train_step(self, y_true, dictionary, log_losses=True):
        """
        Performs a single FISTA training step.
        
        FISTA incorporates a momentum term to accelerate convergence.
        
        Args:
            y_true (torch.Tensor): Ground truth or target vector.
            dictionary (torch.Tensor): Input dictionary for predictions.
            log_losses (bool): Whether to log the computed losses.
            
        Returns:
            torch.Tensor: Computed loss.
        """
        # Move input tensors to the device
        y_true = y_true.to(self.device)
        dictionary = dictionary.to(self.device)
        
        # Increment the iteration counter
        self.iter_count += 1
        
        # 1. Compute the gradient using the auxiliary variable y
        with torch.no_grad():
            # Backup the current value of h
            h_current = self.h.clone()
            
            # Temporarily assign y to h to compute the gradient
            self.h.data = self.y_prev.clone()
        
        # Forward and backward pass to compute the gradient
        self.optimizer.zero_grad()
        loss = self.forward(y_true, dictionary, log_losses)
        loss.backward()
        
        # 2. Update h using the gradient and shrinkage
        with torch.no_grad():
            # Restore h
            self.h.data = h_current
            
            # Gradient step using y
            h_next = self.y_prev - self.alpha * self.h.grad
            
            # Apply shrinkage to promote sparsity
            h_next = self.shrinkage(h_next)
            
            # 3. Update y using the momentum term
            # Calculate the new value of t (momentum parameter)
            t_next = (1 + torch.sqrt(1 + 4 * self.t**2)) / 2
            
            # Calculate the momentum coefficient
            momentum_coef = (self.t - 1) / t_next
            
            # Update y with momentum
            self.y_prev = h_next + momentum_coef * (h_next - self.h.data)
            
            # Update t for the next iteration
            self.t = t_next
            
            # Update h
            self.h.data = h_next
            
            # Optional: Periodic momentum restart for stability
            if self.restart_every > 0 and self.iter_count % self.restart_every == 0:
                self.t = 1.0
                self.y_prev = self.h.data.clone()
                if self.debug:
                    self.logger.debug(f"FISTA: Restarting momentum at iteration {self.iter_count}")
        
        # Call the parameter hook if defined
        if self.parameter_hook is not None:
            hook_info = {
                'h': self.h.clone().detach(),
                'h_grad': self.h.grad.clone().detach() if self.h.grad is not None else None,
                'y': self.y_prev.clone().detach(),
                't': self.t,
                'loss': loss.item(),
                'iter': self.iter_count
            }
            self.parameter_hook(hook_info)
            
        return loss
    
    def partial_fit(self, y, epochs, dictionary, log_losses=True) -> None:
        """
        Performs a partial fit over multiple epochs.

        Args:
            y (torch.Tensor): Ground truth or target vector.
            epochs (int): Number of epochs to train.
            dictionary (torch.Tensor): Input dictionary for predictions.
            log_losses (bool): Whether to log the computed losses.
        """
        y = y.to(self.device)
        dictionary = dictionary.to(self.device)
        for _ in range(epochs):
            self.train_step(y, dictionary, log_losses)