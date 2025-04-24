'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica

ISTA Layer Class

Provides the layer in charge of finding h, the sparse vector 
that chooses words in a dictionary to build a surrogate 
function.

Authors: The SESM Team 

License: 
'''

import logging
import torch
from typing import Callable, Iterator, Optional

class ISTALayer(torch.nn.Module):
    """
    A custom PyTorch module implementing a sparse vector layer with learnable parameters.

    This layer is designed for tasks such as surrogate modeling and function approximation,
    integrating sparsity through shrinkage operations and custom regularization techniques.

    Attributes:
        n_functions (int): Number of basis functions or components in the sparse vector.
        alpha (float): Learning rate for parameter updates.
        lambd (float): Regularization parameter controlling the strength of shrinkage operations.
        criterion (torch.nn.Module): Loss function used for training (default: Mean Squared Error).
        # (deprecated) optimizer (torch.optim.Optimizer): Optimizer builder (default: None (uses SGD)).
        h (torch.nn.Parameter): Sparse vector maintained and updated by the layer.
        losses (list): List storing the computed losses during training.
        evaluation_func (callable): Function to compute predictions from input and parameters.
        threshold (float): Threshold for custom regularization (default: 10.0).
        penalty_weight (float): Weight of the penalty term in custom regularization (default: 0.01).

    Methods:
        setup(h: torch.Tensor) -> None:
            Initializes the sparse vector `h` as a learnable parameter.
        calculate_alpha(dictionary: torch.Tensor) -> float:
            Calculates the optimal step size based on the Lipschitz constant.
        train_step(y: torch.Tensor, dictionary: torch.Tensor, log_losses: bool = True) -> torch.Tensor:
            Performs a single ISTA iteration.
        partial_fit(y: torch.Tensor, epochs: int, dictionary: torch.Tensor, log_losses: bool = True) -> None:
            Performs a partial fit over a specified number of epochs, updating `h`.
        forward(y: torch.Tensor, dictionary: torch.Tensor, log_losses: bool = True) -> torch.Tensor:
            Executes a forward pass, computes the total loss (including regularization),
            and updates parameters using backpropagation.
    """

    def __init__(
            self,
            n_functions: int,
            alpha: float,
            lambd: float,
            evaluation_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            logger: logging.Logger,
            h (torch.Tensor, optional): Initial sparse vector. If not provided, it is initialized randomly = None,
            debug: bool = False,
            criterion=None,
            # optimizer: Callable[[Iterator[torch.nn.Parameter],float], torch.optim.Optimizer] = None,
            parameter_hook: Optional[Callable[[dict], None]] = None,
            device = None,
            adaptive_alpha: bool = True
    ):
        """
        Initializes the ISTALayer with the specified hyperparameters and components.

        Args:
            n_functions (int): Number of basis functions in the dictionary.
            alpha (float): Learning rate for parameter updates.
            adaptive_alpha (bool): Whether to calculate alpha adaptively (recommended).
            lambd (float): Regularization parameter controlling sparsity in shrinkage operations.
            evaluation_func (Callable[[Iterator[torch.nn.Parameter],float], torch.optim.Optimizer]) : eval f(D,h)
            criterion (torch.nn.Module, optional): Loss function used for training (default: MSELoss).
            optimizer (torch.optim.Optimizer, optional): Optimizer for parameter updates (default: SGD).
            calculate_y_pred (callable, optional): Function to compute predictions (default: None).
            parameter_hook (Callable, optional): Callback function to inspect the current parameter state.
        """
        super(ISTALayer, self).__init__()
        self.n_functions = n_functions
        self.alpha = alpha
        self.adaptive_alpha = adaptive_alpha
        self.lambd = lambd
        self.evaluation_func = evaluation_func
        self.losses = []
        self.device = device
        self.to(self.device) # Move the layer to the assigned device for ISTA_LAYER
        self.logger = logger
        self.debug = debug
        self.parameter_hook = parameter_hook
 
        if criterion is None:
            self.criterion = torch.nn.MSELoss()
        else:
            self.criterion = criterion
            
        self.setup(h)
        self.to(self.device)

    def setup(self, h: torch.Tensor = None) -> None:
        """
        Initializes the sparse vector `h`.
        
        Args:
            h (torch.Tensor, optional): Initial value for the sparse vector.
                If not provided, initialized with zeros.
        """
        if h is not None:
            # Ensure h is 2D
            if h.dim() != 2:
                h = h.reshape(-1, 1)
            self.h = torch.nn.Parameter(h.to(self.device), requires_grad=False)
        else:
            # Initialize h as zeros (common for ISTA)
            self.h = torch.nn.Parameter(
                torch.zeros(self.n_functions, 1).to(self.device), 
                requires_grad=False
            )

    def calculate_alpha(self, dictionary: torch.Tensor) -> float:
        """
        Calculates the optimal step size based on the Lipschitz constant.
        
        For MSE loss, the Lipschitz constant is the largest eigenvalue of D^T D.
        
        Args:
            dictionary (torch.Tensor): The dictionary matrix.
            
        Returns:
            float: The calculated step size.
        """
        with torch.no_grad():
            # Compute D^T D (gram matrix)
            gram = torch.matmul(dictionary.T, dictionary)
            
            # Get the largest eigenvalue
            eigenvalues = torch.linalg.eigvalsh(gram)
            L = torch.max(eigenvalues)
            
            # Step size should be <= 1/L for convergence
            alpha = 1.0 / (L + 1e-8)  # Adding small constant for stability
            
            if self.debug:
                self.logger.debug(f"Calculated alpha: {alpha}, Lipschitz constant: {L}")
                
        return alpha

    def soft_threshold(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Applies soft thresholding operation.
        
        Args:
            x (torch.Tensor): Input tensor.
            threshold (float): Threshold value.
            
        Returns:
            torch.Tensor: Soft thresholded tensor.
        """
        return torch.sign(x) * torch.maximum(
            torch.abs(x) - threshold,
            torch.zeros_like(x, device=self.device)
        )

    def train_step(self, y: torch.Tensor, dictionary: torch.Tensor, log_losses: bool = True) -> torch.Tensor:
        """
        Performs a single ISTA iteration.
        
        Args:
            y (torch.Tensor): Target vector.
            dictionary (torch.Tensor): Dictionary matrix.
            log_losses (bool): Whether to log losses.
            
        Returns:
            torch.Tensor: The current loss.
        """
        # Move tensors to correct device
        y = y.to(self.device)
        dictionary = dictionary.to(self.device)
        
        # Calculate adaptive step size if needed
        if self.adaptive_alpha:
            self.alpha = self.calculate_alpha(dictionary)
        
        with torch.no_grad():
            # Forward pass
            y_pred = self.evaluation_func(dictionary, self.h)
            
            # Compute loss
            loss = self.criterion(y, y_pred)
            if log_losses:
                self.losses.append(loss.item())
            
            # Compute gradient (for MSE: 2 * D^T * (y_pred - y))
            error = y_pred - y
            gradient = 2 * torch.matmul(dictionary.T, error)
            
            # ISTA update: h = soft_threshold(h - alpha * gradient, alpha * lambda)
            h_update = self.h - self.alpha * gradient
            self.h.data = self.soft_threshold(h_update, self.alpha * self.lambd)
            
            # Call parameter hook if provided
            if self.parameter_hook is not None:
                hook_info = {
                    'h': self.h.clone().detach(),
                    'gradient': gradient.clone().detach(),
                    'loss': loss.item(),
                    'alpha': self.alpha
                }
                self.parameter_hook(hook_info)
            
        return loss

    def forward(self, y, dictionary, log_losses=True):
        """
        Performs a forward pass 

        Args:
            y (torch.Tensor): Ground truth or target vector.
            dictionary (torch.Tensor): Input dictionary for predictions.
            log_losses (bool): Whether to log the computed losses (default: True).

        Returns:
            torch.Tensor: Estimated loss after forward step
        """

        # Ensure all tensors are on the right device
        y = y.to(self.device)
        dictionary = dictionary.to(self.device)

        # Combine the words in the dictionary using self.h
        y_pred = self.evaluation_func(dictionary, self.h)
        assert y_pred.shape == y.shape, f"Shape mismatch: y_pred {y_pred.shape} != y {y.shape}"
        
        loss = self.criterion(y, y_pred)

        if log_losses:
            self.losses.append(loss.item())

        return loss
    
    def partial_fit(self, y: torch.Tensor, epochs: int, dictionary: torch.Tensor, log_losses: bool = True) -> None:
        """
        Performs multiple ISTA iterations.
        
        Args:
            y (torch.Tensor): Target vector.
            epochs (int): Number of iterations.
            dictionary (torch.Tensor): Dictionary matrix.
            log_losses (bool): Whether to log losses.
        """
        for epoch in range(epochs):
            loss = self.train_step(y, dictionary, log_losses)
            
            if self.debug and (epoch == 0 or (epoch + 1) % 100 == 0 or epoch == epochs - 1):
                self.logger.debug(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")
                
            # Optional early stopping based on convergence
            if epoch > 0 and len(self.losses) >= 2:
                if abs(self.losses[-1] - self.losses[-2]) < 1e-6:
                    if self.debug:
                        self.logger.debug(f"Early stopping at epoch {epoch + 1}, loss converged")
                    break

