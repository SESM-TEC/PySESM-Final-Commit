from torch import Tensor
from torch.nn import Module
from typing import Callable
import torch


class ISTALayer(Module):
    """
    A custom PyTorch module for implementing a sparse vector layer with learnable parameters.

    This layer is designed for surrogate modeling and function approximation tasks, integrating
    sparsity through shrinkage operations and custom regularization to control parameter values.

    Attributes:
        n_functions (int): Number of basis functions or components in the sparse vector.
        random_seed (int): Random seed for reproducibility.
        weight_decay (float): Weight decay coefficient for regularization.
        alpha (float): Learning rate for parameter updates.
        lambd (float): Regularization parameter for shrinkage operations.
        criterion (torch.nn.Module): Loss function used for training (default: Mean Squared Error).
        optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters (default: SGD).
        h (torch.nn.Parameter): Sparse vector maintained and updated by the layer.
        losses (list): List storing the computed losses during training.
        calculate_y_pred (callable): Function to compute predictions from input and parameters.
        threshold (float): Threshold for custom regularization (default: 10.0).
        penalty_weight (float): Weight of the penalty term in custom regularization (default: 0.01).

    Methods:
        initialize_h_vector(h: torch.Tensor):
            Initializes the sparse vector `h` as a learnable parameter.
        get_custom_regularization() -> torch.Tensor:
            Computes the custom regularization term based on the threshold and penalty weight.
        shrinkage() -> torch.Tensor:
            Applies the shrinkage operation to enforce sparsity in `h`.
        partial_fit(y, epochs, dictionary, log_losses=True) -> None:
            Performs a partial fit for a given number of epochs, updating `h`.
        forward(y, dictionary, log_losses=True):
            Performs a forward pass, computes the total loss (including custom regularization),
            and updates parameters using backpropagation.

    """

    def __init__(self, n_functions: int, random_seed: int, weight_decay: float, alpha: float, lambd: float,
                 criterion=None, optimizer=None, h: Tensor = None, calculate_y_pred: Callable[[Tensor, Tensor], Tensor] =None):
        """
        Initializes the ISTALayer with the specified hyperparameters and components.

        Args:
            n_functions (int): Number of basis functions or components in the sparse vector.
            random_seed (int): Random seed for reproducibility.
            weight_decay (float): Weight decay coefficient for regularization.
            alpha (float): Learning rate for parameter updates.
            lambd (float): Regularization parameter for shrinkage operations.
            criterion (torch.nn.Module, optional): Loss function used for training (default: MSELoss).
            optimizer (torch.optim.Optimizer, optional): Optimizer for parameter updates (default: SGD).
            h (torch.Tensor, optional): Initial sparse vector. If not provided, it is initialized randomly.
            calculate_y_pred (callable, optional): Function to compute predictions (default: None).
        """
        super(ISTALayer, self).__init__()
        self.n_functions = n_functions
        self.random_seed = random_seed
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.lambd = lambd
        self.calculate_y_pred = calculate_y_pred
        self.losses = []
        torch.manual_seed(random_seed)
        self.initialize_h_vector(h)

        if criterion is None:
            self.criterion = torch.nn.MSELoss()
        else:
            self.criterion = criterion

        if optimizer is None:
            self.optimizer = torch.optim.SGD(self.parameters(), lr=alpha, weight_decay=weight_decay)
        else:
            self.optimizer = optimizer(parameters=self.parameters(), lr=alpha, weight_decay=weight_decay)
        
        self.calculate_y_pred = calculate_y_pred
        
        self.threshold = 11  
        self.penalty_weight = 0.05  

    def initialize_h_vector(self, h: torch.Tensor) -> None:
        """
        Initializes the sparse vector `h` as a learnable parameter.

        Args:
            h (torch.Tensor): Optional initial value for the sparse vector. If not provided, it is initialized randomly.
        """
        if h is not None:
            self.h = torch.nn.Parameter(h)
        else:
            self.h = torch.nn.Parameter(torch.rand(self.n_functions), requires_grad=True)
            self.h.data /= self.h.data.sum()
    
    def get_custom_regularization(self) -> torch.Tensor:
        """
        Computes the custom regularization term for parameters exceeding the defined threshold.

        Returns:
            torch.Tensor: The computed regularization penalty.
        """
        reg_loss = 0.0
        for param in self.parameters():
            penalty = torch.relu(torch.abs(param) - self.threshold)
            reg_loss += torch.sum(penalty ** 2)
        return self.penalty_weight * reg_loss
    
    def shrinkage(self) -> torch.Tensor:
        """
        Performs the shrinkage operation to enforce sparsity in the layer's parameters.

        Returns:
            torch.Tensor: The updated sparse vector.
        """
        return torch.sign(self.h) * torch.max(torch.abs(self.h) - self.alpha * self.lambd, torch.zeros_like(self.h))

    def partial_fit(self, y, epochs, dictionary, log_losses=True) -> None:        
        """
        Performs a partial fit, iteratively updating the sparse vector `h`.

        Args:
            y (torch.Tensor): Ground truth or target vector.
            epochs (int): Number of epochs to train.
            dictionary (torch.Tensor): Input dictionary for the forward pass.
            log_losses (bool): Whether to log the computed losses (default: True).
        """
        for _ in range(epochs):
            new_h = self.forward(y, dictionary, log_losses)
            if new_h is not None: self.h.data = new_h

    def forward(self, y, dictionary, log_losses=True):
        """
        Performs a forward pass and updates the layer's parameters using backpropagation.

        Args:
            y (torch.Tensor): Ground truth or target vector.
            dictionary (torch.Tensor): Input dictionary for predictions.
            log_losses (bool): Whether to log the computed losses (default: True).

        Returns:
            torch.Tensor: Updated sparse vector after applying shrinkage.
        """
        y_pred = self.calculate_y_pred(dictionary, self.h)        
        loss = self.criterion(y_pred, y)
        
        reg_loss = self.get_custom_regularization()
        total_loss = loss + reg_loss
        
        self.optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        self.optimizer.step()

        if log_losses:
            self.losses.append(total_loss.item())

        with torch.no_grad():
            return self.shrinkage()
