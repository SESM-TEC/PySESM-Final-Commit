import logging

from torch import Tensor
from torch.nn import Module
from typing import Callable
import torch


class ISTALayer(Module):
    """
    A custom PyTorch module implementing a sparse vector layer with learnable parameters.

    This layer is designed for tasks such as surrogate modeling and function approximation,
    integrating sparsity through shrinkage operations and custom regularization techniques.

    Attributes:
        n_functions (int): Number of basis functions or components in the sparse vector.
        alpha (float): Learning rate for parameter updates.
        lambd (float): Regularization parameter controlling the strength of shrinkage operations.
        criterion (torch.nn.Module): Loss function used for training (default: Mean Squared Error).
        optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters (default: SGD).
        h (torch.nn.Parameter): Sparse vector maintained and updated by the layer.
        losses (list): List storing the computed losses during training.
        evaluation_func (callable): Function to compute predictions from input and parameters.
        threshold (float): Threshold for custom regularization (default: 10.0).
        penalty_weight (float): Weight of the penalty term in custom regularization (default: 0.01).

    Methods:
        setup(h: torch.Tensor) -> None:
            Initializes the sparse vector `h` as a learnable parameter.
        get_custom_regularization() -> torch.Tensor:
            Computes a custom regularization term for parameters exceeding the threshold.
        shrinkage() -> torch.Tensor:
            Applies the shrinkage operation (soft thresholding) to enforce sparsity in `h`.
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
            evaluation_func: Callable[[Tensor, Tensor], Tensor],
            logger: logging.Logger,
            debug: bool = False,
            criterion=None,
            optimizer=None,
    ):
        """
        Initializes the ISTALayer with the specified hyperparameters and components.

        Args:
            n_functions (int): Number of basis functions or components in the sparse vector.
            alpha (float): Learning rate for parameter updates.
            lambd (float): Regularization parameter for shrinkage operations.
            criterion (torch.nn.Module, optional): Loss function used for training (default: MSELoss).
            optimizer (torch.optim.Optimizer, optional): Optimizer for parameter updates (default: SGD).
            h (torch.Tensor, optional): Initial sparse vector. If not provided, it is initialized randomly.
            calculate_y_pred (callable, optional): Function to compute predictions (default: None).
        """
        super(ISTALayer, self).__init__()
        self.h = None
        self.n_functions = n_functions
        self.alpha = alpha
        self.lambd = lambd
        self.evaluation_func = evaluation_func
        self.losses = []

        self.setup()

        if criterion is None:
            # self.criterion = torch.nn.L1Loss()  # L1 to promote sparsity
            self.criterion = torch.nn.MSELoss()
        else:
            self.criterion = criterion

        # It is particularly dangerous to activate weight decay with ISTA, because then
        # h will be enforce to be closer to zero, and that goes against all the logic.
        # A previous bug tought us to better enforce weight_decay=0 here, no matter what.
        if optimizer is None:
            self.optimizer = torch.optim.SGD(
                self.parameters(), lr=alpha, weight_decay=0, momentum=0.1
            )
        else:
            self.optimizer = optimizer(
                parameters=self.parameters(), lr=alpha, weight_decay=0
            )

        # TODO: add this in the arguments too.  These are used in the custom regularization
        self.threshold = 11
        self.penalty_weight = 0 # 0.05  # Deactivated for now

    def setup(self, h: torch.Tensor = None) -> None:
        """
        Initializes the sparse vector `h` as a learnable parameter.

        If an initial value for `h` is provided, it is used directly. Otherwise, `h` is randomly
        initialized with values normalized to sum to one.

        Args:
            h (torch.Tensor): Optional initial value for the sparse vector. If not provided, 
            a random vector of size `n_functions` is created.

        Returns:
            None
        """
        if h is not None:
            # Ensure h is 2D
            if h.dim() != 2:
                h = h.reshape(-1, 1)  # Default to column vector if reshaping needed
            self.h = torch.nn.Parameter(h)
        else:
            # Initialize h as 2D
            self.h = torch.nn.Parameter(
                torch.rand(self.n_functions,1), requires_grad=True
            )
            self.h.data /= self.h.data.sum()

    def get_custom_regularization(self) -> float:
        """
        Computes the custom regularization term to penalize parameters exceeding the defined threshold.

        This term encourages sparsity and controls the magnitude of parameters by penalizing their
        values when they exceed `self.threshold`.

        Returns:
            torch.Tensor: The computed regularization penalty.
        """
        params = torch.cat([p.view(-1) for p in self.parameters()])
        penalty = torch.relu(torch.abs(params) - self.threshold)
        return self.penalty_weight * torch.sum(penalty ** 2)

    def shrinkage(self) -> torch.Tensor:
        """
        Applies soft thresholding: zeros out elements below lambda threshold,
        leaves other elements unchanged
        """
        return torch.where(
            torch.abs(self.h) < self.lambd,
            torch.zeros_like(self.h),
            self.h
    )

    def forward(self, y, dictionary, log_losses=True):
        """
        Performs a forward pass and updates the layer's parameters using backpropagation.

        Args:
            y (torch.Tensor): Ground truth or target vector.
            dictionary (torch.Tensor): Input dictionary for predictions.
            log_losses (bool): Whether to log the computed losses (default: True).

        Returns:
            torch.Tensor: Estimated loss after forward step
        """
        y_pred = self.evaluation_func(dictionary, self.h)

        assert y_pred.shape == y.shape, f"Shape mismatch: y_pred {y_pred.shape} != y {y.shape}"

        # DEBUG
        # print("y:", y[:5])  # First few values
        # print("y_pred:", y_pred[:5])
        # print("dict:", dictionary[:5])  # First few values of the dictionary


        loss = self.criterion(y, y_pred)

        reg_loss = self.get_custom_regularization()
        total_loss = loss + reg_loss

        if log_losses:
            self.losses.append(total_loss.item())

        return total_loss

    def train_step(self, y, dictionary, log_losses=True):
        """
        Performs a single training step: forward, backward, and optimization.
        """
        self.optimizer.zero_grad()
        loss = self.forward(y, dictionary, log_losses)
        loss.backward()

        # DEBUG
        # print("h:", self.h)
        # print("gradient:", self.h.grad)  # After backward()


        self.optimizer.step()

        with torch.no_grad():
            self.h.data = self.shrinkage()

        return loss
    
    def partial_fit(self, y, epochs, dictionary, log_losses=True) -> None:
        """
        Performs a partial fit, iteratively updating the sparse vector `h` over a specified number of epochs.

        During training, the method logs the total loss, including both the prediction error
        and the regularization penalty, for each epoch.

        Args:
            y (torch.Tensor): Ground truth or target vector.
            epochs (int): Number of epochs to train.
            dictionary (torch.Tensor): Input matrix or dictionary used to compute predictions.
            log_losses (bool, optional): Whether to log the computed losses during training (default: True).

        Returns:
            None
        """
        for _ in range(epochs):
            self.train_step(y, dictionary, log_losses)


