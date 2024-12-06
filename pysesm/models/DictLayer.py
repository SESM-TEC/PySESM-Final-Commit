import logging

from pysesm.functions import SurrogateFunction, SurrogateFunctionFactory
from pysesm.enums import SurrogateFunctionEnum

import torch
from typing import Union, Callable


class DictLayer(torch.nn.Module):
    """
    A custom PyTorch module for implementing a dictionary layer with learnable parameters.

    This layer is designed for surrogate modeling and function approximation tasks. It uses
    a surrogate function (psi) to compute a dictionary based on input data and learnable parameters.

    Attributes:
        n_samples (int): Number of samples in the input dataset.
        n_features (int): Number of features or dimensions in the input data.
        n_functions (int): Number of functions or basis elements in the dictionary.
        psi (SurrogateFunction): Surrogate function used to generate the dictionary.
        theta_parameter_vector (torch.nn.Parameter): Learnable parameter tensor used by `psi`.
        dictionary (torch.Tensor): Output dictionary computed by the layer.
        alpha (float): Learning rate for the optimizer.
        evaluation_func (Callable): Function used to evaluate predictions from the dictionary.
        losses (list): History of loss values, recorded if `log_losses` is enabled.
        criterion (torch.nn.modules.loss._Loss): Loss function used for training (default: Mean Squared Error).
        optimizer (torch.optim.Optimizer): Optimizer used to train the layer (default: SGD with learning rate `alpha`).
    """

    psi: SurrogateFunction

    def __init__(self, n_samples: int, n_features: int, n_functions: int,
                 psi: Union[SurrogateFunction, SurrogateFunctionEnum], alpha: float,
                 evaluation_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 logger: logging.Logger,
                 criterion: torch.nn.modules.loss._Loss = None, optimizer: torch.optim.Optimizer = None, **kwargs):
        """
        Initialize the `DictLayer` instance.

        This constructor sets up the layer, initializes parameters and the surrogate function,
        and configures the optimizer and loss function.

        Args:
            n_samples (int): Number of samples in the input dataset.
            n_features (int): Number of features or dimensions in the input data.
            n_functions (int): Number of functions or basis elements in the dictionary.
            psi (Union[SurrogateFunction, SurrogateFunctionEnum]): Surrogate function or its enumeration type.
            alpha (float): Learning rate for the optimizer.
            evaluation_func (Callable): Function used to evaluate predictions.
            logger (logging.Logger): Logger instance to capture runtime information.
            criterion (torch.nn.modules.loss._Loss, optional): Loss function (default: Mean Squared Error).
            optimizer (torch.optim.Optimizer, optional): Optimizer for training (default: SGD with learning rate `alpha`).
            **kwargs: Additional arguments passed to the surrogate function factory.
        """

        super(DictLayer, self).__init__()

        self.n_samples = n_samples
        self.n_features = n_features
        self.n_functions = n_functions
        self.alpha = alpha
        self.evaluation_func = evaluation_func

        self.psi = psi if issubclass(type(psi), SurrogateFunction) else SurrogateFunctionFactory.make(psi, n_functions=n_functions, n_features=n_functions, logger=logger, **kwargs)

        self.losses = []

        self.theta_parameter_vector = self.psi.initialize()

        self.dictionary = None

        if criterion is None:
            self.criterion = torch.nn.MSELoss()
        else:
            self.criterion = criterion

        if optimizer is None:
            self.optimizer = torch.optim.SGD(self.parameters(), lr=alpha)
        else:
            self.optimizer = optimizer

    def initialize_layer(self, X: torch.Tensor) -> None:
        """
        Initialize the dictionary for the layer.

        This method uses the surrogate function (`psi`) to compute the dictionary based on
        the provided input data and the current parameter vector.

        Args:
            X (torch.Tensor): Input data of shape `(n_samples, n_features)`.

        Returns:
            None
        """
        self.dictionary = self.psi.__call__(X.mT, self.theta_parameter_vector)

    def partial_fit(self, X: torch.Tensor, y: torch.Tensor, h: torch.Tensor, epochs: int, max_points_in_block: int = 0,
                    active_blocks_count: int = 0, rho_flag: bool = False,
                    mu_flag: bool = False, log_losses: bool = True) -> None:
        """
        Perform a partial fit on the layer, updating the dictionary and weights iteratively.

        This method performs a training loop for a specified number of epochs, computes the
        predicted outputs, evaluates the loss, and updates parameters using backpropagation.

        Args:
            X (torch.Tensor): Input data of shape `(n_samples, n_features)`.
            y (torch.Tensor): Target data of shape `(n_samples,)`.
            h (torch.nn.Parameter): Sparse vector holding the weights for the target function.
            epochs (int): Number of training epochs.
            max_points_in_block (int, optional): Maximum points in each block (default: 0).
            active_blocks_count (int, optional): Number of active blocks (default: 0).
            rho_flag (bool, optional): Whether to update rho values (default: False).
            mu_flag (bool, optional): Whether to update mu values (default: False).
            log_losses (bool, optional): Whether to log the loss values for each epoch (default: True).

        Returns:
            None
        """
        for _ in range(epochs):
            self.dictionary = self.forward(X, max_points_in_block, active_blocks_count, rho_flag, mu_flag)

            # y_pred = torch.bmm(self.dictionary, h).squeeze(-1).flatten()
            y_pred = self.evaluation_func(self.dictionary, h)

            loss = self.criterion(y_pred, y)
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            if log_losses:
                self.losses.append(loss.item())

    def forward(self, X: torch.Tensor, max_points_in_block: int, active_blocks_count: int, rho_flag: bool = False,
                mu_flag: bool = False):
        """
        Compute the forward pass for the dictionary layer.

        This method applies the surrogate function (`psi`) to compute the evaluated dictionary
        based on the input data and learnable parameters. If block-based partitioning is used,
        the dictionary is reshaped accordingly.

        Args:
            X (torch.Tensor): Input data of shape `(n_samples, n_features)`.
            max_points_in_block (int, optional): Maximum points in each block (default: 0).
            active_blocks_count (int, optional): Number of active blocks (default: 0).
            rho_flag (bool, optional): Whether to update rho values (default: False).
            mu_flag (bool, optional): Whether to update mu values (default: False).

        Returns:
            torch.Tensor: Evaluated dictionary of shape `(n_samples, n_functions)`,
            or reshaped to `(active_blocks_count, max_points_in_block, n_functions)` if partitioning is applied.
        """

        evaluated_dictionary = self.psi.__call__(X.mT, self.theta_parameter_vector, rho_flag, mu_flag)
        if not max_points_in_block and not active_blocks_count:
            return evaluated_dictionary
        else:
            return evaluated_dictionary.view((active_blocks_count, max_points_in_block, self.n_functions))
