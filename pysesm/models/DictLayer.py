import logging

from pysesm.customization_factories import SurrogateFunctionFactory
from pysesm.functions import SurrogateFunction
from pysesm.enums import SurrogateFunctionEnum

import torch
from typing import Optional, Callable, Union, Dict

class DictLayer(torch.nn.Module):
    """
    A custom PyTorch module for implementing a dictionary layer with learnable parameters.

    This layer is designed for surrogate modeling and function approximation tasks. It uses
    a surrogate function (psi) to compute a dictionary based on input data and learnable parameters.

    Attributes:
        psi (SurrogateFunction): The function used for generating the layer's output.
        theta_parameter_vector (torch.nn.Parameter): Learnable parameter tensor for the layer's functions.
        dictionary (torch.Tensor): The computed output of the layer.
        n_features (int): The number of input features or dimensions.
        n_functions (int): The number of functions or basis functions.
        losses (list): History of loss values, recorded if `log_losses` is enabled.
        criterion (torch.nn.modules.loss._Loss): Loss function used for training (default: Mean Squared Error).
        optimizer (torch.optim.Optimizer): Optimizer used to train the layer (default: SGD with learning rate `alpha`).
        parameter_hook (Optional): Callback function to inspect the current parameter state
    """

    # These are type hints (not class attributes) for instance attributes:
    psi: SurrogateFunction
    theta_parameter_vector: torch.nn.Parameter
    dictionary: torch.Tensor = None
    n_features: int
    n_functions: int
    losses: list
    criterion: Callable
    optimizer: torch.optim.Optimizer

    def __init__(
        self,
        n_features: int,
        n_functions: int,
        psi: Union[SurrogateFunction, SurrogateFunctionEnum],
        alpha: float,
        evaluation_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        logger: logging.Logger,
        momentum: float = 0,
        criterion: Union[Callable] = None,
        optimizer: torch.optim.Optimizer = None,
        parameter_hook: Optional[Callable[[dict], None]] = None,
        **kwargs
    ):
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
            momentum (float): Learning momentum.
            criterion (torch.nn.modules.loss._Loss, optional): Loss function (default: Mean Squared Error).
            optimizer (torch.optim.Optimizer, optional): Optimizer for training (default: SGD with learning rate `alpha`).
            **kwargs: Additional arguments passed to the surrogate function factory.
        """

        super(DictLayer, self).__init__()

        self.n_features = n_features
        self.n_functions = n_functions
        self.alpha = alpha
        self.momentum = momentum
        self.evaluation_func = evaluation_func

        self.psi = (
            psi
            if issubclass(type(psi), SurrogateFunction)
            else SurrogateFunctionFactory.make(
                psi,
                n_functions=n_functions,
                n_features=n_features,
                logger=logger,
                **kwargs
            )
        )

        self.losses = []
        self.theta_parameter_vector = self.psi.initialize()

        self.dictionary = None

        self.criterion = torch.nn.MSELoss() if criterion is None else criterion

        if optimizer is None:
            self.optimizer = torch.optim.SGD(
                self.parameters(), lr=alpha, weight_decay=0, momentum=momentum
            )
        else:
            self.optimizer = optimizer(
                parameters=self.parameters(), lr=alpha, weight_decay=0,momentum=momentum
            )


        self.parameter_hook = parameter_hook

    def setup(self, X: torch.Tensor) -> None:
        """
        Initialize the dictionary for the layer.

        This method uses the surrogate function (`psi`) to compute the dictionary based on
        the provided input data and the current parameter vector.

        Args:
            X (torch.Tensor): Input data of shape `(n_samples, n_features)`.

        Returns:
            None
        """
        if self.dictionary is None:
            self.dictionary = self.psi.__call__(X.mT, self.theta_parameter_vector)

    def partial_fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        h: torch.Tensor,
        epochs: int,
        dictionary_shape: tuple = None,
        rho_flag: bool = False,
        mu_flag: bool = False,
        log_losses: bool = True,
    ) -> None:
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
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            self.dictionary = self.forward(
                X, dictionary_shape, rho_flag, mu_flag
            )

            # y_pred = torch.bmm(self.dictionary, h).squeeze(-1).flatten()
            y_pred = self.evaluation_func(self.dictionary, h)

            loss = self.criterion(y_pred, y)
            loss.backward(retain_graph=False)
            self.optimizer.step()

            if log_losses:
                self.losses.append(loss.item())

            # After parameter update, if we have a hook, call it with useful info
            if self.parameter_hook is not None:
                # Create info dictionary with detached clones
                hook_info = {
                    'epoch': epoch,
                    'theta': self.theta_parameter_vector.clone().detach(),
                    'loss': loss.item(),
                    'mu': self.theta_parameter_vector[-self.n_features:].clone().detach(),
                    'rho': self.theta_parameter_vector[:-self.n_features].clone().detach(),
                    'mu_flag': mu_flag,
                    'rho_flag': rho_flag
                }
                self.parameter_hook(hook_info)

    def forward(
        self,
        X: torch.Tensor,
        dictionary_shape: tuple = None,
        rho_flag: bool = False,
        mu_flag: bool = False,
    ):
        """
        Compute the forward pass for the dictionary layer.

        This method applies the surrogate function (`psi`) to compute the evaluated dictionary
        based on the input data and learnable parameters. If block-based partitioning is used,
        the dictionary is reshaped accordingly.

        Args:
            dictionary_shape:
            X (torch.Tensor): Input data of shape (n_samples, n_features).
            rho_flag (bool): Whether the rho values will be updated or not.
            mu_flag (book): Whether the mu values will be updated or not.
        Returns:
            torch.Tensor: Evaluated dictionary of shape `(n_samples, n_functions)`,
            or reshaped to `(active_blocks_count, max_points_in_block, n_functions)` if partitioning is applied.
        """
        evaluated_dictionary = self.psi.__call__(
            X.mT, self.theta_parameter_vector, rho_flag, mu_flag
        )
        return evaluated_dictionary if not dictionary_shape else evaluated_dictionary.view(dictionary_shape)

