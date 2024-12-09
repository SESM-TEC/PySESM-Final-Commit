import logging

from pysesm.customization_factories import SurrogateFunctionFactory
from pysesm.functions import SurrogateFunction
from pysesm.enums import SurrogateFunctionEnum

import torch
from typing import Union, Callable


class DictLayer(torch.nn.Module):
    """
    A custom PyTorch module for implementing a dictionary layer with learnable parameters.

    This layer is designed for use in surrogate modeling and function approximation tasks.

    Attributes:
        psi (SurrogateFunction): The function used for generating the layer's output.
        theta_parameter_vector (torch.nn.Parameter): Learnable parameter tensor for the layer's functions.
        dictionary (torch.Tensor): The computed output of the layer.
        n_features (int): The number of input features or dimensions.
        n_functions (int): The number of functions or basis functions.
        losses (list): Losses history if log_losses parameter is set to ture
        criterion (torch.nn.modules.loss._Loss): Loss function used to compute the loss of the model.
        optimizer (torch.optim.Optimizer): Preferred optimizer used in the training of the model.
    """

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
        seed: int,
        logger: logging.Logger,
        criterion: Union[Callable] = None,
        optimizer: torch.optim.Optimizer = None,
        **kwargs
    ):
        """
        Method

        Args:
            n_samples:
            psi:
            alpha:
            criterion:
            optimizer:
        """
        super(DictLayer, self).__init__()

        self.n_features = n_features
        self.n_functions = n_functions
        self.alpha = alpha
        self.evaluation_func = evaluation_func

        self.psi = (
            psi
            if issubclass(type(psi), SurrogateFunction)
            else SurrogateFunctionFactory.make(
                psi,
                n_functions=n_functions,
                n_features=n_features,
                seed=seed,
                logger=logger,
                **kwargs
            )
        )

        self.losses = []
        self.theta_parameter_vector = self.psi.initialize()

        if criterion is None:
            self.criterion = torch.nn.MSELoss()
        else:
            self.criterion = criterion

        if optimizer is None:
            self.optimizer = torch.optim.SGD(self.parameters(), lr=alpha)
        else:
            self.optimizer = optimizer

    def setup(self, X: torch.Tensor) -> None:
        """
        Method that initializes the dictionary of the layer.
        It needs an input value to do so, so it can be done at __init__.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).
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
        Method that does a partial fit on the model without redefining the weights of its layers.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).
            y (torch.Tensor): Target data of shape (n_samples,).
            h (torch.nn.Parameter): Sparse vector which holds the dictionary weights for the target function
            epochs (int): Number of training epochs for the layer.
            rho_flag (bool): Whether the rho values will be updated or not.
            mu_flag (book): Whether the mu values will be updated or not.
            log_losses (bool): Whether the losses of each epoch should be recorded.
        """

        for _ in range(epochs):
            self.dictionary = self.forward(
                X, dictionary_shape, rho_flag, mu_flag
            )

            # y_pred = torch.bmm(self.dictionary, h).squeeze(-1).flatten()
            y_pred = self.evaluation_func(self.dictionary, h)

            loss = self.criterion(y_pred, y)
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            if log_losses:
                self.losses.append(loss.item())

    def forward(
        self,
        X: torch.Tensor,
        dictionary_shape: tuple = None,
        rho_flag: bool = False,
        mu_flag: bool = False,
    ):
        """
        Method that computes the forward pass for the current layer.

        Args:
            dictionary_shape:
            X (torch.Tensor): Input data of shape (n_samples, n_features).
            rho_flag (bool): Whether the rho values will be updated or not.
            mu_flag (book): Whether the mu values will be updated or not.

        Returns:
            torch.Tensor: Input values evaluated using the psi function.
        """
        evaluated_dictionary = self.psi.__call__(
            X.mT, self.theta_parameter_vector, rho_flag, mu_flag
        )
        return evaluated_dictionary if not dictionary_shape else evaluated_dictionary.view(dictionary_shape)
