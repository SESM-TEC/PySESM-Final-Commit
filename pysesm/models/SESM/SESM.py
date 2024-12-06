from pysesm.functions import SurrogateFunction
from pysesm.enums import SurrogateFunctionEnum, EvaluationFuncEnum
from pysesm.models.DictLayer import DictLayer
from pysesm.models.ISTALayer import ISTALayer
from pysesm.validation.sesm_validation import validate_sesm_partial_fit

import logging
import torch
import numpy as np
import time
from typing import Union, Callable


class SESM(torch.nn.Module):
    """
    A PyTorch module implementing a Surrogate Ensemble Sparse Model (SESM).

    SESM combines sparse coding techniques with dictionary learning to approximate complex functions.
    It uses two key layers:
    - ISTA layer for learning sparse representations.
    - Dictionary layer for constructing a dictionary of basis functions.

    Attributes:
        n_samples (int): Number of samples in the input dataset.
        n_features (int): Number of features or input dimensions.
        n_functions (int): Number of basis functions for the dictionary.
        model_epochs (int): Number of epochs for the overall model training.
        ista_epochs (int): Number of training epochs for the ISTA layer.
        ista_alpha (float): Learning rate for the ISTA layer.
        ista_lambd (float): Regularization parameter for the ISTA layer.
        mu_epochs (int): Epochs for updating the `mu` parameter in the dictionary layer.
        rho_epochs (int): Epochs for updating the `rho` parameter in the dictionary layer.
        dictionary_alpha (float): Learning rate for the dictionary layer.
        weight_decay (float): Weight decay penalty to reduce overfitting.
        seed (int): Random seed for reproducibility.
        debug (bool): If True, enables debug logging.
        logger (logging.Logger): Logger instance for logging events.
        evaluation_func (Callable): Function for evaluating predictions based on the dictionary and sparse vector.
        ista_layer (ISTALayer): Module responsible for learning sparse vectors.
        dictionary_layer (DictLayer): Module responsible for learning the dictionary.
        loss_stats (dict): Statistics about dictionary losses during training, with keys:
            - `loss_mean` (list): Mean loss values for each epoch.
            - `loss_std` (list): Standard deviation of loss values for each epoch.
            - `loss_max` (list): Maximum loss values for each epoch.
            - `loss_min` (list): Minimum loss values for each epoch.
        elapsed_time (float): Total time consumed during the model fitting process.
    """
    evaluation_func_registry: dict[EvaluationFuncEnum, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
        EvaluationFuncEnum.BMM_MULT: lambda dictionary, h: torch.bmm(dictionary, h).squeeze(-1).flatten(),
        EvaluationFuncEnum.TWOD_MULT: lambda dictionary, h: torch.matmul(dictionary, h),
        EvaluationFuncEnum.DEFAULT: lambda dictionary, h: torch.matmul(dictionary, h)
    }

    def __init__(self, n_samples: int, n_features: int, n_functions:int, psi: Union[SurrogateFunction, SurrogateFunctionEnum], model_epochs: int,
                 ista_epochs: int, ista_alpha: float, ista_lambd: float, mu_epochs: int, rho_epochs: int,
                 dictionary_alpha: float, weight_decay: float,  seed: int, logger:logging.Logger, debug: bool = False,  h=None, evaluation_func: EvaluationFuncEnum = EvaluationFuncEnum.DEFAULT, **kwargs):
        """
        Initialize the SESM model.

        Sets up the ISTA and dictionary layers, along with evaluation functions and training parameters.

        Args:
            n_samples (int): Number of samples in the input dataset.
            n_features (int): Number of input dimensions.
            n_functions (int): Number of basis functions in the dictionary.
            psi (Union[SurrogateFunction, SurrogateFunctionEnum]): Function for generating the dictionary.
            model_epochs (int): Number of training epochs for the model.
            ista_epochs (int): Number of epochs for the ISTA layer.
            ista_alpha (float): Learning rate for the ISTA layer.
            ista_lambd (float): Regularization parameter for the ISTA layer.
            mu_epochs (int): Epochs for updating the `mu` parameter in the dictionary layer.
            rho_epochs (int): Epochs for updating the `rho` parameter in the dictionary layer.
            dictionary_alpha (float): Learning rate for the dictionary layer.
            weight_decay (float): Regularization term to prevent overfitting.
            seed (int): Random seed for reproducibility.
            logger (logging.Logger): Logger for logging events.
            debug (bool, optional): If True, enables debug logging (default: False).
            h (torch.Tensor, optional): Initial sparse vector for the ISTA layer (default: None).
            evaluation_func (EvaluationFuncEnum, optional): Evaluation function for predictions (default: EvaluationFuncEnum.DEFAULT).
            **kwargs: Additional parameters for the dictionary layer.
        """

        super(SESM, self).__init__()

        self.n_samples = n_samples
        self.n_features = n_features
        self.n_functions = n_functions
        self.model_epochs = model_epochs
        self.ista_alpha = ista_alpha
        self.ista_epochs = ista_epochs
        self.ista_lambd = ista_lambd
        self.mu_epochs = mu_epochs
        self.rho_epochs = rho_epochs
        self.dictionary_alpha = dictionary_alpha
        self.weight_decay = weight_decay
        self.seed = seed
        self.debug = debug
        self.logger = logger
        self.losses_ISTA = []
        self.losses_Dictionary = []
        self.elapsed_time = 0
        self.evaluation_func = self.evaluation_func_registry[evaluation_func]
        print(evaluation_func)
        print(self.evaluation_func_registry[evaluation_func])

        self.loss_stats = {
            "loss_mean": [],
            "loss_std": [],
            "loss_max": [],
            "loss_min": [],
        }

        # Instantiate model layers
        self.ista_layer = ISTALayer(
            n_functions=self.n_functions,
            random_seed=self.seed,
            alpha=self.ista_alpha,
            lambd=self.ista_lambd,
            weight_decay=self.weight_decay,
            evaluation_func=self.evaluation_func,
            h=h
        )

        self.dictionary_layer = DictLayer(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_functions=self.n_functions,
            alpha=self.dictionary_alpha,
            evaluation_func=self.evaluation_func,
            logger=logger,
            psi=psi,
            **kwargs
        )

    @property
    def ista_layer_losses(self):
        """Returns the losses for the ISTA layer and acts as an attribute due to the @property decorator."""
        return self.ista_layer.losses

    @property
    def dictionary_layer_losses(self):
        """Returns the losses for the Dictionary layer and acts as an attribute due to the @property decorator."""
        return self.dictionary_layer.losses

    def fit(self, X: torch.Tensor, y: torch.Tensor, h: torch.Tensor = None):
        """
        Train the SESM model.

        Trains both the ISTA and dictionary layers for a specified number of epochs, learning sparse representations
        and a dictionary that approximates the input-to-output mapping.

        Args:
            X (torch.Tensor): Input tensor of shape `(n_samples, n_features)`.
            y (torch.Tensor): Target tensor of shape `(n_samples,)`.
            h (torch.Tensor, optional): Initial sparse vector for the ISTA layer (default: None).

        Returns:
            None
        """
        self.dictionary_layer.initialize_layer(X)
        self.ista_layer.initialize_h_vector(h)

        for epoch in range(self.model_epochs):
            epoch_start_time = time.time()

            self.forward(X, y)

            epoch_end_time = time.time()
            self.elapsed_time += (epoch_end_time - epoch_start_time)

            logging.info(
                "Epoch {} - Fit: Loss ISTA Layer: {:.6f}, Loss Dictionary Layer: {:.6f}".format(
                    epoch + 1, self.ista_layer_losses[-1], self.dictionary_layer_losses[-1]
                )
            )

    def partial_fit(self, X: torch.Tensor, y: torch.Tensor, max_points_in_block: int = 0,
                    active_blocks_count: int = 0) -> None:
        """
        Perform partial training on the SESM model.

        Updates the ISTA and dictionary layers without reinitializing their weights, enabling iterative refinement.

        Args:
            X (torch.Tensor): Input tensor of shape `(n_samples, n_features)`.
            y (torch.Tensor): Target tensor of shape `(n_samples,)`.
            max_points_in_block (int, optional): Maximum points per block for partitioning (default: 0).
            active_blocks_count (int, optional): Number of active blocks for partitioning (default: 0).

        Returns:
            None
        """


        validate_sesm_partial_fit(self, X, y)

        if self.dictionary_layer.dictionary is None:
            logging.warning("[SESM] Initializing dictionary layer with first iteration of model")
            self.dictionary_layer.initialize_layer(X)

        for epoch in range(self.model_epochs):
            epoch_start_time = time.time()

            self.forward(X, y, max_points_in_block, active_blocks_count)

            epoch_end_time = time.time()
            self.elapsed_time += (epoch_end_time - epoch_start_time)

            logging.info(
                "Epoch {} - Partial Fit: Loss ISTA: {:.6f}, Loss Dictionary: {:.6f}".format(
                    epoch + 1, self.ista_layer_losses[-1], self.dictionary_layer_losses[-1]
                )
            )

    def forward(self, X: torch.Tensor, y: torch.Tensor, max_points_in_block: int = 0,
                active_blocks_count: int = 0) -> None:
        """
        Perform a forward pass through the SESM model.

        Updates the dictionary and sparse vector using the ISTA and dictionary layers.

        Args:
            X (torch.Tensor): Input tensor of shape `(n_samples, n_features)`.
            y (torch.Tensor): Target tensor of shape `(n_samples,)`.
            max_points_in_block (int, optional): Maximum points per block for partitioning (default: 0).
            active_blocks_count (int, optional): Number of active blocks for partitioning (default: 0).

        Returns:
            None
        """


        self.dictionary_layer.partial_fit(
            X=X,
            y=y,
            epochs=self.mu_epochs,
            h=self.ista_layer.h,
            mu_flag=True,
            max_points_in_block=max_points_in_block,
            active_blocks_count=active_blocks_count
        )

        self.loss_analysis(self.mu_epochs)

        self.dictionary_layer.partial_fit(
            X=X,
            y=y,
            epochs=self.rho_epochs,
            h=self.ista_layer.h,
            rho_flag=True,
            max_points_in_block=max_points_in_block,
            active_blocks_count=active_blocks_count
        )

        self.loss_analysis(self.rho_epochs)

        self.ista_layer.partial_fit(
            y=y,
            epochs=self.ista_epochs,
            dictionary=self.dictionary_layer.dictionary
        )

        self.losses_ISTA.append(self.ista_layer.losses[-1])
        self.losses_Dictionary.append(self.dictionary_layer.losses[-1])

    def predict(self, X: torch.Tensor, max_points_in_block: int = 0, active_blocks_count: int = 0,
                custom_ista_layer: ISTALayer = None) -> torch.Tensor:
        """
        Predict outputs using the trained SESM model.

        Generates predictions based on the learned dictionary and sparse vector.

        Args:
            X (torch.Tensor): Input tensor of shape `(n_samples, n_features)`.
            max_points_in_block (int, optional): Maximum points per block for partitioning (default: 0).
            active_blocks_count (int, optional): Number of active blocks for partitioning (default: 0).
            custom_ista_layer (ISTALayer, optional): A custom ISTA layer for prediction (default: None).

        Returns:
            torch.Tensor: Predicted values of shape `(n_samples,)`.
        """


        if custom_ista_layer is not None:
            self.ista_layer = custom_ista_layer

        with torch.no_grad():
            self.dictionary_layer.dictionary = self.dictionary_layer.forward(X, max_points_in_block,
                                                                             active_blocks_count)

        dictionary = self.dictionary_layer.dictionary.double()
        h = self.ista_layer.h.double()

        return self.evaluation_func(dictionary, h)

    def loss_analysis(self, dict_epochs: int) -> None:
        """
        Analyze and record loss statistics for the dictionary layer.

        Args:
            dict_epochs (int): Number of epochs used for dictionary training.

        Returns:
            None
        """
        current_loss = self.dictionary_layer.losses[-dict_epochs:]
        self.loss_stats["loss_mean"].append(np.mean(current_loss))
        self.loss_stats["loss_std"].append(np.std(current_loss))
        self.loss_stats["loss_max"].append(np.max(current_loss))
        self.loss_stats["loss_min"].append(np.min(current_loss))
