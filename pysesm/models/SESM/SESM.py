import logging
import time
import numpy as np
import torch
from typing import Union, Callable

from pysesm.enums import SurrogateFunctionEnum, EvaluationFuncEnum
from pysesm.validation import validate_sesm_partial_fit
from pysesm.functions import SurrogateFunction
from pysesm.models.DictLayer import DictLayer
from pysesm.models.ISTALayer import ISTALayer


class SESM(torch.nn.Module):
    """
    A custom PyTorch module for implementing a surrogate model using the SESM (Sparse-Encoded Surrogate Model) architecture.

    The SESM architecture is designed for surrogate modeling and function approximation tasks, leveraging sparse encoding
    techniques and dictionary-based representations for efficient training and inference. This module provides the core
    functionalities for building and training an SESM model, including the integration of ISTA (Iterative Shrinkage-Thresholding
    Algorithm) and dictionary learning techniques.

    Attributes:
        ista_layer (ISTALayer):
            A PyTorch module responsible for managing and adjusting sparse representations through the ISTA algorithm.
            This layer is crucial for learning the sparse encodings during training.

        dictionary_layer (DictLayer):
            A PyTorch module that manages the dictionary words or functions parameters, which are learned and adjusted during
            training to represent the function approximations more effectively.

        n_features (int):
            The number of input features or dimensions that the model will work with. Each input tensor `X` is expected
            to have this many features.

        model_epochs (int):
            The number of epochs used for training the overall SESM model, including both dictionary learning and sparse
            representation adjustments.

        ista_epochs (int):
            The number of epochs dedicated specifically to training the ISTA layer, focusing on the sparse encoding.

        ista_alpha (float):
            The learning rate for the ISTA layer, controlling how much to adjust the sparse encodings during training.

        ista_lambd (float):
            The regularization parameter for the ISTA layer, which controls the sparsity of the learned representations.

        dictionary_alpha (float):
            The learning rate for the dictionary layer, which influences how the dictionary parameters are updated during
            training.

        dictionary_momentum (float):
            Momentum to be used by the optimizer in the dictionary layer.

        seed (int):
            The random seed used for initialization and reproducibility of training processes, including random weight
            initialization and other stochastic processes.

        debug (bool):
            A boolean flag that, when set to `True`, enables additional debugging output during model training and evaluation.
            This helps track internal processes and identify issues during development.

        logger (logging.Logger):
            A custom logger instance used to record runtime information during the execution of the model.
            This enables detailed tracking of key events, debugging, and monitoring during model use.

        loss_stats (dict):
            A dictionary that tracks the history of loss statistics during the training process. It contains:
                - 'loss_mean' (list): A history of the mean loss values over the epochs.
                - 'loss_std' (list): A history of the standard deviation of loss values.
                - 'loss_max' (list): A history of the maximum loss values across epochs.
                - 'loss_min' (list): A history of the minimum loss values across epochs.

        elapsed_time (float):
            The total time (in seconds) consumed by the fitting process of the SESM model.

        ## TODO: GAUSSIAN FUNCTION SPECIFIC ATTRIBUTES THAT MUST BE ABSTRACTED

        mu_epochs (int):
            The number of epochs dedicated to adjusting the `μ` parameter in the dictionary layer. This parameter helps
            define the dictionary's characteristics.

        rho_epochs (int):
            The number of epochs focused on adjusting the `ρ` parameter in the dictionary layer. This parameter influences
            how well the dictionary represents the data.
    """

    # Type hints for instance attributes: (not class attributes)
    ista_layer: ISTALayer
    dictionary_layer: DictLayer
    n_features: int
    model_epochs: int
    ista_epochs: int
    ista_alpha: float
    ista_lambd: float
    dictionary_alpha: float
    dictionary_momentum: float
    seed: int
    debug: bool
    logger: logging.Logger
    evaluation_fun: Callable
    loss_stats: dict
    elapsed_time: float
    partial_fit_count: int
    evaluation_func_registry: dict
    mu_epochs: int
    rho_epochs: int


    # Constant class attributes
    evaluation_func_registry: dict[
        EvaluationFuncEnum, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ] = {
        EvaluationFuncEnum.BMM_MULT: lambda dictionary, h: torch.bmm(dictionary, h)
        .squeeze(-1)
        .flatten(),
        EvaluationFuncEnum.TWOD_MULT: lambda dictionary, h: torch.matmul(dictionary, h),
        EvaluationFuncEnum.DEFAULT: lambda dictionary, h: torch.matmul(dictionary, h),
    }

    def __init__(
            self,
            n_features: int,
            n_functions: int,
            psi: Union[SurrogateFunction, SurrogateFunctionEnum],
            model_epochs: int,
            ista_epochs: int,
            ista_alpha: float,
            ista_lambd: float,
            dictionary_alpha: float,
            dictionary_momentum: float,
            mu_epochs: int,
            rho_epochs: int,
            seed: int,
            logger: logging.Logger,
            debug: bool = False,
            evaluation_func: EvaluationFuncEnum = EvaluationFuncEnum.DEFAULT,
            **kwargs
    ):
        """
        Initializes the SESM model with the given parameters.

        Args:
            n_features (int): The number of input features or dimensions.
                Represents the number of variables the model works with for each input point.
                For instance, if `n_features=3`, each input point has three values.

            n_functions (int):
                The number of individual functions available in the internal dictionary.
                These functions can be combined linearly as part of the SESM framework to approximate complex
                surrogate behaviors.

            psi (Union[SurrogateFunction, SurrogateFunctionEnum]):
                This parameter defines the surrogate function used in the model. It can either be:
                - A `SurrogateFunctionEnum`, which is a predefined enum that specifies the type of surrogate
                  function to use. In this case, the constructor will map the enum to the corresponding class
                  and initialize it with the provided `kwargs`.
                - A `SurrogateFunction` class (or object), in which case the constructor will directly assign it to `self.psi`.

            model_epochs (int):
                The number of epochs used for training the overall SESM model, including both dictionary learning and sparse
                representation adjustments.

            ista_epochs (int):
                The number of epochs dedicated specifically to training the ISTA layer, focusing on the sparse encoding.

            ista_alpha (float):
                The learning rate for the ISTA layer, controlling how much to adjust the sparse encodings during training.

            ista_lambd (float):
                The regularization parameter for the ISTA layer, which controls the sparsity of the learned representations.

            dictionary_alpha (float):
                The learning rate for the dictionary layer, which influences how the dictionary parameters are updated during
                training.

            dictionary_momentum (float):
                Momentum to be used by the optimizer in the dictionary layer.
        
            mu_epochs (int):
                The number of epochs dedicated to adjusting the `μ` parameter in the dictionary layer. This parameter helps
                define the dictionary's characteristics.

            rho_epochs (int):
                The number of epochs focused on adjusting the `ρ` parameter in the dictionary layer. This parameter influences
                how well the dictionary represents the data.

            seed (int):
                The random seed used for initialization and reproducibility of training processes, including random weight
                initialization and other stochastic processes.

            debug (bool):
                A boolean flag that, when set to `True`, enables additional debugging output during model training and evaluation.
                This helps track internal processes and identify issues during development.

            logger (logging.Logger):
                A custom logger instance used to record runtime information during the execution of the model.
                This enables detailed tracking of key events, debugging, and monitoring during model use.

            **kwargs: Additional keyword arguments passed to the constructor of the `SurrogateFunction` class
                      (if `psi` is a `SurrogateFunction` class). These kwargs can be used to specify configuration
                      parameters specific to the surrogate function being used.
        """

        super(SESM, self).__init__()

        self.n_features = n_features
        self.n_functions = n_functions
        self.model_epochs = model_epochs
        self.ista_alpha = ista_alpha
        self.ista_epochs = ista_epochs
        self.ista_lambd = ista_lambd
        self.mu_epochs = mu_epochs
        self.rho_epochs = rho_epochs
        self.dictionary_alpha = dictionary_alpha
        self.dictionary_momentum = dictionary_momentum
        self.seed = seed
        self.debug = debug
        self.logger = logger
        self.evaluation_func = self.evaluation_func_registry[evaluation_func]

        if self.seed is not None and self.seed != "None":
            torch.manual_seed(self.seed)

        self.losses_ISTA = []
        self.losses_Dictionary = []
        self.elapsed_time = 0

        self.loss_stats = {
            "loss_mean": [],
            "loss_std": [],
            "loss_max": [],
            "loss_min": [],
        }

        # Instantiate ISTA Layer
        self.ista_layer = ISTALayer(
            n_functions=n_features,
            alpha=self.ista_alpha,
            lambd=self.ista_lambd,
            evaluation_func=self.evaluation_func,
            logger=logger
        )

        # Instantiate Dictionary Layer
        self.dictionary_layer = DictLayer(
            n_features=n_features,
            n_functions=n_functions,
            alpha=self.dictionary_alpha,
            momentum=self.dictionary_momentum,
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

    def fit(self, X: torch.Tensor, y: torch.Tensor, dictionary_shape: tuple = None, h: torch.Tensor = None):
        """
        Trains the model by learning a sparse vector and a dictionary that approximates the original function.
        This method updates the weights with each call, improving the representation of the target function.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features), where `n_samples` is the number of
                              samples and `n_features` is the number of features for each sample.
            y (torch.Tensor): Target data of shape (n_samples,), representing the values corresponding to the input data.
            dictionary_shape (tuple, optional): Specifies the shape of the evaluated dictionary before
                                                computing the loss. If not provided, the default shape is used.
            h (torch.Tensor, optional): Custom sparse vector used to initialize the ISTA layer. If not provided,
                                         the ISTA layer initializes with a random vector.
        """

        self.dictionary_layer.setup(X)
        self.ista_layer.setup(h)

        for epoch in range(self.model_epochs):
            epoch_start_time = time.time()

            self.forward(X, y, dictionary_shape)

            epoch_end_time = time.time()
            self.elapsed_time += epoch_end_time - epoch_start_time

            logging.info(
                "Epoch {} - Fit: Loss ISTA Layer: {:.6f}, Loss Dictionary Layer: {:.6f}".format(
                    epoch + 1,
                    self.ista_layer_losses[-1],
                    self.dictionary_layer_losses[-1],
                )
            )

    def partial_fit(
            self,
            X: torch.Tensor,
            y: torch.Tensor,
            dictionary_shape: tuple = None,
    ) -> None:
        """
        Perform partial training on the SESM model.

        Updates the ISTA and dictionary layers without reinitializing their weights, enabling iterative refinement.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features), where `n_samples` is the number of
                              samples and `n_features` is the number of features for each sample.
            y (torch.Tensor): Target data of shape (n_samples,), representing the values corresponding to the input data.
            dictionary_shape (tuple, optional): Specifies the shape of the evaluated dictionary before
                                                computing the loss. If not provided, the default shape is used.

        Returns:
            None
        """

        # Ensure y is 2D
        if y.dim() == 1:
            y = y.unsqueeze(-1)

        validate_sesm_partial_fit(self, X, y)

        self.dictionary_layer.setup(X)

        # Add shape validation here before the training loop
        assert self.ista_layer.h.dim() == 2, \
            f"ISTA layer h parameter must be 2D tensor, got {self.ista_layer.h.shape}"



        for epoch in range(self.model_epochs):
            epoch_start_time = time.time()

            self.train_step(X, y, dictionary_shape)

            self.elapsed_time += time.time() - epoch_start_time

            logging.info(
                "Epoch {} - Partial Fit: Loss ISTA: {:.6f}, Loss Dictionary: {:.6f}".format(
                    epoch + 1,
                    self.ista_layer_losses[-1],
                    self.dictionary_layer_losses[-1],
                )
            )

        self.losses_ISTA = self.ista_layer.losses
        self.losses_Dictionary = self.dictionary_layer.losses


    def train_step(
            self,
            X: torch.Tensor,
            y: torch.Tensor,
            dictionary_shape: tuple = None,
    ):
        """
        Perform a forward pass through the SESM model.

        Updates the dictionary and sparse vector using the ISTA and dictionary layers.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features), where `n_samples` is the number of
                              samples and `n_features` is the number of features for each sample.
            y (torch.Tensor): Target data of shape (n_samples,), representing the values corresponding to the input data.
            dictionary_shape (tuple, optional): Specifies the shape of the evaluated dictionary before
                                                computing the loss. If not provided, the default shape is used.
        """

        # Detach h before dictionary optimization to prevent unwanted gradient flows
        h_detached = self.ista_layer.h.clone().detach()

        self.dictionary_layer.partial_fit(
            X=X,
            y=y,
            epochs=self.mu_epochs,
            h=h_detached,
            mu_flag=True,
            dictionary_shape=dictionary_shape,
        )

        self.loss_analysis(self.mu_epochs)

        self.dictionary_layer.partial_fit(
            X=X,
            y=y,
            epochs=self.rho_epochs,
            h=h_detached,
            rho_flag=True,
            dictionary_shape=dictionary_shape,
        )

        self.loss_analysis(self.rho_epochs)

         # Detach dictionary before passing to ISTA layer
        dictionary_for_ista = self.dictionary_layer.dictionary.detach()
    

        self.ista_layer.partial_fit(
            y=y, 
            epochs=self.ista_epochs, 
            dictionary=dictionary_for_ista
        )

    def predict(
            self,
            X: torch.Tensor,
            dictionary_shape: tuple = None,
            custom_h: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Predict outputs using the trained SESM model.

        Generates predictions based on the learned dictionary and sparse vector.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features), where `n_samples` is the number of
                              samples and `n_features` is the number of features for each sample.
            dictionary_shape (tuple, optional): Specifies the shape of the evaluated dictionary before
                                                computing the loss. If not provided, the default shape is used.
            custom_h (torch.Tensor, optional): Custom sparse vector to be used on the prediction calculation. If not provided,
                                         the model uses the current one.

        Returns:
            torch.Tensor: Predicted values of shape `(n_samples,)`.
        """

        with torch.no_grad():
            self.dictionary_layer.dictionary = self.dictionary_layer.forward(X, dictionary_shape)

        dictionary = self.dictionary_layer.dictionary
        h = custom_h if custom_h is not None else self.ista_layer.h

        return self.evaluation_func(dictionary, h)

    def loss_analysis(self, dict_epochs: int) -> None:
        """
        Analyzes and stores statistical information about the losses from the dictionary layer
        over the specified number of recent epochs.

        Args:
            dict_epochs (int): The number of most recent epochs to consider for loss analysis.

        Returns:
            None: The method updates the `loss_stats` dictionary in place, adding the computed
                  mean, standard deviation, maximum, and minimum loss values.
        """
        current_loss = self.dictionary_layer.losses[-dict_epochs:]
        self.loss_stats["loss_mean"].append(np.mean(current_loss))
        self.loss_stats["loss_std"].append(np.std(current_loss))
        self.loss_stats["loss_max"].append(np.max(current_loss))
        self.loss_stats["loss_min"].append(np.min(current_loss))
