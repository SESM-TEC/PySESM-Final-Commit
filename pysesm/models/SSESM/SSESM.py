'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica

SSESM Base Class

Provides the basic functionality of the Sequential SESM.

Authors: The SESM Team 

License: 
'''


import logging
import numpy as np
import torch
from pysesm.functions import SurrogateFunction
from pysesm.blocks import UniformPartitionManager
from pysesm.models.SESM.SESM import SESM
from typing import Callable, Iterator, Optional
from sklearn.metrics import mean_squared_error
from pysesm.device_manager.DeviceManager import DeviceManager
from pysesm.customization_factories.ISTALayerFactory import ISTALayerFactory
from pysesm.enums.ISTALayerEnum import ISTALayerEnum
class SSESM(SESM):
    """
    A PyTorch module extending the SESM architecture to implement a surrogate model
    using a sequential and block-partitioned approach. This class is designed for
    function approximation and surrogate modeling tasks with dynamic sub-block partitioning.

    """

    def __init__(
        self,
        n_features: int,
        n_functions: int,
        model_epochs: int,
        ista_epochs: int,
        rho_epochs: int,
        mu_epochs: int,
        ista_alpha: float,
        ista_lambd: float,
        dictionary_alpha: float,
        psi: SurrogateFunction,
        permutation_times: int,
        dfngroup,
        seed: int,
        logger: logging.Logger,
        dictionary_optimizer: Callable[[Iterator[torch.nn.Parameter], float], torch.optim.Optimizer] = None,
        ista_optimizer: Callable[[Iterator[torch.nn.Parameter],float], torch.optim.Optimizer] = None,        
        iter: int = 0,
        initial_bounds=None,
        debug=True,
        device_map=None,
        
        ista_layer_type: ISTALayerEnum = None,

        dict_layer_hook: Optional[Callable[[dict], None]] = None,
        ista_layer_hook: Optional[Callable[[dict], None]] = None,   
        sesm_hook: Optional[Callable[[dict], None]] = None,  
        
        **kwargs
    ):
        """
        Initialize the SSESM model with a sequential, block-based approach.

        This constructor sets up the surrogate model, initializes the partition manager for
        sub-block operations, and configures hyperparameters for ISTA and dictionary layers.

        Args:
            n_samples (int): Number of samples in the dataset.
            n_features (int): Number of input features.
            n_functions (int): Number of latent functions used in the model.
            eig_range: Eigenvalue range for the surrogate function's parameter initialization.
            mu_range: Range for the mu parameter in dictionary learning.
            model_epochs (int): Number of epochs for the overall model training.
            ista_epochs (int): Number of epochs for training the ISTA layer.
            rho_epochs (int): Number of epochs for adjusting the rho parameter.
            mu_epochs (int): Number of epochs for adjusting the mu parameter.
            ista_alpha (float): Learning rate for the ISTA layer.
            ista_lambd (float): Regularization parameter for the ISTA layer.
            dictionary_alpha (float): Learning rate for the dictionary layer.
            surrogate_function (SurrogateFunction): Function used to create the dictionary for modeling.
            permutation_times (int): Number of times to permute the dataset for training.
            dfngroup: Grouping information for function blocks (specific to implementation).
            iter (int, optional): Experiment iteration count (default: 1).
            seed (int): Random seed for reproducibility.
            logger (logging.Logger): Logger instance for runtime monitoring.
            dictionary_optimizer (lambda): factory to build the dictionary optimizer
            ista_optimizer (lambda): factor to build the ISTA optimizer
            T (list[int]): Scaling factors for normalization.
            initial_bounds (optional): Initial bounds for partitioning (default: None).
            debug (bool, optional): Enables or disables debug mode (default: True).
            **kwargs: Additional keyword arguments passed to the base class.
        """
        self.ista_layer_hook = ista_layer_hook
        self.ista_layer_type = ista_layer_type

        self.device_manager = DeviceManager(logger, device_map=device_map)
        self.permutation_times = permutation_times
        self.dfngroup = dfngroup
        self.partition_manager = UniformPartitionManager(
            logger, kwargs.get("T"), 
            n_functions=n_functions, 
            initial_bounds=initial_bounds,
            device_manager=self.device_manager)

        super().__init__(
            n_features=n_features,
            n_functions=n_functions,
            psi=psi,
            seed=seed,
            model_epochs=model_epochs,
            ista_epochs=ista_epochs,
            ista_alpha=ista_alpha,
            ista_lambd=ista_lambd,
            dictionary_alpha=dictionary_alpha,
            mu_epochs=mu_epochs,
            rho_epochs=rho_epochs,
            logger=logger,
            dictionary_optimizer=dictionary_optimizer,
            ista_optimizer=ista_optimizer,
            debug=debug,
            device_manager=self.device_manager,
            sesm_hook = sesm_hook,
            dict_layer_hook = dict_layer_hook,
            ista_layer_hook = ista_layer_hook,
            ista_layer_type = ista_layer_type,
            **kwargs
        )

    def partial_fit(self, X: torch.Tensor, y: torch.Tensor, initial_h: torch.Tensor = None, *_):
        """
        Perform a partial fit on the model, iteratively updating parameters using active sub-blocks.

        The method permutes the dataset and processes each block sequentially to update the
        ISTA layer. Active blocks are dynamically retrieved and used to compute the partial fit.

        Args:
            X (torch.Tensor): Input features for training.
            y (torch.Tensor): Target values.
            initial_h (torch.Tensor): Initial h value or None for random initialization
            *_: Additional unused positional arguments.

        Returns:
            None
        """
        # Ensure y is 2D
        if y.dim() == 1:
            y = y.unsqueeze(-1)

        self.partition_manager.add_points(X, y)
        self.partition_manager.init_ista_per_block(
            n_functions=self.n_functions,
            ista_alpha=self.ista_alpha,
            ista_lambd=self.ista_lambd,
            ista_optimizer=self.ista_optimizer,
            evaluation_func=self.evaluation_func,
            initial_h=initial_h,
            ista_layer_hook=self.ista_layer_hook,
            ista_layer_type=self.ista_layer_type
        )
        active_blocks = self.partition_manager.retrieve_active_blocks()

        for _ in range(self.permutation_times):
            selected_indices = np.random.permutation(len(active_blocks))
            permuted_list_sub_blocks = [active_blocks[i] for i in selected_indices]
            for block in permuted_list_sub_blocks:
                self.ista_layer = block.ista_layer

                # DEBUG: Checking state of optimizer here
                # print("Optimizer state:", [p.shape for p in self.ista_layer.optimizer.param_groups[0]['params']])

                X_torch = block.normalized_X.clone().detach().requires_grad_(False)
                super().partial_fit(X_torch, block.target)

    def predict(self, X, y, *_):
        """
        Predict the output using the trained SSESM model with active sub-blocks.

        The prediction process involves:
        1. Retrieving active sub-blocks from the partition manager.
        2. Making predictions for each sub-block using the associated ISTA layer.
        3. Aggregating predictions to reconstruct the final output vector.

        Args:
            X (torch.Tensor): Input features for prediction.
            y (torch.Tensor): Target values, used to identify active blocks.
            *_: Additional unused positional arguments.

        Returns:
            torch.Tensor: Predicted values for the input dataset.
        """

        active_blocks = self.partition_manager.retrieve_test_active_blocks(X, y)

        y_pred_per_block = [0 for _ in range(len(y))]
        for block in active_blocks:
            X_torch = block.normalized_X.clone().detach()
            # super().predict does linear combination of dict and h.  anti-squeeze needed.
            block_pred = super().predict(X_torch, custom_h=block.ista_layer.h)/block.amplitude
            for i, pos in enumerate(block.positions):
                y_pred_per_block[pos] = block_pred[i]

        return torch.tensor(y_pred_per_block, dtype=X.dtype)

    def performance_stats(self, X: torch.Tensor, y: torch.Tensor):
        """
        Evaluate the model's performance on a given dataset using active sub-blocks.

        The method computes predictions, tracks the elapsed training time, and calculates
        the Mean Squared Error (MSE) between the predictions and true values.

        Args:
            X (torch.Tensor): Input features for evaluation.
            y (torch.Tensor): Target values.

        Returns:
            tuple: A tuple containing:
                - y_pred (torch.Tensor): Predicted values for the dataset.
                - time (float): Total elapsed time for training (in minutes).
                - mse (float): Mean Squared Error between predictions and true target values.
        """

        y_pred = self.predict(X, y)
        time = self.elapsed_time / 60
        mse = mean_squared_error(y_pred.clone().detach(), y)
        return y_pred, time, mse
