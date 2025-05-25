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
from typing import Callable, Iterator, Optional
from sklearn.metrics import mean_squared_error

from ..functions import SurrogateFunction
from ..blocks import UniformPartitionManager
from ..models.SESM.SESM import SESM, SESMConfig
from ..models.SparseCodingBaseLayer import SparseCodingBaseLayer, SparseCodingConfig
from ..device_manager.DeviceManager import DeviceManager
from ..factories.SparseCodingFactory import SparseCodingFactory


class SSESM(SESM):
    """
    A PyTorch module extending the SESM architecture to implement a surrogate model
    using a sequential and block-partitioned approach. This class is designed for
    function approximation and surrogate modeling tasks with dynamic sub-block partitioning.
    """

    def __init__(
        self,
        config: SESMConfig,
        logger: logging.Logger,
        device_manager: DeviceManager,
        permutation_times: int,
        dfngroup,
        initial_bounds=None,
        T=None,
        dict_layer_hook: Optional[Callable[[dict], None]] = None,
        sparse_coding_layer_hook: Optional[Callable[[dict], None]] = None,
        sesm_hook: Optional[Callable[[dict], None]] = None,
        **kwargs
    ):
        """
        Initialize the SSESM model with a sequential, block-based approach.

        This constructor sets up the surrogate model, initializes the partition manager for
        sub-block operations, and configures hyperparameters.

        Args:
            config (SESMConfig): Configuration object containing all SESM parameters.
            logger (logging.Logger): Logger instance for runtime monitoring.
            device_manager (DeviceManager): Device manager for GPU/CPU allocation.
            permutation_times (int): Number of times to permute the dataset for training.
            dfngroup: Grouping information for function blocks (specific to implementation).
            initial_bounds (optional): Initial bounds for partitioning (default: None).
            T: Number of blocks per dimension or total blocks.
            dict_layer_hook: Optional callback for dictionary layer monitoring.
            sparse_coding_layer_hook: Optional callback for sparse coding layer monitoring.
            sesm_hook: Optional callback for SESM-level monitoring.
            **kwargs: Additional keyword arguments.
        """
        # Initialize parent SESM
        super().__init__(
            config=config,
            logger=logger,
            device_manager=device_manager,
            sesm_hook=sesm_hook,
            dict_layer_hook=dict_layer_hook,
            sparse_coding_layer_hook=sparse_coding_layer_hook,
            **kwargs
        )

        self.permutation_times = permutation_times
        self.dfngroup = dfngroup

        # Create partition manager
        self.partition_manager = UniformPartitionManager(
            logger=logger, 
            T=T, 
            n_functions=config.n_functions, 
            initial_bounds=initial_bounds,
            device_manager=device_manager,
            sparse_coding_layer_hook=sparse_coding_layer_hook
        )


    def evaluation_func(self, dictionary: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Concrete implementation of the evaluation function for SSESM.
        This performs standard 2D matrix multiplication.
        """
        return torch.matmul(dictionary, h)    

    
    def partial_fit(self, X: torch.Tensor, y: torch.Tensor, initial_h: torch.Tensor = None, *_):
        """
        Perform a partial fit on the model, iteratively updating parameters using active sub-blocks.

        The method permutes the dataset and processes each block sequentially to update the
        sparse coding layer. Active blocks are dynamically retrieved and used to compute the partial fit.

        Args:
            X (torch.Tensor): Input features for training.
            y (torch.Tensor): Target values.
            initial_h (torch.Tensor): Initial h value or None for random initialization.
            *_: Additional unused positional arguments.

        Returns:
            None
        """
        # Ensure y is 2D
        if y.dim() == 1:
            y = y.unsqueeze(-1)

        # Add points to partition manager and initialize blocks
        self.partition_manager.add_points(X, y)
        self.partition_manager.init_sparse_coding_per_block(config=self.sparse_coding_config)
        active_blocks = self.partition_manager.retrieve_active_blocks()

        # Train blocks with permutation
        for permutation in range(self.permutation_times):
            selected_indices = np.random.permutation(len(active_blocks))
            permuted_blocks = [active_blocks[i] for i in selected_indices]
            
            for block in permuted_blocks:
                # Use the new clean interface - pass the whole block
                super().partial_fit(block)
                
                if self.debug and permutation == 0:
                    self.logger.debug(
                        f"Block {block.block_index} training completed. "
                        f"Sparse vector sparsity: {(block.sparse_coding_layer.h == 0).sum().item()}/{self.n_functions}"
                    )

    def predict(self, X: torch.Tensor, y: torch.Tensor, *_) -> torch.Tensor:
        """
        Predict the output using the trained SSESM model with active sub-blocks.

        The prediction process involves:
        1. Retrieving active sub-blocks from the partition manager.
        2. Making predictions for each sub-block using the associated sparse coding layer.
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
            
            # Use parent's predict with the block's sparse coding h
            block_pred = super().predict(
                X_torch, 
                custom_h=block.sparse_coding_layer.h
            ) / block.amplitude
            
            # Map predictions back to original positions
            for i, pos in enumerate(block.positions):
                y_pred_per_block[pos] = block_pred[i].item() if block_pred[i].dim() == 0 else block_pred[i]

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
        mse = mean_squared_error(y_pred.cpu().numpy(), y.cpu().numpy())
        return y_pred, time, mse
