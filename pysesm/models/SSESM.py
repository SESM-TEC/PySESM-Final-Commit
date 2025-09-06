'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica

SSESM Base Class

Provides the sequential version of SESM

Authors: The SESM Team 

License: 
'''
from __future__ import annotations

import logging
import warnings

from collections.abc import Callable
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error

import numpy as np
import torch
import time

from ..models.SESM import SESM, SESMConfig
from ..device_manager.DeviceManager import DeviceManager
from ..factories.BlockManagerFactory import BlockManagerFactory

@dataclass
class SSESMConfig(SESMConfig):
    """
    Configuration for SSESM model, extending base SESMConfig.
    
    Attributes:
        permutation_times (int): Number of times to permute the dataset for
                                 training in the sequential partial fit loop.
    """
    permutation_times: int = 1


class SSESM(SESM):
    """
    A PyTorch module extending the SESM architecture to implement
    a surrogate model using a sequential and block-partitioned
    approach. This class is designed for function approximation and
    surrogate modeling tasks with dynamic sub-block partitioning.
    """

    CONFIG_CLASS = SSESMConfig

    def __init__(
        self,
        config: SSESMConfig,
        logger: logging.Logger,
        device_manager: DeviceManager = None,
        dict_layer_hook: Callable[[dict], None] | None = None,
        sparse_coding_layer_hook: Callable[[dict], None] | None = None,
        sesm_hook: Callable[[dict], None] | None = None,
        **kwargs
    ):
        """
        Initialize the SSESM model with a sequential, block-based approach.

        This constructor sets up the surrogate model, initializes the partition manager for
        sub-block operations, and configures hyperparameters.

        Args:
            config (SSESMConfig): Configuration object containing all SSESM parameters.
            logger (logging.Logger): Logger instance for runtime monitoring.
            device_manager (DeviceManager): Device manager for GPU/CPU allocation.
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

        self.permutation_times = config.permutation_times

        # Create partition manager
        self.partition_manager = BlockManagerFactory.create(
            config.partition_config,
            logger=logger, 
            device_manager=self.device_manager,
            sparse_coding_layer_hook=sparse_coding_layer_hook
        )


    def evaluation_func(self, dictionary: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Concrete implementation of the evaluation function for SSESM.
        This performs standard 2D matrix multiplication.
        """
        return torch.matmul(dictionary, h)    

    
    def partial_fit(self, X: torch.Tensor, y: torch.Tensor, *_):
        """
        Perform a partial fit on the model, iteratively updating parameters using active sub-blocks.

        The method permutes the dataset and processes each block sequentially to update the
        sparse coding layer. Active blocks are dynamically retrieved and used to compute the partial fit.

        Args:
            X (torch.Tensor): Input features for training.
            y (torch.Tensor): Target values.
            *_: Additional unused positional arguments.

        """
        start_time = time.time()
        # Ensure y is 2D
        if y.dim() == 1:
            y = y.unsqueeze(-1)

        # Add points to partition manager and initialize blocks
        self.partition_manager.add_points(X, y)
        self.partition_manager.init_sparse_coding_per_block(config=self.sparse_coding_config,
                                                            evaluation_func=self.evaluation_func)
        active_blocks = self.partition_manager.retrieve_active_blocks()

        # Train blocks with permutation
        for permutation in range(self.permutation_times):
            selected_indices = np.random.permutation(len(active_blocks))
            permuted_blocks = [active_blocks[i] for i in selected_indices]
            
            for block in permuted_blocks:
                # Reuse the SESM approach at the parent class
                super()._train_block(block, permutation)
                
                self.logger.debug(
                    f"Block {block.block_index}/{len(active_blocks)} processed. "
                    f"Sparse vector sparsity: {(block.sparse_coding_layer.h == 0).sum().item()}/{self.n_functions}"
                )

            self.logger.debug(f"Permutation {permutation}/{self.permutation_times} done.")
            
        end_time = time.time()
        return end_time - start_time

    def predict(self, X: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        Predict the output using the trained SSESM model with active sub-blocks.

        The prediction process involves:
        1. Retrieving active sub-blocks from the partition manager.
        2. Making predictions for each sub-block using the associated sparse coding layer.
        3. Aggregating predictions to reconstruct the final output vector.

        Args:
            X (torch.Tensor): Input features for prediction.
            y (torch.Tensor):  Deprecated parameter.  No longer needed.

        Returns:
            torch.Tensor: Predicted values for the input dataset.
        """

        if y is not None:
            warnings.warn("Deprecated behaviour: predict does not need y values",
                          DeprecationWarning, stacklevel=2)

        
        active_blocks = self.partition_manager.retrieve_inference_blocks(X)

        # Create tensor directly with correct shape and device
        y_pred = torch.zeros(X.shape[0], 1, dtype=X.dtype, device=X.device)
        
        for block in active_blocks:
            # Use parent's predict with the block's sparse coding h
            block_pred = self._predict_block(block) / block.amplitude
            
            # Map predictions back to original positions
            for i, pos in enumerate(block.positions):
                y_pred[pos, 0] = block_pred[i].item()

        return y_pred

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
        y_pred = self.predict(X)
        time = self.elapsed_time / 60
        mse = mean_squared_error(y_pred.cpu().numpy(), y.cpu().numpy())
        return y_pred, time, mse
