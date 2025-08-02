'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica
BSESM Base Class
Provides the batched version of SESM
Authors: The SESM Team 
License: 
'''

import logging
import torch
import time
from typing import Callable, Iterator, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import mean_squared_error

from pysesm.functions import SurrogateFunction
from pysesm.blocks import PartitionBlock
from pysesm.models.SESM import SESM, SESMConfig
from pysesm.sparse_coding.SparseCodingBaseLayer import SparseCodingBaseLayer, SparseCodingConfig
from pysesm.device_manager.DeviceManager import DeviceManager
from pysesm.factories.SparseCodingFactory import SparseCodingFactory
from pysesm.enums.DeviceTargetEnum import DeviceTarget

@dataclass
class BSESMConfig(SESMConfig):
    """
    Configuration for BSESM model, extending base SESMConfig.
    """
    pass

class BSESM(SESM):
    """
    A Batch-based Sparse-Encoded Surrogate Models (BSESM).
    
    This class extends the SESM model by incorporating a batch processing approach
    where the dictionary and sparse codes (h vectors for all active blocks) are
    trained jointly in a single global optimization step per model epoch.
    
    Unlike SSESM, which trains blocks sequentially, BSESM aggregates all active
    training data. It updates the dictionary using gradient accumulation from all
    blocks and solves for the sparse codes using a block-diagonal matrix formulation.
    """

    CONFIG_CLASS = BSESMConfig

    def __init__(
        self,
        config: BSESMConfig,
        logger: logging.Logger,
        device_manager: Optional[DeviceManager] = None,
        dict_layer_hook: Optional[Callable[[dict], None]] = None,
        sparse_coding_layer_hook: Optional[Callable[[dict], None]] = None,
        sesm_hook: Optional[Callable[[dict], None]] = None,
        **kwargs,
    ):
        """
        Initializes the BSESM model.

        Args:
            config (BSESMConfig): Configuration object containing all BSESM parameters.
            logger (logging.Logger): Logger instance for runtime monitoring.
            device_manager (DeviceManager): Device manager for GPU/CPU allocation.
            dict_layer_hook: Optional callback for dictionary layer monitoring.
            sparse_coding_layer_hook: Optional callback for sparse coding layer monitoring.
            sesm_hook: Optional callback for SESM-level monitoring.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            config=config,
            logger=logger,
            device_manager=device_manager,
            sesm_hook=sesm_hook,
            dict_layer_hook=dict_layer_hook,
            sparse_coding_layer_hook=sparse_coding_layer_hook,
            **kwargs,
        )

        # The global sparse coding layer will operate on the large block-diagonal matrix,
        # so it uses standard matrix multiplication.
        self.global_sparse_coding_layer = SparseCodingFactory.create(
            config=self.sparse_coding_config,
            evaluation_func=self.evaluation_func,
            logger=self.logger,
            parameter_hook=self.sparse_coding_layer_hook,
            device=self.device_manager.get_device(DeviceTarget.SPARSE_CODING_LAYER)
        )
        self.logger.info(f"Global Sparse Coding Layer: {type(self.global_sparse_coding_layer).__name__}")

    def evaluation_func(self, dictionary: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Concrete implementation of the evaluation function. 
        For BSESM, this always performs a simple matrix multiplication,
        as batching is handled by loops inside the training methods.
        """
        return torch.matmul(dictionary, h)

    def _aggregate_block_data(self, blocks: List[PartitionBlock]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Aggregates data from active blocks into lists of tensors, without padding.
        
        This prepares data for the batched dictionary evaluation and the construction
        of the block-diagonal system.
        
        Args:
            blocks (List[PartitionBlock]): List of active blocks.
            
        Returns:
            Tuple containing three lists:
                - X_list (List[torch.Tensor]): Normalized input features for each block.
                - y_list (List[torch.Tensor]): Target values for each block.
                - h_initial_list (List[torch.Tensor]): Initial `h` vectors for each block.
        """
        if not blocks:
            return [], [], []

        X_list = [block.normalized_X for block in blocks]
        y_list = [block.target for block in blocks]
        h_initial_list = [block.sparse_coding_layer.h for block in blocks]
        
        return X_list, y_list, h_initial_list

    def partial_fit(self, X: torch.Tensor, y: torch.Tensor, *_):
        """
        Perform a partial fit on the BSESM model, training dictionary and sparse codes globally.

        This method uses a block-diagonal matrix strategy:
        1. Aggregates data from all active blocks into lists.
        2. Updates the shared dictionary by accumulating gradients from all blocks in a single optimizer step.
        3. Constructs a large block-diagonal "mega-matrix" from the evaluated dictionaries.
        4. Concatenates all `y` and `h` vectors to solve the sparse coding problem globally.
        5. Unpacks the optimized `H_mega` back into individual `h` vectors for each block.

        Args:
            X (torch.Tensor): Input features for training (full dataset or a batch).
            y (torch.Tensor): Target values for training.
            *_: Additional unused positional arguments.
        """
        if y.dim() == 1:
            y = y.unsqueeze(-1)

        self.partition_manager.add_points(X, y)
        self.partition_manager.init_sparse_coding_per_block(
            config=self.sparse_coding_config,
            evaluation_func=self.evaluation_func
        )
        
        active_blocks = self.partition_manager.retrieve_active_blocks()
        if not active_blocks:
            self.logger.warning("No active blocks found. Skipping training.")
            return

        # Step 1: Aggregate data into lists
        X_list, y_list, h_list = self._aggregate_block_data(active_blocks)
        
        # Wrap in a nested_tensor for efficient dictionary evaluation in a single call
        X_nested = torch.nested.nested_tensor(X_list, layout=torch.jagged,
                                              device=self.device_manager.get_device(DeviceTarget.DICTIONARY_LAYER))
        
        for epoch in range(self.config.model_epochs):
            # --- Dictionary Update Step with Manual Batching / Gradient Accumulation ---
            dict_optimizer = self.dictionary_layer.optimizer
            dict_criterion = self.dictionary_layer.criterion
            
            # Zero gradients once before accumulating
            dict_optimizer.zero_grad()
            
            # Evaluate the dictionary for all blocks in a single, efficient call.
            # self.dictionary_layer(X_nested) returns a list of evaluated dictionaries [D_1, D_2, ...].
            dict_list = self.dictionary_layer(X_nested)
            
            total_loss_value = 0.0
            total_samples = 0
            
            # This loop accumulates gradients from all blocks.
            for i in range(len(active_blocks)):
                dict_i = dict_list[i]
                y_i = y_list[i]
                h_i = h_list[i].detach() # Use detached h for dict training

                y_pred_i = self.evaluation_func(dict_i, h_i)
                loss_i = dict_criterion(y_pred_i, y_i)

                # The .backward() call on each loss accumulates gradients in the dictionary's parameters.
                loss_i.backward() 
                
                total_loss_value += loss_i.item() * y_i.shape[0]
                total_samples += y_i.shape[0]
            
            # Step the optimizer once after all gradients have been accumulated.
            dict_optimizer.step()
            
            avg_loss = total_loss_value / total_samples if total_samples > 0 else 0.0
            self.dictionary_layer.losses.append(avg_loss)
            self._dict_losses.append(avg_loss)

            # --- Sparse Coding Update Step (Mega-Matrix) ---
            with torch.no_grad():
                # Re-evaluate the dictionary with the newly updated parameters.
                dict_list_updated = self.dictionary_layer(X_nested)

            # Step 2: Pack data into the mega-matrix formulation
            Y_mega = torch.cat(y_list)
            D_mega = torch.block_diag(*dict_list_updated)
            
            # The number of functions for the global SC layer is the total number of columns in D_mega.
            self.global_sparse_coding_layer.config.n_functions = D_mega.shape[1]
            
            H_mega_initial = torch.cat(h_list)
            self.global_sparse_coding_layer.setup(H_mega_initial)

            # Step 3: Solve the single, large sparse coding problem
            self.global_sparse_coding_layer.partial_fit(y=Y_mega, dictionary=D_mega)
            H_mega_optimizado = self.global_sparse_coding_layer.h.detach()
            
            # Step 4: Unpack the results and update each block's h for the next epoch
            h_split_sizes = [h.shape[0] for h in h_list]
            h_optimizado_list = torch.split(H_mega_optimizado, h_split_sizes)

            for i, block in enumerate(active_blocks):
                # Update the original h-vector in the list for the next iteration
                h_list[i].data = h_optimizado_list[i].to(block.sparse_coding_layer.device)
                # Also update the reference in the block object itself
                block.sparse_coding_layer.h.data = h_list[i].data

        self.partial_fit_count += 1


    def predict(self, X: torch.Tensor, y: torch.Tensor, *_) -> torch.Tensor:
        """
        Predict the output using the trained BSESM model with active sub-blocks.

        Args:
            X (torch.Tensor): Input features for prediction.
            y (torch.Tensor): Target values (used to identify active blocks).
            *_: Additional unused positional arguments.

        Returns:
            torch.Tensor: Predicted values for the input dataset.
        """
        if y.dim() == 1:
            y = y.unsqueeze(-1)

        test_active_blocks = self.partition_manager.retrieve_test_active_blocks(X, y)
        if not test_active_blocks:
            self.logger.warning("No active test blocks found. Returning empty prediction.")
            return torch.empty_like(y, device=X.device, dtype=X.dtype)

        # Aggregate test data into a nested_tensor for efficient batch evaluation
        X_list_test = [block.normalized_X for block in test_active_blocks]
        X_nested_test = torch.nested.nested_tensor(X_list_test, layout=torch.jagged, device=self.device_manager.get_device(DeviceTarget.DICTIONARY_LAYER))

        with torch.no_grad():
            # Evaluate the dictionary for all test blocks at once -> returns a list of matrices
            dict_list_test = self.dictionary_layer(X_nested_test)
            
            # Perform prediction for each block using its specific evaluated dictionary and learned h
            y_pred_normalized_list = [
                self.evaluation_func(dict_i, block.sparse_coding_layer.h)
                for dict_i, block in zip(dict_list_test, test_active_blocks)
            ]

        # Denormalize and reconstruct the final output tensor
        # Ensure y_final_predictions is created with the correct shape and device
        y_final_predictions = torch.zeros(y.shape[0], device=X.device, dtype=X.dtype)
        
        for i, block in enumerate(test_active_blocks):
            block_preds_unnormalized = y_pred_normalized_list[i] / block.amplitude
            y_final_predictions[block.positions] = block_preds_unnormalized.squeeze()
            
        return y_final_predictions.view_as(y)

    def performance_stats(self, X: torch.Tensor, y: torch.Tensor):
        """
        Evaluate the model's performance on a given dataset using active sub-blocks.
        """
        y_pred = self.predict(X, y)
        time = self.elapsed_time / 60
        mse = mean_squared_error(y_pred.cpu().numpy(), y.cpu().numpy())
        return y_pred, time, mse
