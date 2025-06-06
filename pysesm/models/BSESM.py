'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica

BSESM Class

Provides the batch version of SESM

Authors: The SESM Team 

License: 
'''

import logging
import torch
from typing import Callable, Iterator, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np # For np.ndindex
from sklearn.metrics import mean_squared_error

from pysesm.functions import SurrogateFunction
from pysesm.blocks import PartitionBlock # Import PartitionBlock explicitly for type hinting
from pysesm.models.SESM import SESM, SESMConfig # Import SESM and its config
from pysesm.sparse_coding.SparseCodingBaseLayer import SparseCodingBaseLayer, SparseCodingConfig
from pysesm.device_manager.DeviceManager import DeviceManager # Ensure DeviceManager is imported
from pysesm.factories.SparseCodingFactory import SparseCodingFactory
from pysesm.factories.BlockManagerFactory import BlockManagerFactory

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
    training data points (X, y) and their corresponding h vectors, then trains
    a global dictionary and all h vectors simultaneously.
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
        # The parent SESM.__init__ will set up the dictionary_layer and partition_manager
        # and store all hooks.
        super().__init__(
            config=config,
            logger=logger,
            device_manager=device_manager,
            sesm_hook=sesm_hook,
            dict_layer_hook=dict_layer_hook,
            sparse_coding_layer_hook=sparse_coding_layer_hook,
            **kwargs,
        )

        # BSESM needs a single sparse coding layer for the global training step.
        # In contrast to SSESM where each block has its own.
        # This global_sparse_coding_layer will manage the batch of h vectors.
        # It needs to be a regular SparseCodingBaseLayer instance.
        self.global_sparse_coding_layer = SparseCodingFactory.create(
            config=self.sparse_coding_config, # Use the shared sparse coding config
            evaluation_func=self._global_evaluation_func, # BSESM needs its own special eval func for global training
            logger=self.logger,
            parameter_hook=self.sparse_coding_layer_hook,
            device=self.device_manager.get_device(DeviceTarget.SPARSE_CODING_LAYER)
        )
        self.logger.info(f"Global Sparse Coding Layer: {type(self.global_sparse_coding_layer).__name__}")
        
        # Override parent's sparse_coding_layer with the global one (for _train_step)
        # self.sparse_coding_layer = self.global_sparse_coding_layer
        # The parent SESM has a `sparse_coding_layer` attribute which is `None` by default.
        # We need to ensure it points to the correct sparse coding layer during global training.
        # However, for `partial_fit` in BSESM, we will explicitly pass the `global_sparse_coding_layer`.
        # So we don't need to override `self.sparse_coding_layer` here.
        # The `_train_step` will receive it as `sparsecoding` parameter.


    def evaluation_func(self, dictionary: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Concrete implementation of the evaluation function for BSESM.
        
        This method combines a batch of dictionary evaluations `D` (from `(N_total_points, N_functions)`
        or `(N_blocks, N_points_per_block, N_functions)`) with a batch of sparse vectors `h`
        (from `(N_functions, 1)` or `(N_blocks, N_functions, 1)`) to produce predictions.

        For BSESM's global training, `h` will be `(N_functions, 1)` (after being reshaped internally
        by `_global_evaluation_func` to be a batch if needed).
        This `evaluation_func` from `SESM` (which is `torch.matmul(D, h)`) is used by `dictionary_layer`
        to combine its output `D` with `h_detached`.
        """
        # This `evaluation_func` is for the `dictionary_layer` and `SESM` base class.
        # It expects D (N_samples, N_functions) and h (N_functions, 1).
        # We will use a *different* `_global_evaluation_func` for the `global_sparse_coding_layer`.
        return torch.matmul(dictionary, h)


    def _global_evaluation_func(self, dictionary: torch.Tensor, h_batch: torch.Tensor) -> torch.Tensor:
        """
        Special evaluation function for BSESM's global sparse coding layer.
        
        This combines a dictionary evaluated for *all* points (`(N_total_points, N_functions)`)
        with a *batch* of h vectors (`(N_blocks, N_functions, 1)`), to produce a batch of predictions.
        
        It implicitly assumes that `dictionary` is ordered such that its rows correspond
        to the concatenated `X` from all blocks in the same order as `h_batch`'s blocks.
        
        Args:
            dictionary (torch.Tensor): Dictionary matrix for all points. Shape: (N_total_points, N_functions).
            h_batch (torch.Tensor): Stacked h vectors for all blocks. Shape: (N_blocks, N_functions, 1).
            
        Returns:
            torch.Tensor: Predicted values for all points, structured as a batch.
                          Shape: (N_total_points, output_dim) or (N_total_points, 1).
        """
        # We need to know how many points are in each block to correctly partition the `dictionary`
        # and apply the corresponding `h` from `h_batch`.
        # This requires `max_points_in_block` and `num_blocks` information from `_fill_block_points`.
        # This implies `_global_evaluation_func` needs more context than standard `eval_func`.
        # Or, we design `h_batch` to be already expanded to match the dictionary's structure.

        # Let's simplify this. `h_batch` should be structured for `bmm`.
        # `h_batch` comes as `(N_blocks, N_functions, 1)`.
        # `dictionary` from `dictionary_layer.forward(X_batch)` is `(N_total_points, N_functions)`.

        # To perform BMM, we need D to be (N_blocks, N_points_per_block, N_functions).
        # And h_batch to be (N_blocks, N_functions, 1).
        # The result will be (N_blocks, N_points_per_block, 1).
        # We also need a way to know `N_points_per_block`.

        # This indicates a potential issue in the current `_fill_block_points` output.
        # It returns `filled_active_blocks_X` (N_total_points, N_features)
        # and `max_points_in_block`.

        # Redesign:
        # 1. `_fill_block_points` should return a list of (X, y) tensors for each block,
        #    padded individually if needed, or structured for batching.
        # 2. `_global_evaluation_func` will then iterate over blocks or use explicit batching.

        # Given `dictionary` is `(N_total_points, N_functions)`, it's a concatenated matrix.
        # `h_batch` is `(N_blocks, N_functions, 1)`.
        # We need to split `dictionary` by blocks and multiply each part with its corresponding `h`.

        # This requires knowing the `block_sizes_in_samples` for each block, including padding.
        # This is where `max_points_in_block` from `_fill_block_points` is crucial.

        num_blocks = h_batch.shape[0]
        num_functions = h_batch.shape[1] # Should be N_functions
        
        # We assume `dictionary` comes from `dictionary_layer.forward(X_batch_normalized)`.
        # X_batch_normalized is `(N_total_points, N_features)`, which comes from `_fill_block_points`.
        # `_fill_block_points` also returns `max_points_in_block`.
        points_per_block = dictionary.shape[0] // num_blocks
        
        # Reshape dictionary to be a batch for BMM
        # (N_total_points, N_functions) -> (N_blocks, points_per_block, N_functions)
        dictionary_reshaped = dictionary.view(num_blocks, points_per_block, num_functions)
        
        # Perform batch matrix multiplication
        # (N_blocks, points_per_block, N_functions) @ (N_blocks, N_functions, 1)
        # Result: (N_blocks, points_per_block, 1)
        y_pred_batch = torch.bmm(dictionary_reshaped, h_batch)
        
        # Flatten back to (N_total_points, 1) to match concatenated target_y.
        return y_pred_batch.view(-1, 1)


    def _aggregate_block_data(self, blocks: List[PartitionBlock]) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Aggregates and pads data from a list of PartitionBlocks into batch tensors.
        
        This prepares data for global training in BSESM.
        
        Args:
            blocks (List[PartitionBlock]): List of active blocks.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, int]:
                - X_batch_normalized (torch.Tensor): Normalized input features, stacked and padded.
                                                   Shape: (num_blocks * max_points_in_block, n_features).
                - y_batch_target (torch.Tensor): Target values, stacked and padded.
                                                Shape: (num_blocks * max_points_in_block, output_dim).
                - max_points_in_block (int): The maximum number of points in any single block.
        """
        if not blocks:
            # Return empty tensors with correct shapes if no active blocks
            return (
                torch.empty(0, self.n_features, device=self.device_manager.get_device(DeviceTarget.GLOBAL)),
                torch.empty(0, 1, device=self.device_manager.get_device(DeviceTarget.GLOBAL)), # Assuming output_dim=1
                0
            )

        max_points_in_block = max(len(block.X) for block in blocks)
        
        # Prepare padding tensors
        # Assuming n_features and output_dim (1) for padding
        padding_X_row = torch.zeros(1, self.n_features, device=self.device_manager.get_device(DeviceTarget.GLOBAL))
        padding_y_row = torch.zeros(1, 1, device=self.device_manager.get_device(DeviceTarget.GLOBAL)) # Assuming output_dim=1

        # Lists to hold padded tensors for concatenation
        padded_X_list = []
        padded_y_list = []

        for block in blocks:
            # Ensure block's normalized_X and target are tensors and on device
            current_X = block.normalized_X.to(self.device_manager.get_device(DeviceTarget.GLOBAL))
            current_y = block.target.to(self.device_manager.get_device(DeviceTarget.GLOBAL))

            # Pad X
            num_padding_X = max_points_in_block - current_X.shape[0]
            if num_padding_X > 0:
                padded_X = torch.cat([current_X, padding_X_row.repeat(num_padding_X, 1)], dim=0)
            else:
                padded_X = current_X
            padded_X_list.append(padded_X)

            # Pad y
            num_padding_y = max_points_in_block - current_y.shape[0]
            if num_padding_y > 0:
                padded_y = torch.cat([current_y, padding_y_row.repeat(num_padding_y, 1)], dim=0)
            else:
                padded_y = current_y
            padded_y_list.append(padded_y)

        X_batch_normalized = torch.cat(padded_X_list, dim=0)
        y_batch_target = torch.cat(padded_y_list, dim=0)
        
        return X_batch_normalized, y_batch_target, max_points_in_block


    def partial_fit(self, X: torch.Tensor, y: torch.Tensor, *_):
        """
        Perform a partial fit on the BSESM model, training dictionary and sparse codes globally.

        This method aggregates all active training points, initializes sparse codes for each block,
        and then iteratively updates the global dictionary and all sparse codes simultaneously.

        Args:
            X (torch.Tensor): Input features for training (full dataset or a batch).
            y (torch.Tensor): Target values for training.
            *_: Additional unused positional arguments.

        Returns:
            None
        """
        # Ensure y is 2D
        if y.dim() == 1:
            y = y.unsqueeze(-1)

        # 1. Add points to partition manager and initialize blocks
        self.partition_manager.add_points(X, y)
        
        # 2. Initialize sparse coding layer for each block
        # Each block will get a fresh SC layer if it's new/active, or reuse existing one.
        # This gives each block its own h tensor.
        self.partition_manager.init_sparse_coding_per_block(
            config=self.sparse_coding_config,
            evaluation_func=self.evaluation_func # Standard eval_func for individual block's SC layer
        )
        
        # 3. Retrieve all currently active blocks (those with data points)
        active_blocks = self.partition_manager.retrieve_active_blocks()
        
        if not active_blocks:
            self.logger.warning("No active blocks found. Skipping training.")
            return

        # 4. Aggregate all normalized X and target y from active blocks into batch tensors
        X_batch_normalized, y_batch_target, max_points_in_block = self._aggregate_block_data(active_blocks)

        # 5. Prepare the initial batch of h vectors for global training
        # These are stacked into a single tensor for the global sparse coding layer.
        # Shape: (N_blocks, N_functions, 1)
        h_batch_initial = torch.stack([
            block.sparse_coding_layer.h.to(self.device_manager.get_device(DeviceTarget.SPARSE_CODING_LAYER))
            for block in active_blocks
        ])
        
        # Initialize the global sparse coding layer with this aggregated h
        self.global_sparse_coding_layer.setup(h_batch_initial)
        
        # 6. Perform global training over model_epochs
        # This calls the parent SESM's _train_step, but with our global SC layer and batch data.
        # The `_train_step` expects `sparsecoding` to be the layer managing `h`.
        # Here, `self.global_sparse_coding_layer` is that manager for the *batch* of `h`s.
        
        for epoch in range(self.config.model_epochs):
            epoch_start_time = time.time()
            
            # _train_step optimizes:
            # 1. Dictionary (using X_batch_normalized, y_batch_target, and detached h_batch)
            # 2. h_batch (using dictionary, y_batch_target)
            
            # The dictionary_layer.forward(X_batch_normalized) will produce (N_total_points, N_functions)
            # The global_sparse_coding_layer needs D (N_total_points, N_functions) and h (N_blocks, N_functions, 1).
            # Its internal evaluation_func (`_global_evaluation_func`) will handle this.
            
            super()._train_step(
                X=X_batch_normalized,
                y=y_batch_target,
                sparsecoding=self.global_sparse_coding_layer # Pass the global SC layer
            )
            
            self.elapsed_time += time.time() - epoch_start_time
            
            if ( (self.config.log_interval > 0) and
                 ( (epoch + 1) % self.config.log_interval == 0 or
                   epoch == 0 or
                   epoch == self.config.model_epochs - 1 ) ):
                self.logger.info(
                    f"BSESM Global Epoch {epoch + 1}/{self.config.model_epochs}: "
                    f"Loss Global SC: {self.global_sparse_coding_layer.losses[-1]:.6f}, "
                    f"Loss Dictionary: {self.dictionary_layer_losses[-1]:.6f}"
                )
        
        self.partial_fit_count += 1
        
        # 7. After global training, distribute the learned h_batch back to individual blocks
        # h_batch is now self.global_sparse_coding_layer.h, shape (N_blocks, N_functions, 1)
        learned_h_batch = self.global_sparse_coding_layer.h.detach().cpu()
        
        for i, block in enumerate(active_blocks):
            block.sparse_coding_layer.h.data = learned_h_batch[i].to(block.sparse_coding_layer.device)
            if self.logger.level <= logging.DEBUG:
                 # Check sparsity
                 h_tensor = block.sparse_coding_layer.h.detach().cpu()
                 non_zero_components = torch.sum(torch.abs(h_tensor) > 1e-6).item()
                 total_components = h_tensor.numel()
                 sparsity_ratio = (total_components - non_zero_components) / total_components * 100
                 self.logger.debug(
                    f"Block {block.block_index} 'h' updated. Sparsity: {sparsity_ratio:.2f}%"
                 )


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
        # Ensure y is 2D
        if y.dim() == 1:
            y = y.unsqueeze(-1)

        # Retrieve active blocks for testing. These are new PartitionBlock instances
        # but they point to the *original* sparse_coding_layer (with the trained h)
        # and the amplitude from the training phase.
        test_active_blocks = self.partition_manager.retrieve_test_active_blocks(X, y)

        if not test_active_blocks:
            self.logger.warning("No active test blocks found. Returning empty prediction.")
            return torch.empty(y.shape[0], 1, device=X.device, dtype=X.dtype) # Return appropriate empty tensor

        # Aggregate X_test and y_test similarly to training, for a single forward pass
        X_test_batch_normalized, y_test_batch_target, max_points_in_block = self._aggregate_block_data(test_active_blocks)

        # Create a batch of h vectors for prediction.
        # These come directly from the `sparse_coding_layer` of each test block.
        h_predict_batch = torch.stack([
            block.sparse_coding_layer.h.to(self.device_manager.get_device(DeviceTarget.SPARSE_CODING_LAYER))
            for block in test_active_blocks
        ])
        
        # Perform the prediction using the _global_evaluation_func and the global dictionary.
        # The `dictionary_layer.forward(X_test_batch_normalized)` evaluates D for all points.
        # The `_global_evaluation_func` then re-batches D and multiplies with h_predict_batch.
        with torch.no_grad():
            evaluated_D = self.dictionary_layer.forward(X_test_batch_normalized)
            y_pred_batch_normalized = self._global_evaluation_func(evaluated_D, h_predict_batch)

        # Denormalize predictions by dividing by block amplitude
        # y_pred_batch_normalized is (N_total_points, 1). We need to split it back by block.
        # The `_aggregate_block_data` returned the total points.
        
        y_pred_unnormalized_list = []
        current_point_idx = 0
        for block_idx, block in enumerate(test_active_blocks):
            num_points_in_block_original = len(block.X) # Number of actual points, not padded
            num_points_in_block_padded = max_points_in_block # Number of points including padding

            # Extract the relevant (non-padded) predictions for this block
            block_normalized_preds = y_pred_batch_normalized[
                current_point_idx : current_point_idx + num_points_in_block_original
            ]
            
            # Apply denormalization using the block's amplitude
            block_unnormalized_preds = block_normalized_preds / block.amplitude
            y_pred_unnormalized_list.append(block_unnormalized_preds)
            
            current_point_idx += num_points_in_block_padded # Advance by padded size

        # Map predictions back to original global positions (based on original `y` indices)
        y_final_predictions = torch.zeros_like(y, device=X.device, dtype=X.dtype) # Initialize with zeros
        
        point_in_batch_idx = 0
        for block_idx, block in enumerate(test_active_blocks):
            num_actual_points = len(block.X) # Number of actual points in this block
            
            # Get the predictions for the actual points of this block
            preds_for_this_block = y_pred_unnormalized_list[block_idx]
            
            # Assign them to their original global positions
            for i in range(num_actual_points):
                original_pos = block.positions[i] # Get the original index from the test set X
                y_final_predictions[original_pos] = preds_for_this_block[i].cpu()

        return y_final_predictions


    # performance_stats can remain as it is, as it calls predict.
    def performance_stats(self, X: torch.Tensor, y: torch.Tensor):
        """
        Evaluate the model's performance on a given dataset using active sub-blocks.
        ...
        """
        y_pred = self.predict(X, y)
        time = self.elapsed_time / 60
        mse = mean_squared_error(y_pred.cpu().numpy(), y.cpu().numpy())
        return y_pred, time, mse
