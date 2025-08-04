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
from pysesm.base_types import TensorBatch 


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
        # Delegate to SurrogateFunction's __call__ which handles TensorBatch
        # and its internal matmul logic.
        if (getattr(dictionary, "is_nested", False) and
            getattr(h, "is_nested", False)):
            results = [torch.matmul(d, hi)
                       for d, hi in zip(dictionary.unbind(), h.unbind())]
            return torch.nested.as_nested_tensor(results,
                                                  layout=dictionary.layout,
                                                  device=dictionary.device,
                                                  dtype=results[0].dtype)
        elif isinstance(dictionary, torch.Tensor) and dictionary.dim() <= 2:
            return torch.matmul(dictionary, h)
        elif isinstance(dictionary, torch.Tensor) and dictionary.dim() == 3:
            return torch.vmap(torch.matmul)(dictionary, h)
        elif isinstance(dictionary, list) and isinstance(h, list):
            results = [torch.matmul(d, hi) for d, hi in zip(dictionary, h)]
            return results
        else:
            raise TypeError("Unsupported TensorBatch types for evaluation_func: "
                            f"D={type(dictionary)}, h={type(h)}")

    def _aggregate_block_data(self, blocks: List[PartitionBlock]
                              ) -> Tuple[TensorBatch, TensorBatch, TensorBatch]:
        """
        Aggregates data from active blocks into nested_tensors for efficient
        batched processing without padding.
        
        Args:
            blocks (List[PartitionBlock]): List of active blocks.
            
        Returns:
            Tuple containing three nested_tensors:
                - X_nested (TensorBatch): Normalized input features for each block.
                - y_nested (TensorBatch): Target values for each block.
                - h_nested (TensorBatch): Initial `h` vectors for each block.
        """
        if not blocks:
            # Return empty nested tensors. This handles cases where no
            # active blocks are found.
            empty_tensor_list = [torch.empty(0, self.n_features,
                                             device=self.device_manager.get_device(
                                                 DeviceTarget.DICTIONARY_LAYER))]
            empty_y_list = [torch.empty(0, 1,
                                        device=self.device_manager.get_device(
                                            DeviceTarget.DICTIONARY_LAYER))]
            empty_h_list = [torch.empty(0, self.n_functions, 1,
                                        device=self.device_manager.get_device(
                                            DeviceTarget.DICTIONARY_LAYER))]
            return (torch.nested.nested_tensor(empty_tensor_list),
                    torch.nested.nested_tensor(empty_y_list),
                    torch.nested.nested_tensor(empty_h_list))

        X_list = [block.normalized_X for block in blocks]
        y_list = [block.target for block in blocks]
        # h needs to be detached here because DictBaseLayer.partial_fit
        # expects h to be detached for dictionary training.
        h_list = [block.sparse_coding_layer.h.detach() for block in blocks] 
        
        X_nested = torch.nested.nested_tensor(
            X_list, layout=torch.jagged,
            device=self.device_manager.get_device(DeviceTarget.DICTIONARY_LAYER))
        y_nested = torch.nested.nested_tensor(
            y_list, layout=torch.jagged,
            device=self.device_manager.get_device(DeviceTarget.DICTIONARY_LAYER))
        h_nested = torch.nested.nested_tensor(
            h_list, layout=torch.jagged,
            device=self.device_manager.get_device(DeviceTarget.DICTIONARY_LAYER))
        
        return X_nested, y_nested, h_nested


    def _global_train_step(self,
                           X_nested: TensorBatch,
                           y_nested: TensorBatch,
                           h_nested: TensorBatch,
                           active_blocks: List[PartitionBlock],
                           current_epoch: int):
        """
        Performs a single global training step for the BSESM model.
        This includes training the dictionary on all active blocks' data
        and then training the global sparse coding layer.

        Args:
            X_nested (TensorBatch): Aggregated normalized input features for all active blocks.
            y_nested (TensorBatch): Aggregated target values for all active blocks.
            h_nested (TensorBatch): Aggregated sparse coding vectors for all active blocks.
            active_blocks (List[PartitionBlock]): List of PartitionBlock objects currently active.
            current_epoch (int): The current model epoch (for logging or hook context).
        """
        # Step 1: Optimize dictionary with fixed h (from previous iteration or initial)
        # The h_nested must be detached for dictionary training (already handled by _aggregate_block_data or DictBaseLayer)
        self.dictionary_layer.partial_fit(
            X=X_nested,
            y=y_nested,
            h=h_nested # h is detached internally by DictBaseLayer for dict training
        )
        # Dictionary losses are internally tracked by dictionary_layer.losses.
        # They will be logged at the BSESM model_epoch level.
        
        # Step 2: Optimize h with fixed dictionary (Mega-Matrix approach)
        # Ensure no_grad here, as dictionary parameters are fixed for SC phase.
        with torch.no_grad():
            # dictionary_layer.forward will return a NestedTensor here.
            dict_nested_updated = self.dictionary_layer.forward(X_nested)
            
            # Unbind NestedTensors to get lists for torch.cat and block_diag
            # y_nested is already a NestedTensor, get its unbound list of tensors
            y_list_unbound = y_nested.unbind()
            dict_list_unbound = dict_nested_updated.unbind()

            # Pack data into the mega-matrix formulation
            Y_mega = torch.cat(y_list_unbound) # Concatenate all y_i into a single column vector
            D_mega = torch.block_diag(*dict_list_unbound) # Create block-diagonal matrix D_mega
            
            # The number of functions for the global SC layer is the total
            # number of columns in D_mega (sum of n_functions for each block).
            # This needs to be set dynamically if blocks have different n_functions,
            # but for now, we assume all blocks share the same n_functions from config.
            # So, total n_functions = len(active_blocks) * self.n_functions
            self.global_sparse_coding_layer.config.n_functions = D_mega.shape[1]
            
            # Prepare initial H_mega from aggregated h_nested values (these are the current h values for each block)
            # h_nested.values() gives the concatenated tensor of all h_i values
            H_mega_initial = h_nested.values()
            
            # Setup the global sparse coding layer with the combined h vector.
            # This re-initializes its internal h.
            self.global_sparse_coding_layer.setup(H_mega_initial)

            # Solve the single, large sparse coding problem
            # The global sparse coding layer will run its own internal partial_fit loop
            # for `self.sparse_coding_config.epochs` iterations.
            self.global_sparse_coding_layer.partial_fit(y=Y_mega,
                                                        dictionary=D_mega)
            
            # Retrieve the optimized H_mega from the global sparse coding layer
            H_mega_optimizado = self.global_sparse_coding_layer.h.detach()
            
            # Unpack the results and update each block's h for the next epoch
            # h_split_sizes must match the original n_functions for each block
            h_split_sizes = [block.sparse_coding_layer.h.shape[0]
                             for block in active_blocks]
            h_optimizado_list = torch.split(H_mega_optimizado,
                                            h_split_sizes, dim=0) # Specify dim=0 for splitting H_mega_optimizado

            for i, block in enumerate(active_blocks):
                # Update the h-vector of the individual sparse coding layer within each block object.
                # Ensure it's moved to the correct device (which it should already be, but good practice).
                block.sparse_coding_layer.h.data = \
                    h_optimizado_list[i].to(block.sparse_coding_layer.device)

        # Losses are logged by partial_fit in the BSESM model_epochs loop.
        # The internal losses of dictionary_layer and global_sparse_coding_layer
        # are already being collected by those layers.



    def partial_fit(self, X: torch.Tensor, y: torch.Tensor, *_):
        """
        Perform a partial fit on the BSESM model, training dictionary and
        sparse codes globally.

        This method uses a block-diagonal matrix strategy:
        1. Aggregates data from all active blocks into nested tensors.
        2. Delegates dictionary training to the dictionary layer.
        3. Constructs a large block-diagonal "mega-matrix" from the
           evaluated dictionaries.
        4. Concatenates all `y` and `h` vectors to solve the sparse
           coding problem globally.
        5. Unpacks the optimized `H_mega` back into individual `h` vectors
           for each block.

        Args:
            X (torch.Tensor): Input features for training (full dataset or a
                              batch).
            y (torch.Tensor): Target values for training.
            *_: Additional unused positional arguments.
        """
        if y.dim() == 1:
            y = y.unsqueeze(-1)

        # Add points to partition manager and initialize blocks if needed
        self.partition_manager.add_points(X, y)
        self.partition_manager.init_sparse_coding_per_block(
            config=self.sparse_coding_config,
            evaluation_func=self.evaluation_func
        )
        
        # Retrieve active blocks (these will be the same across model_epochs for this call to partial_fit)
        active_blocks = self.partition_manager.retrieve_active_blocks()
        if not active_blocks:
            self.logger.warning("No active blocks found. Skipping training.")
            return

        # Step 1: Aggregate data into nested_tensors
        X_nested, y_nested, h_nested = self._aggregate_block_data(active_blocks)
        
        # Main training loop for BSESM model_epochs
        for epoch in range(self.model_epochs):
            epoch_start_time = time.time()

            # Perform a single global training step for dictionary and sparse codes
            self._global_train_step(X_nested, y_nested, h_nested, active_blocks, epoch)

            self.elapsed_time += time.time() - epoch_start_time
            
            # Log progress for BSESM model_epochs
            if ( (self.config.log_interval > 0) and
                 ( (epoch + 1) % self.config.log_interval == 0 or
                   epoch == 0 or
                   epoch == self.model_epochs - 1 ) ):
                
                # Retrieve last losses for logging
                dict_loss = self.dictionary_layer.losses[-1] if self.dictionary_layer.losses else float('nan')
                sc_loss = self.global_sparse_coding_layer.losses[-1] if self.global_sparse_coding_layer.losses else float('nan')

                self.logger.info(
                    f"BSESM Epoch {epoch + 1}/{self.model_epochs}: "
                    f"Dict Loss: {dict_loss:.6f}, SC Loss: {sc_loss:.6f}"
                )
                
            # Call SESM hook if provided for monitoring after each SESM model epoch
            if self.sesm_hook is not None:
                hook_info = {
                    'partial_fit_call_count': self.partial_fit_count,
                    'model_epoch': epoch,
                    'dict_losses': self.dictionary_layer.losses,
                    'sparse_coding_losses': self.global_sparse_coding_layer.losses,
                    'h_mega': self.global_sparse_coding_layer.h.detach().clone(),
                    'dictionary_params': self.dictionary_layer.theta_params.detach().clone()
                }
                hook_info['h_per_block'] = [block.sparse_coding_layer.h.detach().clone() for block in active_blocks]
                self.sesm_hook(hook_info)

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

        test_active_blocks = self.partition_manager.retrieve_test_active_blocks(
            X, y)
        if len(test_active_blocks) == 0:
            self.logger.warning("No active test blocks found. "
                                "Returning empty prediction.")
            return torch.empty_like(y, device=X.device, dtype=X.dtype)

        # Aggregate test data into a nested_tensor for efficient batch evaluation
        X_list_test = [block.normalized_X for block in test_active_blocks]
        X_nested_test = torch.nested.nested_tensor(
            X_list_test, layout=torch.jagged,
            device=self.device_manager.get_device(
                DeviceTarget.DICTIONARY_LAYER))

        with torch.no_grad():
            # Evaluate the dictionary for all test blocks at once -> returns a
            # NestedTensor of matrices.
            dict_nested_test = self.dictionary_layer.forward(X_nested_test)
            dict_list_test = dict_nested_test.unbind()
            
            # Perform prediction for each block using its specific evaluated
            # dictionary and learned h
            y_pred_normalized_list = [
                self.evaluation_func(dict_i, block.sparse_coding_layer.h)
                for dict_i, block in zip(dict_list_test, test_active_blocks)
            ]

        # Denormalize and reconstruct the final output tensor
        # Ensure y_final_predictions is created with the correct shape and device
        y_final_predictions = torch.zeros(y.shape[0], device=X.device,
                                          dtype=X.dtype)
        
        for i, block in enumerate(test_active_blocks):
            # The y_pred_normalized_list elements are (N_samples_in_block, 1)
            block_preds_unnormalized = \
                y_pred_normalized_list[i] / block.amplitude
            y_final_predictions[block.positions] = \
                block_preds_unnormalized.squeeze()
            
        return y_final_predictions.view_as(y)

    def performance_stats(self, X: torch.Tensor, y: torch.Tensor):
        """
        Evaluate the model's performance on a given dataset using active
        sub-blocks.
        """
        y_pred = self.predict(X, y)
        time = self.elapsed_time / 60
        mse = mean_squared_error(y_pred.cpu().numpy(), y.cpu().numpy())
        return y_pred, time, mse
