'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica
BSESM Base Class
Provides the batched version of SESM
Authors: The SESM Team 
License: 
'''
from __future__ import annotations

import logging
import time
import warnings

from collections.abc import Callable

from dataclasses import dataclass
from sklearn.metrics import mean_squared_error
import torch

from pysesm.blocks import PartitionBlock
from pysesm.models.SESM import SESM, SESMConfig
from pysesm.device_manager.DeviceManager import DeviceManager
from pysesm.factories.SparseCodingFactory import SparseCodingFactory
from pysesm.enums.DeviceTargetEnum import DeviceTarget
from pysesm.base_types import TensorBatch 


@dataclass
class BSESMConfig(SESMConfig):
    """
    Configuration for BSESM model, extending base SESMConfig.
    """


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
        device_manager: DeviceManager | None = None,
        dict_layer_hook: Callable[[dict], None] | None = None,
        sparse_coding_layer_hook: Callable[[dict], None] | None = None,
        sesm_hook: Callable[[dict], None] | None = None,
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
        if isinstance(dictionary, torch.Tensor) and dictionary.dim() <= 2:
            return torch.matmul(dictionary, h)
        if isinstance(dictionary, torch.Tensor) and dictionary.dim() == 3:
            return torch.vmap(torch.matmul)(dictionary, h)
        if isinstance(dictionary, list) and isinstance(h, list):
            results = [torch.matmul(d, hi) for d, hi in zip(dictionary, h)]
            return results

        raise TypeError("Unsupported TensorBatch types for evaluation_func: "
                        f"D={type(dictionary)}, h={type(h)}")

    def _aggregate_block_data(self, blocks: list[PartitionBlock]
                              ) -> tuple[TensorBatch, TensorBatch, TensorBatch]:
        """
        Aggregates data from active blocks into nested_tensors for efficient
        batched processing without padding.
        
        Args:
            blocks (List[PartitionBlock]): List of active blocks.
            
        Returns:
            Tuple containing three nested_tensors:
                - X_nested (TensorBatch): Normalized input features for each block.
                - y_nested (TensorBatch): Target values for each block.
                - h_nested (TensorBatch): `h` vectors for each block.
        """
        if not blocks:
            # Return empty nested tensors. This handles cases where no
            # active blocks are found.
            # PyTorch's nested_tensor constructor for empty lists actually creates
            # a nested_tensor that contains a single zero-sized tensor.
            # So, .unbind() on it will return a list with one empty tensor.
            # This is the expected behavior for now, to avoid RuntimeError.
            device = self.device_manager.get_device(DeviceTarget.DICTIONARY_LAYER)
            empty_tensor_list = [torch.empty(0, self.n_features,
                                             device=device)]
            empty_y_list = [torch.empty(0, 1,
                                        device=device)]
            empty_h_list = [torch.empty(0, self.n_functions, 1,
                                        device=device)]
            return (torch.nested.nested_tensor(empty_tensor_list,
                                               layout=torch.jagged,
                                               device=device),
                    torch.nested.nested_tensor(empty_y_list,
                                               layout=torch.jagged,
                                               device=device),
                    torch.nested.nested_tensor(empty_h_list,
                                               layout=torch.jagged,
                                               device=device))

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
                           active_blocks: list[PartitionBlock],
                           epoch: int):
        """
        Performs a single global training step for the BSESM model.
        This includes training the dictionary on all active blocks' data
        and then training the global sparse coding layer for the current epoch.

        Args:
            X_nested (TensorBatch): Aggregated normalized input features for all active blocks.
            y_nested (TensorBatch): Aggregated target values for all active blocks.
            h_nested (TensorBatch): Aggregated sparse coding vectors for all active blocks.
            active_blocks (List[PartitionBlock]): List of PartitionBlock objects currently active.
            epoch (int): The current SESM model epoch.

        Returns:
            TensorBatch: Updated h_nested after sparse coding optimization.
        """
        
        # Step 1: Optimize dictionary with fixed h
        self.dictionary_layer.partial_fit(
            X=X_nested,
            y=y_nested,
            h=h_nested  # h is detached internally by DictBaseLayer for dict training
        )
        # Dictionary losses are internally tracked by dictionary_layer.losses.
        self._dict_losses.append(self.dictionary_layer.losses[-1])

        # Step 2: Optimize h with fixed dictionary (Mega-Matrix approach)
        # Use the already calculated dictionary (like SSESM)
        dict_nested_detached = self.dictionary_layer.dictionary.detach()

        # Unbind NestedTensors to get lists for torch.cat and block_diag
        y_list_unbound = y_nested.unbind()
        dict_list_unbound = dict_nested_detached.unbind()

        # Pack data into the mega-matrix formulation
        Y_mega = torch.cat(y_list_unbound).detach()  # Concatenate all y_i into a single column vector
        D_mega = torch.block_diag(*dict_list_unbound).detach()  # Create block-diagonal matrix D_mega

        # Update the number of functions for the global SC layer dynamically
        self.global_sparse_coding_layer.config.n_functions = D_mega.shape[1]

        # Solve the single, large sparse coding problem
        self.global_sparse_coding_layer.partial_fit(y=Y_mega,
                                                    dictionary=D_mega,
                                                    reset_state=(epoch==0))

        self._sparse_coding_losses.append(self.global_sparse_coding_layer.losses[-1])

        # Update h_nested directly with optimized values - zero extra allocations
        # Both have identical memory layout: (total_functions, 1)
        h_nested.values().data.copy_(self.global_sparse_coding_layer.h.detach())

        return h_nested


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

        # Step 1: Aggregate data into nested_tensors (X and y don't change during training)
        X_nested, y_nested, h_nested = self._aggregate_block_data(active_blocks)

        # Initialize/Update self.global_sparse_coding_layer's h
        H_mega_initial = h_nested.values()

        # Update n_functions for the global SC layer's config
        self.global_sparse_coding_layer.config.n_functions = H_mega_initial.shape[0]

        # Decide if we need to force a full setup for global SC layer (recreating h Parameter)
        if self.global_sparse_coding_layer.h.shape[0] != H_mega_initial.shape[0]:
            self.logger.debug(f"Global sparse coding layer's h dimension changed from "
                              f"{self.global_sparse_coding_layer.h.shape[0]} to "
                              f"{H_mega_initial.shape[0]}. Forcing setup.")
            self.global_sparse_coding_layer.setup(H_mega_initial)
        else:
            self.global_sparse_coding_layer.h.data = H_mega_initial.to(self.global_sparse_coding_layer.device)

        # Main training loop for BSESM model_epochs
        for epoch in range(self.model_epochs):
            epoch_start_time = time.time()

            # Perform a single global training step and get updated h_nested
            h_nested = self._global_train_step(X_nested, y_nested, h_nested, active_blocks, epoch)

            self.elapsed_time += time.time() - epoch_start_time

            # Log progress for BSESM model_epochs
            if ( (self.config.log_interval > 0) and
                 ( (epoch + 1) % self.config.log_interval == 0 or
                   epoch == 0 or
                   epoch == self.model_epochs - 1 ) ):

                # Retrieve last losses for logging
                dict_loss = self.dictionary_layer.losses[-1] if self.dictionary_layer.losses else float('nan')
                sc_loss = self._sparse_coding_losses[-1] if self._sparse_coding_losses else float('nan')

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
                    'sparse_coding_losses': self._sparse_coding_losses,
                    'h_mega': self.global_sparse_coding_layer.h.detach().clone(),
                    'dictionary_params': self.dictionary_layer.theta_params.detach().clone()
                }
                # Extract current h values from h_nested for monitoring
                h_list_current = h_nested.unbind()
                hook_info['h_per_block'] = [h.detach().clone() for h in h_list_current]
                self.sesm_hook(hook_info)

        # Step 2: After training is complete, distribute final h values to individual blocks
        h_list_final = h_nested.unbind()
        for i, block in enumerate(active_blocks):
            # Update the h-vector of the individual sparse coding layer within each block object.
            block.sparse_coding_layer.h.data = h_list_final[i].to(block.sparse_coding_layer.device)

        self.partial_fit_count += 1


    def predict(self, X: torch.Tensor, _y: torch.Tensor = None) -> torch.Tensor:
        """
        Predict the output using the trained BSESM model with active sub-blocks.

        Args:
            X (torch.Tensor): Input features for prediction.
            _y (torch.Tensor): Target values (used to identify active blocks).

        Returns:
            torch.Tensor: Predicted values for the input dataset.
        """

        if _y is not None:
            warnings.warn("Deprecated behaviour: predict does not need y values",
                          DeprecationWarning, stacklevel=2)

        
        active_blocks = self.partition_manager.retrieve_inference_blocks(X)
        
        if len(active_blocks) == 0:
            self.logger.warning("No active test blocks found. "
                                "Returning empty prediction.")
            return torch.zeros(X.shape[0], 1, device=X.device, dtype=X.dtype)

        # Create final tensor directly with correct shape
        y_final_predictions = torch.zeros(X.shape[0], 1,
                                          device=X.device,
                                          dtype=X.dtype)

        
        # Aggregate test data into a nested_tensor for efficient batch evaluation
        X_list_test = [block.normalized_X for block in active_blocks]
        X_nested_test = torch.nested.nested_tensor(
            X_list_test, layout=torch.jagged,
            device=self.device_manager.get_device(DeviceTarget.DICTIONARY_LAYER))

        with torch.no_grad():
            # Evaluate the dictionary for all test blocks at once -> returns a
            # NestedTensor of matrices.
            dict_nested_test = self.dictionary_layer.forward(X_nested_test)
            dict_list_test = dict_nested_test.unbind()
            
            # Perform prediction for each block using its specific evaluated
            # dictionary and learned h
            y_pred_normalized_list = [
                self.evaluation_func(dict_i, block.sparse_coding_layer.h) \
                for dict_i, block in zip(dict_list_test, active_blocks)
            ]


        # Fill predictions directly into final tensor
        for i, block in enumerate(active_blocks):
            # The y_pred_normalized_list elements are (N_samples_in_block, 1)
            block_preds_unnormalized = y_pred_normalized_list[i] / block.amplitude
            y_final_predictions[block.positions, 0] = block_preds_unnormalized[:,0]
        
        return y_final_predictions

    def performance_stats(self, X: torch.Tensor, y: torch.Tensor):
        """
        Evaluate the model's performance on a given dataset using active
        sub-blocks.
        """
        y_pred = self.predict(X)
        current_time = self.elapsed_time / 60
        mse = mean_squared_error(y_pred.cpu().numpy(), y.cpu().numpy())
        return y_pred, current_time, mse
