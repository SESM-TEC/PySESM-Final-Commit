"""
Batched Sparse-Encoded Surrogate Model (BSESM).

Implements the Batched version of the SESM architecture, which processes
data in aggregated batches using a global dictionary and a block-diagonal
sparse coding strategy for parallel optimization.

Copyright (c) 2023-2025, Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
Version: 0.1.1
"""
import logging
import time
from enum import Enum, auto

from collections.abc import Callable

import copy
from dataclasses import dataclass
import torch
import torch.nn.functional as F

from pysesm.blocks import PartitionBlock
from pysesm.models.SESM import SESM, SESMConfig
from pysesm.factories.SparseCodingFactory import SparseCodingFactory
from pysesm.base_types import TensorBatch, TensorProxy 

class BSESMSolverStrategy(Enum):
    MEGA_MATRIX = auto()
    SEQUENTIAL = auto()

@dataclass
class BSESMConfig(SESMConfig):
    """
    Configuration for BSESM model, extending base SESMConfig.
    """
    solver_strategy: BSESMSolverStrategy = BSESMSolverStrategy.SEQUENTIAL

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

    @staticmethod
    def _tensor_mb(tensor: torch.Tensor) -> float:
        return (tensor.numel() * tensor.element_size()) / (1024 ** 2)

    @staticmethod
    def _shape_tuple(tensor: torch.Tensor) -> tuple[int, ...]:
        return tuple(int(dim) for dim in tensor.shape)

    def __init__(
        self,
        config: BSESMConfig,
        logger: logging.Logger,
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
            dict_layer_hook: Optional callback for dictionary layer monitoring.
            sparse_coding_layer_hook: Optional callback for sparse coding layer monitoring.
            sesm_hook: Optional callback for SESM-level monitoring.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            config=config,
            logger=logger,
            sesm_hook=sesm_hook,
            dict_layer_hook=dict_layer_hook,
            sparse_coding_layer_hook=sparse_coding_layer_hook,
            **kwargs,
        )

        # The global sparse coding layer will operate on the large block-diagonal matrix,
        # so it uses standard matrix multiplication.
        self.global_sparse_coding_layer = None
        if self.config.solver_strategy == BSESMSolverStrategy.MEGA_MATRIX:
            global_sc_config = copy.deepcopy(self.sparse_coding_config)
            self.global_sparse_coding_layer = SparseCodingFactory.create(
                config=global_sc_config,
                evaluation_func=self.evaluation_func,
                logger=self.logger,
                parameter_hook=self.sparse_coding_layer_hook
            )
            self.logger.info(f"Global Sparse Coding Layer: {type(self.global_sparse_coding_layer).__name__}")
        else:
            self.logger.info("Using Sequential Sparse Coding Solver Strategy. No global layer allocated.")

        # Structure metrics captured during training
        self.structure_metrics: dict[str, str | None] = {
            'theta_shape': None,
            'theta_mb': None,
            'd_mega_shape': None,
            'y_mega_shape': None,
        }
        self._max_dict_eval_mb = 0.0
        self._max_d_mega_mb = 0.0

    def _capture_structure_metrics(
        self,
        theta_params: torch.Tensor,
        dict_eval_mb: float,
        D_mega: torch.Tensor,
        Y_mega: torch.Tensor,
    ) -> None:
        """Captura métricas estructurales y conserva los máximos observados.

        Este método se invoca en cada época del entrenamiento global.
        Actualiza siempre los metadatos actuales de `theta` y mantiene los
        picos de memoria para `dict_eval_mb` y `D_mega`.

        Args:
            theta_params (torch.Tensor): Parámetros actuales del diccionario.
            dict_eval_mb (float): Memoria total (MB) de diccionarios evaluados en la época.
            D_mega (torch.Tensor): Matriz bloque-diagonal usada en sparse coding global.
            Y_mega (torch.Tensor): Vector/objetivo concatenado correspondiente a `D_mega`.
        """
        # Tamaños actuales en esta época.
        current_theta_mb = self._tensor_mb(theta_params)
        current_d_mega_mb = self._tensor_mb(D_mega)

        # Estado actual de theta (no pico).
        self.structure_metrics['theta_shape'] = str(self._shape_tuple(theta_params))
        self.structure_metrics['theta_mb'] = f"{current_theta_mb:.6f}"

        # Pico acumulado de evaluación de diccionario.
        if dict_eval_mb >= self._max_dict_eval_mb:
            self._max_dict_eval_mb = dict_eval_mb
            self.structure_metrics['dict_eval_mb_max'] = f"{dict_eval_mb:.6f}"

        # Pico acumulado de la mega-matriz y su shape asociado.
        if current_d_mega_mb >= self._max_d_mega_mb:
            self._max_d_mega_mb = current_d_mega_mb
            self.structure_metrics['d_mega_shape'] = str(self._shape_tuple(D_mega))
            self.structure_metrics['d_mega_mb_max'] = f"{current_d_mega_mb:.6f}"
            self.structure_metrics['y_mega_shape'] = str(self._shape_tuple(Y_mega))
            
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

    def _aggregate_block_data(self, 
                              blocks: list[PartitionBlock],
                              device: str | torch.device
                              ) -> tuple[TensorBatch, TensorBatch, TensorBatch]:
        """
        Aggregates data from active blocks into nested_tensors for efficient
        batched processing without padding.
        
        Args:
            blocks (List[PartitionBlock]): List of active blocks.
            device (str | torch.device): The target device for the aggregated tensors

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

        X_list = [block.normalized_X.get_for_device(device) for block in blocks]
        y_list = [block.target.get_for_device(device) for block in blocks]
        # h needs to be detached here because DictBaseLayer.partial_fit
        # expects h to be detached for dictionary training.
        h_list = [block.sparse_coding_layer.h.detach().to(device) for block in blocks] 
        
        X_nested = torch.nested.nested_tensor(
            X_list, layout=torch.jagged,
            device=device)
        y_nested = torch.nested.nested_tensor(
            y_list, layout=torch.jagged,
            device=device)
        h_nested = torch.nested.nested_tensor(
            h_list, layout=torch.jagged,
            device=device)
        
        return X_nested, y_nested, h_nested


    def _global_train_step(self,
                           X_nested_proxy: TensorProxy,
                           y_nested_proxy: TensorProxy,
                           h_nested: TensorBatch,
                           epoch: int,
                           active_blocks: list[PartitionBlock]):
        """
        Performs a single global training step for the BSESM model.
        This includes training the dictionary on all active blocks' data
        and then training the global sparse coding layer for the current epoch.

        Args:
            X_nested_proxy (TensorProxy): Proxy to the aggregated normalized input features.
            y_nested_proxy (TensorProxy): Proxy to the aggregated target values.
            h_nested (TensorBatch): The current state of aggregated sparse vectors.
            epoch (int): The current SESM model epoch.
            active_blocks (list[PartitionBlock]): List of active blocks to access their sparse coders.

        Returns:
            TensorBatch: Updated h_nested after sparse coding optimization.
        """
        
        dict_device = self.dictionary_layer.device
        sc_device = self.global_sparse_coding_layer.device if self.global_sparse_coding_layer else active_blocks[0].sparse_coding_layer.device

        # Step 1: Optimize dictionary with fixed h
        self.dictionary_layer.partial_fit(
            X=X_nested_proxy.get_for_device(dict_device),
            y=y_nested_proxy.get_for_device(dict_device),
            h=h_nested  # h is detached internally by DictBaseLayer for dict training
        )
        # Dictionary losses are internally tracked by dictionary_layer.losses.
        self._dict_losses.append(self.dictionary_layer.losses[-1])

        # Step 2: Optimize h with fixed dictionary (Mega-Matrix approach)
        
        # Use the already calculated dictionary (like SSESM)
        dict_nested_detached = self.dictionary_layer.dictionary.detach()

        if self.config.solver_strategy == BSESMSolverStrategy.MEGA_MATRIX:
            # Unbind the dictionary's nested tensor, moving components only if necessary.
            if sc_device == dict_nested_detached.device:
                dict_list_sc = dict_nested_detached.unbind()
            else:
                dict_list_sc =[d.to(sc_device) for d in dict_nested_detached.unbind()]
            D_mega = torch.block_diag(*dict_list_sc)

            # Transfer y_nested and get its contiguous values on the target device
            Y_mega = y_nested_proxy.get_for_device(sc_device).values()
            
            # Solve the single, large sparse coding problem
            self.global_sparse_coding_layer.partial_fit(y=Y_mega,
                                                        dictionary=D_mega,
                                                        reset_state=(epoch==0))

            self._sparse_coding_losses.append(self.global_sparse_coding_layer.losses[-1])

            # Update h_nested directly with optimized values - zero extra allocations
            # Both have identical memory layout: (total_functions, 1)
            h_nested.values().data.copy_(self.global_sparse_coding_layer.h.detach().to(h_nested.device))
        elif self.config.solver_strategy == BSESMSolverStrategy.SEQUENTIAL:
            dict_list_sc =[d.to(sc_device) for d in dict_nested_detached.unbind()]
            y_list_sc =[y.to(sc_device) for y in y_nested_proxy.get_for_device(sc_device).unbind()]
            h_list_sc = h_nested.unbind()
            
            weighted_loss_sum = 0.0
            total_samples = 0
            for i, block in enumerate(active_blocks):
                sc_layer = block.sparse_coding_layer
                sc_layer.h.data.copy_(h_list_sc[i].to(sc_layer.device))
                sc_layer.partial_fit(y=y_list_sc[i], dictionary=dict_list_sc[i], reset_state=(epoch==0))
                h_list_sc[i].data.copy_(sc_layer.h.detach().to(h_nested.device))
                n_i = y_list_sc[i].shape[0]
                weighted_loss_sum += sc_layer.losses[-1] * n_i
                total_samples += n_i
            
            self._sparse_coding_losses.append(weighted_loss_sum / total_samples if total_samples > 0 else 0.0)
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

        # --- Step 1: Aggregate data ONCE and create proxies for static data ---
        dict_device = self.dictionary_layer.device
        sc_device = self.global_sparse_coding_layer.device if self.global_sparse_coding_layer else active_blocks[0].sparse_coding_layer.device 

        # Aggregate all data onto the dictionary's device initially. This is the single, expensive operation.
        X_nested, y_nested, h_nested = self._aggregate_block_data(active_blocks, device=dict_device)

        # Wrap the static aggregated data (X, y) in proxies for efficient device access later.
        X_nested_proxy = TensorProxy(X_nested)
        y_nested_proxy = TensorProxy(y_nested)

        # Transfer the contiguous block of h data efficiently to the sparse coding device.
        H_mega_initial = h_nested.values().to(sc_device)

        if self.config.solver_strategy == BSESMSolverStrategy.MEGA_MATRIX:
            self.global_sparse_coding_layer.config.n_functions = H_mega_initial.shape[0]

            # Decide if we need to force a full setup for global SC layer (recreating h Parameter)
            if self.global_sparse_coding_layer.h.shape[0] != H_mega_initial.shape[0]:
                self.logger.debug(f"Global sparse coding layer's h dimension changed from "
                                  f"{self.global_sparse_coding_layer.h.shape[0]} to "
                                  f"{H_mega_initial.shape[0]}. Forcing setup.")
                self.global_sparse_coding_layer.setup(H_mega_initial)
            else:
                self.global_sparse_coding_layer.h.data.copy_(H_mega_initial)

        # Main training loop for BSESM model_epochs
        for epoch in range(self.model_epochs):
            epoch_start_time = time.time()

            # Perform a single global training step and get updated h_nested
            h_nested = self._global_train_step(X_nested_proxy, y_nested_proxy, h_nested, epoch, active_blocks)

            self.training_time += time.time() - epoch_start_time

            # Log progress for BSESM model_epochs
            if ( (self.config.log_interval > 0) and
                 ( (epoch + 1) % self.config.log_interval == 0 or
                   epoch == 0 or
                   epoch == self.model_epochs - 1 ) ):

                self.logger.info(
                    f"BSESM Epoch {epoch + 1}/{self.model_epochs}: training step completed"
                )

            # Call SESM hook if provided for monitoring after each SESM model epoch
            if self.sesm_hook is not None:
                # Extract current h values from h_nested for monitoring
                h_list_current = h_nested.unbind()
                h_mega = self.global_sparse_coding_layer.h.detach().clone() if self.global_sparse_coding_layer else h_nested.values().detach().clone()
                hook_info = {
                    'partial_fit_call_count': self.partial_fit_count,
                    'model_epoch': epoch,
                    'dictionary_losses': self.dictionary_layer.losses,
                    'sparse_coding_losses': self._sparse_coding_losses,
                    'h_mega': h_mega,
                    'dictionary_params': self.dictionary_layer.theta_params.detach().clone(),
                    'h_per_block': [h.detach().clone() for h in h_list_current],
                }
                self.sesm_hook(hook_info)

        # Step 2: After training is complete, distribute final h values to individual blocks
        h_list_final = h_nested.unbind()
        for i, block in enumerate(active_blocks):
            # Update the h-vector of the individual sparse coding layer within each block object.
            block.sparse_coding_layer.h.data = h_list_final[i].to(block.sparse_coding_layer.device)

        self.partial_fit_count += 1


    def predict(self, X: torch.Tensor, custom_h: torch.Tensor | None = None) -> torch.Tensor:
        """
        Predict the output using the trained BSESM model with active sub-blocks.

        Args:
            X (torch.Tensor): Input features for prediction.
            custom_h (torch.Tensor | None): A custom "mega" h-vector containing the
                concatenated weights for all active blocks. Used for debugging.

        Returns:
            torch.Tensor: Predicted values for the input dataset.
        """

       
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
        X_list_test = [block.normalized_X.get_for_device(self.dictionary_layer.device) 
                       for block in active_blocks]
        X_nested_test = torch.nested.nested_tensor(
            X_list_test, layout=torch.jagged,
            device=self.dictionary_layer.device)

        with torch.no_grad():
            # Evaluate the dictionary for all test blocks at once -> returns a
            # NestedTensor of matrices.
            dict_nested_test = self.dictionary_layer.forward(X_nested_test)
            dict_list_test = dict_nested_test.unbind()

            # Prepare h vectors for prediction, handling the optional custom_h.
            if custom_h is not None:
                h_sizes = [b.sparse_coding_layer.h.shape[0] for b in active_blocks]
                if custom_h.shape[0] != sum(h_sizes):
                    raise ValueError(f"custom_h size ({custom_h.shape[0]}) must match "
                                     f"total size of all block h's ({sum(h_sizes)})")
                h_sections = torch.split(custom_h, h_sizes)
                h_to_use_list = [h.to(self.dictionary_layer.device) for h in h_sections]
            else:
                h_to_use_list = [b.sparse_coding_layer.h.to(self.dictionary_layer.device) for b in active_blocks]
            
            # Perform prediction for each block using its specific evaluated
            # dictionary and learned h
            y_pred_normalized_list = [
                self.evaluation_func(
                    dict_i, h_i
                ) 
                for dict_i, h_i in zip(dict_list_test, h_to_use_list)
            ]


        # Fill predictions directly into final tensor
        for i, block in enumerate(active_blocks):
            # The y_pred_normalized_list elements are (N_samples_in_block, 1)
            block_preds_unnormalized = y_pred_normalized_list[i] / block.amplitude # Still on dict_layer's device
            # Move prediction to the final tensor's device (CPU) before assignment
            y_final_predictions[block.positions, 0] = block_preds_unnormalized.to(y_final_predictions.device)[:,0]
            
        return y_final_predictions

    def performance_stats(self, X: torch.Tensor, y: torch.Tensor):
        """
        Evaluate the model's performance on a given dataset using active
        sub-blocks.
        """
        y_pred = self.predict(X)

        current_time = self.training_time / 60
        
        # Ensure both tensors are on the same device for the calculation.
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        
        mse = F.mse_loss(y_pred,y.to(y_pred.device))
        return y_pred, current_time, mse.item()
