"""
Dictionary Base Layer.

Defines the abstract base class `DictBaseLayer` and its configuration
`DictConfig`, providing the common interface and functionality for all
learnable dictionary implementations (e.g., Gaussian, Polynomial).

Copyright (c) 2023-2025, Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Callable, Iterator
import logging
from typing import Optional

import torch
from torch.utils.data import TensorDataset, DataLoader
from pysesm.base_types import BaseConfig, TensorBatch
from pysesm.utils.batch_ops import recursive_cat


@dataclass(kw_only=True)
class DictConfig(BaseConfig):
    """Base configuration for all dictionary types"""
    epochs: int
    batch_size: int | None = None
    alpha: float
    criterion: torch.nn.Module | None = None
    # Custom regularization function that takes the layer instance as input
    regularization_func: Optional[Callable[["DictBaseLayer"], torch.Tensor]] = None
    regularization_gamma: float = 0.01
    optimizer_factory: Callable[[Iterator[torch.nn.Parameter], float], torch.optim.Optimizer] | None = None
    

class DictBaseLayer(torch.nn.Module, ABC):
    """
    Abstract base class for all dictionary implementations.
    
    This class provides the common interface and functionality that all
    dictionary types must implement, while allowing each type to define
    its own specific training strategies and parameter initialization.
    """
    
    # Each subclass must define this to specify which config type it expects
    CONFIG_CLASS: type[DictConfig] = DictConfig
    
    theta_params: torch.nn.Parameter

    def __init__(
        self,
        config: DictConfig,
        n_features: int,
        n_functions: int,
        evaluation_func: Callable[[TensorBatch, TensorBatch], TensorBatch],
        logger: logging.Logger,
        parameter_hook: Callable[[dict], None] | None = None,
        **kwargs  # For subclass-specific parameters like 'psi' for Gaussian
    ):
        """
        Initialize the dictionary base layer.
        
        Args:
            config: Configuration object specific to the dictionary type
            n_features: Number of input features
            n_functions: Number of functions in the dictionary
            evaluation_func: Function to evaluate dictionary * h
            logger: Logger instance
            parameter_hook: Optional callback for parameter monitoring
            **kwargs: Additional arguments passed to subclasses
        """
        super().__init__()
        
        # Type check for config
        if not isinstance(config, self.CONFIG_CLASS):
            raise TypeError(f"Expected config of type {self.CONFIG_CLASS.__name__}, "
                           f"got {type(config).__name__}")
        
        self.config = config
        self.n_features = n_features
        self.n_functions = n_functions
        self.evaluation_func = evaluation_func
        self.logger = logger
        self.parameter_hook = parameter_hook
        self.device = torch.device(config.device) if config.device else torch.device('cpu')
        self.losses = []
        self.dictionary = None
        
        # Initialize parameters (subclass-specific)
        self.theta_params = self._initialize_parameters(**kwargs)
        
        # Setup criterion and optimizer
        self._setup_criterion()
        self._setup_optimizer()

        self.to(self.device)
    
    @abstractmethod
    def _initialize_parameters(self, **kwargs) -> torch.nn.Parameter:
        """
        Initialize the learnable parameters specific to this dictionary type.
        
        Args:
            **kwargs: Subclass-specific arguments
            
        Returns:
            torch.nn.Parameter: The initialized parameters for this dictionary
        """
        
    
    @abstractmethod
    def _evaluate_dictionary(self, X: TensorBatch, parameters: torch.Tensor, **kwargs) -> TensorBatch:
        """
        Evaluate the dictionary at given points with current parameters.
        
        Args:
            X: Input coordinates where to evaluate the dictionary
            parameters: Current parameter values
            **kwargs: Additional evaluation arguments (e.g., rho_flag, mu_flag for Gaussian)
            
        Returns:
            TensorBatch: Evaluated dictionary matrix
        """
        
    
    @abstractmethod
    def _train_with_strategy(self, X: TensorBatch, y: TensorBatch,
                             h: TensorBatch, log_losses: bool):
        """
        Implement the specific training strategy for this dictionary type.
        
        This method encapsulates the training logic specific to each dictionary
        type. For example, Gaussian dictionaries might split mu/rho training,
        while polynomial dictionaries might use unified training.
        
        Args:
            X: Input data
            y: Target data  
            h: Sparse coding vector (detached)
            log_losses: Whether to log training losses
        """
        
    
    def _setup_criterion(self):
        """Setup the loss criterion"""
        if self.config.criterion is None:
            self.criterion = torch.nn.MSELoss()
        else:
            self.criterion = self.config.criterion
    
    def _setup_optimizer(self):
        """Setup the optimizer"""
        if self.config.optimizer_factory is None:
            # Nota: self.parameters() devuelve un iterador de todos los parámetros del módulo.
            # Aquí, solo necesitamos pasar nuestro parámetro específico.
            self.optimizer = torch.optim.SGD([self.theta_params],
                                             lr=self.config.alpha,
                                             weight_decay=0)            
        else:
            self.optimizer = self.config.optimizer_factory([self.theta_params],
                                                           lr=self.config.alpha)
            
    def _train_epoch(self, X: TensorBatch, y: TensorBatch, h: TensorBatch, 
                    log_losses: bool, **eval_kwargs) -> None:
        if self.config.batch_size is not None:
            full_dataset = TensorDataset(X, y)
            data_loader = DataLoader(full_dataset, self.config.batch_size, shuffle=False)
            all_dicts = []
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                self._train_epoch_for_batch(x_batch, y_batch, h, log_losses, **eval_kwargs)
                all_dicts.append(self.dictionary)
            self.dictionary = recursive_cat(all_dicts)
        else:
            self._train_epoch_for_batch(X, y, h, log_losses, **eval_kwargs)

    def _train_epoch_for_batch(self, X: TensorBatch, y: TensorBatch, h: TensorBatch, 
                    log_losses: bool, **eval_kwargs) -> None:
        """
        Perform a single training epoch with batch input support.
        
        This is a common training step that can be used by subclasses.
        X, y, h are expected to be TensorBatch.
        """

        # Ensure all components within TensorBatch are on the correct device if not already
        # This is handled by evaluation_func and criterion expecting device-correct tensors.
        # No need for explicit .to(self.device) here on X, y, h directly.

        self.optimizer.zero_grad()
        
        # self.dictionary will be a TensorBatch (output of psi.__call__)
        # eval_kwargs are propagated here, so GaussianDictLayer can pass
        # mu_flag/rho_flag
        self.dictionary = self.forward(X, **eval_kwargs)
        
        # evaluation_func needs to handle TensorBatch inputs for D and h
        # The output y_pred will also be a TensorBatch
        # Ensure h is detached for dictionary training (gradients should
        # not flow back to h)
        h_detached = self._detach_tensor_batch(h)
        y_pred = self.evaluation_func(self.dictionary, h_detached)
        
        loss_recon = self._calculate_batch_loss(y_pred, y)

        # Calculate regularization loss if a function was provided
        loss_reg = torch.tensor(0.0, device=self.device, dtype=torch.float32, requires_grad=True)
        if self.config.regularization_func is not None:
            loss_reg = self.config.regularization_func(self)

        # Total loss
        total_loss = loss_recon + self.config.regularization_gamma * loss_reg

        total_loss.backward(retain_graph=False)
        self.optimizer.step()
        
        if log_losses:
            self.losses.append(total_loss.item())
        
        # Call parameter hook if provided
        if self.parameter_hook is not None:
            hook_info = {
                'epoch': len(self.losses),
                'theta_params': self.theta_params.clone().detach(),
                'loss': total_loss.item(),
                'loss_reconstruction': loss_recon.item(),
                'loss_regularization': loss_reg.item()
            }
            # Subclasses can add more specific info to hook_info
            self._add_hook_info(hook_info, **eval_kwargs)
            self.parameter_hook(hook_info)

    def _calculate_batch_loss(self, y_pred: TensorBatch,
                               y: TensorBatch) -> torch.Tensor:
        """
        Helper method to calculate the overall loss for TensorBatch inputs.
        It flattens all TensorBatch types into a single 2D torch.Tensor
        before passing them to the criterion for a unified loss calculation.
        """
        y_pred_flat: torch.Tensor
        y_flat: torch.Tensor
 
        if getattr(y_pred, "is_nested", False) and \
           getattr(y, "is_nested", False):
            # For NestedTensor inputs, use .values() to get the contiguous
            # flattened tensor. This is efficient (O(1) view operation).
            y_pred_flat = y_pred.values()
            y_flat = y.values()
        elif isinstance(y_pred, list) and \
             isinstance(y, list):
            # For List[torch.Tensor] inputs, concatenate them. This creates
            # a new tensor (O(N) operation), which is the less efficient
            # fallback, but robust.
            y_pred_flat = torch.cat(y_pred, dim=0)
            y_flat = torch.cat(y, dim=0)
        elif isinstance(y_pred, torch.Tensor) and isinstance(y, torch.Tensor):
            # For torch.Tensor inputs (2D or 3D from vmap), flatten them
            # to (Total_Samples, Output_Dim).
            y_pred_flat = y_pred.flatten(0, -2)
            y_flat = y.flatten(0, -2)
        else:
            raise TypeError(
                "Mismatched or unsupported TensorBatch types for loss "
                f"calculation: y_pred={type(y_pred)}, y={type(y)}"
            )
        
        # Assert that the flattened tensors have compatible shapes
        # (e.g., (N, C) == (N, C)). This catches broadcasting issues.
        assert y_pred_flat.shape == y_flat.shape, \
            f"Shape mismatch after flattening for loss calculation: " \
            f"y_pred_flat.shape={y_pred_flat.shape}, y_flat.shape={y_flat.shape}"

        # Now, with y_pred_flat and y_flat consistently 2D tensors,
        # pass them to the criterion. The criterion's 'reduction'
        # (e.g., 'mean' or 'sum') will handle the final aggregation.
        return self.criterion(y_pred_flat, y_flat)



    def _detach_tensor_batch(self, tensor_batch: TensorBatch) -> TensorBatch:
        """Detaches all tensors within a TensorBatch."""
        if isinstance(tensor_batch, torch.Tensor):
            return tensor_batch.detach()
        if getattr(tensor_batch,"is_nested",False):
            # Creating a new nested_tensor from detached components
            detached_tensors = [t.detach() for t in tensor_batch.unbind()]
            return torch.nested.as_nested_tensor(detached_tensors,
                                                  layout=tensor_batch.layout)
        if isinstance(tensor_batch, list):
            return [t.detach() for t in tensor_batch]
        
        raise TypeError("Unsupported TensorBatch type for detach: "
                        f"{type(tensor_batch)}")


    def _add_hook_info(self, hook_info: dict, **eval_kwargs):
        """
        Add dictionary-specific information to the parameter hook.
        Subclasses can override this to add specialized info.
        """
        
    
    def setup(self, X: torch.Tensor) -> None:
        """
        Initialize the dictionary for the layer.
        
        Args:
            X: Input data to initialize dictionary evaluation
        """
        X = X.to(self.device)
        if self.dictionary is None:
            self.dictionary = self._evaluate_dictionary(X, self.theta_params)
    
    def partial_fit(self, X: TensorBatch, y: TensorBatch, h: TensorBatch, 
                    log_losses: bool = True) -> None:
        """
        Public interface for training the dictionary, supporting TensorBatch inputs.
        
        This method delegates to the specific training strategy implemented
        by each dictionary type.

        It expects X, y, and h to be `TensorBatch` types (torch.Tensor (2D/3D),
        nested_tensor, or List[torch.Tensor]).
        
        Args:
            X (TensorBatch): Input data points.
            y (TensorBatch): Target data.
            h (TensorBatch): Sparse coding vectors (should be detached,
                             but the layer will handle this).
            log_losses: Whether to log training losses
        """
        self._train_with_strategy(X, y, h, log_losses)
    
    def forward(self, X: TensorBatch, **kwargs) -> TensorBatch:
        """
        Evaluate dictionary at given points.
        
        Args:
            X: Input coordinates
            **kwargs: Additional evaluation arguments
            
        Returns:
            TensorBatch: Evaluated dictionary
        """
        evaluated_dictionary = self._evaluate_dictionary(X, self.theta_params,
                                                         **kwargs)
        return evaluated_dictionary
