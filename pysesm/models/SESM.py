'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica

SESM Base Class

Provides the basic functionality of the Sparse-Encoded Surrogate Model.

Authors: The SESM Team 

License: 
'''
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch

from ..dictionaries.DictBaseLayer import DictBaseLayer, DictConfig
from ..sparse_coding.SparseCodingBaseLayer import SparseCodingBaseLayer, SparseCodingConfig
from ..factories.SparseCodingFactory import SparseCodingFactory
from ..factories.DictFactory import DictFactory
from ..factories.BlockManagerFactory import BlockManagerFactory
from ..blocks.PartitionBlock import PartitionBlock
from ..blocks.BlockManager import BlockManager, BlockManagerConfig
from ..base_types import BaseConfig, TensorProxy

@dataclass(kw_only=True)
class SESMConfig(BaseConfig):
    """
    Configuration for SESM model.
    
    This dataclass encapsulates all configuration parameters required to initialize
    and train a SESM model, promoting cleaner interfaces and easier parameter management.
    
    Attributes:
        n_features (int): The number of input features or dimensions that the model will work with.
            Each input tensor X is expected to have this many features.
                
        model_epochs (int): The number of epochs for the overall SESM training loop,
            including both dictionary learning and sparse representation adjustments.
        
        sparse_coding_config (SparseCodingConfig): Configuration for the sparse coding layer
            (e.g., ISTA, FISTA, ADMM). Contains algorithm-specific parameters and the number
            of epochs for sparse coding optimization, and most importantly, how many words
            the dictionary has (n_functions).
        
        dict_config (DictConfig): Configuration for the dictionary layer. Contains parameters
            specific to the dictionary type (e.g., GaussianDictConfig for Gaussian dictionaries).

        partition_config (BlockManagerConfig): Configuration for the block manager, or input space
            partition strategy.
        
        seed (int): Random seed for reproducibility of training processes, including
            weight initialization and other stochastic operations.
        
        evaluation_func (Union[str, Callable]): Function to combine dictionary and sparse vector.
            Can be a string key (EVAL_DEFAULT, EVAL_TWOD_MULT, EVAL_BMM_MULT) for standard
            functions, or a custom callable with signature (dictionary, h) -> predictions.
            Default is EVAL_DEFAULT which uses standard matrix multiplication.
        
        log_interval (int): Every how many epochs of the main SESM loop should we log progress.
    """
    n_features: int
    model_epochs: int
    sparse_coding_config: SparseCodingConfig
    dict_config: DictConfig
    partition_config: BlockManagerConfig
    seed: int = None
    log_interval: int = 1


class SESM(torch.nn.Module, ABC):
    """
    Sparse-Encoded Surrogate Model (SESM) - Base implementation.
    
    The SESM architecture is designed for surrogate modeling and function approximation tasks,
    leveraging sparse encoding techniques and dictionary-based representations for efficient
    training and inference. This module provides the core functionalities for building and
    training a SESM model, including the integration of sparse coding algorithms (ISTA, FISTA,
    ADMM) and dictionary learning techniques.
    
    The model learns a dictionary D and sparse codes h such that y ≈ D @ h, where:
    - D is the learned dictionary (managed by dictionary_layer)
    - h is the sparse representation (managed by sparse_coding_layer)
    - y is the target function to approximate
    
    Attributes:
        sparse_coding_layer (SparseCodingBaseLayer): Layer responsible for finding and
            maintaining sparse representations through algorithms like ISTA, FISTA, or ADMM.
            In partitioned settings, each block has its own sparse coding layer.
        
        dictionary_layer (DictBaseLayer): Layer that manages the dictionary functions,
            learned and adjusted during training. The dictionary is global across all blocks.
        
        n_features (int): Number of input dimensions the model works with.
        
        model_epochs (int): Number of training epochs for the overall SESM model.
        
        sparse_coding_config (SparseCodingConfig): Configuration for the sparse coding layer.
        
        dict_config (DictConfig): Configuration for the dictionary layer.
        
        seed (int): Random seed for reproducibility.
        
        debug (bool): Whether to enable debug logging.
        
        logger (logging.Logger): Logger instance for runtime information.
        
        training_time (float): Total training time in seconds.
        
        partial_fit_count (int): Number of partial_fit calls made.
    """
    
    # Type hints for instance attributes
    config: SESMConfig
    sparse_coding_layer: SparseCodingBaseLayer
    dictionary_layer: DictBaseLayer
    partition_manager: BlockManager
    n_features: int
    model_epochs: int
    sparse_coding_config: SparseCodingConfig
    dict_config: DictConfig
    seed: int
    log_interval: int
    logger: logging.Logger
    training_time: float
    partial_fit_count: int

    def __init__(
            self,
            config: SESMConfig,
            logger: logging.Logger,
            dict_layer_hook: Callable[[dict], None] | None = None,
            sparse_coding_layer_hook: Callable[[dict], None] | None = None,  
            sesm_hook: Callable[[dict], None] | None = None,
            **kwargs
    ):
        """
        Initialize the SESM model with the given configuration.

        Args:
            config (SESMConfig): Configuration object containing all SESM parameters.
                See SESMConfig documentation for details on required fields.
            
            logger (logging.Logger): Logger instance for recording runtime information,
                debugging, and monitoring during model execution.
            
            dict_layer_hook (Callable[[dict], None], optional): Callback function called
                during dictionary training. Receives a dict with information like parameters,
                gradients, loss, etc. Useful for monitoring or logging to external systems.
            
            sparse_coding_layer_hook (Callable[[dict], None], optional): Callback function
                called during sparse coding optimization. Receives information about h,
                gradients, and optimization progress.
            
            sesm_hook (Callable[[dict], None], optional): Callback function called at the
                SESM level after each training step. Receives aggregated information about
                both dictionary and sparse coding progress.
            
            **kwargs: Additional keyword arguments passed to the dictionary layer constructor.
                Can include dictionary-specific parameters not in the config.
        """
        super().__init__()
        
        self.config = config
        self.n_features = config.n_features
        self.n_functions = config.sparse_coding_config.n_functions
        self.model_epochs = config.model_epochs
        self.sparse_coding_config = config.sparse_coding_config
        self.dict_config = config.dict_config
        self.seed = config.seed
        self.logger = logger

        # The 'device' in SESMConfig acts as the global default.
        global_device = config.device or "cpu"

        # Assign devices to component configs, respecting overrides.
        self.dict_config.device = self.dict_config.device or global_device
        self.sparse_coding_config.device = self.sparse_coding_config.device or global_device
        self.config.partition_config.device = self.config.partition_config.device or global_device

        # Store hooks for monitoring
        self.sesm_hook = sesm_hook
        self.dict_layer_hook = dict_layer_hook
        self.sparse_coding_layer_hook = sparse_coding_layer_hook

        # Set random seed for reproducibility
        if self.seed is not None and self.seed != "None":
            torch.manual_seed(self.seed)

        # Initialize tracking variables
        self.training_time = 0
        self.partial_fit_count = 0

        # Lists for epoch-wise losses
        self._sparse_coding_losses = []
        self._dict_losses = []
                
        # Sparse coding layer will be set from block in partial_fit
        self.sparse_coding_layer = None

        sparse_coding_class = SparseCodingFactory._registered_by_config.get(type(self.sparse_coding_config))
        if sparse_coding_class:
            self.logger.info(f"Sparse Coding Layer: {sparse_coding_class.__name__}")
        else:
            self.logger.warning(f"No Sparse Coding Layer registered for config type: {type(self.sparse_coding_config).__name__}")

        
        # Create dictionary layer using factory pattern
        self.dictionary_layer = DictFactory.create(
            config=self.dict_config,
            n_features=self.n_features,
            n_functions=self.n_functions,
            evaluation_func=self.evaluation_func,
            logger=self.logger,
            parameter_hook=self.dict_layer_hook,
            **kwargs
        )
        self.logger.info(f"Dictionary Layer: {type(self.dictionary_layer).__name__}")

        
        self.partition_manager = BlockManagerFactory.create(
            config.partition_config,
            logger=self.logger,
            sparse_coding_layer_hook=self.sparse_coding_layer_hook)
        self.logger.info(f"Block Manager: {type(self.partition_manager).__name__}")

    @abstractmethod
    def evaluation_func(self, dictionary: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to define how the dictionary (D) and sparse vector (h) are combined
        to produce predictions (y_pred).
        
        Subclasses (SSESM, BSESM) must implement this to provide the specific
        combination logic (e.g., standard matrix multiplication, batch matrix multiplication).
        
        Args:
            dictionary (torch.Tensor): The dictionary matrix.
            h (torch.Tensor): The sparse code vector/matrix.
            
        Returns:
            torch.Tensor: The predicted output (D @ h).
        """
        
    @property
    def sparse_coding_layer_losses(self):
        """
        Returns the loss history for the sparse coding layer.
        
        Returns:
            list: Loss values from sparse coding optimization. Empty list if no
                sparse coding layer is set.
        """
        return self.sparse_coding_layer.losses if self.sparse_coding_layer else self._sparse_coding_losses

    @property
    def dictionary_layer_losses(self):
        """
        Returns the loss history for the dictionary layer.
        
        Returns:
            list: Loss values from dictionary training.
        """
        return self._dict_losses

    def fit(self, X: torch.Tensor, y: torch.Tensor, h: torch.Tensor = None):
        """
        Train the model by learning a sparse vector and dictionary that approximates the target function.
        
        Note: This method is kept for backward compatibility and non-partitioned use cases.
        For partitioned training (SSESM, BSESM), use partial_fit.
        
        This method creates a single sparse coding layer and trains both dictionary and
        sparse representations over the specified number of model epochs.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features), where n_samples
                is the number of samples and n_features matches config.n_features.
            
            y (torch.Tensor): Target data of shape (n_samples,) or (n_samples, 1),
                representing the function values to approximate.
            
            h (torch.Tensor, optional): Initial sparse vector of shape (n_functions, 1).
                If not provided, the sparse coding layer initializes it (usually as zeros).
        """
        # Create a sparse coding layer for non-partitioned training
        if self.sparse_coding_layer is None:
            self.sparse_coding_layer = SparseCodingFactory.create(
                config=self.sparse_coding_config,
                evaluation_func=self.evaluation_func,
                logger=self.logger,
                parameter_hook=self.sparse_coding_layer_hook
            )

        # Initialize layers
        X_proxy = TensorProxy(X)
        y_proxy = TensorProxy(y)

        self.dictionary_layer.setup(X_proxy.get_for_device(self.dictionary_layer.device).detach())

        self.sparse_coding_layer.setup(h)

        # Main training loop
        for epoch in range(self.model_epochs):
            epoch_start_time = time.time()

            self._train_step(X_proxy, y_proxy, self.sparse_coding_layer, epoch)

            epoch_end_time = time.time()
            self.training_time += epoch_end_time - epoch_start_time

            self.logger.info(
                f"Epoch {epoch + 1} - Fit: Loss Sparse Coding: {self.sparse_coding_layer_losses[-1]:.6f}, "
                f"Loss Dictionary: {self.dictionary_layer_losses[-1]:.6f}"
            )

    @abstractmethod
    def partial_fit(self, X: torch.Tensor, y: torch.Tensor, *_):
        """
        Perform a partial fit on the model, iteratively updating parameters using active sub-blocks.

        All derived classes must implement this as the entry point for (partial) trainings using
        input space partitioning.
         

        Args:
            X (torch.Tensor): Input features for training.
            y (torch.Tensor): Target values.
            *_: Additional unused positional arguments.

        Returns:
            None
        """

            
    def _train_block(self, block: PartitionBlock, permutation: int) -> None:
        """
        Perform partial training on the SESM model using a PartitionBlock.
        
        This corresponds to a "block step" in the block strategy, which is usually implemented
        in the derived classes (SSESM, BSESM).
        
        The method implements the key insight of SESM: a global dictionary with local
        sparse representations per block, enabling scalable function approximation.

        Args:
            block (PartitionBlock): A partition block containing:
                - normalized_X: Input data normalized to the block's coordinate system
                - target: Normalized target values (y * amplitude)
                - sparse_coding_layer: The block's dedicated sparse coding layer
                - h: Initial sparse vector for this block
                - amplitude: Scaling factor for this block
            permutation (int): the number of permutation this training belongs to.

        Note:
            This method modifies the dictionary_layer parameters and the block's
            sparse_coding_layer.h in place.
        """
        # Setup dictionary if this is the first block with the proxy's tensor
        #  on the correct device
        dict_device = self.dictionary_layer.device
        self.dictionary_layer.setup(block.normalized_X.get_for_device(dict_device).detach())
        
        # Ensure sparse coding layer is initialized with block's h
        if not hasattr(block.sparse_coding_layer, 'h') or block.sparse_coding_layer.h is None:
            raise RuntimeError(
                f"Critical error: block.sparse_coding_layer.h of the {block.block_index}-th block "
                "is not initialized yet."
            )

        # Ensure the target y has the proper dimensions
        # if block.target.dim() == 1:
        #    block.target = block.target.unsqueeze(-1)

        # Train for the specified number of epochs
        
        for epoch in range(self.model_epochs):
            epoch_start_time = time.time()

            self._train_step(X=block.normalized_X,
                             y=block.target,
                             sparsecoding=block.sparse_coding_layer,
                             epoch=permutation + epoch)

            self.training_time += time.time() - epoch_start_time
            
            if ( (self.config.log_interval>0) and
                 ( (epoch + 1) % self.config.log_interval == 0 or
                   epoch == 0 or
                   epoch == self.model_epochs - 1 ) ):
                self.logger.info(
                    f"Block {block.block_index} - Epoch {epoch + 1}/{self.model_epochs}: "
                    f"Dict Loss: {self.dictionary_layer_losses[-1]:.6f}, "
                    f"SC Loss: {block.sparse_coding_layer.losses[-1]:.6f}"
                )
                
        self.partial_fit_count += 1

    def _train_step(self,
                    X: TensorProxy,
                    y: TensorProxy,
                    sparsecoding: SparseCodingBaseLayer,
                    epoch: int
    ):
        """
        Perform a single training step through the SESM model.
        
        This method implements the alternating optimization strategy:
        1. Fix h, optimize dictionary parameters
        2. Fix dictionary, optimize h
        
        The dictionary layer may implement its own internal alternating strategy
        (e.g., GaussianDictLayer alternates between mu and rho parameters).

        Args:
            X (TensorProxy): Input data of shape (n_samples, n_features) through proxy.
            
            y (TensorProxy): Target data of shape (n_samples,) or (n_samples, 1) through proxy.

            sparsecoding (SparseCodingBaseLayer): The sparse coding layer responsible for.
            
            epoch (int): number of epoch this training step belongs to.
        """

        # --- Phase 1: Optimize dictionary ---
        dict_device = self.dictionary_layer.device
        X_dict = X.get_for_device(dict_device)
        y_dict = y.get_for_device(dict_device)
        h_detached_dict = sparsecoding.h.detach().to(dict_device)

        self.dictionary_layer.partial_fit(X=X_dict, y=y_dict, h=h_detached_dict)

        self._dict_losses.append(self.dictionary_layer.losses[-1])
        
        # --- Phase 2: Optimize sparse coding ---
        sc_device = sparsecoding.device
        y_sc = y.get_for_device(sc_device)
        dictionary_for_sparse = self.dictionary_layer.dictionary.detach().to(sc_device)

        sparsecoding.partial_fit(y=y_sc,
                                 dictionary=dictionary_for_sparse,
                                 #reset_state=(epoch == 0))
                                 reset_state=True)

        self._sparse_coding_losses.append(sparsecoding.losses[-1])
        
        # Call SESM hook if provided for monitoring
        if self.sesm_hook is not None:
            hook_info = {
                'partial_fit_count': self.partial_fit_count,
                'sparse_coding_losses': sparsecoding.losses[-sparsecoding.config.epochs:],
                'dictionary_losses': self.dictionary_layer_losses[-self.dictionary_layer.config.epochs:],
                'h': sparsecoding.h.detach().clone(),
                'dictionary_params': self.dictionary_layer.theta_params.detach().clone(),
                'model_epoch': epoch
            }
            self.sesm_hook(hook_info)
            
    def predict(self,
                X: torch.Tensor,
                custom_h: torch.Tensor | None = None) -> torch.Tensor:
        """
        Generate predictions using the trained SESM model with fit, i.e. a single block.
        
        Computes y_pred = evaluation_func(dictionary, h) where the dictionary is
        evaluated at the input points X.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features) where
                predictions are needed.
            custom_h (torch.Tensor, optional): Custom sparse vector to use for
                predictions instead of the learned h. Shape should be (n_functions, 1).
                Useful for evaluating different sparse representations.

        Returns:
            torch.Tensor: Predicted values of shape (n_samples,) or shape matching
                the evaluation function output.
                
        Raises:
            ValueError: If no sparse vector is available (neither trained nor custom_h).
        """
        return self._predict(X,self.sparse_coding_layer,custom_h)

    def _predict(
            self,
            X: torch.Tensor,
            sparsecoding: SparseCodingBaseLayer,
            custom_h: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Generate predictions using the trained SESM model, but for one block only.
        It assumes the X data points lie within the range the given sparse coding layer
        was train into.

        
        Computes y_pred = evaluation_func(dictionary, h) where the dictionary is
        evaluated at the input points X.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features) where
                predictions are needed.

            sparsecoding (SparseCodingBaseLayer): the sparse coding layer of the block
                in charge of the X points.
            
            custom_h (torch.Tensor, optional): Custom sparse vector to use for
                predictions instead of the learned h. Shape should be (n_functions, 1).
                Useful for evaluating different sparse representations.

        Returns:
            torch.Tensor: Predicted values of shape (n_samples,) or shape matching
                the evaluation function output.
                
        Raises:
            ValueError: If no sparse vector is available (neither trained nor custom_h).
        """
        # Prediction should happen on the dictionary's device.
        device = self.dictionary_layer.device
        X = X.to(device)

        # Evaluate dictionary at input points
        with torch.no_grad():
            self.dictionary_layer.dictionary = self.dictionary_layer.forward(X)

        # The dictionary is already on the correct device from the forward pass.
        dictionary = self.dictionary_layer.dictionary

        # Determine which h to use
        if custom_h is not None:
            h = custom_h.to(device).detach()
        elif sparsecoding is not None and sparsecoding.h is not None:
            h = sparsecoding.h.to(device).detach()
        else:
            raise ValueError("No sparse vector available for prediction. Train the model first or provide custom_h.")
        
        # Combine dictionary and h using evaluation function
        return self.evaluation_func(dictionary, h)
    
    def _predict_block(
            self,
            block: PartitionBlock,
            custom_h: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Generate predictions using the trained SESM model for a particular block

        This is an interface function that extracts the information from the given block
        and calls the protected method _predict.
        
        Args:
            block (PartitionBlock): Block with input data and sparse coding layer
            
            custom_h (torch.Tensor, optional): Custom sparse vector to use for
                predictions instead of the learned h. Shape should be (n_functions, 1).
                Useful for evaluating different sparse representations.

        Returns:
            torch.Tensor: Predicted values of shape (n_samples,) or shape matching
                the evaluation function output.
                
        Raises:
            ValueError: If no sparse vector is available (neither trained nor custom_h).
        """
        # Get the tensor on the dictionary's device for prediction.
        X = block.normalized_X.get_for_device(self.dictionary_layer.device).detach()
        return self._predict(X,block.sparse_coding_layer,custom_h)
    
