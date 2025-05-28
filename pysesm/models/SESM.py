'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica

SESM Base Class

Provides the basic functionality of the Sparse-Encoded Surrogate Model.

Authors: The SESM Team 

License: 
'''
import logging
import time
import numpy as np
import torch
from typing import Dict, Union, Callable, Iterator, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..validation import validate_sesm_partial_fit
from ..functions import SurrogateFunction
from ..dictionaries.DictBaseLayer import DictBaseLayer, DictConfig
from ..sparse_coding.SparseCodingBaseLayer import SparseCodingBaseLayer, SparseCodingConfig
from ..enums.DeviceTargetEnum import DeviceTarget
from ..factories.SparseCodingFactory import SparseCodingFactory
from ..factories.DictFactory import DictFactory
from ..factories.BlockManagerFactory import BlockManagerFactory
from ..blocks.PartitionBlock import PartitionBlock
from ..blocks.BlockManager import BlockManager, BlockManagerConfig
from ..base_types import BaseConfig
from ..device_manager.DeviceManager import DeviceManager

@dataclass
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
        
        log_interva (int): Every how many epochs of the main SESM loop should we log progress.
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
        
        device_manager: Manages device allocation for different components.
        
        loss_stats (dict): Tracks loss statistics during training:
            - 'loss_mean': History of mean loss values over epochs
            - 'loss_std': History of standard deviation of loss values
            - 'loss_max': History of maximum loss values
            - 'loss_min': History of minimum loss values
        
        elapsed_time (float): Total training time in seconds.
        
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
    loss_stats: dict
    elapsed_time: float
    partial_fit_count: int

    def __init__(
            self,
            config: SESMConfig,
            logger: logging.Logger,
            device_manager=None,
            dict_layer_hook: Optional[Callable[[dict], None]] = None,
            sparse_coding_layer_hook: Optional[Callable[[dict], None]] = None,  
            sesm_hook: Optional[Callable[[dict], None]] = None,
            **kwargs
    ):
        """
        Initialize the SESM model with the given configuration.

        Args:
            config (SESMConfig): Configuration object containing all SESM parameters.
                See SESMConfig documentation for details on required fields.
            
            logger (logging.Logger): Logger instance for recording runtime information,
                debugging, and monitoring during model execution.
            
            device_manager (optional): Device manager for GPU/CPU allocation. If provided,
                manages device placement for dictionary and sparse coding layers.
            
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
        super(SESM, self).__init__()
        
        self.config = config
        self.n_features = config.n_features
        self.n_functions = config.sparse_coding_config.n_functions
        self.model_epochs = config.model_epochs
        self.sparse_coding_config = config.sparse_coding_config
        self.dict_config = config.dict_config
        self.seed = config.seed
        self.logger = logger
        
        
        if device_manager is None:
            self.device_manager = DeviceManager(logger=self.logger, default_device="cpu")
            self.logger.info("No DeviceManager provided. Creating a default DeviceManager (CPU only).")
        else:
            self.device_manager = device_manager
        
        # Store hooks for monitoring
        self.sesm_hook = sesm_hook
        self.dict_layer_hook = dict_layer_hook
        self.sparse_coding_layer_hook = sparse_coding_layer_hook

        # Set random seed for reproducibility
        if self.seed is not None and self.seed != "None":
            torch.manual_seed(self.seed)

        # Initialize tracking variables
        self.elapsed_time = 0
        self.partial_fit_count = 0

        self.loss_stats = {
            "loss_mean": [],
            "loss_std": [],
            "loss_max": [],
            "loss_min": [],
        }
                
        # Sparse coding layer will be set from block in partial_fit
        self.sparse_coding_layer = None
        
        # Create dictionary layer using factory pattern
        self.dictionary_layer = DictFactory.create(
            config=self.dict_config,
            n_features=self.n_features,
            n_functions=self.n_functions,
            evaluation_func=self.evaluation_func,
            logger=self.logger,
            parameter_hook=self.dict_layer_hook,
            device=self.device_manager.get_device(DeviceTarget.DICTIONARY_LAYER),
            **kwargs
        )

        self.partition_manager = BlockManagerFactory.create(
            config.partition_config,
            logger=self.logger,
            device_manager=self.device_manager,
            sparse_coding_layer_hook=self.sparse_coding_layer_hook)

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
        pass
        
    @property
    def sparse_coding_layer_losses(self):
        """
        Returns the loss history for the sparse coding layer.
        
        Returns:
            list: Loss values from sparse coding optimization. Empty list if no
                sparse coding layer is set.
        """
        return self.sparse_coding_layer.losses if self.sparse_coding_layer else []

    @property
    def dictionary_layer_losses(self):
        """
        Returns the loss history for the dictionary layer.
        
        Returns:
            list: Loss values from dictionary training.
        """
        return self.dictionary_layer.losses

    def fit(self, X: torch.Tensor, y: torch.Tensor, dictionary_shape: tuple = None, h: torch.Tensor = None):
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
            
            dictionary_shape (tuple, optional): If provided, reshapes the evaluated dictionary
                before computing predictions. Useful for structured dictionaries.
            
            h (torch.Tensor, optional): Initial sparse vector of shape (n_functions, 1).
                If not provided, the sparse coding layer initializes it (usually as zeros).
        """
        # Create a sparse coding layer for non-partitioned training
        if self.sparse_coding_layer is None:
            self.sparse_coding_layer = SparseCodingFactory.create(
                config=self.sparse_coding_config,
                evaluation_func=self.evaluation_func,
                logger=self.logger,
                parameter_hook=self.sparse_coding_layer_hook,
                device=self.device_manager.get_device(DeviceTarget.SPARSE_CODING_LAYER) if self.device_manager else None
            )

        # Initialize layers
        self.dictionary_layer.setup(X)
        self.sparse_coding_layer.setup(h)

        # Main training loop
        for epoch in range(self.model_epochs):
            epoch_start_time = time.time()

            self._train_step(X, y, self.sparse_coding_layer,dictionary_shape)

            epoch_end_time = time.time()
            self.elapsed_time += epoch_end_time - epoch_start_time

            self.logger.info(
                f"Epoch {epoch + 1} - Fit: Loss Sparse Coding: {self.sparse_coding_layer_losses[-1]:.6f}, "
                f"Loss Dictionary: {self.dictionary_layer_losses[-1]:.6f}"
            )

    @abstractmethod
    def partial_fit(self, X: torch.Tensor, y: torch.Tensor, initial_h: torch.Tensor = None, *_):
        """
        Perform a partial fit on the model, iteratively updating parameters using active sub-blocks.

        All derived classes must implement this as the entry point for (partial) trainings using
        input space partitioning.
         

        Args:
            X (torch.Tensor): Input features for training.
            y (torch.Tensor): Target values.
            initial_h (torch.Tensor): Initial h value or None for random initialization.
            *_: Additional unused positional arguments.

        Returns:
            None
        """
        pass
            
    def _train_block(self, block: PartitionBlock) -> None:
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

        Returns:
            None
            
        Note:
            This method modifies the dictionary_layer parameters and the block's
            sparse_coding_layer.h in place.
        """
        # Setup dictionary if this is the first block
        self.dictionary_layer.setup(block.normalized_X.clone().detach().requires_grad_(False))
        
        # Ensure sparse coding layer is initialized with block's h
        if not hasattr(block.sparse_coding_layer, 'h') or block.sparse_coding_layer.h is None:
            block.sparse_coding_layer.setup(block.h)

        # Ensure the target y have the proper dimensions
        if block.target.dim() == 1:
            block.target = block.target.unsqueeze(-1)

        # Train for the specified number of epochs
        
        for epoch in range(self.model_epochs):
            epoch_start_time = time.time()

            self._block_train_step(block, dictionary_shape=None)

            self.elapsed_time += time.time() - epoch_start_time

            if ( (self.config.log_interval>0) and
                 ( (epoch + 1) % self.config.log_interval == 0 or
                   epoch == 0 or
                   epoch == self.model_epochs - 1 ) ):
                self.logger.info(
                    f"Block {block.block_index} - Epoch {epoch + 1}/{self.model_epochs}: "
                    f"Loss Sparse Coding: {block.sparse_coding_layer.losses[-1]:.6f}, "
                    f"Loss Dictionary: {self.dictionary_layer_losses[-1]:.6f}"
                )
                
        self.partial_fit_count += 1

    def _block_train_step(self,
                          block: PartitionBlock,
                          dictionary_shape: tuple = None,
    ):
        """
        Perform a single training step through the SESM model for one block.
        """

        X = block.normalized_X.clone().detach().requires_grad_(False)
        y = block.target

        self._train_step(X,y,block.sparse_coding_layer,dictionary_shape)


    def _train_step(self,
                    X: torch.Tensor,
                    y: torch.Tensor,
                    sparsecoding: SparseCodingBaseLayer,
                    dictionary_shape: tuple = None,
    ):
        """
        Perform a single training step through the SESM model.
        
        This method implements the alternating optimization strategy:
        1. Fix h, optimize dictionary parameters
        2. Fix dictionary, optimize h
        
        The dictionary layer may implement its own internal alternating strategy
        (e.g., GaussianDictLayer alternates between mu and rho parameters).

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).
            
            y (torch.Tensor): Target data of shape (n_samples,) or (n_samples, 1).
            
            dictionary_shape (tuple, optional): Shape for dictionary evaluation.
                If provided, the dictionary is reshaped before use.
        """
        # Step 1: Optimize dictionary with fixed h
        # Detach h to prevent gradient flow during dictionary optimization
        # Ensure y is 2D for consistent matrix operations

        h_detached = sparsecoding.h.detach().clone()

        # Train dictionary layer with its own training strategy
        self.dictionary_layer.partial_fit(
            X=X,
            y=y,
            h=h_detached,
            dictionary_shape=dictionary_shape,
        )

        # Step 2: Optimize h with fixed dictionary
        # Detach dictionary to prevent gradient flow during sparse coding
        dictionary_for_sparse = self.dictionary_layer.dictionary.detach()
        sparsecoding.partial_fit(y=y, 
                                 dictionary=dictionary_for_sparse)
        
        # Call SESM hook if provided for monitoring
        if self.sesm_hook is not None:
            hook_info = {
                'partial_fit_count': self.partial_fit_count,
                'sparse_coding_losses': sparsecoding.losses[-sparsecoding.config.epochs:],
                'dictionary_losses': self.dictionary_layer_losses[-self.dictionary_layer.config.epochs:],
                'h': sparsecoding.h.detach().clone(),
                'dictionary_params': self.dictionary_layer.parameters.detach().clone()
            }
            self.sesm_hook(hook_info)
            
    def predict(
            self,
            X: torch.Tensor,
            dictionary_shape: tuple = None,
            custom_h: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Generate predictions using the trained SESM model with fit, i.e. a single block.
        
        Computes y_pred = evaluation_func(dictionary, h) where the dictionary is
        evaluated at the input points X.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features) where
                predictions are needed.
            
            dictionary_shape (tuple, optional): Shape for dictionary evaluation.
                If provided, the dictionary is reshaped before predictions.
            
            custom_h (torch.Tensor, optional): Custom sparse vector to use for
                predictions instead of the learned h. Shape should be (n_functions, 1).
                Useful for evaluating different sparse representations.

        Returns:
            torch.Tensor: Predicted values of shape (n_samples,) or shape matching
                the evaluation function output.
                
        Raises:
            ValueError: If no sparse vector is available (neither trained nor custom_h).
        """
        return _predict(X,self.sparse_coding_layer,dictionary_shape,custom_h)

    def _predict(
            self,
            X: torch.Tensor,
            sparsecoding: SparseCodingBaseLayer,
            dictionary_shape: tuple = None,
            custom_h: torch.Tensor = None
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
            
            dictionary_shape (tuple, optional): Shape for dictionary evaluation.
                If provided, the dictionary is reshaped before predictions.
            
            custom_h (torch.Tensor, optional): Custom sparse vector to use for
                predictions instead of the learned h. Shape should be (n_functions, 1).
                Useful for evaluating different sparse representations.

        Returns:
            torch.Tensor: Predicted values of shape (n_samples,) or shape matching
                the evaluation function output.
                
        Raises:
            ValueError: If no sparse vector is available (neither trained nor custom_h).
        """
        device = self.device_manager.get_device(DeviceTarget.GLOBAL) if self.device_manager else X.device

        # Evaluate dictionary at input points
        with torch.no_grad():
            self.dictionary_layer.dictionary = self.dictionary_layer.forward(X, dictionary_shape)

        dictionary = self.dictionary_layer.dictionary.to(device)
        
        # Determine which h to use
        if custom_h is not None:
            h = custom_h.to(device)
        elif sparsecoding.h is not None:
            h = sparsecoding.h.to(device)
        else:
            raise ValueError("No sparse vector available for prediction. Train the model first or provide custom_h.")
        
        # Combine dictionary and h using evaluation function
        return self.evaluation_func(dictionary, h)
    
    def _predict_block(
            self,
            block: PartitionBlock,
            dictionary_shape: tuple = None,
            custom_h: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Generate predictions using the trained SESM model for a particular block

        This is an interface function that extracts the information from the given block
        and calls the protected method _predict.
        
        Args:
            block (PartitionBlock): Block with input data and sparse coding layer
            
            dictionary_shape (tuple, optional): Shape for dictionary evaluation.
                If provided, the dictionary is reshaped before predictions.
            
            custom_h (torch.Tensor, optional): Custom sparse vector to use for
                predictions instead of the learned h. Shape should be (n_functions, 1).
                Useful for evaluating different sparse representations.

        Returns:
            torch.Tensor: Predicted values of shape (n_samples,) or shape matching
                the evaluation function output.
                
        Raises:
            ValueError: If no sparse vector is available (neither trained nor custom_h).
        """
        X = block.normalized_X.clone().detach().requires_grad_(False)
        return self._predict(X,block.sparse_coding_layer,dictionary_shape,custom_h)
    
