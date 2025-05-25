'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica

Dictionary Base Layer

Abstract base class for all dictionary implementations.

Authors: The SESM Team 

License: 
'''

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable, Iterator, Type
import torch
import logging

import ..base_types import BaseConfig


@dataclass
class DictConfig(BaseConfig):
    """Base configuration for all dictionary types"""
    epochs: int
    alpha: float
    criterion: Optional[torch.nn.Module] = None
    optimizer_factory: Optional[Callable[[Iterator[torch.nn.Parameter], float], torch.optim.Optimizer]] = None


class DictBaseLayer(torch.nn.Module, ABC):
    """
    Abstract base class for all dictionary implementations.
    
    This class provides the common interface and functionality that all
    dictionary types must implement, while allowing each type to define
    its own specific training strategies and parameter initialization.
    """
    
    # Each subclass must define this to specify which config type it expects
    CONFIG_CLASS: Type[DictConfig] = DictConfig
    
    def __init__(
        self,
        config: DictConfig,
        n_features: int,
        n_functions: int,
        evaluation_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        logger: logging.Logger,
        parameter_hook: Optional[Callable[[dict], None]] = None,
        device = None,
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
            device: Device for computation
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
        self.device = device
        self.losses = []
        self.dictionary = None
        
        # Initialize parameters (subclass-specific)
        self.parameters = self._initialize_parameters(**kwargs)
        
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
        pass
    
    @abstractmethod
    def _evaluate_dictionary(self, X: torch.Tensor, parameters: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Evaluate the dictionary at given points with current parameters.
        
        Args:
            X: Input coordinates where to evaluate the dictionary
            parameters: Current parameter values
            **kwargs: Additional evaluation arguments (e.g., rho_flag, mu_flag for Gaussian)
            
        Returns:
            torch.Tensor: Evaluated dictionary matrix
        """
        pass
    
    @abstractmethod
    def _train_with_strategy(self, X: torch.Tensor, y: torch.Tensor, h: torch.Tensor, 
                           dictionary_shape: tuple, log_losses: bool):
        """
        Implement the specific training strategy for this dictionary type.
        
        This method encapsulates the training logic specific to each dictionary type.
        For example, Gaussian dictionaries might split mu/rho training, while
        polynomial dictionaries might use unified training.
        
        Args:
            X: Input data
            y: Target data  
            h: Sparse coding vector (detached)
            dictionary_shape: Shape for dictionary evaluation
            log_losses: Whether to log training losses
        """
        pass
    
    def _setup_criterion(self):
        """Setup the loss criterion"""
        if self.config.criterion is None:
            self.criterion = torch.nn.MSELoss()
        else:
            self.criterion = self.config.criterion
    
    def _setup_optimizer(self):
        """Setup the optimizer"""
        if self.config.optimizer_factory is None:
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.config.alpha, weight_decay=0)
        else:
            self.optimizer = self.config.optimizer_factory(self.parameters(), lr=self.config.alpha)
    
    def _train_epoch(self, X: torch.Tensor, y: torch.Tensor, h: torch.Tensor, 
                    dictionary_shape: tuple, log_losses: bool, **eval_kwargs):
        """
        Perform a single training epoch.
        
        This is a common training step that can be used by subclasses.
        """
        X = X.to(self.device)
        y = y.to(self.device)
        h = h.to(self.device)
        
        self.optimizer.zero_grad()
        
        self.dictionary = self.forward(X, dictionary_shape, **eval_kwargs)
        
        # Calculate prediction using the current dictionary and h
        # IMPORTANT: h should already be detached to prevent gradient conflicts
        y_pred = self.evaluation_func(self.dictionary, h)
        
        loss = self.criterion(y_pred, y)
        loss.backward(retain_graph=False)
        self.optimizer.step()
        
        if log_losses:
            self.losses.append(loss.item())
        
        # Call parameter hook if provided
        if self.parameter_hook is not None:
            hook_info = {
                'epoch': len(self.losses),
                'parameters': self.parameters.clone().detach(),
                'loss': loss.item(),
                'dictionary_shape': dictionary_shape
            }
            # Subclasses can add more specific info to hook_info
            self._add_hook_info(hook_info, **eval_kwargs)
            self.parameter_hook(hook_info)
        
        return loss
    
    def _add_hook_info(self, hook_info: dict, **eval_kwargs):
        """
        Add dictionary-specific information to the parameter hook.
        Subclasses can override this to add specialized info.
        """
        pass
    
    def setup(self, X: torch.Tensor) -> None:
        """
        Initialize the dictionary for the layer.
        
        Args:
            X: Input data to initialize dictionary evaluation
        """
        X = X.to(self.device)
        if self.dictionary is None:
            self.dictionary = self._evaluate_dictionary(X, self.parameters)
    
    def partial_fit(self, X: torch.Tensor, y: torch.Tensor, h: torch.Tensor, 
                   dictionary_shape: tuple = None, log_losses: bool = True) -> None:
        """
        Public interface for training the dictionary.
        
        This method delegates to the specific training strategy implemented
        by each dictionary type.
        
        Args:
            X: Input data
            y: Target data
            h: Sparse coding vector (should be detached)
            dictionary_shape: Optional shape for dictionary evaluation
            log_losses: Whether to log training losses
        """
        self._train_with_strategy(X, y, h, dictionary_shape, log_losses)
    
    def forward(self, X: torch.Tensor, dictionary_shape: tuple = None, **kwargs) -> torch.Tensor:
        """
        Evaluate dictionary at given points.
        
        Args:
            X: Input coordinates
            dictionary_shape: Optional shape for output
            **kwargs: Additional evaluation arguments
            
        Returns:
            torch.Tensor: Evaluated dictionary
        """
        X = X.to(self.device)
        evaluated_dictionary = self._evaluate_dictionary(X, self.parameters, **kwargs)
        return evaluated_dictionary if not dictionary_shape else evaluated_dictionary.view(dictionary_shape)
