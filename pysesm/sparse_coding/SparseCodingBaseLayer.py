'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica

Base class for all sparse coding layers

Provides the abstract definitions required for all sparse coding layers like ISTA, FISTA, etc.

Authors: The SESM Team 

License: 
'''
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar,Generic
from collections.abc import Callable
import logging

import torch

from ..base_types import BaseConfig

@dataclass(kw_only=True)
class SparseCodingConfig(BaseConfig):
    """
    Configuration parameters for all sparse coding algorithms.
    
    This class encapsulates all configuration parameters for the sparse coding algorithms a clean
    interface and easier parameter management.
    
    Attributes:
        n_functions (int): Number of words in the dictionary, i.e. dimension of h. It MUST be given.
        epochs (int): Number of epochs to train the sparse coding layer.
        initial_h: Initial h value of dimension n_functions x 1.
        criterion (torch.nn.Module): Loss function used for training (default: Mean Squared Error).

    """
    n_functions: int   # Required, no default
    epochs: int = 100  # Number of training epochs
    initial_h: torch.Tensor | None = None  # Initial sparse vector (optional)
    criterion: torch.nn.Module | None  = None

T_Config = TypeVar('T_Config', bound=SparseCodingConfig)

class SparseCodingBaseLayer(torch.nn.Module, Generic[T_Config], ABC):
    """
    Abstract base class for sparse coding algorithm implementations.
    Inherits from torch.nn.Module for PyTorch integration and ABC for abstract functionality.
    
    All concrete implementations (ISTALayer, FISTALayer, etc.) must inherit from this class
    and implement the abstract methods.
    """
    
    # Class variable to store the expected config class
    CONFIG_CLASS: type[SparseCodingConfig] = SparseCodingConfig

    @abstractmethod
    def __init__(self,
                 config: T_Config,
                 evaluation_func:  Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 logger: logging.Logger | None = None,
                 debug: bool = False,
                 parameter_hook: Callable | None = None,
                 device: torch.device | None = None):
        """
        Base initialization. Child classes must call super().__init__()
        """
        super().__init__()

        # Type check for config
        if not isinstance(config, self.CONFIG_CLASS):
            raise TypeError(f"Expected config of type {self.CONFIG_CLASS.__name__}, "
                           f"got {type(config).__name__}")

        self.config = config

        if self.config.n_functions is None:
            raise ValueError("n_functions must be specified in SparseCodingConfig")
        
        self.device = device
        self.logger = logger
        self.debug = debug
        self.evaluation_func = evaluation_func

        self.parameter_hook = parameter_hook

        if self.config.criterion is None:
            self.criterion = torch.nn.MSELoss()
        else:
            self.criterion = self.config.criterion
            
        self.setup(config.initial_h)
        self.to(self.device)        
        
    @abstractmethod
    def setup(self, h: torch.Tensor = None) -> None:
        """
        Initializes the sparse vector h.
        
        Args:
            h (torch.Tensor, optional): Initial value for the sparse vector. 
                   If None, it will be randomly initialized.
        """
        
    
    @abstractmethod
    def forward(self, y: torch.Tensor, dictionary: torch.Tensor, 
                log_losses: bool = True) -> torch.Tensor:
        """
        Performs the forward pass.
        
        Args:
            y (torch.Tensor): Target/ground truth vector
            dictionary (torch.Tensor): Dictionary matrix for prediction
            log_losses (bool): If True, logs the computed losses
            
        Returns:
            torch.Tensor: Computed loss
        """
        
    
    @abstractmethod
    def train_step(self, y: torch.Tensor, dictionary: torch.Tensor, 
                   log_losses: bool = True) -> torch.Tensor:
        """
        Performs a complete training step (forward + backward + optimization).
        
        Args:
            y (torch.Tensor): Target/ground truth vector
            dictionary (torch.Tensor): Dictionary matrix for prediction
            log_losses (bool): If True, logs the computed losses
            
        Returns:
            torch.Tensor: Computed loss
        """
        
    
    @abstractmethod
    def partial_fit(self, y: torch.Tensor,
                    dictionary: torch.Tensor,
                    log_losses: bool = True,
                    reset_state: bool = True) -> None:
        """
        Performs multiple train steps (the given number of epochs)
        
        Args:
            y (torch.Tensor): Target/ground truth vector
            dictionary (torch.Tensor): Dictionary matrix for prediction
            log_losses (bool): If True, logs the computed losses
            reset_state (bool): Reset layer state if True
            
        Returns:
            None
        """
        
