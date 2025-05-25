'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica

Gaussian Dictionary Layer

Dictionary layer implementation using Gaussian basis functions.

Authors: The SESM Team 

License: 
'''

from dataclasses import dataclass
from typing import Optional, Callable, Iterator
import torch
import logging

from .DictBaseLayer import DictBaseLayer, DictConfig
from ..functions.GaussianFunction import GaussianFunction


@dataclass
class GaussianDictConfig(DictConfig):
    """Configuration specific to Gaussian dictionaries"""
    mu_epochs: int = 10
    rho_epochs: int = 10
    split_mu_rho: bool = True
    # Parameters for GaussianFunction initialization
    eig_range: list = None
    mu_range: list = None


class GaussianDictLayer(DictBaseLayer):
    """Dictionary layer using Gaussian basis functions"""
    
    CONFIG_CLASS = GaussianDictConfig
    
    def __init__(
        self,
        config: GaussianDictConfig,
        n_features: int,
        n_functions: int,
        evaluation_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        logger: logging.Logger,
        parameter_hook: Optional[Callable[[dict], None]] = None,
        device = None,
        **kwargs
    ):
        # Create the Gaussian surrogate function
        self.psi = GaussianFunction(
            n_features=n_features,
            n_functions=n_functions,
            eig_range=config.eig_range,
            mu_range=config.mu_range,
            device=device
        )
        
        super().__init__(
            config=config,
            n_features=n_features,
            n_functions=n_functions,
            evaluation_func=evaluation_func,
            logger=logger,
            parameter_hook=parameter_hook,
            device=device,
            **kwargs
        )
