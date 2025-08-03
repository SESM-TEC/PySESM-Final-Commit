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
from pysesm.functions.GaussianFunction import GaussianFunction
from pysesm.base_types import TensorBatch

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
        evaluation_func: Callable[[TensorBatch, TensorBatch], TensorBatch],
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
            logger=logger
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

    # --- Implement abstract methods from DictBaseLayer ---

    def _initialize_parameters(self, **kwargs) -> torch.nn.Parameter:
        """
        Initializes the parameters (theta) for the Gaussian functions using self.psi.
        """
        # self.psi is the GaussianFunction instance
        return self.psi.initialize().to(self.device)


    def _evaluate_dictionary(self, X: TensorBatch, parameters: torch.Tensor, **kwargs) -> TensorBatch:
        """
        Evaluates the Gaussian dictionary functions at given points X using the current parameters.
        Passes kwargs like rho_flag, mu_flag to the GaussianFunction's __call__.
        """
        # self.psi is the GaussianFunction instance
        return self.psi(X, parameters, **kwargs)


    def _train_with_strategy(self, X: TensorBatch, y: TensorBatch, h: TensorBatch, 
                             log_losses: bool = True): 
        """
        Implements the training strategy for Gaussian dictionary parameters (mu and rho).
        Uses a split training strategy based on `split_mu_rho` config.
        """
        if self.config.split_mu_rho:
            # Training mu (mean) parameters
            if self.config.mu_epochs > 0:
                for epoch in range(self.config.mu_epochs):
                    loss = self._train_epoch(X=X, y=y, h=h, 
                                             log_losses=log_losses, mu_flag=True, rho_flag=False)

            # Training rho (covariance) parameters
            if self.config.rho_epochs > 0:
                for epoch in range(self.config.rho_epochs):
                    loss = self._train_epoch(X=X, y=y, h=h, 
                                             log_losses=log_losses, mu_flag=False, rho_flag=True)
        else:
            # Joint training of all parameters
            for epoch in range(self.config.epochs):
                loss = self._train_epoch(X=X, y=y, h=h, 
                                         log_losses=log_losses, mu_flag=True, rho_flag=True)

    # --- Override _add_hook_info to provide mu/rho specific info ---
    def _add_hook_info(self, hook_info: dict, **eval_kwargs):
        """
        Adds Gaussian-specific information (mu/rho parameters, flags) to the parameter hook.
        """
        # Extract mu and rho from the full parameters vector (theta)
        # Assuming theta has rho first, then mu.
        num_rho_params = self.n_features * (self.n_features + 1) // 2
        
        hook_info['rho_params'] = hook_info['theta_params'][:num_rho_params, :].clone()
        hook_info['mu_params'] = hook_info['theta_params'][-self.n_features:, :].clone()
        
        # Add flags from eval_kwargs to indicate which parameters are being optimized in this epoch
        hook_info['mu_flag'] = eval_kwargs.get('mu_flag', False)
        hook_info['rho_flag'] = eval_kwargs.get('rho_flag', False)
