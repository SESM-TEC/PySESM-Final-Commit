'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica
Gaussian Dictionary Layer
Dictionary layer implementation using Gaussian basis functions.
Authors: The SESM Team 
License: 
'''
from __future__ import annotations


import logging

from dataclasses import dataclass
from collections.abc import Callable
import torch

from pysesm.functions.GaussianFunction import GaussianFunction
from pysesm.base_types import TensorBatch
from .DictBaseLayer import DictBaseLayer, DictConfig

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
        parameter_hook: Callable[[dict], None] | None = None,
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
                for _ in range(self.config.mu_epochs):
                    self._train_epoch(X=X, y=y, h=h, 
                                      log_losses=log_losses, mu_flag=True, rho_flag=False)

            # Training rho (covariance) parameters
            if self.config.rho_epochs > 0:
                for _ in range(self.config.rho_epochs):
                    self._train_epoch(X=X, y=y, h=h, 
                                      log_losses=log_losses, mu_flag=False, rho_flag=True)
        else:
            # Joint training of all parameters
            for _ in range(self.config.epochs):
                self._train_epoch(X=X, y=y, h=h, 
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

    @staticmethod
    def gram_regularization(layer: "GaussianDictLayer") -> torch.Tensor:
        """Penalizes the dictionary's coherence using the Gram matrix."""
        if layer.dictionary is None:
            return torch.tensor(0.0, device=layer.device)
        
        # Normalize the dictionary's columns (words) to L2 norm = 1
        D_norm = torch.nn.functional.normalize(layer.dictionary, p=2, dim=0)
        
        # Calculate the Gram matrix: G = D_norm^T * D_norm
        G = D_norm.T @ D_norm
        
        # The goal is for G to be the identity matrix. Penalize the difference.
        identity = torch.eye(G.shape[0], device=layer.device)
        return torch.norm(G - identity, p='fro')**2

    @staticmethod
    def electrostatic_regularization(layer: "GaussianDictLayer") -> torch.Tensor:
        """Penalizes the closeness of Gaussian means using an efficient proxy."""
        n_features = layer.n_features
        n_functions = layer.n_functions
        epsilon = 1e-8
        
        num_rho_params = n_features * (n_features + 1) // 2
        rho = layer.theta_params[:num_rho_params, :]
        mu = layer.theta_params[-n_features:, :]

        # The "charge" is inversely proportional to the "precision," which is
        # the sum of the squares of the rho parameters (a proxy for trace(G)).
        trace_G = torch.sum(rho**2, dim=0)
        charges = 1.0 / (trace_G + epsilon)

        # Calculate all pairwise distances in a vectorized way
        mu_t = mu.T # Shape: (n_functions, n_features)
        diffs = mu_t.unsqueeze(1) - mu_t.unsqueeze(0) # Shape: (n_func, n_func, n_feat)
        dist_sq = torch.sum(diffs**2, dim=-1) # Shape: (n_func, n_func)
        
        # Calculate charge products and the potential energy
        charge_prods = charges.unsqueeze(1) * charges.unsqueeze(0)
        potential_matrix = charge_prods / (dist_sq + epsilon)
        
        # Sum only the upper triangle to avoid double-counting pairs
        return torch.triu(potential_matrix, diagonal=1).sum()
