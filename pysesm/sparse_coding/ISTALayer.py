'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica

ISTA Layer Class

Provides the layer in charge of finding h, the sparse vector 
that chooses words in a dictionary to build a surrogate 
function.

Authors: The SESM Team 

License: 
'''
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import torch

from .SparseCodingBaseLayer import SparseCodingBaseLayer, SparseCodingConfig
from .sparse_coding_utils import StepSizeMethod, soft_threshold, calculate_step_size

__all__ = ['ISTALayer', 'ISTAConfig', 'StepSizeMethod']

@dataclass
class ISTAConfig(SparseCodingConfig):
    """
    Configuration parameters for the ISTA algorithm.
    
    This class encapsulates all configuration parameters for the Iterative
    Shrinkage-Thresholding Algorithm (ISTA) to provide a cleaner interface
    and easier parameter management.
    
    Attributes:
        alpha (float): Learning rate for parameter updates. If step_size_method is MANUAL,
                     this value is used directly as the fixed step size.
        lambd (float): Regularization parameter controlling sparsity (L1 penalty strength).
        step_size_method (StepSizeMethod): Method to determine step size during ISTA iterations.
        power_iterations (int): Number of iterations for power method (if used).
        early_stopping (bool): Whether to enable early stopping based on loss convergence.
        early_stopping_tol (float): Tolerance threshold for early stopping.
    """
    alpha: float = 0.1
    lambd: float = 0.00001
    step_size_method: StepSizeMethod = StepSizeMethod.POWER_ITERATION
    power_iterations: int = 10
    early_stopping: bool = False
    early_stopping_tol: float = 1e-6


# Define the ISTA layer
class ISTALayer(SparseCodingBaseLayer):
    """
    A custom PyTorch module implementing a sparse vector layer with learnable parameters.

    This layer implements the Iterative Shrinkage-Thresholding Algorithm (ISTA) for sparse coding.
    ISTA is used to find a sparse representation (vector h) that linearly combines elements from 
    a dictionary to approximate a target function with L1 regularization for sparsity.

    Attributes:
        config (ISTAConfig): Configuration parameters for the ISTA algorithm.
        h (torch.nn.Parameter): Sparse vector maintained and updated by the layer.
        losses (list): List storing the computed losses during training.
        logger (logging.Logger): Logger for recording debug information.
        debug (bool): Whether to enable detailed debug logging.
        parameter_hook (Callable): Optional callback function to monitor internal state.
        device: Device to run computations on.
        last_eigenvector (torch.Tensor): Last computed eigenvector for warm-starting calculations.
    """

    CONFIG_CLASS = ISTAConfig
    
    def __init__(
            self,
            config: ISTAConfig,
            evaluation_func:  Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            logger: logging.Logger | None = None,
            debug: bool = False,
            parameter_hook: Callable[[dict], None] | None = None,
            device = None):
        """
        Initializes the ISTALayer with the specified hyperparameters and components.

        Args:
            config (ISTAConfig): Configuration parameters for the ISTA algorithm.
            logger (logging.Logger, optional): Logger for recording debug information.
            debug (bool, optional): Whether to enable detailed debug logging.
            parameter_hook (Callable, optional): Callback function to inspect the current parameter state.
            device: Device for computation (CPU/GPU).
        """
        super().__init__(config=config,
                         evaluation_func=evaluation_func,
                         logger=logger,
                         debug=debug,
                         parameter_hook=parameter_hook,
                         device=device)

        self.losses = []
        self.last_eigenvector = None  # For warm starting calculations

    def setup(self, h: torch.Tensor = None) -> None:
        """
        Initializes the sparse vector `h`.
        
        Args:
            h (torch.Tensor, optional): Initial value for the sparse vector.
                If not provided, initialized with zeros.
        """
        if h is not None:
            # Ensure h is 2D
            if h.dim() != 2:
                h = h.reshape(-1, 1)

            if h.shape[0] != self.config.n_functions:
                raise ValueError(f"Dimension mismatch: h has {h.shape[0]} rows but n_functions is {self.config.n_functions}")
                
            self.h = torch.nn.Parameter(h.to(self.device), requires_grad=False)
        else:
            num_ones_en_h = 1

            h_inicial_disperso = torch.zeros(self.config.n_functions, 1, device=self.device)

            if self.config.n_functions > 0 and num_ones_en_h > 0:
                # Asegurar que no intentas poner más unos que elementos disponibles
                k_elementos_activos = min(num_ones_en_h, self.config.n_functions)

                # Seleccionar k_elementos_activos índices aleatorios sin reemplazo
                indices_activos = torch.randperm(self.config.n_functions, device=self.device)[:k_elementos_activos]

                # Establecer esos índices a 1.0 en la columna 0
                h_inicial_disperso[indices_activos, 0] = 1.0

            self.h = torch.nn.Parameter(h_inicial_disperso, requires_grad=False)


    def train_step(self, y: torch.Tensor, dictionary: torch.Tensor, log_losses: bool = True) -> torch.Tensor:
        """
        Performs a single ISTA iteration.
        
        This implements one step of the iterative shrinkage-thresholding algorithm:
        1. Calculate the gradient of the loss function
        2. Take a gradient step: h' = h - α * ∇L(h)
        3. Apply soft thresholding: h_new = S_{αλ}(h')
        
        Args:
            y (torch.Tensor): Target vector.
            dictionary (torch.Tensor): Dictionary matrix.
            log_losses (bool): Whether to log losses.
            
        Returns:
            torch.Tensor: The current loss.
        """
        # Move tensors to correct device
        y = y.to(self.device)
        dictionary = dictionary.to(self.device)
        
        # Calculate step size based on selected method
        step_size, self.last_eigenvector = calculate_step_size(
            dictionary, 
            self.config.step_size_method,
            self.config.alpha,
            self.config.power_iterations,
            self.last_eigenvector,
            self.debug,
            self.logger
        )
        
        # Manual ISTA update
        with torch.no_grad():
            # Forward pass
            y_pred = self.evaluation_func(dictionary, self.h)
            
            # Compute loss
            loss = self.criterion(y_pred, y)
            if log_losses:
                self.losses.append(loss.item())
            
            # Compute gradient for MSE: 2 * D^T * (y_pred - y)
            error = y_pred - y
            gradient = 2 * torch.matmul(dictionary.T, error)
            
            # ISTA update: h = soft_threshold(h - alpha * gradient, alpha * lambda)
            h_update = self.h - step_size * gradient
            h_new = soft_threshold(h_update, step_size * self.config.lambd, self.device)
            
            # Update h parameter
            self.h.data = h_new
            
            # Call parameter hook if provided
            if self.parameter_hook is not None:
                hook_info = {
                    'h': self.h.detach().clone(),
                    'gradient': gradient.detach().clone(),
                    'loss': loss.item(),
                    'alpha': step_size
                }
                self.parameter_hook(hook_info)
            
        return loss

    def forward(self, y: torch.Tensor, dictionary: torch.Tensor, log_losses: bool = True) -> torch.Tensor:
        """
        Performs a forward pass without updating parameters.

        Args:
            y (torch.Tensor): Ground truth or target vector.
            dictionary (torch.Tensor): Input dictionary for predictions.
            log_losses (bool): Whether to log the computed losses (default: True).

        Returns:
            torch.Tensor: Estimated loss after forward step
        """
        # Ensure all tensors are on the right device
        y = y.to(self.device)
        dictionary = dictionary.to(self.device)

        # Combine the words in the dictionary using self.h
        with torch.no_grad():
            y_pred = self.evaluation_func(dictionary, self.h)
            assert y_pred.shape == y.shape, f"Shape mismatch: y_pred {y_pred.shape} != y {y.shape}"
            
            loss = self.criterion(y_pred, y)

            if log_losses:
                self.losses.append(loss.item())

        return loss
    
    def partial_fit(self, y: torch.Tensor,
                    dictionary: torch.Tensor,
                    log_losses: bool = True,
                    reset_state: bool = True) -> None:
        """
        Performs multiple ISTA iterations.
        
        This method runs the ISTA algorithm for the specified number of epochs,
        effectively optimizing the sparse vector h to minimize the loss while
        maintaining sparsity through the soft thresholding operation.
        
        Args:
            y (torch.Tensor): Target vector.
            dictionary (torch.Tensor): Dictionary matrix.
            log_losses (bool): Whether to log losses.
        """
        epochs = self.config.epochs

        # Reset ISTA state
        if reset_state:
            self.last_eigenvector = None # Reset eigenvector for step estimation
            self.losses = [] # Reset losses for a new partial_fit cycle
        
        for epoch in range(epochs):
            loss = self.train_step(y, dictionary, log_losses)
            
            if self.debug and (epoch == 0 or (epoch + 1) % 100 == 0 or epoch == epochs - 1):
                self.logger.debug(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")
                
            # Optional early stopping if enabled and has converged
            if self.config.early_stopping and epoch > 0 and len(self.losses) >= 2:
                if abs(self.losses[-1] - self.losses[-2]) < self.config.early_stopping_tol:
                    if self.debug:
                        self.logger.debug(f"Early stopping at epoch {epoch + 1}, loss converged within tolerance {self.config.early_stopping_tol}")
                    break
