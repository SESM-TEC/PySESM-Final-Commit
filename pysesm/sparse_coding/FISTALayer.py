'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica

FISTA Layer Class

Provides an implementation of the Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)
for solving L1-regularized least squares problems more efficiently than standard ISTA.

Authors: The SESM Team 

License: 
'''

import logging
import torch
from dataclasses import dataclass
from typing import Callable, Optional
from enum import Enum, auto
from math import sqrt

from .SparseCodingBaseLayer import SparseCodingBaseLayer, SparseCodingConfig
from .sparse_coding_utils import StepSizeMethod, soft_threshold, calculate_step_size

class RestartStrategy(Enum):
    """Enumeration of restart strategies for FISTA algorithm."""
    NONE = auto()      # No restart
    ADAPTIVE = auto()  # Restart when monotonicity is violated
    FIXED = auto()     # Restart after fixed number of iterations

class MomentumScheme(Enum):
    """Enumeration of momentum update schemes for FISTA algorithm."""
    ORIGINAL = auto()   # Standard FISTA scheme: t_{k+1} = (1 + sqrt(1 + 4*t_k^2)) / 2
    MONOTONIC = auto()  # Alternative scheme for better stability: t_{k+1} = (1 + sqrt(1 + 8*t_k^2)) / 4
    
@dataclass
class FISTAConfig(SparseCodingConfig):
    """
    Configuration parameters for the FISTA algorithm.
    
    This class extends the base SparseCodingConfig with FISTA-specific parameters
    to control its behavior and convergence properties.
    
    Attributes:
        alpha (float): Learning rate for parameter updates. If step_size_method is MANUAL,
                     this value is used directly as the fixed step size.
        lambd (float): Regularization parameter controlling sparsity (L1 penalty strength).
        step_size_method (StepSizeMethod): Method to determine step size during FISTA iterations.
        power_iterations (int): Number of iterations for power method (if used).
        early_stopping (bool): Whether to enable early stopping based on loss convergence.
        early_stopping_tol (float): Tolerance threshold for early stopping.
        restart_strategy (RestartStrategy): Strategy for restarting momentum in FISTA.
        restart_period (int): Number of iterations between restarts for FIXED strategy.
        momentum_scheme (MomentumScheme): Scheme for computing momentum parameter (ORIGINAL or MONOTONIC).
    """
    alpha: float = 0.1
    lambd: float = 0.01
    step_size_method: StepSizeMethod = StepSizeMethod.POWER_ITERATION
    power_iterations: int = 10
    early_stopping: bool = False
    early_stopping_tol: float = 1e-6
    restart_strategy: RestartStrategy = RestartStrategy.NONE
    restart_period: int = 50
    momentum_scheme: MomentumScheme = MomentumScheme.ORIGINAL  


class FISTALayer(SparseCodingBaseLayer):
    """
    A custom PyTorch module implementing the Fast Iterative Shrinkage-Thresholding Algorithm (FISTA).
    
    FISTA accelerates the convergence of ISTA by incorporating a momentum term,
    achieving a convergence rate of O(1/k²) compared to ISTA's O(1/k). This implementation
    also supports various restart strategies to improve convergence in practice.
    
    Attributes:
        config (FISTAConfig): Configuration parameters for the FISTA algorithm.
        h (torch.nn.Parameter): Sparse vector maintained and updated by the layer.
        losses (list): List storing the computed losses during training.
        logger (logging.Logger): Logger for recording debug information.
        debug (bool): Whether to enable detailed debug logging.
        parameter_hook (Callable): Optional callback function to monitor internal state.
        device: Device to run computations on.
        t (float): Momentum parameter for FISTA.
        z (torch.Tensor): Auxiliary variable for momentum acceleration.
        prev_h (torch.Tensor): Previous iteration's sparse vector.
        iter_count (int): Iteration counter for fixed restart strategy.
        last_eigenvector (torch.Tensor): Last computed eigenvector for warm-starting calculations.
        prev_loss (float): Previous iteration's loss value for adaptive restart.
    """

    CONFIG_CLASS = FISTAConfig
    
    def __init__(
            self,
            config: FISTAConfig,
            evaluation_func:  Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            logger: Optional[logging.Logger] = None,
            debug: bool = False,
            parameter_hook: Optional[Callable[[dict], None]] = None,
            device = None):
        """
        Initializes the FISTALayer with the specified hyperparameters and components.

        Args:
            config (FISTAConfig): Configuration parameters for the FISTA algorithm.
            logger (logging.Logger, optional): Logger for recording debug information.
            debug (bool, optional): Whether to enable detailed debug logging.
            parameter_hook (Callable, optional): Callback function to inspect the current parameter state.
            device: Device to run computations on.
        """
        super().__init__(config=config,
                         evaluation_func=evaluation_func,
                         logger=logger,
                         debug=debug,
                         parameter_hook=parameter_hook,
                         device=device)

        self.losses = []
        self.last_eigenvector = None  # For warm starting calculations
        
        # FISTA-specific variables
        self.t = 1.0  # Initial momentum parameter
        self.iter_count = 0  # For fixed restart strategy
        self.prev_loss = float('inf')  # For adaptive restart strategy
 
    def setup(self, h: torch.Tensor = None) -> None:
        """
        Initializes the sparse vector `h` and auxiliary variables for FISTA.
        
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
            # Initialize h as zeros (common for ISTA/FISTA)
            self.h = torch.nn.Parameter(
                torch.zeros(self.config.n_functions, 1).to(self.device), 
                requires_grad=False
            )
        
        # Initialize auxiliary variables for FISTA
        self.z = self.h.clone().detach()
        self.prev_h = self.h.clone().detach()

    def _check_restart_condition(self, loss: float) -> bool:
        """
        Determines whether to restart the momentum based on the configured strategy.
        
        Args:
            loss (float): Current loss value.
            
        Returns:
            bool: True if momentum should be restarted, False otherwise.
        """
        if self.config.restart_strategy == RestartStrategy.NONE:
            return False
            
        if self.config.restart_strategy == RestartStrategy.FIXED:
            # Restart every restart_period iterations
            restart = (self.iter_count > 0) and (self.iter_count % self.config.restart_period == 0)
            self.iter_count += 1
            return restart
            
        if self.config.restart_strategy == RestartStrategy.ADAPTIVE:
            # Restart when loss increases (monotonicity violated)
            restart = loss > self.prev_loss
            self.prev_loss = loss
            return restart
            
        return False
        
    def _update_momentum_parameter(self, restart: bool = False) -> None:
        """
        Updates the momentum parameter t based on the selected scheme.
        
        Args:
            restart (bool): Whether to restart momentum (set t=1).
        """
        if restart:
            self.t = 1.0
            return
            
        if self.config.momentum_scheme == MomentumScheme.ORIGINAL:
            # Original FISTA scheme: t_{k+1} = (1 + sqrt(1 + 4*t_k^2)) / 2
            self.t = (1.0 + sqrt(1.0 + 4.0 * self.t**2)) / 2.0
        elif self.config.momentum_scheme == MomentumScheme.MONOTONIC:
            # Alternative scheme that ensures monotonic decrease in objective
            # This is more stable in some cases: t_{k+1} = (1 + sqrt(1 + 8*t_k^2)) / 4
            self.t = (1.0 + sqrt(1.0 + 8.0 * self.t**2)) / 4.0

    def train_step(self, y: torch.Tensor, dictionary: torch.Tensor, log_losses: bool = True) -> torch.Tensor:
        """
        Performs a single FISTA iteration.
        
        This implements one step of the Fast Iterative Shrinkage-Thresholding Algorithm:
        1. Compute gradient at current auxiliary point z
        2. Take a gradient step: h_new = soft_threshold(z - α * ∇L(z), αλ)
        3. Update momentum parameter t
        4. Update auxiliary variable: z = h_new + ((t_{k-1} - 1)/t_k) * (h_new - h_prev)
        
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
        
        # Manual FISTA update
        with torch.no_grad():
            # Forward pass using the auxiliary point z
            z_pred = self.evaluation_func(dictionary, self.z)
            
            # Compute loss
            loss = self.criterion(z_pred, y)
            if log_losses:
                self.losses.append(loss.item())
            
            # Check if we should restart the momentum
            restart = self._check_restart_condition(loss.item())
            if restart and self.debug:
                self.logger.debug(f"Restarting FISTA momentum (t=1) at iteration {self.iter_count}")
            
            # Compute gradient at z: 2 * D^T * (z_pred - y)
            error = z_pred - y
            gradient = 2 * torch.matmul(dictionary.T, error)
            
            # Store previous h for momentum calculation
            self.prev_h = self.h.clone()
            
            # FISTA update: h = soft_threshold(z - alpha * gradient, alpha * lambda)
            h_update = self.z - step_size * gradient
            h_new = soft_threshold(h_update, step_size * self.config.lambd, self.device)
            
            # Update h parameter
            self.h.data = h_new
            
            # Update momentum parameter
            old_t = self.t
            self._update_momentum_parameter(restart)
            
            # Update auxiliary variable z with momentum
            # z = h_new + ((old_t - 1) / new_t) * (h_new - h_prev)
            momentum_factor = (old_t - 1) / self.t if not restart else 0.0
            self.z = h_new + momentum_factor * (h_new - self.prev_h)
            
            # Call parameter hook if provided
            if self.parameter_hook is not None:
                hook_info = {
                    'h': self.h.detach().clone(),
                    'z': self.z.detach().clone(),
                    't': self.t,
                    'gradient': gradient.detach().clone(),
                    'loss': loss.item(),
                    'alpha': step_size,
                    'restart': restart
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
    
    def partial_fit(self, y: torch.Tensor, dictionary: torch.Tensor, log_losses: bool = True) -> None:
        """
        Performs multiple FISTA iterations.
        
        This method runs the FISTA algorithm for the specified number of epochs,
        effectively optimizing the sparse vector h to minimize the loss while
        maintaining sparsity through the soft thresholding operation.
        
        Args:
            y (torch.Tensor): Target vector.
            dictionary (torch.Tensor): Dictionary matrix.
            log_losses (bool): Whether to log losses.
        """
        epochs = self.config.epochs
        # Reset iteration counter for fixed restart strategy
        self.iter_count = 0
        self.prev_loss = float('inf')
        
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
