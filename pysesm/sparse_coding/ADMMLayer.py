'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica

ADMM Layer Class

Provides an implementation of the Alternating Direction Method of Multipliers
for solving L1-regularized least squares problems in sparse coding.

Authors: The SESM Team 

License: 
'''
from __future__ import annotations

import logging
from dataclasses import dataclass

from collections.abc import Callable
import torch

from .SparseCodingBaseLayer import SparseCodingBaseLayer, SparseCodingConfig
from .sparse_coding_utils import soft_threshold

@dataclass
class ADMMConfig(SparseCodingConfig):
    """
    Configuration parameters for the ADMM algorithm.
    
    This class extends the base SparseCodingConfig with ADMM-specific parameters
    to control its behavior and convergence properties.
    
    Attributes:
        lambd (float): L1 regularization parameter that controls sparsity.
        rho (float): Penalty parameter for the augmented Lagrangian. Controls the 
                    weight of the constraint violation in the objective function.
        alpha (float): Relaxation parameter for the updates (typically between 1.0 and 1.8).
                      Values > 1.0 can accelerate convergence through over-relaxation.
        abs_tol (float): Absolute tolerance for the stopping criteria.
        rel_tol (float): Relative tolerance for the stopping criteria.
        lambda_scaling (float): Scaling factor for the L1 regularization to make it independent of rho.
    """
    lambd: float = 0.01      # L1 regularization parameter
    rho: float = 1.0         # Penalty parameter
    alpha: float = 1.0       # Relaxation parameter (1.0 = standard ADMM, >1.0 = over-relaxation)
    abs_tol: float = 1e-4    # Absolute tolerance for stopping criteria
    rel_tol: float = 1e-2    # Relative tolerance for stopping criteria
    lambda_scaling: float = 1.0  # Scaling factor for lambda to make it independent of rho

class ADMMLayer(SparseCodingBaseLayer):
    """
    A custom PyTorch module implementing the ADMM algorithm for sparse coding.
    
    ADMM (Alternating Direction Method of Multipliers) solves the sparse coding problem
    by breaking it into smaller, more manageable subproblems. It has excellent convergence
    properties and is often more robust than first-order methods like ISTA.
    
    The ADMM formulation for sparse coding is:
        minimize (1/2)||Dh - y||^2 + λ||z||_1
        subject to h = z
    
    where h is the coefficient vector, D is the dictionary, y is the target,
    and z is an auxiliary variable.
    
    Attributes:
        config (ADMMConfig): Configuration parameters for the ADMM algorithm.
        h (torch.nn.Parameter): Sparse vector maintained and updated by the layer.
        z (torch.Tensor): Auxiliary variable for the ADMM algorithm.
        u (torch.Tensor): Scaled dual variable for the ADMM algorithm.
        losses (list): List storing the computed losses during training.
        logger (logging.Logger): Logger for recording debug information.
        debug (bool): Whether to enable detailed debug logging.
        parameter_hook (Callable): Optional callback function to monitor internal state.
        device: Device to run computations on.
        cached_factorization (tuple): Cached matrix factorization for efficiency.
    """

    CONFIG_CLASS = ADMMConfig
    
    def __init__(
            self,
            config: ADMMConfig,
            evaluation_func:  Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            logger: logging.Logger | None = None,
            debug: bool = False,
            parameter_hook: Callable[[dict], None] | None = None,
            device = None):
        """
        Initializes the ADMMLayer with the specified hyperparameters and components.

        Args:
            config (ADMMConfig): Configuration parameters for the ADMM algorithm.
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
        self.cached_factorization = None
        
    def setup(self, h: torch.Tensor = None) -> None:
        """
        Initializes the sparse vector `h` and auxiliary variables for ADMM.
        
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
            # Initialize h as zeros
            self.h = torch.nn.Parameter(
                torch.zeros(self.config.n_functions, 1, device=self.device), 
                requires_grad=False
            )
        
        # Initialize auxiliary variables for ADMM
        self.z = torch.zeros_like(self.h.data)
        self.u = torch.zeros_like(self.h.data)  # Scaled dual variable
    
    def _factorize_system_matrix(self, dictionary: torch.Tensor) -> tuple:
        """
        Factorizes the system matrix (D^T D + ρI) for efficient solving.
        
        This precomputation significantly speeds up the h-update step in ADMM.
        
        Args:
            dictionary (torch.Tensor): Dictionary matrix.
            
        Returns:
            Tuple: Cached factorization for later use in linear system solving.
        """
        # Compute D^T D
        gram = torch.matmul(dictionary.T, dictionary)
        
        # Add ρI to get (D^T D + ρI)
        system_matrix = gram + self.config.rho * torch.eye(
            gram.shape[0], device=self.device
        )
        
        # Cholesky factorization for solving linear systems
        # We'll use L L^T = (D^T D + ρI)
        L = torch.linalg.cholesky(system_matrix) # pylint: disable=not-callable
        # Pylinter have false positive with all torch.linalg functions
        # Other persons have reported this issue since 2023 so I disable the checker:
        # https://github.com/pytorch/pytorch/issues/149639
        # https://github.com/pylint-dev/pylint/issues/9218
        
        return (L, gram)

    def _update_h(self, y: torch.Tensor, dictionary: torch.Tensor) -> torch.Tensor:
        """
        Performs the h-update step of ADMM.
        
        This solves the linear system (D^T D + ρI)h = D^T y + ρ(z - u)
        
        Args:
            y (torch.Tensor): Target vector.
            dictionary (torch.Tensor): Dictionary matrix.
            
        Returns:
            torch.Tensor: Updated h vector.
        """
        # Compute the right-hand side of the linear system
        Dt_y = torch.matmul(dictionary.T, y)
        z_minus_u = self.z - self.u
        rhs = Dt_y + self.config.rho * z_minus_u
        
        # Use cached factorization if available, otherwise compute it
        if self.cached_factorization is None:
            self.cached_factorization = self._factorize_system_matrix(dictionary)
        
        L, _  = self.cached_factorization
        
        # Solve the linear system (D^T D + ρI)h = rhs using Cholesky factorization
        # First solve L y = rhs, then L^T h = y
        y_intermediate = torch.linalg.solve_triangular(L, rhs, upper=False) # pylint: disable=not-callable 
        h_new = torch.linalg.solve_triangular(L.T, y_intermediate, upper=True) # pylint: disable=not-callable
        
        return h_new
    
    def _update_z(self, h_tilde: torch.Tensor) -> torch.Tensor:
        """
        Performs the z-update step of ADMM using soft thresholding.
        
        Args:
            h_tilde (torch.Tensor): Relaxed h + u value.
            
        Returns:
            torch.Tensor: Updated z vector.
        """
        # Apply the soft-thresholding operator
        threshold = self.config.lambd * self.config.lambda_scaling / self.config.rho
        z_new = soft_threshold(h_tilde, threshold, self.device)
        
        return z_new
    
    def _compute_residuals(self, h_new: torch.Tensor, z_new: torch.Tensor) -> tuple[float, float]:
        """
        Computes the primal and dual residuals for convergence checking.
        
        Args:
            h_new (torch.Tensor): Updated h vector.
            z_new (torch.Tensor): Updated z vector.
            
        Returns:
            Tuple[float, float]: Primal and dual residuals.
        """
        # Primal residual: ||h - z||
        primal_residual = torch.norm(h_new - z_new).item()
        
        # Dual residual: ρ||z - z_prev||
        dual_residual = self.config.rho * torch.norm(z_new - self.z).item()
        
        return primal_residual, dual_residual
    
    def _check_stopping_criteria(self, 
                                 primal_residual: float, 
                                 dual_residual: float,
                                 h_new: torch.Tensor, 
                                 z_new: torch.Tensor) -> bool:
        """
        Checks if the ADMM algorithm has converged.
        
        Args:
            primal_residual (float): Primal residual.
            dual_residual (float): Dual residual.
            h_new (torch.Tensor): Updated h vector.
            z_new (torch.Tensor): Updated z vector.
            
        Returns:
            bool: True if converged, False otherwise.
        """
        # Compute the tolerance thresholds
        h_norm = torch.norm(h_new).item()
        z_norm = torch.norm(z_new).item()
        u_norm = torch.norm(self.u).item()
        
        eps_primal = self.config.abs_tol + self.config.rel_tol * max(h_norm, z_norm)
        eps_dual = self.config.abs_tol + self.config.rel_tol * u_norm
        
        # Check if both residuals are below their respective thresholds
        return primal_residual < eps_primal and dual_residual < eps_dual
    
    def _update_rho(self, primal_residual: float, dual_residual: float) -> None:
        """
        Updates the penalty parameter rho based on the primal and dual residuals.
        
        This implements an adaptive penalty parameter strategy to improve convergence.
        
        Args:
            primal_residual (float): Primal residual.
            dual_residual (float): Dual residual.
        """
        # Only update rho occasionally to avoid computational overhead
        if hasattr(self, 'iter_count') and self.iter_count % 10 == 0:
            mu = 10.0  # Multiplier for rho updates
            tau = 2.0  # Ratio threshold
            
            if primal_residual > mu * dual_residual:
                # Primal residual much larger than dual: increase rho
                self.config.rho *= tau
                self.u /= tau  # Update scaled dual variable to maintain u = y/rho
                # Invalidate cached factorization since rho changed
                self.cached_factorization = None
                if self.debug:
                    self.logger.debug(f"Increased rho to {self.config.rho:.6f}")
                    
            elif dual_residual > mu * primal_residual:
                # Dual residual much larger than primal: decrease rho
                self.config.rho /= tau
                self.u *= tau  # Update scaled dual variable to maintain u = y/rho
                # Invalidate cached factorization since rho changed
                self.cached_factorization = None
                if self.debug:
                    self.logger.debug(f"Decreased rho to {self.config.rho:.6f}")


    def train_step(self, y: torch.Tensor, dictionary: torch.Tensor, log_losses: bool = True) -> torch.Tensor:
        """
        Performs a single ADMM iteration step.

        This implements one step of the ADMM algorithm:
        1. h-update: Solve linear system
        2. z-update: Apply soft thresholding
        3. u-update: Update dual variable

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

        # Ensure z and u are initialized
        if not hasattr(self, 'z') or self.z is None:
            self.z = torch.zeros_like(self.h.data, device=self.device)
        if not hasattr(self, 'u') or self.u is None:
            self.u = torch.zeros_like(self.h.data, device=self.device)

        # Initialize factorization if needed
        if self.cached_factorization is None:
            self.cached_factorization = self._factorize_system_matrix(dictionary)

        # Perform a single ADMM iteration
        with torch.no_grad():
            # 1. Update h by solving the linear system
            h_new = self._update_h(y, dictionary)

            # 2. Apply relaxation and update z with soft thresholding
            h_relaxed = self.config.alpha * h_new + (1 - self.config.alpha) * self.z
            h_tilde = h_relaxed + self.u
            z_new = self._update_z(h_tilde)

            # 3. Update the dual variable u
            self.u = self.u + h_relaxed - z_new

            # 4. Compute residuals for monitoring (but don't use for stopping here)
            primal_residual, dual_residual = self._compute_residuals(h_new, z_new)

            # 5. Update variables for next iteration
            self.h.data = h_new
            self.z = z_new

            # Compute current loss for tracking
            y_pred = self.evaluation_func(dictionary, self.h)
            loss = self.criterion(y_pred, y)

            if log_losses:
                self.losses.append(loss.item())

            # Call parameter hook if provided
            if self.parameter_hook is not None:
                # Create info dictionary with relevant iteration data
                hook_info = {
                    'h': self.h.detach().clone(),
                    'z': self.z.detach().clone(),
                    'u': self.u.detach().clone(),
                    'loss': loss.item(),
                    'primal_residual': primal_residual,
                    'dual_residual': dual_residual,
                    'l1_norm': torch.norm(self.h.data, p=1).item(),
                    'sparsity_ratio': torch.sum(torch.abs(self.h.data) > 1e-6).item() / self.h.data.numel()
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
        Performs multiple ADMM iterations.

        This method runs the ADMM algorithm for the specified number of epochs,
        with each epoch performing a single ADMM iteration step.

        Args:
            y (torch.Tensor): Target vector.
            dictionary (torch.Tensor): Dictionary matrix.
            log_losses (bool): Whether to log losses.
        """
        epochs=self.config.epochs
        # Initialize cached factorization for efficiency across iterations
        self.cached_factorization = self._factorize_system_matrix(dictionary)

        # Track convergence metrics
        converged = False
        best_loss = float('inf')
        best_h = None
        no_improvement_count = 0

        for epoch in range(epochs):
            # Perform a single ADMM iteration
            loss = self.train_step(y, dictionary, log_losses)

            # Debug logging
            if self.debug and (epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1):
                h_nnz = torch.sum(torch.abs(self.h.data) > 1e-6).item()
                sparsity = h_nnz / self.h.data.numel() * 100
                self.logger.debug(
                    f"ADMM Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}, "
                    f"Non-zeros: {h_nnz}/{self.h.data.numel()} ({sparsity:.1f}%)"
                )

            # Track best solution (optional)
            if loss < best_loss:
                best_loss = loss.item()
                best_h = self.h.data.clone()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Check convergence using residuals
            if epoch > 0:  # Skip first iteration
                # Compute primal and dual residuals
                primal_residual, dual_residual = self._compute_residuals(self.h.data, self.z)

                # Check if converged
                converged = self._check_stopping_criteria(primal_residual, dual_residual, self.h.data, self.z)
                if converged and self.debug:
                    self.logger.debug(f"ADMM converged after {epoch + 1} iterations")
                    break

            # Optional early stopping if enabled
            if hasattr(self.config, 'early_stopping') and self.config.early_stopping:
                if no_improvement_count >= 10:  # No improvement for 10 consecutive iterations
                    if self.debug:
                        self.logger.debug(f"Early stopping at epoch {epoch + 1}, no improvement for 10 iterations")
                    break

        # Use best solution if tracking was enabled and a better solution was found
        if best_h is not None and best_loss < loss.item():
            self.h.data = best_h
            if self.debug:
                self.logger.debug(f"Using best solution with loss {best_loss:.6f}")

        # Clean up cached factorization to free memory
        self.cached_factorization = None
