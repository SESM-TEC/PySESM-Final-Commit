'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica

Sparse Coding Utilities

Shared utility functions for sparse coding algorithms like ISTA and FISTA.

Authors: The SESM Team 

License: 
'''
from __future__ import annotations

import logging
from enum import Enum, auto

import torch

class StepSizeMethod(Enum):
    """Enumeration of methods for determining the step size in iterative shrinkage algorithms."""
    MANUAL = auto()        # Use fixed alpha value provided by user
    EXACT = auto()         # Use LOBPCG to find largest eigenvalue of D^T D (accurate but slow)
    POWER_ITERATION = auto()  # Power iteration approximation (balanced)
    FROBENIUS = auto()     # Frobenius norm upper bound (fast but conservative)


def soft_threshold(x: torch.Tensor, threshold: float, device: str | None =None) -> torch.Tensor:
    """
    Applies soft thresholding operation (proximal operator for L1 norm).
    
    This implements the operation: S_λ(x) = sign(x) * max(|x| - λ, 0)
    
    Args:
        x (torch.Tensor): Input tensor.
        threshold (float): Threshold value λ.
        device: Computation device (default: same as input tensor).
            
    Returns:
        torch.Tensor: Soft thresholded tensor.
    """
    if device is None:
        device = x.device
    return torch.sign(x) * torch.maximum(
        torch.abs(x) - threshold,
        torch.zeros_like(x, device=device)
    )


def calculate_step_size(
    dictionary: torch.Tensor, 
    method: StepSizeMethod, 
    alpha: float = 0.1, 
    power_iterations: int = 10,
    last_eigenvector: torch.Tensor | None = None,
    debug: bool = False, 
    logger: logging.Logger | None = None
) -> tuple[float, torch.Tensor | None]:
    """
    Calculates the step size for iterative shrinkage algorithms based on the selected method.
    
    The method used for calculation depends on the step_size_method:
    - MANUAL: Simply returns the fixed alpha value provided
    - EXACT: Computes the largest eigenvalue of D^T D using LOBPCG
    - POWER_ITERATION: Uses power iteration to approximate the largest eigenvalue
    - FROBENIUS: Uses the Frobenius norm as an upper bound
    
    Args:
        dictionary (torch.Tensor): The dictionary matrix.
        method (StepSizeMethod): Method to use for step size calculation.
        alpha (float): Fixed learning rate (used if method is MANUAL).
        power_iterations (int): Number of iterations for power method.
        last_eigenvector (torch.Tensor, optional): Previous eigenvector for warm start.
        debug (bool): Whether to print debug information.
        logger (logging.Logger, optional): Logger for debug messages.
            
    Returns:
        Tuple[float, torch.Tensor]: A tuple containing:
            - The calculated step size.
            - The updated eigenvector (for warm starting next iteration) or None.
    """
    device = dictionary.device
    
    with torch.no_grad():
        # For MANUAL method, just return the fixed alpha value
        if method == StepSizeMethod.MANUAL:
            return alpha, last_eigenvector
                
        L_estimate = 0.0
        updated_eigenvector = last_eigenvector
        
        if method == StepSizeMethod.EXACT:
            # Compute D^T D (gram matrix)
            gram = torch.matmul(dictionary.T, dictionary)
            
            # Create initial guess vector or use previous eigenvector for warm start
            k = 1  # We only want the largest eigenvalue
            n = gram.shape[0]
            
            if last_eigenvector is not None:
                # Use warm start if we have a previous eigenvector
                X = last_eigenvector.clone()
            else:
                # Otherwise, use random initialization
                X = torch.randn(n, k, device=device)
                X = X / torch.norm(X)
            
            # Compute largest eigenvalue using LOBPCG
            eigenvalues, eigenvectors = torch.lobpcg(A=gram, k=1, X=X, largest=True)
            
            # Store eigenvector for warm starting next time
            updated_eigenvector = eigenvectors
            
            L_estimate = eigenvalues[0].item()
            
        elif method == StepSizeMethod.POWER_ITERATION:
            # Transpose for matrix multiplication
            d_t = dictionary.T
            
            # Initialize vector - use warm start if available
            if last_eigenvector is not None:
                v = last_eigenvector.clone()
            else:
                v = torch.randn(dictionary.shape[1], 1, device=device)
                v = v / torch.norm(v)
            
            # Power iteration
            for _ in range(power_iterations):
                v = torch.matmul(d_t, torch.matmul(dictionary, v))
                v_norm = torch.norm(v)
                if v_norm > 0:
                    v = v / v_norm
            
            # Store vector for warm starting next time
            updated_eigenvector = v.clone()
            
            # Compute Rayleigh quotient
            L_estimate = torch.matmul(v.T, torch.matmul(torch.matmul(d_t, dictionary), v)).item()                

        elif method == StepSizeMethod.FROBENIUS:
            # Frobenius norm upper bound (fastest but less tight)
            # For MSE loss, L <= 2 * ||D||_F^2
            frob_norm_squared = torch.sum(dictionary * dictionary)
            L_estimate = 2.0 * frob_norm_squared
        
        # Step size should be <= 1/L for convergence
        step_size = 1.0 / (L_estimate + 1e-8)  # Adding small constant for stability
        
        if debug and logger is not None:
            logger.debug(
                f"Calculated step size: {step_size}, Lipschitz estimate: {L_estimate}, Method: {method.name}"
            )
            
    return step_size, updated_eigenvector
