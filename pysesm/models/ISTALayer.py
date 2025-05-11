'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica

ISTA Layer Class

Provides the layer in charge of finding h, the sparse vector 
that chooses words in a dictionary to build a surrogate 
function.

Authors: The SESM Team 

License: 
'''

import logging
import torch
from typing import Callable, Iterator, Optional
from enum import Enum, auto
from dataclasses import dataclass, field
from pysesm.models.SparseCodingBaseLayer import SparseCodingBaseLayer, SparseCodingConfig

class StepSizeMethod(Enum):
    """Enumeration of methods for determining the ISTA step size."""
    MANUAL = auto()        # Use fixed alpha value provided by user
    EXACT = auto()         # Use LOBPCG to find largest eigenvalue of D^T D (accurate but slow)
    POWER_ITERATION = auto()  # Power iteration approximation (balanced)
    FROBENIUS = auto()     # Frobenius norm upper bound (fast but conservative)

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
    lambd: float = 0.01
    step_size_method: StepSizeMethod = StepSizeMethod.POWER_ITERATION
    power_iterations: int = 10
    early_stopping: bool = False
    early_stopping_tol: float = 1e-6

# Define the ISTA layer - here's the important part    
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
        last_eigenvector (torch.Tensor): Last computed eigenvector for warm-starting LOBPCG.

    Methods:
        setup(h: torch.Tensor) -> None:
            Initializes the sparse vector `h` as a learnable parameter.
        calculate_step_size(dictionary: torch.Tensor) -> float:
            Calculates the optimal step size based on the chosen method.
        soft_threshold(x: torch.Tensor, threshold: float) -> torch.Tensor:
            Applies soft thresholding operation for L1 proximal mapping.
        train_step(y: torch.Tensor, dictionary: torch.Tensor, log_losses: bool) -> torch.Tensor:
            Performs a single ISTA iteration.
        partial_fit(y: torch.Tensor, epochs: int, dictionary: torch.Tensor, log_losses: bool) -> None:
            Performs multiple ISTA iterations for fitting.
        forward(y: torch.Tensor, dictionary: torch.Tensor, log_losses: bool) -> torch.Tensor:
            Computes the current loss without updating parameters.
    """

    CONFIG_CLASS = ISTAConfig
    
    def __init__(
            self,
            config: ISTAConfig,
            logger: logging.Logger,
            debug: bool = False,
            parameter_hook: Optional[Callable[[dict], None]] = None,
            device = None):
        """
        Initializes the ISTALayer with the specified hyperparameters and components.

        Args:
            config (ISTAConfig): Configuration parameters for the ISTA algorithm.
            logger (logging.Logger): Logger for recording debug information.
            debug (bool): Whether to enable detailed debug logging.
            criterion (torch.nn.Module, optional): Loss function used for training (default: MSELoss).
            parameter_hook (Callable, optional): Callback function to inspect the current parameter state.
            device: Device to run computations on.
        """
        super().__init__(config=config,
                         logger=logger,
                         debug=debug,
                         parameter_hook=parameter_hook,
                         device=device)

        self.losses = []
               
        self.last_eigenvector = None  # For warm starting LOBPCG
 
        

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
                raise ValueError(f"Dimension mismatch: h has {h_to_use.shape[0]} rows but n_functions is {self.config.n_functions}")
                
            self.h = torch.nn.Parameter(h.to(self.device), requires_grad=True)
        else:
            # Initialize h as zeros (common for ISTA)
            self.h = torch.nn.Parameter(
                torch.zeros(self.config.n_functions, 1).to(self.device), 
                requires_grad=True
            )

    def calculate_step_size(self, dictionary: torch.Tensor) -> float:
        """
        Calculates the step size for ISTA iterations based on the selected method.
        
        The method used for calculation depends on config.step_size_method:
        - MANUAL: Simply returns the fixed alpha value provided in the config
        - EXACT: Computes the largest eigenvalue of D^T D using LOBPCG
        - POWER_ITERATION: Uses power iteration to approximate the largest eigenvalue
        - FROBENIUS: Uses the Frobenius norm as an upper bound
        
        Args:
            dictionary (torch.Tensor): The dictionary matrix.
            
        Returns:
            float: The calculated step size.
        """
        with torch.no_grad():
            # For MANUAL method, just return the fixed alpha value from config
            if self.config.step_size_method == StepSizeMethod.MANUAL:
                return self.config.alpha
                
            L_estimate = 0.0
            
            if self.config.step_size_method == StepSizeMethod.EXACT:
                # Compute D^T D (gram matrix)
                gram = torch.matmul(dictionary.T, dictionary)
                
                # Create initial guess vector or use previous eigenvector for warm start
                k = 1  # We only want the largest eigenvalue
                n = gram.shape[0]
                
                if self.last_eigenvector is not None:
                    # Use warm start if we have a previous eigenvector
                    X = self.last_eigenvector.clone()
                else:
                    # Otherwise, use random initialization
                    X = torch.randn(n, k, device=self.device)
                    X = X / torch.norm(X)
                
                # Compute largest eigenvalue using LOBPCG
                eigenvalues, eigenvectors = torch.lobpcg(A=gram, k=1, X=X, largest=True)
                
                # Store eigenvector for warm starting next time
                self.last_eigenvector = eigenvectors
                
                L_estimate = eigenvalues[0].item()
                
            elif self.config.step_size_method == StepSizeMethod.POWER_ITERATION:
                # Transpose for matrix multiplication
                d_t = dictionary.T
                
                # Initialize vector - use warm start if available
                if hasattr(self, 'last_power_vector') and self.last_power_vector is not None:
                    v = self.last_power_vector.clone()
                else:
                    v = torch.randn(dictionary.shape[1], 1, device=self.device)
                    v = v / torch.norm(v)
                
                # Power iteration
                for _ in range(self.config.power_iterations):
                    v = torch.matmul(d_t, torch.matmul(dictionary, v))
                    v_norm = torch.norm(v)
                    if v_norm > 0:
                        v = v / v_norm
                
                # Store vector for warm starting next time
                self.last_power_vector = v.clone()
                
                # Compute Rayleigh quotient
                L_estimate = torch.matmul(v.T, torch.matmul(torch.matmul(d_t, dictionary), v)).item()                

            elif self.config.step_size_method == StepSizeMethod.FROBENIUS:
                # Frobenius norm upper bound (fastest but less tight)
                # For MSE loss, L <= 2 * ||D||_F^2
                frob_norm_squared = torch.sum(dictionary * dictionary)
                L_estimate = 2.0 * frob_norm_squared
            
            # Step size should be <= 1/L for convergence
            alpha = 1.0 / (L_estimate + 1e-8)  # Adding small constant for stability
            
            if self.debug:
                self.logger.debug(f"Calculated alpha: {alpha}, Lipschitz estimate: {L_estimate}, Method: {self.config.step_size_method.name}")
                
        return alpha

    def soft_threshold(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Applies soft thresholding operation (proximal operator for L1 norm).
        
        This implements the operation: S_λ(x) = sign(x) * max(|x| - λ, 0)
        
        Args:
            x (torch.Tensor): Input tensor.
            threshold (float): Threshold value λ.
            
        Returns:
            torch.Tensor: Soft thresholded tensor.
        """
        return torch.sign(x) * torch.maximum(
            torch.abs(x) - threshold,
            torch.zeros_like(x, device=self.device)
        )

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
        step_size = self.calculate_step_size(dictionary)
        
        # Manual ISTA update
        with torch.no_grad():
            # Forward pass
            y_pred = self.config.evaluation_func(dictionary, self.h)
            
            # Compute loss
            loss = self.criterion(y_pred, y)
            if log_losses:
                self.losses.append(loss.item())
            
            # Compute gradient for MSE: 2 * D^T * (y_pred - y)
            error = y_pred - y
            gradient = 2 * torch.matmul(dictionary.T, error)
            
            # ISTA update: h = soft_threshold(h - alpha * gradient, alpha * lambda)
            h_update = self.h - step_size * gradient
            h_new = self.soft_threshold(h_update, step_size * self.config.lambd)
            
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

    def forward(self, y, dictionary, log_losses=True):
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
            y_pred = self.config.evaluation_func(dictionary, self.h)
            assert y_pred.shape == y.shape, f"Shape mismatch: y_pred {y_pred.shape} != y {y.shape}"
            
            loss = self.criterion(y_pred, y)

            if log_losses:
                self.losses.append(loss.item())

        return loss
    
    def partial_fit(self, y: torch.Tensor, epochs: int, dictionary: torch.Tensor, log_losses: bool = True) -> None:
        """
        Performs multiple ISTA iterations.
        
        This method runs the ISTA algorithm for the specified number of epochs,
        effectively optimizing the sparse vector h to minimize the loss while
        maintaining sparsity through the soft thresholding operation.
        
        Args:
            y (torch.Tensor): Target vector.
            epochs (int): Number of iterations.
            dictionary (torch.Tensor): Dictionary matrix.
            log_losses (bool): Whether to log losses.
        """
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
