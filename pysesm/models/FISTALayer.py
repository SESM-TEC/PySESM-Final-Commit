'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica

FISTA Layer Class

Provides the layer implementing Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)
for finding h, the sparse vector that chooses words in a dictionary to build a surrogate 
function. FISTA provides accelerated convergence compared to standard ISTA.

Authors: The SESM Team 

License: 
'''

import logging
import torch
import math
from typing import Callable, Optional
from dataclasses import dataclass
from pysesm.models.SparseCodingBaseLayer import SparseCodingBaseLayer, SparseCodingConfig
from pysesm.customization_factories.SparseCodingFactory import SparseCodingFactory
from pysesm.models.ISTALayer import StepSizeMethod

@dataclass
class FISTAConfig(SparseCodingConfig):
    """
    Configuration parameters for the FISTA algorithm.
    
    This class encapsulates all configuration parameters for the Fast Iterative
    Shrinkage-Thresholding Algorithm (FISTA) to provide a cleaner interface
    and easier parameter management.
    
    Attributes:
        alpha (float): Learning rate for parameter updates. If step_size_method is MANUAL,
                     this value is used directly as the fixed step size.
        lambd (float): Regularization parameter controlling sparsity (L1 penalty strength).
        step_size_method (StepSizeMethod): Method to determine step size during FISTA iterations.
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

@SparseCodingFactory.register("classic_fista")
class FISTALayer(SparseCodingBaseLayer):
    """
    A custom PyTorch module implementing FISTA (Fast Iterative Shrinkage-Thresholding Algorithm).

    This layer implements an accelerated version of ISTA for sparse coding, which converges faster
    by using momentum. FISTA finds a sparse representation (vector h) that linearly combines elements 
    from a dictionary to approximate a target function with L1 regularization for sparsity.

    Attributes:
        config (FISTAConfig): Configuration parameters for the FISTA algorithm.
        h (torch.nn.Parameter): Sparse vector maintained and updated by the layer.
        z (torch.Tensor): Auxiliary variable for momentum updates.
        t (float): FISTA momentum parameter.
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
            Performs a single FISTA iteration.
        partial_fit(y: torch.Tensor, epochs: int, dictionary: torch.Tensor, log_losses: bool) -> None:
            Performs multiple FISTA iterations for fitting.
        forward(y: torch.Tensor, dictionary: torch.Tensor, log_losses: bool) -> torch.Tensor:
            Computes the current loss without updating parameters.
    """
    CONFIG_CLASS = FISTAConfig
    def __init__(
            self,
            config: FISTAConfig,
            logger: logging.Logger,
            debug: bool = False,
            parameter_hook: Optional[Callable[[dict], None]] = None,
            device=None):
        super().__init__(config=config,
                         logger=logger,
                         debug=debug,
                         parameter_hook=parameter_hook,
                         device=device)

        self.losses = []
        self.last_eigenvector = None
        self.h_prev = None
        self.t_prev = 1.0

    def setup(self, h: torch.Tensor = None) -> None:
        """
        Initialize the sparse vector h as a learnable parameter.

        Args:
            h (torch.Tensor, optional): Initial value for the sparse vector.
                If None, initialized to zeros. Defaults to None.

        Raises:
            ValueError: If h dimensions don't match configuration.
        """
        if h is not None:
            if h.dim() != 2:
                h = h.reshape(-1, 1)
            if h.shape[0] != self.config.n_functions:
                raise ValueError(f"Dimension mismatch: h has {h.shape[0]} rows but n_functions is {self.config.n_functions}")
            self.h = torch.nn.Parameter(h.to(self.device), requires_grad=True)
        else:
            self.h = torch.nn.Parameter(
                torch.zeros(self.config.n_functions, 1).to(self.device),
                requires_grad=True
            )
        self.h_prev = self.h.clone().detach()

    def calculate_step_size(self, dictionary: torch.Tensor) -> float:
        """
        Calculates the step size for FISTA iterations based on the selected method.
        
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
            if self.config.step_size_method == StepSizeMethod.MANUAL:
                return self.config.alpha

            L_estimate = 0.0
            if self.config.step_size_method == StepSizeMethod.EXACT:
                gram = torch.matmul(dictionary.T, dictionary)
                n = gram.shape[0]
                X = self.last_eigenvector if self.last_eigenvector is not None else torch.randn(n, 1, device=self.device)
                X = X / torch.norm(X)
                eigenvalues, eigenvectors = torch.lobpcg(A=gram, k=1, X=X, largest=True)
                self.last_eigenvector = eigenvectors
                L_estimate = eigenvalues[0].item()

            elif self.config.step_size_method == StepSizeMethod.POWER_ITERATION:
                d_t = dictionary.T
                v = getattr(self, 'last_power_vector', torch.randn(dictionary.shape[1], 1, device=self.device))
                v = v / torch.norm(v)
                for _ in range(self.config.power_iterations):
                    v = torch.matmul(d_t, torch.matmul(dictionary, v))
                    v = v / torch.norm(v)
                self.last_power_vector = v.clone()
                L_estimate = torch.matmul(v.T, torch.matmul(torch.matmul(d_t, dictionary), v)).item()

            elif self.config.step_size_method == StepSizeMethod.FROBENIUS:
                frob_norm_squared = torch.sum(dictionary * dictionary)
                L_estimate = 2.0 * frob_norm_squared

            return 1.0 / (L_estimate + 1e-8)

    def soft_threshold(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Apply soft thresholding operation (proximal operator for L1 norm).

        Implements the operation: S_λ(x) = sign(x) * max(|x| - λ, 0)

        Args:
            x (torch.Tensor): Input tensor.
            threshold (float): Threshold value λ.

        Returns:
            torch.Tensor: Soft-thresholded tensor.
        """
        return torch.sign(x) * torch.maximum(
            torch.abs(x) - threshold,
            torch.zeros_like(x, device=self.device)
        )

    def train_step(self, y: torch.Tensor, dictionary: torch.Tensor, log_losses: bool = True) -> torch.Tensor:
        """
        Perform a single FISTA iteration.

        This implements one step of the Fast Iterative Shrinkage-Thresholding Algorithm:
        1. Calculate momentum point y_momentum based on previous iterations
        2. Evaluate gradient at the momentum point
        3. Take a gradient step from the momentum point
        4. Apply soft thresholding to enforce sparsity
        5. Update momentum parameter and prepare for next iteration

        Args:
            y (torch.Tensor): Target vector.
            dictionary (torch.Tensor): Dictionary matrix.
            log_losses (bool, optional): Whether to record loss values. Defaults to True.

        Returns:
            torch.Tensor: Current loss value.
        """
        y = y.to(self.device)
        dictionary = dictionary.to(self.device)
        step_size = self.calculate_step_size(dictionary)

        if self.h_prev is None:
            self.h_prev = self.h.clone().detach()

        with torch.no_grad():
            # FISTA acceleration: compute momentum step
            t_next = (1 + math.sqrt(1 + 4 * self.t_prev ** 2)) / 2
            momentum = ((self.t_prev - 1) / t_next)
            y_momentum = self.h + momentum * (self.h - self.h_prev)

            y_pred = self.config.evaluation_func(dictionary, y_momentum)
            loss = self.criterion(y_pred, y)
            if log_losses:
                self.losses.append(loss.item())

            error = y_pred - y
            gradient = 2 * torch.matmul(dictionary.T, error)

            h_update = y_momentum - step_size * gradient
            h_new = self.soft_threshold(h_update, step_size * self.config.lambd)

            self.h_prev = self.h.clone().detach()
            self.h.data = h_new
            self.t_prev = t_next

            if self.parameter_hook is not None:
                self.parameter_hook({
                    'h': self.h.detach().clone(),
                    'gradient': gradient.detach().clone(),
                    'loss': loss.item(),
                    'alpha': step_size
                })

        return loss

    def forward(self, y, dictionary, log_losses=True):
        """
        Compute loss without updating parameters.

        Args:
            y (torch.Tensor): Target vector.
            dictionary (torch.Tensor): Dictionary matrix.
            log_losses (bool, optional): Whether to record loss values. Defaults to True.

        Returns:
            torch.Tensor: Current loss value.
        """
        y = y.to(self.device)
        dictionary = dictionary.to(self.device)
        with torch.no_grad():
            y_pred = self.config.evaluation_func(dictionary, self.h)
            loss = self.criterion(y_pred, y)
            if log_losses:
                self.losses.append(loss.item())
        return loss

    def partial_fit(self, y: torch.Tensor, epochs: int, dictionary: torch.Tensor, log_losses: bool = True) -> None:
        """
        Perform multiple FISTA iterations to find optimal sparse coding.

        This runs the FISTA algorithm for a specified number of epochs,
        updating the sparse vector h to minimize reconstruction error
        while maintaining sparsity.

        Args:
            y (torch.Tensor): Target vector.
            epochs (int): Number of iterations to perform.
            dictionary (torch.Tensor): Dictionary matrix.
            log_losses (bool, optional): Whether to record loss values. Defaults to True.
        """
        for epoch in range(epochs):
            loss = self.train_step(y, dictionary, log_losses)
            if self.debug and (epoch == 0 or (epoch + 1) % 100 == 0 or epoch == epochs - 1):
                self.logger.debug(f"[FISTA] Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")
            if self.config.early_stopping and epoch > 0 and len(self.losses) >= 2:
                if abs(self.losses[-1] - self.losses[-2]) < self.config.early_stopping_tol:
                    if self.debug:
                        self.logger.debug(f"[FISTA] Early stopping at epoch {epoch + 1}")
                    break
