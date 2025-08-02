'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica

Unnormalized Gaussian Function Set

This is a surrogate function set composed of unnormalized gaussians.
It just helps with two things: randomly initialize the words and
evaluate points on each word, where a word is an unnormalized gaussian.

Authors: The SESM Team 

License: 
'''


import numpy as np
import torch
from typing import List, Union, TypeAlias
from pysesm.base_types import TensorBatch
from pysesm.functions.SurrogateFunction import SurrogateFunction
from pysesm.utils.linalg import (
    to_triu_matrix,
    generate_random_vectors,
    get_upper_triangle,
    gram_schmidt,
)

# Define a descriptive type alias for the range structure
RangeType: TypeAlias = Union[List[float], List[List[float]]]

class GaussianFunction(SurrogateFunction):
    """
    A surrogate function composed of a set of unnormalized Gaussian functions.

    This computes the unnormalized Gaussians for a given set of points.
    Each Gaussian corresponds to a dictionary word, and it is equivalent
    to:

    exp(-0.5 (x-µ)' G (x-µ)

    where G stands for the inverse of the covariance matrix, also
    called the precision matrix.

    Since G is symmetric and positive definite, it can be split with a
    Cholesky decompomposition into G=A'A, where A is an upper
    triangular matrix (the usual notation is G=LL', with L a lower
    triangular matrix, but its equivalent).  This way, the parameters
    required for each Gaussian are µ and rho, where rho are the upper
    triangular elements of A (i.e. without the zeros).   

    Theta holds the parameters for all words in a dictionary.

    This class implements the `evaluate` method for a single 2D tensor of data,
    as required by the SurrogateFunction base class. The polymorphic `__call__`
    method, which handles batched inputs (like NestedTensors), is inherited.

    Parameters
    ----------
    mu_range : list or array-like
        Specifies the range for initializing the means (μ) of the Gaussians.
        Can be provided in two formats:
        1. As a list of two values [min, max] - This range will be used for all dimensions
           Example: [-1, 1] will initialize all mean components uniformly in [-1, 1]
        2. As a list of ranges per dimension: [[min1, max1], [min2, max2], ...]
           Example: [[0, 1], [-1, 0]] specifies different ranges for each dimension
           Must have length n_features if provided in this format.

    eig_range : list[float] or array-like
        A list of two values [min, max] specifying the range for the eigenvalues
        of the covariance matrix. These values represent variances, not precisions.
        It can also be a list of ranges, as mu_range.
        Example: [0.5, 0.5] will create Gaussians with variance 0.5 in all directions
                [0.1, 1.0] will create Gaussians with variances randomly chosen 
                between 0.1 and 1.0 for each eigendirection
    """

    eig_range: RangeType
    mu_range: RangeType

    def __init__(self, n_features, n_functions, logger, mu_range=[-1, 1], eig_range=[0.1, 0.5]):
        super().__init__(n_features=n_features, 
                         n_functions=n_functions, 
                         logger=logger)
        self.theta_size = int(n_features * (n_features + 3) / 2)
        self.mu_range = self._fix_range(mu_range)
        self.eig_range = self._fix_range(eig_range)
        

    def _fix_range(self, range):
        # Expand the range one for each function
        fixed = torch.tensor(range)
        if fixed.ndim == 1:
            if len(fixed) != 2:
                raise ValueError(
                    f"If providing a single range, it must be [from, to], got {range}"
                )
            # Expand single range to all dimensions
            fixed = fixed.expand(self.n_features, 2)
        elif fixed.shape != (self.n_features, 2):
            raise ValueError(
                f"range must be either [from, to] or have shape ({self.n_features}, 2) "
                f"for per-dimension ranges, got shape {fixed.shape}"
            )
        return fixed


    def initialize(self) -> torch.nn.Parameter:
        """
        Initialize the parameter vector Theta maintaining specific
        statistical properties given at construction time:
        mu_range: 
        - Means (mu) uniformly distributed in mu_range
        - Covariance matrices with controlled eigenvalue distribution
        """
               
        # Reminder tensor has (planes,rows,columns) 

        # Vectorized initialization of mu
        mu_min_vals = self.mu_range[:, 0]
        mu_spans = self.mu_range[:, 1] - mu_min_vals
        mu = torch.rand(self.n_features, self.n_functions) * mu_spans.unsqueeze(1) + mu_min_vals.unsqueeze(1)

        # Generate random orthogonal matrices Q_all
        Q_all = torch.rand(self.n_functions, self.n_features, self.n_features)
        Q_all, _ = torch.linalg.qr(Q_all)

        # Vectorized initialization of eigenvalues (D_all)
        eig_min_vals = self.eig_range[:, 0]
        eig_spans = self.eig_range[:, 1] - eig_min_vals
        D_all = torch.rand(self.n_functions, self.n_features) * eig_spans + eig_min_vals
        
        # Convert variances to precision eigenvalues
        D_precision_vec = 1.0 / D_all  # Shape: (n_functions, n_features)

        # Paso 1: Calcular Q @ D eficientemente escalando las columnas de Q.
        # unsqueeze(2) -> (B, N, 1) escala las columnas.
        Temp = Q_all * D_precision_vec.unsqueeze(2)

        # Paso 2: Multiplicar el resultado por Q.T
        Sigma_inv_all = torch.bmm(Temp, Q_all.transpose(1, 2))
        
        # Batch Cholesky decomposition
        L_all = torch.linalg.cholesky(Sigma_inv_all).transpose(1, 2)

        # Extract upper triangular elements from all matrices at once
        n = L_all.shape[1]
        indices = torch.triu_indices(n, n, offset=0)
        Rho = L_all[:, indices[0], indices[1]].T  # Shape: (n_upper_elements, n_functions)

        # Log debug information
        self.logger.info(f"Shape of Mu: {mu.shape}")
        self.logger.info(f"Shape of Rho: {Rho.shape}")
        
        # Create final Theta parameter
        Theta = torch.nn.Parameter(torch.cat([Rho, mu], dim=0))
        
        self.logger.info(f"Shape of Theta: {Theta.shape}")
        
        # Additional debug statistics
        with torch.no_grad():
            self.logger.info(f"Mu statistics - min: {mu.min():.4f}, max: {mu.max():.4f}, mean: {mu.mean():.4f}")
            self.logger.info(f"Rho statistics - min: {Rho.min():.4f}, max: {Rho.max():.4f}, mean: {Rho.mean():.4f}")
            eigvals = torch.linalg.eigvalsh(Sigma_inv_all)  # Get all eigenvalues at once
            self.logger.info(f"Sigma_inv eigenvalues - min: {eigvals.abs().min():.4f}, max: {eigvals.abs().max():.4f}")

        return Theta


    def evaluate(self, X: torch.Tensor, 
                 Theta: torch.nn.Parameter,
                 rho_flag=False, mu_flag=False) -> torch.Tensor:
        """
        Evaluates a set of Gaussian functions on a single batch of data points,
        packed as a tensor of dim=2  (N_samples, N_features)

        Computes exp(-0.5 || A'*(x-µ) ||_2^2) for all data points
        and all functions.        

        This method implements the core logic for an input of shape
        (n_samples, n_features) and returns a tensor of shape (n_samples, n_functions).
        For evaluation on irregular batches, use the inherited `__call__` method.

        Args:
            X (torch.Tensor): The input tensor of shape (n_samples, n_features).
            Theta (torch.nn.Parameter): The parameter tensor containing rho and mu.
            rho_flag (bool): If True, gradients for rho parameters will be enabled.
            mu_flag (bool): If True, gradients for mu parameters will be enabled.

        Returns:
            torch.Tensor: The output tensor of shape (n_samples, n_functions).
        """
        if X.dim() != 2 or X.shape[1] != self.n_features:
            raise ValueError(
                f"Input X must be a 2D tensor of shape (n_samples, n_features). "
                f"Got shape {X.shape} for n_features={self.n_features}."
            )
        
        n_samples = X.shape[0]

        # Split parameters
        num_rho_params_per_func = self.n_features * (self.n_features + 1) // 2
        rho = Theta[:num_rho_params_per_func, :]
        mu = Theta[-self.n_features:, :]

        # Handle gradient flow control
        if not rho_flag:
            rho = rho.detach()
        if not mu_flag:
            mu = mu.detach()

        # Reshape mu for broadcasting: (n_features, n_functions) -> (n_functions, n_features)
        mu_expanded = mu.T

        # Expand X and mu to align for element-wise subtraction and batching
        # X: (n_samples, n_features) -> (n_functions, n_samples, n_features)
        # mu: (n_functions, n_features) -> (n_functions, 1, n_features)
        x_expanded = X.unsqueeze(0).expand(self.n_functions, n_samples, self.n_features)
        mu_expanded = mu_expanded.unsqueeze(1).expand(self.n_functions, n_samples, self.n_features)

        # Compute x-µ for all functions and samples at once
        # Result: (n_functions, n_samples, n_features)
        x_mu = x_expanded - mu_expanded

        # Create batch of triangular matrices A directly
        n = self.n_features
        A = torch.zeros(self.n_functions, n, n, device=Theta.device)
        indices = torch.triu_indices(n, n, offset=0)
        # Following the pattern from deprecated_call_
        A[:, indices[1], indices[0]] = rho.T  # Use rho.T to match batch dimension: (n_functions, n_rho_params_per_func)

        # Compute A'*(x-µ) using bmm
        # Original logic was torch.bmm(x_mu, A).
        # x_mu: (n_functions, n_samples, n_features)
        # A: (n_functions, n_features, n_features) -> A is (B, M, P) and x_mu is (B, N, M)
        # torch.bmm(x_mu, A) computes (x_mu @ A_T) element-wise for the batch.
        # This is preserved.
        A_t_x_mu = torch.bmm(x_mu, A)
        
        # Compute squared L2 norm: ||A'*(x-µ)||_2^2
        # Sum over the last dimension (n_features)
        # Result: (n_functions, n_samples)
        squared_norms = torch.sum(A_t_x_mu ** 2, dim=-1)
        # Compute final exponential
        # Result: (N_batches, n_functions, n_samples_in_batch)
        result = torch.exp(-0.5 * squared_norms).T

        return result

