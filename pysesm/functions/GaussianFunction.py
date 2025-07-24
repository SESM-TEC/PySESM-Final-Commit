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

from pysesm.functions.SurrogateFunction import SurrogateFunction
from pysesm.utils.linalg import (
    to_triu_matrix,
    generate_random_vectors,
    get_upper_triangle,
    gram_schmidt,
)


class GaussianFunction(SurrogateFunction):
    """
    Compute the unnormalized Gaussians for a given set of points.
    Each Gaussian corresponds to a dictionary word, and is equivalent
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

    eig_range : list[float]
        A list of two values [min, max] specifying the range for the eigenvalues
        of the covariance matrix. These values represent variances, not precisions.
        Example: [0.5, 0.5] will create Gaussians with variance 0.5 in all directions
                [0.1, 1.0] will create Gaussians with variances randomly chosen 
                between 0.1 and 1.0 for each eigendirection
    """

    eig_range: list[float]
    mu_range: list[float]

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


    def __call__(self, x, Theta, rho_flag=False, mu_flag=False):
        """
        Computes exp(-0.5 || A'*(x-µ) ||_2^2) for all data points
        and all functions. Handles 2D (N_samples, N_features) or
        3D (N_batches, N_samples_per_batch, N_features) input x.
        """
        original_x_dim = x.dim()
        if original_x_dim == 2:
            # Reshape x from (N_samples, N_features) to (1, N_samples, N_features) for consistent batching
            x = x.unsqueeze(0)
        elif original_x_dim != 3:
            raise ValueError(f"Input x must be 2D (N_samples, N_features) or 3D (N_batches, N_samples_per_batch, N_features), got {x.shape}")

        n_batches, n_samples_in_batch, n_features_in_x = x.shape
        assert n_features_in_x == self.n_features, f"Input x features mismatch. Expected {self.n_features}, got {n_features_in_x}"

        # Split parameters
        num_rho_params_per_func = self.n_features * (self.n_features + 1) // 2
        rho = Theta[:num_rho_params_per_func, :]  # (n_rho_params_per_func, n_functions)
        mu_raw = Theta[-self.n_features:, :] # (n_features, n_functions)

        # Handle gradient flow control
        if not rho_flag:
            rho = rho.detach()
        if not mu_flag:
            mu_raw = mu_raw.detach()

        # Reshape mu for batching: (n_features, n_functions) -> (1, n_functions, 1, n_features)
        # So it broadcasts with x (N_batches, n_samples_in_batch, n_features)
        mu = mu_raw.T.unsqueeze(0).unsqueeze(2) # (1, n_functions, 1, n_features)

        # Expand x and mu to align for element-wise subtraction and batching
        # x: (N_batches, n_samples_in_batch, n_features) -> (N_batches, 1, n_samples_in_batch, n_features)
        # mu: (1, n_functions, 1, n_features)
        x_expanded = x.unsqueeze(1) # (N_batches, 1, n_samples_in_batch, n_features)

        # Compute x-µ for all functions and samples at once
        # Result: (N_batches, n_functions, n_samples_in_batch, n_features)
        x_mu = x_expanded - mu

        # Create batch of triangular matrices A directly
        n = self.n_features
        A = torch.zeros(self.n_functions, n, n, device=Theta.device)
        indices = torch.triu_indices(n, n, offset=0)
        # Following the pattern from deprecated_call_
        A[:, indices[1], indices[0]] = rho.T  # Use rho.T to match batch dimension: (n_functions, n_rho_params_per_func)

        # Reshape for batch matrix multiplication
        # We need to match the pattern from deprecated_call_ where bmm operates on:
        # x_mu: (n_functions, n_samples, n_features) 
        # A: (n_functions, n_features, n_features)

        # Current x_mu shape: (N_batches, n_functions, n_samples_in_batch, n_features)
        # We need to reshape to combine batches and functions for bmm
        # Then separate them back out

        # Reshape x_mu to (N_batches * n_functions, n_samples_in_batch, n_features)
        x_mu_reshaped = x_mu.reshape(n_batches * self.n_functions, n_samples_in_batch, self.n_features)

        # Expand A to match: (n_functions, n_features, n_features) -> (N_batches * n_functions, n_features, n_features)
        A_expanded = A.repeat(n_batches, 1, 1)

        # Compute A'*(x-µ) using bmm as in deprecated_call_
        A_t_x_mu = torch.bmm(x_mu_reshaped, A_expanded)

        # Reshape back to (N_batches, n_functions, n_samples_in_batch, n_features)
        A_t_x_mu = A_t_x_mu.reshape(n_batches, self.n_functions, n_samples_in_batch, self.n_features)

        # Compute squared L2 norm: ||A'*(x-µ)||_2^2
        # torch.sum(..., dim=-1) sums over the last dimension (n_features)
        # Result: (N_batches, n_functions, n_samples_in_batch)
        squared_norms = torch.sum(A_t_x_mu ** 2, dim=-1)

        # Compute final exponential
        # Result: (N_batches, n_functions, n_samples_in_batch)
        result = torch.exp(-0.5 * squared_norms)

        # Reshape to (N_total_points, N_functions) or (N_batches, N_samples_per_batch, N_functions)
        # based on original_x_dim for consistency with DictBaseLayer.forward expectations.
        if original_x_dim == 2:
            return result.permute(0, 2, 1).squeeze(0) # (1, N_samples, N_functions) -> (N_samples, N_functions)
        else: # original_x_dim == 3
            return result.permute(0, 2, 1) # (N_batches, N_samples_per_batch, N_functions)
    
    
    def old__call__(self, x, Theta, rho_flag=False, mu_flag=False):
        """
        DEPRECATED
        Computes exp(-0.5 || A'*(x-µ) ||_2^2) for all data points
        and all functions; this is minus one half of the square of the
        L2 norm of the triangular matrix A transposed times x minus
        the mean value.

        This is equivalent to the traditional Mahalanobis distance but
        much more efficient to compute.
        
        Args:
            x: Input tensor as traditional design matrix n_samples x n_features
            Theta: Parameter tensor containing rho and mu
            rho_flag: Whether rho parameters are being optimized
            mu_flag: Whether mu parameters are being optimized
        
        Returns:
            Tensor of shape (n_samples, n_functions) containing the evaluated gaussians

        """

        assert x.shape[1] == self.n_features, f"Input x should have shape [n_samples, n_features], got {x.shape}"

        # Split parameters
        rho = Theta[:-self.n_features, :]  # (n_rho_params, n_functions)

        # Starting with Theta[-self.n_features:, :] which is [n_features, n_functions]
        mu = Theta[-self.n_features:, :].T  # [n_functions, n_features]
        mu = mu.reshape(self.n_functions, 1, self.n_features)  # [n_functions, 1, n_features]
        
        # Handle gradient flow control
        if not rho_flag:
            rho = rho.detach()
        if not mu_flag:
            mu = mu.detach()
        
        # Convert x to shape (n_functions, n_features, n_samples) by repeating        
        x_expanded = x.unsqueeze(0)                     # Add functions dim -> [1, n_samples, n_features]
        x_expanded = x_expanded.repeat(self.n_functions, 1, 1)  # [n_functions, n_samples, n_features]

        # Compute x-µ for all functions at once
        x_mu = x_expanded - mu  # (n_functions, n_features, n_samples)
        
        # Create batch of triangular matrices A directly
        n = self.n_features
        A = torch.zeros(self.n_functions, n, n, device=Theta.device)
        indices = torch.triu_indices(n, n, offset=0)
        A[:, indices[1], indices[0]] = rho.T  # Use rho.T to match batch dimension
        
        # Compute A'*(x-µ) for all functions and samples at once
        A_t_x_mu = torch.bmm(x_mu,A)
        
        # Compute squared L2 norm: ||A'*(x-µ)||_2^2
        squared_norms = torch.sum(A_t_x_mu ** 2, dim=2)  # (n_functions, n_samples)
        
        # Compute final exponential
        result = torch.exp(-0.5 * squared_norms).T
        
        # Debug statistics
        #with torch.no_grad():
        #    self.logger.debug(f"Squared norms - min: {squared_norms.min():.4f}, max: {squared_norms.max():.4f}")
        #    self.logger.debug(f"Result - min: {result.min():.4f}, max: {result.max():.4f}")
        
        return result


