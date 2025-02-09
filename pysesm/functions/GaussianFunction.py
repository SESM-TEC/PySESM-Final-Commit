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
    This is a mixture of gaussian functions.

    Theta holds all parameters for all words in a dictionary.
    It has two subsets of parameters:
    mu: mean values for each gaussian
    rho: elements of the triangular matrices.
    
    The inverse of the convariance matrix is computed as A*A' where 
    A is a triangular matrix with the elements in rho.
    
    """

    eig_range: list[float]
    mu_range: list[float]

    def __init__(self, n_features, n_functions, seed, logger, **kwargs):
        super().__init__(
            n_features=n_features, n_functions=n_functions, seed=seed, logger=logger, **kwargs
        )
        self.theta_size = int(n_features * (n_features + 3) / 2)

    def initialize(self) -> torch.nn.Parameter:
        """
        Initialize the parameter vector Theta maintaining specific
        statistical properties:
        - Means (mu) uniformly distributed in mu_range
        - Covariance matrices with controlled eigenvalue distribution
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # Validate and process mu_range
        mu_range = torch.tensor(self.mu_range)
        if mu_range.ndim == 1:
            if len(mu_range) != 2:
                raise ValueError(
                    f"If providing a single range, mu_range must be [from, to], got {mu_range}"
                )
            # Expand single range to all dimensions
            mu_range = mu_range.expand(self.n_features, 2)
        elif mu_range.shape != (self.n_features, 2):
            raise ValueError(
                f"mu_range must be either [from, to] or have shape ({self.n_features}, 2) "
                f"for per-dimension ranges, got shape {mu_range.shape}"
            )
        
        # Initialize means using per-dimension ranges
        mu = torch.zeros(self.n_features, self.n_functions)
        for i in range(self.n_features):
            mu[i] = torch.rand(self.n_functions) * \
                    (mu_range[i, 1] - mu_range[i, 0]) + mu_range[i, 0]

        # Generate completely random matrices and orthogonalize them
        Q_all = torch.rand(self.n_functions, self.n_features, self.n_features)
        Q_all, _ = torch.linalg.qr(Q_all)

        # Generate eigenvalues in specified range
        D_all = torch.rand(self.n_functions, self.n_features) * \
                (self.eig_range[1] - self.eig_range[0]) + self.eig_range[0]
        D_all = torch.diag_embed(D_all)

        # Batch compute all covariance matrices
        Sigma_all = torch.bmm(torch.bmm(Q_all, D_all), Q_all.transpose(1, 2))
        
        # Batch Cholesky decomposition
        L_all = torch.linalg.cholesky(Sigma_all).transpose(1, 2)

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
            eigvals = torch.linalg.eigvalsh(Sigma_all)  # Get all eigenvalues at once
            self.logger.info(f"Sigma_inv eigenvalues - min: {eigvals.abs().min():.4f}, max: {eigvals.abs().max():.4f}")

        return Theta

    def __call__(self, x, Theta, rho_flag=False, mu_flag=False):
        """
        Computes exp(-0.5 || A'*(x-µ) ||_2^2) for all data points and all functions.
        
        Args:
            x: Input tensor (n_features, n_samples)
            Theta: Parameter tensor containing rho and mu
            rho_flag: Whether rho parameters are being optimized
            mu_flag: Whether mu parameters are being optimized
        
        Returns:
            Tensor of shape (n_samples, n_functions) containing the evaluated gaussians
        """
        # Split parameters
        rho = Theta[:-self.n_features, :]  # (n_rho_params, n_functions)
        mu = Theta[-self.n_features:, :].mT.unsqueeze(2)  # (n_functions, n_features, 1)
        
        # Handle gradient flow control
        if not rho_flag:
            rho = rho.detach()
        if not mu_flag:
            mu = mu.detach()

        # Debug statistics
        #with torch.no_grad():
        #    self.logger.debug(f"Rho stats - min: {rho.min():.4f}, max: {rho.max():.4f}, mean: {rho.mean():.4f}")
        #    self.logger.debug(f"Mu stats - min: {mu.min():.4f}, max: {mu.max():.4f}, mean: {mu.mean():.4f}")
        
        # Convert x to shape (n_functions, n_features, n_samples) by repeating
        x_expanded = x.unsqueeze(0).expand(self.n_functions, -1, -1)  # (n_functions, n_features, n_samples)
        
        # Compute x-µ for all functions at once
        x_mu = x_expanded - mu  # (n_functions, n_features, n_samples)
        
        # Create batch of triangular matrices A directly
        n = self.n_features
        A = torch.zeros(self.n_functions, n, n, device=Theta.device)
        indices = torch.triu_indices(n, n, offset=0)
        A[:, indices[0], indices[1]] = rho.T  # Use rho.T to match batch dimension
        
        # Compute A'*(x-µ) for all functions and samples at once
        A_t_x_mu = torch.bmm(A.transpose(1, 2), x_mu)
        
        # Compute squared L2 norm: ||A'*(x-µ)||_2^2
        squared_norms = torch.sum(A_t_x_mu ** 2, dim=1)  # (n_functions, n_samples)
        
        # Compute final exponential
        result = torch.exp(-0.5 * squared_norms).T
        
        # Debug statistics
        #with torch.no_grad():
        #    self.logger.debug(f"Squared norms - min: {squared_norms.min():.4f}, max: {squared_norms.max():.4f}")
        #    self.logger.debug(f"Result - min: {result.min():.4f}, max: {result.max():.4f}")
        
        return result


    # Deprecated function
    def __old_call__(
        self,
        x,
        Theta: torch.nn.Parameter,
        rho_flag: bool = False,
        mu_flag: bool = False,
    ) -> torch.Tensor:
        # Takes the RHO values from the theta vector
        rho = Theta[: -self.n_features, :]
        # Takes the MU values from the theta vector
        mu = Theta[-self.n_features :, :].mT.unsqueeze(2)
        # Detach the computational graph for the RHO parameters tensor
        if not rho_flag:
            rho = rho.detach()
        # Detach the computational graph for the MU parameters tensor
        if not mu_flag:
            mu = mu.detach()
        # Takes the RHO values and packs them as a diagonal matrix of higher order
        A = torch.stack(
            [to_triu_matrix(rho[:, i]) for i in range(self.n_functions)], dim=0
        )
        Sigma_inv = torch.matmul(A, A.mT)
        x_mu = x - mu
        exponent = -0.5 * torch.einsum("bij,bik,bji->jb", x_mu, Sigma_inv, x_mu.mT)
        result = torch.exp(exponent)

        return result

