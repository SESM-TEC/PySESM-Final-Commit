import logging
import numpy as np
import torch

from pysesm.functions.ApproximateSurrogateFunction import ApproximateSurrogateFunction
from pysesm.utils.linalg import to_triu_matrix, generate_random_vectors, get_upper_triangle, gram_schmidt

#Meter un metodo para optimizar los rho y los mu
# Renamed GaussianFunctions -> GaussianApproximateSurrogateFunction
class GaussianApproximateSurrogateFunction(ApproximateSurrogateFunction):
    def __init__(self, n_features, n_functions, logger, eig_range, mu_range, vector_range,seed):
        super().__init__(n_features, n_functions, logger)
        self.eig_range = eig_range
        self.mu_range = mu_range
        self.seed = seed
        self.vector_range = vector_range
        self.theta_size = int(n_features*(n_features+3)/2)

    def initialize(self) -> torch.nn.Parameter:
        """
        Here comes the initialization
        In every interface there comes an initialization
        for i in n_functions:
            generate_random_vectors
            gram_schmidt
            Sigma_inv
            A
            For each gaussian, Rho: 3
            For each gaussian, Myu: 2
            Min will be a vector and max will be a vector
        """
        torch.manual_seed(self.seed)

        Theta = torch.nn.Parameter(torch.normal(mean=0, std=np.sqrt(1/self.theta_size), size=(self.theta_size, self.n_functions), requires_grad=True))

        mu = torch.rand(self.n_features, self.n_functions) * (self.mu_range[1] - self.mu_range[0]) + self.mu_range[0]

        Rho = torch.zeros(self.theta_size - self.n_features, self.n_functions)

        if self.eig_range[1] > 1e1 or self.vector_range[1] > 1e1:
            # Log the warning
            self.logger.setLevel(logging.WARNING)
            self.logger.warning("Recommended range is between 0 and 10")

        if self.mu_range[1] > 2 or self.mu_range[0] != -2:
            # Log the warning
            self.logger.setLevel(logging.WARNING)
            self.logger.warning("Recommended range is between [-1, -2] and [1, 2]")

        # Log the shape of the tensor
        self.logger.setLevel(logging.INFO)
        self.logger.info(f"Shape of Mu: {mu.shape}")
        self.logger.setLevel(logging.WARNING)

        for i in range(self.n_functions):
            Q = generate_random_vectors(self.n_features, self.vector_range[1], self.vector_range[0])
            Q = gram_schmidt(Q)
            D = torch.diag(torch.rand(self.n_features) * (self.eig_range[1] - self.eig_range[0]) + self.eig_range[0])
            Sigma = Q @ D @ Q.mT
            L = torch.linalg.cholesky(Sigma).mT
            rho = get_upper_triangle(L)
            for j in range(self.theta_size - self.n_features):
                Rho[j, i] = rho[j]

        with torch.no_grad():
            Theta[:-self.n_features, :] = Rho
            Theta[-self.n_features:, :] = mu

        # Log the shape of the tensor
        self.logger.setLevel(logging.INFO)
        self.logger.info(f"Shape of Rho: {Rho.shape}")
        self.logger.info(f"Shape of Theta: {Theta.shape}")
        self.logger.setLevel(logging.WARNING)

        return Theta

    def __call__(self, x, Theta: torch.nn.Parameter, rho_flag: bool = False, mu_flag: bool = False):
        # Toma los Rho del Theta que recibe
        rho = Theta[:-self.n_features, :]
        # Toma los Myu del Theta que recibe
        mu = Theta[-self.n_features:, :].mT.unsqueeze(2)
        # Detach the computational graph for the RHO parameters tensor
        if not rho_flag:
            rho = rho.detach()
        # Detach the computational graph for the MU parameters tensor
        elif not mu_flag:
            mu = mu.detach()
        # Toma los Rho y los representa como una matriz diagonal superior
        A = torch.stack([to_triu_matrix(rho[:, i]) for i in range(self.n_functions)], dim = 0)
        Sigma_inv = torch.matmul(A, A.mT)
        x_mu = x - mu
        exponent = -0.5 * torch.einsum('bij,bik,bji->jb', x_mu, Sigma_inv, x_mu.mT)
        result = torch.exp(exponent)

        return result
