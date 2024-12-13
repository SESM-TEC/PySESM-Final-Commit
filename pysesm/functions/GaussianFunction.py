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
    TODO: I am not sure about the eig, mu and vector range
    """

    eig_range: list[float]
    mu_range: list[float]
    vector_range: list[float]

    def __init__(self, n_features, n_functions, seed, logger, **kwargs):
        super().__init__(
            n_features=n_features, n_functions=n_functions, seed=seed, logger=logger, **kwargs
        )
        self.theta_size = int(n_features * (n_features + 3) / 2)

    def initialize(self) -> torch.nn.Parameter:
        torch.manual_seed(self.seed)

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

        Theta = torch.nn.Parameter(
            torch.normal(
                mean=0,
                std=np.sqrt(1 / self.theta_size),
                size=(self.theta_size, self.n_functions),
                requires_grad=True,
            )
        )

        mu = (
            torch.rand(self.n_features, self.n_functions)
            * (self.mu_range[1] - self.mu_range[0])
            + self.mu_range[0]
        )

        Rho = torch.zeros(self.theta_size - self.n_features, self.n_functions)

        # Following range changes will be eliminated:  they depend on the data set!
        # if self.eig_range[1] > 1e1 or self.vector_range[1] > 1e1:
        #    # Log the warning
        #    self.logger.warning("Recommended eig range is between 0 and 10")

        #if self.mu_range[1] > 2 or self.mu_range[0] != -2:
        #    # Log the warning
        #    self.logger.warning("Recommended mu range is between [-1, -2] and [1, 2]")

        # Log the shape of the tensor
        self.logger.info(f"Shape of Mu: {mu.shape}")

        for i in range(self.n_functions):
            Q = generate_random_vectors(
                self.n_features, self.vector_range[1], self.vector_range[0]
            )
            Q = gram_schmidt(Q)
            D = torch.diag(
                torch.rand(self.n_features) * (self.eig_range[1] - self.eig_range[0])
                + self.eig_range[0]
            )
            Sigma = Q @ D @ Q.mT
            L = torch.linalg.cholesky(Sigma).mT
            rho = get_upper_triangle(L)
            for j in range(self.theta_size - self.n_features):
                Rho[j, i] = rho[j]

        with torch.no_grad():
            Theta[: -self.n_features, :] = Rho
            Theta[-self.n_features :, :] = mu

        # Log the shape of the tensor
        self.logger.info(f"Shape of Rho: {Rho.shape}")
        self.logger.info(f"Shape of Theta: {Theta.shape}")

        return Theta

    def __call__(
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
