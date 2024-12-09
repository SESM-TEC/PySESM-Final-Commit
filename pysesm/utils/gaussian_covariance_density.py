# Opcion covariance_matrices.py -> generate_sigma_tensors()
# Opcion density_generation.py -> generate_z() y generate_mu()


import numpy as np
import torch
from scipy.stats import multivariate_normal


# Non-diagonal covariance
def generate_sigma_tensors():
    """
    Generates non-diagonal covariance tensors for 2D Gaussian distributions.

    Returns:
    tuple: Tuple containing three non-diagonal covariance tensors (sigma1, sigma2, sigma3).
    """
    e0 = torch.tensor([1.0, 1.0], dtype=torch.float32)
    e0 = e0 / e0.norm()

    def generate_sigma(rotation_angle, scaling_factors):
        rotation_matrix = torch.tensor(
            [
                [np.cos(rotation_angle), -np.sin(rotation_angle)],
                [np.sin(rotation_angle), np.cos(rotation_angle)],
            ],
            dtype=torch.float32,
        )
        e = torch.mm(rotation_matrix, e0.unsqueeze(1))
        E = torch.cat((e0.unsqueeze(1), e), dim=1)
        D = torch.diag(torch.tensor(scaling_factors, dtype=torch.float32))
        return torch.mm(torch.mm(E, D), E.t())

    sigma1 = generate_sigma(np.pi / 4, [0.4, 0.1])
    sigma2 = generate_sigma(np.pi / 4, [0.05, 0.5])
    sigma3 = generate_sigma(np.pi / 4, [0.2, 0.5])

    return sigma1, sigma2, sigma3


def generate_z(X, sigma, mu):
    """
    Generates a combined density of three multivariate normal distributions evaluated at points X.

    Args:
    - X (torch.Tensor): A tensor containing points where the distributions are evaluated.
    - sigma (list of torch.Tensor): A list of covariance matrices for the distributions.
    - mu (list of torch.Tensor): A list of mean vectors for the distributions.

    Returns:
    - torch.Tensor: The combined density values of the three distributions at points X.
    """
    z1 = torch.tensor(
        multivariate_normal.pdf(X.numpy(), mu[0].numpy(), sigma[0].numpy()),
        dtype=torch.float32,
    )
    z2 = torch.tensor(
        multivariate_normal.pdf(X.numpy(), mu[1].numpy(), sigma[1].numpy()),
        dtype=torch.float32,
    )
    z3 = torch.tensor(
        multivariate_normal.pdf(X.numpy(), mu[2].numpy(), sigma[2].numpy()),
        dtype=torch.float32,
    )

    return z1 + z2 + z3


def generate_mesh(n_points, xl, xr, sigma, mu):
    """
    Generates a 2D mesh grid and evaluates a combined density of three multivariate normal distributions on it.

    Args:
    - n_points (int): Number of points in each dimension of the grid.
    - xl (float): The lower bound of the grid.
    - xr (float): The upper bound of the grid.
    - sigma (list of torch.Tensor): A list of covariance matrices for the distributions.
    - mu (list of torch.Tensor): A list of mean vectors for the distributions.

    Returns:
    - xx (np.ndarray): The x-coordinates of the mesh grid.
    - yy (np.ndarray): The y-coordinates of the mesh grid.
    - zz (torch.Tensor): The combined density values of the three distributions on the mesh grid.
    """
    x = np.linspace(xl, xr, n_points)
    xx, yy = np.meshgrid(x, x)
    X = torch.tensor(np.column_stack([xx.ravel(), yy.ravel()]), dtype=torch.float32)
    zz = generate_z(X, sigma, mu)
    zz = zz.reshape(xx.shape)
    return xx, yy, zz
