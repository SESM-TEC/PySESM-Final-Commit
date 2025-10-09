import numpy as np
import torch
from pysesm.utils_dataset.gaussian_covariance_density import generate_gmm_z


def generate_mesh_samples(n_points, xl, xr, sigma, mu, weights=None, dtype=torch.float32):
    """
    Generates a 2D mesh grid and evaluates a combined density of
    multivariate normal distributions on it.

    Args:
    - n_points (int): Number of points in each dimension of the grid.
    - xl (float): The lower bound of the grid.
    - xr (float): The upper bound of the grid.
    - sigma (list of torch.Tensor): A list of covariance matrices for the distributions.
    - mu (list of torch.Tensor): A list of mean vectors for the distributions.
    - weights  for each gaussian

    Returns:
    - xx (np.ndarray): The x-coordinates of the mesh grid.
    - yy (np.ndarray): The y-coordinates of the mesh grid.
    - zz (torch.Tensor): The combined density values of the three distributions on the mesh grid.

    """
    x = np.linspace(xl, xr, n_points)
    xx, yy = np.meshgrid(x, x)
    X = torch.tensor(np.column_stack([xx.ravel(), yy.ravel()]), dtype=dtype)
    zz = generate_gmm_z(X, sigma, mu, weights,dtype=dtype)
    zz = zz.reshape(xx.shape)
    return xx, yy, zz


def generate_random_samples(n_points, xl, xr, sigma, mu, weights=None, dtype=torch.float32):
    """
    Generates random samples within a specified range and evaluates a
    combined density of several multivariate normal distributions on
    them.

    Args:

    - n_points (int): Number of random points to generate.
    - xl (float): The lower bound of the range for generating random
      points.
    - xr (float): The upper bound of the range for generating random
      points.
    - sigma (list of torch.Tensor): A list of covariance matrices for
      the distributions.
    - mu (list of torch.Tensor): A list of mean vectors for the
      distributions.
    - weights (tensor or None): weights for each distribution
    - generation to ensure reproducibility. Default is None (random).

    If no weights are provided, then all distribution samples are just
    added (as if all weights were 1).

    Returns:
    - xx (np.ndarray): The x-coordinates of the random samples.
    - yy (np.ndarray): The y-coordinates of the random samples.
    - zz (torch.Tensor): The combined density values of the three
      distributions at the random sample points.
    """       
    xx = np.random.uniform(xl, xr, n_points)
    yy = np.random.uniform(xl, xr, n_points)

    X = torch.tensor(np.column_stack([xx.ravel(), yy.ravel()]), dtype=dtype)
    zz = generate_gmm_z(X, sigma, mu, weights, dtype=dtype)

    return xx, yy, zz


def generate_mu(x_center, y_center,dtype=torch.float32):
    """
    Generates a mean vector for a multivariate normal distribution.

    Args:
    - x_center (float): The x-coordinate of the center.
    - y_center (float): The y-coordinate of the center.

    Returns:
    - torch.Tensor: A tensor containing the mean vector [x_center, y_center].
    """
    return torch.tensor([x_center, y_center], dtype=dtype)
