import numpy as np
import torch
from pysesm.utils.gaussian_covariance_density import generate_z

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

def generate_random_samples(n_points, xl, xr, sigma, mu, SEED):
  """
  Generates random samples within a specified range and evaluates a combined density of three multivariate normal distributions on them.

  Args:
  - n_points (int): Number of random points to generate.
  - xl (float): The lower bound of the range for generating random points.
  - xr (float): The upper bound of the range for generating random points.
  - sigma (list of torch.Tensor): A list of covariance matrices for the distributions.
  - mu (list of torch.Tensor): A list of mean vectors for the distributions.
  - SEED (int, optional): Seed for random number generation to ensure reproducibility. Default is 42.

  Returns:
  - xx (np.ndarray): The x-coordinates of the random samples.
  - yy (np.ndarray): The y-coordinates of the random samples.
  - zz (torch.Tensor): The combined density values of the three distributions at the random sample points.
  """
  np.random.seed(SEED)
  xx = np.random.uniform(xl, xr, n_points)
  yy = np.random.uniform(xl, xr, n_points)

  X = torch.tensor(np.column_stack([xx.ravel(), yy.ravel()]), dtype=torch.float32)
  zz = generate_z(X, sigma, mu)

  return xx, yy, zz