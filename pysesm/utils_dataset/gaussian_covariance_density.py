# Opcion covariance_matrices.py -> generate_sigma_tensors()
# Opcion density_generation.py -> generate_z() y generate_mu()


import numpy as np
import torch
from scipy.stats import multivariate_normal


# Non-diagonal covariance
def generate_nondiag_covariance_matrices(dtype=torch.float32):
    """
    Generates three non-diagonal covariance tensors for 2D
    Gaussian distributions.

    Returns: 

    tuple: Tuple containing three non-diagonal covariance
           tensors (sigma1, sigma2, sigma3).
    """

    def generate_sigma(rotation_angle, scaling_factors):
        E = torch.tensor(
            [
                [np.cos(rotation_angle), -np.sin(rotation_angle)],
                [np.sin(rotation_angle), np.cos(rotation_angle)],
            ],
            dtype=dtype,
        )
        D = torch.diag(torch.tensor(scaling_factors, dtype=dtype))
        return torch.mm(torch.mm(E, D), E.t())

    sigma1 = generate_sigma(np.pi / 4, [0.4, 0.1])
    sigma2 = generate_sigma(np.pi / 3, [0.05, 0.5])
    sigma3 = generate_sigma(np.pi / 6, [0.2, 0.5])

    return sigma1, sigma2, sigma3


def generate_gmm_z(X, sigma, mu, weights=None, dtype=torch.float32):
    """
    Generates a combined density of several multivariate normal
    distributions evaluated at points X.

    Args:

    - X (torch.Tensor): A tensor containing points in rows, where the
      distributions are evaluated.
    - sigma (list of torch.Tensor): A list of covariance matrices for
      the distributions.   
    - mu (list of torch.Tensor): A list of mean vectors for the
      distributions.

    Returns:
    
    - torch.Tensor: The combined density values of all distributions
      at points X.

    """
    if len(sigma) != len(mu):
        raise ValueError(f"Number of means ({len(mu)}) must match "
                         f"number of covariances ({len(sigma)})")
    
    # Set default weights if None
    if weights is None:
        weights = [1.0] * len(mu)

    if len(weights) != len(mu):
        raise ValueError(f"Number of weights ({len(weights)}) must match"
                         f"number of covariances ({len(sigma)})")

        
    # Convert X to numpy once since we'll reuse it
    X_np = X.numpy()
    
    # Initialize result with first gaussian to avoid one addition
    max_val = multivariate_normal.pdf(mu[0].numpy(), # eval at the meant
                                      mean=mu[0].numpy(),
                                      cov=sigma[0].numpy())
    result = torch.tensor(multivariate_normal.pdf(X_np,
                                                  mu[0].numpy(),
                                                  sigma[0].numpy()),
                          dtype=dtype)*(weights[0]/max_val)
    
    # Add remaining gaussians
    for i, (mean, cov) in enumerate(zip(mu[1:], sigma[1:]),start=1):
        max_val = multivariate_normal.pdf(mean.numpy(),
                                          mean=mean.numpy(),
                                          cov=cov.numpy())
        result.add_(torch.tensor(multivariate_normal.pdf(X_np,
                                                         mean=mean.numpy(),
                                                         cov=cov.numpy()),
                                 dtype=dtype)*(weights[i]/max_val))
    
    return result


def generate_mesh(n_points, xl, xr, sigma, mu, dtype=torch.float32):
    """
    Generates a 2D regular mesh grid and evaluates a combined
    density of three multivariate normal distributions on it.

    Args:
    - n_points (int): Number of points in each dimension of the grid.
    - xl (float): The lower bound of the grid.
    - xr (float): The upper bound of the grid.
    - sigma (list of torch.Tensor): A list of covariance matrices for
      the distributions.
    - mu (list of torch.Tensor): A list of mean vectors for the distributions.

    Returns:
    - xx (np.ndarray): The x-coordinates of the mesh grid.
    - yy (np.ndarray): The y-coordinates of the mesh grid.

    - zz (torch.Tensor): The combined density values of the three
      distributions on the mesh grid.
    """
    x = np.linspace(xl, xr, n_points)
    xx, yy = np.meshgrid(x, x)
    X = torch.tensor(np.column_stack([xx.ravel(), yy.ravel()]),
                     dtype=dtype)
    zz = generate_gmm_z(X, sigma, mu, dtype=dtype
                        )
    zz = zz.reshape(xx.shape)
    return xx, yy, zz
