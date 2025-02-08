import pytest
import torch
import numpy as np
from scipy.stats import multivariate_normal
from pysesm.utils.gaussian_covariance_density import *


def test_generate_nondiag_covariance_matrices():
    """
    Test the generate_nondiag_covariance_matrices.
    """
    sigma1, sigma2, sigma3 = generate_nondiag_covariance_matrices()

    # Check that three tensors are returned
    assert isinstance(sigma1, torch.Tensor), "sigma1 is not a tensor"
    assert isinstance(sigma2, torch.Tensor), "sigma2 is not a tensor"
    assert isinstance(sigma3, torch.Tensor), "sigma3 is not a tensor"

    # Check tensor shapes
    assert sigma1.shape == (2, 2), "sigma1 shape is not 2x2"
    assert sigma2.shape == (2, 2), "sigma2 shape is not 2x2"
    assert sigma3.shape == (2, 2), "sigma3 shape is not 2x2"

    # Check if matrices are symmetric
    assert torch.allclose(sigma1, sigma1.T, atol=1e-5), "sigma1 is not symmetric"
    assert torch.allclose(sigma2, sigma2.T, atol=1e-5), "sigma2 is not symmetric"
    assert torch.allclose(sigma3, sigma3.T, atol=1e-5), "sigma3 is not symmetric"

    # Check if matrices are positive definite
    def is_positive_definite(matrix):
        # Get eigenvalues using the modern API
        eigenvalues = torch.linalg.eigvalsh(matrix)
        # Check real parts are all positive
        return torch.all(eigenvalues.real > 0)

    assert is_positive_definite(sigma1), "sigma1 is not positive definite"
    assert is_positive_definite(sigma2), "sigma2 is not positive definite"
    assert is_positive_definite(sigma3), "sigma3 is not positive definite"


def test_generate_z():
    """
    Test the generate_z function for correct density calculation.
    """
    sigma1, sigma2, sigma3 = generate_sigma_tensors()

    # Define mean vectors
    mu1 = torch.tensor([0.0, 0.0], dtype=torch.float32)
    mu2 = torch.tensor([1.0, 1.0], dtype=torch.float32)
    mu3 = torch.tensor([-1.0, -1.0], dtype=torch.float32)

    X = torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]], dtype=torch.float32)
    sigma = [sigma1, sigma2, sigma3]
    mu = [mu1, mu2, mu3]

    z = generate_gmm_z(X, sigma, mu)

    # Check the shape of the result
    assert z.shape == (3,), "Output shape is incorrect"

    # Manually calculate expected density
    expected_z1 = multivariate_normal.pdf(X.numpy(), mu1.numpy(), sigma1.numpy())
    expected_z2 = multivariate_normal.pdf(X.numpy(), mu2.numpy(), sigma2.numpy())
    expected_z3 = multivariate_normal.pdf(X.numpy(), mu3.numpy(), sigma3.numpy())

    expected_z = expected_z1 + expected_z2 + expected_z3

    assert np.allclose(z.numpy(), expected_z, atol=1e-5), "Density values are incorrect"


def test_generate_mesh():
    """
    Test the generate_mesh function for correct mesh generation and density evaluation.
    """
    n_points = 100
    xl, xr = -3.0, 3.0
    sigma1, sigma2, sigma3 = generate_sigma_tensors()

    mu1 = torch.tensor([0.0, 0.0], dtype=torch.float32)
    mu2 = torch.tensor([1.0, 1.0], dtype=torch.float32)
    mu3 = torch.tensor([-1.0, -1.0], dtype=torch.float32)

    sigma = [sigma1, sigma2, sigma3]
    mu = [mu1, mu2, mu3]

    xx, yy, zz = generate_mesh(n_points, xl, xr, sigma, mu)

    # Check the dimensions of the mesh grid
    assert xx.shape == (n_points, n_points), "xx shape is incorrect"
    assert yy.shape == (n_points, n_points), "yy shape is incorrect"
    assert zz.shape == (n_points, n_points), "zz shape is incorrect"

    # Check mesh grid bounds
    assert xx.min() == pytest.approx(xl, abs=1e-5), "xx min is incorrect"
    assert xx.max() == pytest.approx(xr, abs=1e-5), "xx max is incorrect"
    assert yy.min() == pytest.approx(xl, abs=1e-5), "yy min is incorrect"
    assert yy.max() == pytest.approx(xr, abs=1e-5), "yy max is incorrect"

    # Check some specific density values
    X = torch.tensor(np.column_stack([xx.ravel(), yy.ravel()]), dtype=torch.float32)
    zz_expected = generate_gmm_z(X, sigma, mu).reshape(xx.shape)

    assert torch.allclose(
        zz, zz_expected, atol=1e-5
    ), "Density values on mesh are incorrect"

    # Test edge cases
    xx_small, yy_small, zz_small = generate_mesh(10, -1.0, 1.0, sigma, mu)
    assert xx_small.shape == (10, 10), "Small mesh xx shape is incorrect"
    assert yy_small.shape == (10, 10), "Small mesh yy shape is incorrect"
    assert zz_small.shape == (10, 10), "Small mesh zz shape is incorrect"

    # Test large grid
    xx_large, yy_large, zz_large = generate_mesh(1000, -10.0, 10.0, sigma, mu)
    assert xx_large.shape == (1000, 1000), "Large mesh xx shape is incorrect"
    assert yy_large.shape == (1000, 1000), "Large mesh yy shape is incorrect"
    assert zz_large.shape == (1000, 1000), "Large mesh zz shape is incorrect"

if __name__ == "__main__":
    from pytest_helper import print_pytest_instructions
    print_pytest_instructions()
