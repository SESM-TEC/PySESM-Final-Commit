import pytest
import torch
import numpy as np
from pysesm.utils.gaussian_covariance_density import generate_gmm_z
from pysesm.utils.mesh_generation import *


def test_generate_mesh():
    """
    Test the generate_mesh function for correct mesh generation and density evaluation.
    """
    n_points = 100
    xl, xr = -3.0, 3.0

    # Mock covariance matrices and mean vectors
    sigma1 = torch.tensor([[0.5, 0.2], [0.2, 0.5]], dtype=torch.float32)
    sigma2 = torch.tensor([[0.3, 0.1], [0.1, 0.3]], dtype=torch.float32)
    sigma3 = torch.tensor([[0.4, 0.0], [0.0, 0.4]], dtype=torch.float32)
    sigma = [sigma1, sigma2, sigma3]

    mu1 = torch.tensor([0.0, 0.0], dtype=torch.float32)
    mu2 = torch.tensor([1.0, 1.0], dtype=torch.float32)
    mu3 = torch.tensor([-1.0, -1.0], dtype=torch.float32)
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

    # Test edge cases with small and large grid sizes
    xx_small, yy_small, zz_small = generate_mesh(10, -1.0, 1.0, sigma, mu)
    assert xx_small.shape == (10, 10), "Small mesh xx shape is incorrect"
    assert yy_small.shape == (10, 10), "Small mesh yy shape is incorrect"
    assert zz_small.shape == (10, 10), "Small mesh zz shape is incorrect"

    xx_large, yy_large, zz_large = generate_mesh(1000, -10.0, 10.0, sigma, mu)
    assert xx_large.shape == (1000, 1000), "Large mesh xx shape is incorrect"
    assert yy_large.shape == (1000, 1000), "Large mesh yy shape is incorrect"
    assert zz_large.shape == (1000, 1000), "Large mesh zz shape is incorrect"


def test_generate_random_samples():
    """
    Test the generate_random_samples function for correct random sample generation and density evaluation.
    """
    n_points = 100
    xl, xr = -3.0, 3.0
    SEED = 42

    # Mock covariance matrices and mean vectors
    sigma1 = torch.tensor([[0.5, 0.2], [0.2, 0.5]], dtype=torch.float32)
    sigma2 = torch.tensor([[0.3, 0.1], [0.1, 0.3]], dtype=torch.float32)
    sigma3 = torch.tensor([[0.4, 0.0], [0.0, 0.4]], dtype=torch.float32)
    sigma = [sigma1, sigma2, sigma3]

    mu1 = torch.tensor([0.0, 0.0], dtype=torch.float32)
    mu2 = torch.tensor([1.0, 1.0], dtype=torch.float32)
    mu3 = torch.tensor([-1.0, -1.0], dtype=torch.float32)
    mu = [mu1, mu2, mu3]

    xx, yy, zz = generate_random_samples(n_points, xl, xr, sigma, mu, SEED)

    # Check the length of random samples
    assert len(xx) == n_points, "xx length is incorrect"
    assert len(yy) == n_points, "yy length is incorrect"
    assert zz.shape == (n_points,), "zz shape is incorrect"

    # Check random sample bounds
    assert xx.min() >= xl and xx.max() <= xr, "xx is out of bounds"
    assert yy.min() >= xl and yy.max() <= xr, "yy is out of bounds"

    # Ensure reproducibility with the seed
    np.random.seed(SEED)
    expected_xx = np.random.uniform(xl, xr, n_points)
    expected_yy = np.random.uniform(xl, xr, n_points)

    assert np.allclose(
        xx, expected_xx, atol=1e-5
    ), "Random samples xx are not reproducible"
    assert np.allclose(
        yy, expected_yy, atol=1e-5
    ), "Random samples yy are not reproducible"

    # Check specific density values
    X = torch.tensor(np.column_stack([xx.ravel(), yy.ravel()]), dtype=torch.float32)
    zz_expected = generate_gmm_z(X, sigma, mu)

    assert torch.allclose(
        zz, zz_expected, atol=1e-5
    ), "Density values for random samples are incorrect"

    # Test edge cases with different seeds
    xx1, yy1, zz1 = generate_random_samples(n_points, xl, xr, sigma, mu, SEED + 1)
    xx2, yy2, zz2 = generate_random_samples(n_points, xl, xr, sigma, mu, SEED + 2)

    # Check if different seeds produce different samples
    assert not np.allclose(xx1, xx2), "Different seeds produce identical xx samples"
    assert not np.allclose(yy1, yy2), "Different seeds produce identical yy samples"
    assert not torch.allclose(
        zz1, zz2
    ), "Different seeds produce identical zz density values"


print("Todo bien")
