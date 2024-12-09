import pytest
import torch
import numpy as np
from pysesm.utils.sampling_data import *


def test_generate_uniform_sampling():
    """
    Test the generate_uniform_sampling function for correct sample generation.
    """
    total_points = 1000
    n_samples = 500
    SEED = 42

    sampled_indices = generate_uniform_sampling(total_points, SEED, n_samples)

    # Check the number of samples
    assert len(sampled_indices) == n_samples, "Number of samples is incorrect"

    # Ensure reproducibility with the seed
    np.random.seed(SEED)
    expected_indices = np.random.permutation(total_points)[:n_samples]
    assert np.array_equal(
        sampled_indices, expected_indices
    ), "Sampled indices are not reproducible"

    # Ensure the indices are unique
    assert len(set(sampled_indices)) == n_samples, "Sampled indices are not unique"

    # Ensure indices are within valid range
    assert all(
        0 <= idx < total_points for idx in sampled_indices
    ), "Sampled indices are out of range"

    # Test edge cases
    sampled_indices_edge = generate_uniform_sampling(total_points, SEED, total_points)
    assert (
        len(sampled_indices_edge) == total_points
    ), "Edge case failed for sampling all points"

    # Check with min_separation (to be implemented)
    sampled_indices_separated = generate_uniform_sampling(
        total_points, SEED, n_samples, min_separation=2
    )
    assert (
        len(sampled_indices_separated) == n_samples
    ), "Number of samples with separation is incorrect"

    # Verify minimum separation constraint
    assert all(
        np.abs(np.diff(np.sort(sampled_indices_separated))) >= 2
    ), "Minimum separation constraint is violated"


def test_sample_data():
    """
    Test the sample_data function for correct data sampling.
    """
    x_values = np.arange(1000)
    y_values = np.arange(1000, 2000)
    z_values = np.arange(2000, 3000)
    sampled_indices = [0, 10, 50, 100, 150]

    X, y = sample_data(x_values, y_values, z_values, sampled_indices)

    # Check the shape of the returned samples
    assert X.shape == (len(sampled_indices), 2), "Sampled X shape is incorrect"
    assert y.shape == (len(sampled_indices),), "Sampled y shape is incorrect"

    # Check the sampled data
    expected_X = torch.tensor(
        [[0, 1000], [10, 1010], [50, 1050], [100, 1100], [150, 1150]],
        dtype=torch.float32,
    )
    expected_y = torch.tensor([2000, 2010, 2050, 2100, 2150], dtype=torch.float32)

    assert torch.allclose(X, expected_X, atol=1e-5), "Sampled X data is incorrect"
    assert torch.allclose(y, expected_y, atol=1e-5), "Sampled y data is incorrect"

    # Test with different indices
    sampled_indices_diff = [999, 500, 250, 750]
    X_diff, y_diff = sample_data(x_values, y_values, z_values, sampled_indices_diff)

    expected_X_diff = torch.tensor(
        [[999, 1999], [500, 1500], [250, 1250], [750, 1750]], dtype=torch.float32
    )
    expected_y_diff = torch.tensor([2999, 2500, 2250, 2750], dtype=torch.float32)

    assert torch.allclose(
        X_diff, expected_X_diff, atol=1e-5
    ), "Sampled X data with different indices is incorrect"
    assert torch.allclose(
        y_diff, expected_y_diff, atol=1e-5
    ), "Sampled y data with different indices is incorrect"


print("Todo bien")
