import numpy as np
import torch
import pytest
from pysesm.utils.design_matrices import *


def test_unstack_design_matrix():
    """
    Test the unstack_design_matrix function to ensure correct separation of columns.
    """
    # Test case: Simple 2-column matrix
    X_test = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
    expected_x = torch.tensor([1, 3, 5], dtype=torch.float32)
    expected_y = torch.tensor([2, 4, 6], dtype=torch.float32)

    x_tensor, y_tensor = unstack_design_matrix(X_test)
    assert torch.equal(x_tensor, expected_x), "Unstacking x-component failed"
    assert torch.equal(y_tensor, expected_y), "Unstacking y-component failed"

    # Test edge case: Empty matrix
    X_test_empty = torch.empty((0, 2), dtype=torch.float32)
    expected_x_empty = torch.tensor([], dtype=torch.float32)
    expected_y_empty = torch.tensor([], dtype=torch.float32)

    x_tensor_empty, y_tensor_empty = unstack_design_matrix(X_test_empty)
    assert torch.equal(
        x_tensor_empty, expected_x_empty
    ), "Unstacking empty matrix x-component failed"
    assert torch.equal(
        y_tensor_empty, expected_y_empty
    ), "Unstacking empty matrix y-component failed"


def test_create_design_matrix_train():
    """
    Test the create_design_matrix_train function to ensure correct
    sampling and matrix creation.
    """
    # Mock input data
    xx = np.array([[1, 2, 3], [4, 5, 6]])
    yy = np.array([[7, 8, 9], [10, 11, 12]])
    zz = np.array([[13, 14, 15], [16, 17, 18]])

    hyperparams = {"n_samples": 4, "T": 1, "seed": None}

    # Generate design matrix with shuffled data points
    X, y = create_design_matrix_train(xx, yy, zz, hyperparams)

    # Ensure correct shape of output
    assert X.shape == (4, 2), "Design matrix X has incorrect shape"
    assert y.shape == (4,), "Target variable y has incorrect shape"

    # Ensure sampled data lies within the input range
    assert np.all(np.isin(X[:, 0], xx.ravel())), "Sampled x-values out of input range"
    assert np.all(np.isin(X[:, 1], yy.ravel())), "Sampled y-values out of input range"
    assert np.all(np.isin(y, zz.ravel())), "Sampled z-values out of input range"

    # Test edge case: n_samples greater than total points
    hyperparams["n_samples"] = 10
    with pytest.raises(ValueError,match="Cannot sample"):
        X, y = create_design_matrix_train(xx, yy, zz, hyperparams)


def test_create_design_matrix_test():
    """
    Test the create_design_matrix_test function to ensure correct creation of the test design matrix.
    """
    # Mock input data
    xx = np.array([[1, 2], [3, 4]])
    yy = np.array([[5, 6], [7, 8]])
    zz = np.array([[9, 10], [11, 12]])

    # Generate test design matrix
    X_test, z_values = create_design_matrix_test(xx, yy, zz)

    # Ensure correct shape of output
    assert X_test.shape == (4, 2), "Test design matrix X_test has incorrect shape"
    assert z_values.shape == (4,), "Target values z_values has incorrect shape"

    # Ensure all values are correctly flattened and paired
    expected_X_test = torch.tensor(
        [[1, 5], [2, 6], [3, 7], [4, 8]], dtype=torch.float32
    )
    expected_z_values = np.array([9, 10, 11, 12])

    assert torch.equal(X_test, expected_X_test), "Test design matrix X_test incorrect"
    assert np.array_equal(z_values, expected_z_values), "Target values z_values incorrect"

if __name__ == "__main__":
    from ..pytest_helper import print_pytest_instructions
    print_pytest_instructions()    
