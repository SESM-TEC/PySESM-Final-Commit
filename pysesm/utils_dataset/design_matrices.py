import torch
from pysesm.utils_dataset.sampling_data import generate_uniform_sampling, sample_data


def unstack_design_matrix(X_test):
    """
    Unstacks the design matrix into its individual components.

    Args:
    - X_test (torch.Tensor): A tensor containing the design matrix with multiple columns.

    Returns:
    - x_tensor (torch.Tensor): A tensor containing the first component (column) of the design matrix.
    - y_tensor (torch.Tensor): A tensor containing the second component (column) of the design matrix.
    """
    x_tensor, y_tensor = torch.unbind(X_test, dim=1)
    return x_tensor, y_tensor


def create_design_matrix_train(xx, yy, zz, n_samples, dtype=torch.float32):
    """
    Creates a design matrix for training by randomly sampling data points.

    Args:
    - xx (numpy.ndarray): The x-coordinates of the grid.
    - yy (numpy.ndarray): The y-coordinates of the grid.
    - zz (numpy.ndarray): The z-values (target values) of the grid.
    - n_samples (int): The number of samples to generate.

    Returns:
    - X (numpy.ndarray): A 2D array containing the sampled x and y coordinates as the design matrix.
    - y (numpy.ndarray): A 1D array containing the sampled z-values as the target variable.
    """
    x_values = xx.ravel()
    y_values = yy.ravel()
    z_values = zz.ravel()

    total_points = len(x_values)

    # Take a permutation of the indices 
    sampled_indices = generate_uniform_sampling(total_points=total_points,
                                                n_samples=n_samples)
    
    X, y = sample_data(x_values, y_values, z_values, sampled_indices, dtype=dtype)

    return X, y


def create_design_matrix_test(xx, yy, zz, dtype=torch.float32):
    """
    Creates a design matrix for testing from given grid coordinates and target values.

    Args:
    - xx (numpy.ndarray): The x-coordinates of the grid.
    - yy (numpy.ndarray): The y-coordinates of the grid.
    - zz (numpy.ndarray): The z-values (target values) of the grid.

    Returns:
    - X_test (torch.Tensor): A tensor containing the design matrix for testing, where each row represents a sample with x and y coordinates.
    - z_values (numpy.ndarray): A 1D array containing the target values (zz) flattened as a vector.
    """
    x_values = xx.ravel()
    y_values = yy.ravel()
    z_values = zz.ravel()

    x_tensor = torch.as_tensor(x_values,dtype=dtype)
    y_tensor = torch.as_tensor(y_values,dtype=dtype)
    X_test = torch.stack((x_tensor, y_tensor), dim=1)

    return X_test, z_values
