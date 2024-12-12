import torch
import numpy as np

def generate_uniform_sampling(total_points, SEED, n_samples=500):
    """
    Generate uniform sampling indices

    Args:
        total_points (int): Total number of data points.
        n_samples (int): Number of samples to generate (default is 500).

    Returns:
        list: List of selected indices.

    Example:
        sampled_indices = generate_uniform_sampling(1000, n_samples=200)
    """

    # Assuming total_points is a tensor with shape (N, M)
    if n_samples > total_points:
        raise ValueError(f"Cannot sample {n_samples} points from tensor with only {total_points} rows")

    np.random.seed(SEED)
    selected_indexes = np.random.permutation(total_points)[:n_samples]
    return selected_indexes


def sample_data(x_values, y_values, z_values, sampled_indices):
    """
    Sample data based on selected indices.

    Args:
        x_values (array-like): X-axis values.
        y_values (array-like): Y-axis values.
        z_values (array-like): Z-axis values.
        sampled_indices (list): List of indices to sample.

    Returns:
        tuple: Tuple containing sampled X (features) and y (labels).

    Example:
        X, y = sample_data(x_values, y_values, z_values, sampled_indices)
    """

    sampled_x = torch.tensor(x_values[sampled_indices], dtype=torch.float32)
    sampled_y = torch.tensor(y_values[sampled_indices], dtype=torch.float32)
    X = torch.stack((sampled_x, sampled_y), dim=1)
    y = z_values[sampled_indices].clone().detach().to(dtype=torch.float32)    
    return X, y
