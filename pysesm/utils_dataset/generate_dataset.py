from typing import Callable, List, Tuple, Dict

import torch

from pysesm.utils_dataset.design_matrices import create_design_matrix_test, create_design_matrix_train
from pysesm.utils_dataset.mesh_generation import generate_mu, generate_mesh_samples, generate_random_samples

def generate_gaussian_dataset(
    n_samples: int,
    means: List[Tuple[float, float]] = None,
    variances: List[float] = None,
    weights: List[float] = None,
    limits: Tuple[float, float] = (-2, 2),
    mesh_divisions: int = 50
) -> Tuple[Dict, torch.Tensor, torch.Tensor, Dict, torch.Tensor, torch.Tensor]:
    """
    Create a dataset from a weighted mixture of gaussians with customizable parameters.

    Args:
        n_samples (int): Number of random samples to generate for training
        means (List[Tuple[float, float]]): List of means for each gaussian component
        variances (List[float]): List of variances for each gaussian component
        weights (List[float]): List of weights for each gaussian component
        limits (Tuple[float, float]): Lower and upper limits for the data range
        mesh_divisions (int): Number of divisions per dimension for the test mesh

    Returns:
        tuple: A tuple containing:
            - trainDataset: Dictionary with keys X,Y,Z, with n_samples random points
            - X_train (torch.Tensor): Training input features
            - y_train (torch.Tensor): Training target values
            - testDataset: The test dataset
            - X_test (torch.Tensor): Test input features
            - y_test (torch.Tensor): Test target values
    """

    if means is None:
        means = [(1, 1), (1, -1), (-1, -1)]
    if variances is None:
        variances = [0.15, 0.5, 0.75]
    if weights is None:
        weights = [1.25, 0.5, 0.75]

    # Validate input lengths
    if not (len(means) == len(variances) == len(weights)):
        raise ValueError("means, variances and weights must have the same length")
    
    # Create diagonal covariance matrices
    sigma_list = [var * torch.eye(2) for var in variances]
    
    # Convert means to tensors
    mu_list = [generate_mu(mu[0], mu[1]) for mu in means]
    
    low_lim, high_lim = limits
    
    # Regular 2D cartesian grid for prediction
    xx, yy, zz = generate_mesh_samples(mesh_divisions, low_lim, high_lim, sigma_list, mu_list, weights)
    
    # Random samples for training
    xx_r, yy_r, zz_r = generate_random_samples(
        n_samples, low_lim, high_lim, sigma_list, mu_list, weights
    )
    
    # Create datasets
    trainDataset = {"X": xx_r.ravel(), "Y": yy_r.ravel(), "Z": zz_r.ravel()}
    testDataset = {"X": xx.ravel(), "Y": yy.ravel(), "Z": zz.ravel()}
    
    # Create design matrices
    X_train, y_train = create_design_matrix_train(xx_r, yy_r, zz_r, n_samples)
    X_test, y_test = create_design_matrix_test(xx, yy, zz)
    
    return trainDataset, X_train, y_train, testDataset, X_test, y_test

def generate_custom_function_dataset(
    n_samples: int,
    function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    function_params: Dict = None,
    limits: Tuple[float, float] = (-2, 2),
    mesh_divisions: int = 50
) -> Tuple[Dict, torch.Tensor, torch.Tensor, Dict, torch.Tensor, torch.Tensor]:
    """
    Generate a dataset using a parametrizable 2D function.

    Args:
        n_samples (int): Number of random samples for training.
        function (Callable): Function f(x, y, **params) -> z.
        function_params (Dict): Parameters to pass to the function.
        limits (Tuple[float, float]): Domain limits.
        mesh_divisions (int): Mesh resolution.

    Returns:
        tuple: Train and test datasets (dicts and tensors).
    """
    if function_params is None:
        function_params={}

    low_lim, high_lim = limits

    # Generate mesh grid
    x_lin = torch.linspace(low_lim, high_lim, mesh_divisions)
    y_lin = torch.linspace(low_lim, high_lim, mesh_divisions)
    xx, yy = torch.meshgrid(x_lin, y_lin, indexing="ij")
    zz = function(xx, yy, **function_params)

    # Generate random samples
    xx_r = low_lim + (high_lim - low_lim) * torch.rand(n_samples)
    yy_r = low_lim + (high_lim - low_lim) * torch.rand(n_samples)
    zz_r = function(xx_r, yy_r, **function_params)

    # Create datasets
    trainDataset = {"X": xx_r.ravel(), "Y": yy_r.ravel(), "Z": zz_r.ravel()}
    testDataset = {"X": xx.ravel(), "Y": yy.ravel(), "Z": zz.ravel()}
    
    # Create design matrices
    X_train, y_train = create_design_matrix_train(xx_r, yy_r, zz_r, n_samples)
    X_test, y_test = create_design_matrix_test(xx, yy, zz)

    return trainDataset, X_train, y_train, testDataset, X_test, y_test

def generate_custom_nd_function_dataset(
    n_samples: int,
    n_dimensions: int,
    function: Callable[[torch.Tensor], torch.Tensor],
    function_params: Dict = None,
    limits: Tuple[float, float] = (-2.0, 2.0),
    mesh_divisions: int = 50
) -> Tuple[Dict, torch.Tensor, torch.Tensor, Dict, torch.Tensor, torch.Tensor]:
    """
    Generate a dataset using a parametrizable N-D function.

    Args:
        n_samples (int): Number of random samples for training.
        n_dimensions (int): Number of input dimensions.
        function (Callable): Function f(X, **params) -> z where X is (batch, N).
        function_params (Dict): Parameters to pass to the function.
        limits (Tuple[float, float]): Domain limits (same for all dims).
        mesh_divisions (int): Mesh resolution per dimension.

    Returns:
        tuple: Train and test datasets (dicts and tensors).
    """
    if function_params is None:
        function_params={}

    low_lim, high_lim = limits

    # Generate mesh grid for test data
    linspaces = [torch.linspace(low_lim, high_lim, mesh_divisions) for _ in range(n_dimensions)]
    mesh = torch.meshgrid(*linspaces, indexing="ij")
    mesh_flat = torch.stack([m.reshape(-1) for m in mesh], dim=-1)  # Shape: (mesh_size^N, N)

    zz = function(mesh_flat, **function_params)  # Output: (mesh_size^N,)

    # Generate random training samples
    rand_samples = low_lim + (high_lim - low_lim) * torch.rand(n_samples, n_dimensions)
    zz_r = function(rand_samples, **function_params)

    # Train and test datasets
    trainDataset = {"X": rand_samples, "Z": zz_r}
    testDataset = {"X": mesh_flat, "Z": zz}

    # Design matrices
    X_train, y_train = rand_samples, zz_r
    X_test, y_test = mesh_flat, zz

    return trainDataset, X_train, y_train, testDataset, X_test, y_test


def print_nd_dataset_info(dataset, name="Dataset"):
    print(f"\n=== {name} Information (N-dimensional) ===")
    print(f"Tipo: {type(dataset)}")
    print("Claves disponibles:", list(dataset.keys()))
    
    for key, value in dataset.items():
        print(f"\n{key}:")
        print(f"  Tipo: {type(value)}")
        if torch.is_tensor(value):
            print(f"  Shape: {value.shape}")
            print(f"  dtype: {value.dtype}")
            
            if len(value.shape) > 1:  # Para X (multidimensional)
                print("  Rango por dimensión:")
                for dim in range(value.shape[1]):
                    print(f"    Dim {dim}: [{value[:, dim].min():.4f}, {value[:, dim].max():.4f}]")
                print(f"  Primeras 3 muestras:\n{value[:3]}")
            else:  # Para Z (unidimensional)
                print(f"  Rango global: [{value.min():.4f}, {value.max():.4f}]")
                print(f"  Primeros 3 valores: {value[:3]}")
        else:
            print(f"  Valor: {value}")

