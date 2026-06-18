"""
Dataset Generation Utilities.

Provides functions to generate synthetic datasets for testing and
experimentation within the SESM framework, including Gaussian mixtures and
custom functions.

Copyright (c) 2023-2025, Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
"""

from collections.abc import Callable
import torch

from pysesm.utils_dataset.design_matrices import create_design_matrix_train, create_design_matrix_test
from pysesm.utils_dataset.mesh_generation import generate_mu, generate_mesh_samples, generate_random_samples

def _discretize_to_grid(
    data: torch.Tensor,
    low_lim: float,
    high_lim: float,
    divisions: int
) -> torch.Tensor:
    """Snap each value to the nearest point on a regular grid."""
    grid = torch.linspace(low_lim, high_lim, divisions)
    indices = torch.bucketize(data.contiguous(), grid).clamp(0, divisions - 1)
    # bucketize returns the right-side index; pick the closer neighbor
    left = indices.clamp(min=1) - 1
    right = indices.clamp(max=divisions - 1)
    dist_left = (data - grid[left]).abs()
    dist_right = (grid[right] - data).abs()
    closest = torch.where(dist_left <= dist_right, left, right)
    return grid[closest]



def generate_gaussian_dataset(
    n_samples: int,
    means: list[tuple[float, float]] | None = None,
    variances: list | None = None,
    weights: list[float] | None = None,
    limits: tuple[float, float] | None = None,
    mesh_divisions: int = 50,
    test_random_samples: int | None = None
) -> tuple[dict, torch.Tensor, torch.Tensor, dict, torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """
    Create a dataset from a weighted mixture of gaussians with customizable parameters.

    There are three fixed gaussians, from which data is sampled.

    The generated dictionaries hold the column vectors you would use with plot
    functions, whereas the X_train, X_test and their corresponding y values hold
    the data as you need for training.


    Args:
        n_samples (int): Number of random samples to generate for training
        means (List[Tuple[float, float]]): List of means for each gaussian component
        variances (List): List of variances to create diagonal covariance matrices,
                          or a list of predefined 2x2 covariance tensors (torch.Tensor)
        weights (List[float]): List of weights for each gaussian component
        limits (Tuple[float, float]): Lower and upper limits for the data range
        mesh_divisions (int): Number of divisions per dimension for the test mesh
        test_random_samples (int | None): If set, generate this many random test
            samples instead of a mesh grid. Defaults to None (mesh behavior).

    Returns:
        tuple: A tuple containing:
            - trainDataset: Dictionary with keys X,Y,Z, with n_samples random points
            - X_train (torch.Tensor): Training input features
            - y_train (torch.Tensor): Training target values
            - testDataset: The test dataset
            - X_test (torch.Tensor): Test input features
            - y_test (torch.Tensor): Test target values
            - mu_list (list[torch.Tensor]): List of mean vectors for the ground truth gaussians.
            - sigma_list (list[torch.Tensor]): List of covariance matrices for the ground truth gaussians.

    """

    # Safe default values of mutable types:
    if means is None:
        means = [(1, 1), (1, -1), (-1, -1)]

    if variances is None:
        variances = [0.15, 0.5, 0.75]

    if weights is None:
        weights = [1.25, 0.5, 0.75]

    if limits is None:
        limits = (-2, 2)

    # Validate input lengths
    if not (len(means) == len(variances) == len(weights)):
        raise ValueError("means, variances and weights must have the same length")
    
    # Create or assign covariance matrices
    if variances and isinstance(variances[0], torch.Tensor):
        # Use pre-defined covariance matrices
        sigma_list = variances
    else:
        # Create diagonal covariance matrices from variances
        sigma_list = [var * torch.eye(2) for var in variances]
   
    # Convert means to tensors
    mu_list = [generate_mu(mu[0], mu[1]) for mu in means]
    
    low_lim, high_lim = limits
    
    # Test data: mesh grid or random samples
    if test_random_samples is not None:
        xx, yy, zz = generate_random_samples(
            test_random_samples, low_lim, high_lim, sigma_list, mu_list, weights
        )
    else:
        xx, yy, zz = generate_mesh_samples(
            mesh_divisions, low_lim, high_lim, sigma_list, mu_list, weights
        )

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
    
    return trainDataset, X_train, y_train, testDataset, X_test, y_test, mu_list, sigma_list

def generate_custom_function_dataset(
    n_samples: int,
    function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    function_params: dict = None,
    limits: tuple[float, float] = (-2, 2),
    mesh_divisions: int = 50,
    test_random_samples: int | None = None
) -> tuple[dict, torch.Tensor, torch.Tensor, dict, torch.Tensor, torch.Tensor]:
    """
    Generate a dataset using a parametrizable 2D function.

    Args:
        n_samples (int): Number of random samples for training.
        function (Callable): Function f(x, y, **params) -> z.
        function_params (Dict): Parameters to pass to the function.
        limits (Tuple[float, float]): Domain limits.
        mesh_divisions (int): Mesh resolution.
        test_random_samples (int | None): If set, generate this many random test
            samples instead of a mesh grid. Defaults to None (mesh behavior).        

    Returns:
        tuple: Train and test datasets (dicts and tensors).
    """

    if function_params is None:
        function_params = {}

    low_lim, high_lim = limits

    # Test data: mesh grid or random samples
    if test_random_samples is not None:
        xx = low_lim + (high_lim - low_lim) * torch.rand(test_random_samples)
        yy = low_lim + (high_lim - low_lim) * torch.rand(test_random_samples)
        zz = function(xx, yy, **function_params)
    else:
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
    function_params: dict = None,
    limits: tuple[float, float] = (-2.0, 2.0),
    mesh_divisions: int = 50,
    test_random_samples: int | None = None
) -> tuple[dict, torch.Tensor, torch.Tensor, dict, torch.Tensor, torch.Tensor]:
    """
    Generate a dataset using a parametrizable N-D function.

    Args:
        n_samples (int): Number of random samples for training.
        n_dimensions (int): Number of input dimensions.
        function (Callable): Function f(X, **params) -> z where X is (batch, N).
        function_params (Dict): Parameters to pass to the function.
        limits (Tuple[float, float]): Domain limits (same for all dims).
        mesh_divisions (int | None): If set, discretize random samples to a
            regular grid with this many divisions.  Also used as the mesh
            resolution when generating a mesh-based test set (only when
            test_random_samples is None).  Defaults to None.
        test_random_samples (int | None): If set, generate this many random
            test samples instead of an N-D mesh grid.  Recommended for
            n_dimensions > 3 to avoid exponential memory usage.
 
    Returns:
        tuple: Train and test datasets (dicts and tensors).
    """

    if function_params is None:
        function_params = {}

    if mesh_divisions is None and test_random_samples is None:
        raise ValueError(
            "Either mesh_divisions or test_random_samples must be specified "
            "to generate test data.  For n_dimensions > 3, use "
            "test_random_samples to avoid exponential memory usage."
        )

    low_lim, high_lim = limits

    
    # --- Training data (always random) ---
    rand_samples = low_lim + (high_lim - low_lim) * torch.rand(n_samples, n_dimensions)
    
    if mesh_divisions is not None:
        rand_samples = _discretize_to_grid(rand_samples, low_lim, high_lim, mesh_divisions)
 
    zz_r = function(rand_samples, **function_params)

    # --- Test data ---
    if test_random_samples is not None:
        test_samples = low_lim + (high_lim - low_lim) * torch.rand(
            test_random_samples, n_dimensions
        )
        if mesh_divisions is not None:
            test_samples = _discretize_to_grid(
                test_samples, low_lim, high_lim, mesh_divisions
            )
        zz_test = function(test_samples, **function_params)
    else:
        # Mesh-based test set (caller explicitly requested via mesh_divisions)
        linspaces = [
            torch.linspace(low_lim, high_lim, mesh_divisions)
            for _ in range(n_dimensions)
        ]
        mesh = torch.meshgrid(*linspaces, indexing="ij")
        test_samples = torch.stack([m.reshape(-1) for m in mesh], dim=-1)
        zz_test = function(test_samples, **function_params)

    # --- Datasets ---
    trainDataset = {"X": rand_samples, "Z": zz_r}
    testDataset = {"X": test_samples, "Z": zz_test}

    X_train, y_train = rand_samples, zz_r
    X_test, y_test = test_samples, zz_test

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

