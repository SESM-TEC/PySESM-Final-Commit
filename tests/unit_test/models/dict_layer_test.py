import torch
import pytest
import logging
from pysesm.models.DictLayer import DictLayer
from pysesm.functions.GaussianFunction import GaussianFunction
import numpy as np
from scipy.stats import multivariate_normal

def test_dict_layer_find_mu_only():
    """Test dictionary layer's ability to find correct mean with fixed covariance"""
    # Setup
    n_features = 2
    n_functions = 1
    logger = logging.getLogger('test')
    
    # Create synthetic data from a known 2D Gaussian
    n_samples = 100
    true_mean = np.array([0.5, -0.3])
    # Fixed identity covariance
    fixed_cov = np.eye(2)
    
    # Generate random points
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    X = rng.multivariate_normal(true_mean, fixed_cov, n_samples)
    
    # Calculate true unnormalized Gaussian values
    gaussian_values = multivariate_normal.pdf(X, mean=true_mean, cov=fixed_cov)
    peak_value = multivariate_normal.pdf(true_mean, mean=true_mean, cov=fixed_cov)
    y = torch.tensor(gaussian_values / peak_value, dtype=torch.float32).reshape(-1, 1)
    X = torch.tensor(X, dtype=torch.float32)
    
    # Initialize dictionary layer with fixed covariance (identity)
    dict_layer = DictLayer(
        n_features=n_features,
        n_functions=n_functions,
        psi=GaussianFunction(
            n_features=n_features,
            n_functions=n_functions,
            logger=logger,
            eig_range=[1.0, 1.0],  # Force eigenvalues to 1 (identity covariance)
            mu_range=[[-2.0, 2.0], [-2.0, 2.0]]  # Wide range for mean
        ),
        alpha=0.1,  # Learning rate
        evaluation_func=lambda d, h: torch.matmul(d, h),
        logger=logger
    )
    
    # Initialize h to [1] since we only have one Gaussian
    h = torch.ones((1, 1), dtype=torch.float32)
    
    # Train for several epochs
    n_epochs = 100
    dict_layer.partial_fit(
        X=X,
        y=y,
        h=h,
        epochs=n_epochs,
        mu_flag=True,   # Only optimize mean
        rho_flag=False  # Keep covariance fixed
    )
    
    # Extract learned mean from theta
    learned_mean = dict_layer.theta_parameter_vector[-n_features:, 0].detach().numpy()
    
    # Assert learned mean is close to true mean
    np.testing.assert_allclose(learned_mean, true_mean, rtol=1e-1)
    
    # Verify loss decreased
    assert dict_layer.losses[-1] < dict_layer.losses[0]

def test_dict_layer_find_diagonal_covariance():
    """Test dictionary layer's ability to find diagonal covariance with fixed mean"""
    # Setup similar to previous test but with fixed mean and learnable diagonal covariance
    n_features = 2
    n_functions = 1
    logger = logging.getLogger('test')
    
    # Create synthetic data with diagonal covariance
    n_samples = 100
    fixed_mean = np.array([0.0, 0.0])
    true_cov = np.array([[2.0, 0.0], [0.0, 0.5]])  # Diagonal covariance
    
    rng = np.random.default_rng(42)
    X = rng.multivariate_normal(fixed_mean, true_cov, n_samples)
    
    gaussian_values = multivariate_normal.pdf(X, mean=fixed_mean, cov=true_cov)
    peak_value = multivariate_normal.pdf(fixed_mean, mean=fixed_mean, cov=true_cov)
    y = torch.tensor(gaussian_values / peak_value, dtype=torch.float32).reshape(-1, 1)
    X = torch.tensor(X, dtype=torch.float32)
    
    dict_layer = DictLayer(
        n_features=n_features,
        n_functions=n_functions,
        psi=GaussianFunction(
            n_features=n_features,
            n_functions=n_functions,
            logger=logger,
            eig_range=[0.1, 5.0],  # Allow range for eigenvalues
            mu_range=[[0.0, 0.0], [0.0, 0.0]]  # Fixed mean at origin
        ),
        alpha=0.1,
        evaluation_func=lambda d, h: torch.matmul(d, h),
        logger=logger
    )
    
    h = torch.ones((1, 1), dtype=torch.float32)
    
    # Train focusing on covariance
    n_epochs = 100
    dict_layer.partial_fit(
        X=X,
        y=y,
        h=h,
        epochs=n_epochs,
        mu_flag=False,   # Keep mean fixed
        rho_flag=True    # Optimize covariance
    )
    
    # Extract learned covariance
    rho = dict_layer.theta_parameter_vector[:-n_features, 0].detach()
    A = torch.zeros(n_features, n_features)
    indices = torch.triu_indices(n_features, n_features)
    A[indices[0], indices[1]] = rho
    learned_cov = torch.matmul(A.T, A)
    
    # Convert to numpy for testing
    learned_cov = learned_cov.numpy()
    
    # Verify covariance structure
    # 1. Should be approximately diagonal
    assert np.abs(learned_cov[0, 1]) < 0.1, "Off-diagonal elements should be close to zero"
    assert np.abs(learned_cov[1, 0]) < 0.1, "Off-diagonal elements should be close to zero"
    
    # 2. Diagonal elements should be close to true values (up to scaling)
    ratio = learned_cov[0, 0] / learned_cov[1, 1]
    true_ratio = true_cov[0, 0] / true_cov[1, 1]
    np.testing.assert_allclose(ratio, true_ratio, rtol=0.2)
    
    # Verify loss decreased
    assert dict_layer.losses[-1] < dict_layer.losses[0]

if __name__ == "__main__":
    from pytest_helper import print_pytest_instructions
    print_pytest_instructions()
