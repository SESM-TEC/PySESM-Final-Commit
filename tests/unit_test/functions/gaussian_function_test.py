from pysesm.functions.GaussianFunction import GaussianFunction
import torch
import logging
import numpy as np
from scipy.stats import multivariate_normal

def test_single_gaussian_identity():
    """Test a single Gaussian with zero mean and identity covariance"""
    # Setup
    n_features = 2
    n_functions = 1
    logger = logging.getLogger('test')
    
    gaussian = GaussianFunction(
        n_features=n_features,
        n_functions=n_functions,
        seed=42,
        logger=logger,
        eig_range=[1.0, 1.0],  # Force eigenvalue of 1
        mu_range=[0.0, 0.0],   # Force mean at zero
        vector_range=[1.0, 1.0] # Force identity rotation
    )
    
    # Initialize and verify parameters
    theta = gaussian.initialize()
    
    # Create test points in a grid
    x = torch.linspace(-2, 2, 5)
    y = torch.linspace(-2, 2, 5)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([X.flatten(), Y.flatten()], dim=1).T  # (2, 25)
    
    # Compute Gaussian function values
    values = gaussian(points, theta)
    
    # Expected values for standard normal distribution
    mean = np.zeros(2)
    cov = np.eye(2)
    expected_values = torch.tensor(
        multivariate_normal.pdf(points.T.numpy(), mean=mean, cov=cov)
    ).unsqueeze(1)
    
    # Test forward pass
    assert torch.allclose(values, expected_values, rtol=1e-4)
    
    # Test gradients
    values.sum().backward()
    assert theta.grad is not None
    assert not torch.isnan(theta.grad).any()

def test_single_gaussian_gradient():
    """Test gradients using finite differences"""
    n_features = 2
    n_functions = 1
    logger = logging.getLogger('test')
    
    gaussian = GaussianFunction(
        n_features=n_features,
        n_functions=n_functions,
        seed=42,
        logger=logger,
        eig_range=[1.0, 1.0],
        mu_range=[0.0, 0.0],
        vector_range=[1.0, 1.0]
    )
    
    theta = gaussian.initialize()
    
    # Single test point
    x = torch.tensor([[1.0], [0.0]])  # (2, 1)
    
    def f(params):
        return gaussian(x, params)
    
    # Compute numerical gradient
    eps = 1e-6
    numerical_grad = torch.zeros_like(theta)
    
    for i in range(theta.numel()):
        theta_plus = theta.clone()
        theta_plus.data.flatten()[i] += eps
        theta_minus = theta.clone()
        theta_minus.data.flatten()[i] -= eps
        
        numerical_grad.flatten()[i] = (f(theta_plus) - f(theta_minus)).sum() / (2 * eps)
    
    # Compute analytical gradient
    out = f(theta)
    out.sum().backward()
    analytical_grad = theta.grad
    
    # Compare gradients
    assert torch.allclose(numerical_grad, analytical_grad, rtol=1e-4)

def test_two_gaussians():
    """Test two Gaussians with different means and covariances"""
    n_features = 2
    n_functions = 2
    logger = logging.getLogger('test')
    
    gaussian = GaussianFunction(
        n_features=n_features,
        n_functions=n_functions,
        seed=42,
        logger=logger,
        eig_range=[1.0, 2.0],
        mu_range=[-1.0, 1.0],
        vector_range=[1.0, 1.0]
    )
    
    theta = gaussian.initialize()
    
    # Extract parameters for verification
    rho = theta[:-n_features, :]
    mu = theta[-n_features:, :]
    
    # Verify shapes
    assert rho.shape == (3, 2)  # 3 elements for upper triangular in 2D
    assert mu.shape == (2, 2)   # 2D means for 2 Gaussians
    
    # Create test points
    x = torch.linspace(-2, 2, 5)
    y = torch.linspace(-2, 2, 5)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([X.flatten(), Y.flatten()], dim=1).T
    
    # Compute values
    values = gaussian(points, theta)
    
    # Shape tests
    assert values.shape == (25, 2)  # 25 points, 2 Gaussians
    
    # Value range tests
    assert torch.all(values >= 0)
    assert torch.all(values <= 1)
