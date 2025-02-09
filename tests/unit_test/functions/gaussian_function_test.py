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
        mu_range=[[0.0, 0.0],[0.0, 0.0]],   # Force mean at zero
    )
    
    # Initialize and verify parameters
    theta = gaussian.initialize()
    assert theta.requires_grad,"Theta should require gradient computation"
    
    # Create test points in a grid
    x = torch.linspace(-2, 2, 5)
    y = torch.linspace(-2, 2, 5)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([X.flatten(), Y.flatten()], dim=1).T.requires_grad_(True)  # (2, 25)
    
    # Compute Gaussian function values
    values = gaussian(points, theta)
    
    # Expected values for standard normal distribution
    mean = np.zeros(2)
    cov = np.eye(2)
    expected_values = torch.tensor(
        ( multivariate_normal.pdf(points.T.detach().numpy(), mean=mean, cov=cov) /
          multivariate_normal.pdf(mean, mean=mean, cov=cov) ),
          dtype = values.dtype
    ).unsqueeze(1)
    
    max_diff = torch.max(torch.abs(values - expected_values)).item()

    assert max_diff < 1e-5, "Distribution not similar enough"


    # Check both implementations
    torch_result = torch.allclose(values, expected_values, rtol=1e-4, atol=1e-4)

    # Test forward pass
    assert torch_result, "Distribuitions not similar enough"
    
    # Test mu gradients only
    theta.grad = None
    values = gaussian(points, theta, rho_flag=False, mu_flag=True)
    values.sum().backward()
    assert theta.grad is not None
    mu_grads = theta.grad[-n_features:, :]
    rho_grads = theta.grad[:-n_features, :]
    assert torch.all(rho_grads == 0), "Rho gradients should be zero when rho_flag is False"
    assert not torch.all(mu_grads == 0), "Mu gradients should be non-zero when mu_flag is True"

    # Test rho gradients only
    theta.grad = None
    values = gaussian(points, theta, rho_flag=True, mu_flag=False)
    values.sum().backward()
    assert theta.grad is not None
    mu_grads = theta.grad[-n_features:, :]
    rho_grads = theta.grad[:-n_features, :]
    assert torch.all(mu_grads == 0), "Mu gradients should be zero when mu_flag is False"
    assert not torch.all(rho_grads == 0), "Rho gradients should be non-zero when rho_flag is True"

    # Test both gradients
    theta.grad = None
    values = gaussian(points, theta, rho_flag=True, mu_flag=True)
    values.sum().backward()
    assert theta.grad is not None
    mu_grads = theta.grad[-n_features:, :]
    rho_grads = theta.grad[:-n_features, :]
    assert not torch.all(mu_grads == 0), "Mu gradients should be non-zero when both flags are True"
    assert not torch.all(rho_grads == 0), "Rho gradients should be non-zero when both flags are True"

def test_single_gaussian_gradient():
    """Test gradients using finite differences for a single Gaussian"""
    n_features = 2
    n_functions = 1
    logger = logging.getLogger('test')
    
    gaussian = GaussianFunction(
        n_features=n_features,
        n_functions=n_functions,
        seed=42,
        logger=logger,
        eig_range=[1.0, 1.0],
        mu_range=[[0.0, 0.0], [0.0, 0.0]],
    )
    
    theta = gaussian.initialize()
    assert theta.requires_grad, "Theta should require gradient computation"
    
    # Test points at different locations
    test_points = [
        torch.tensor([[0.0], [0.0]]),  # At mean
        torch.tensor([[1.0], [0.0]]),  # Along x axis
        torch.tensor([[0.0], [1.0]]),  # Along y axis
        torch.tensor([[1.0], [1.0]]),  # Diagonal
        torch.tensor([[-1.0], [-1.0]]), # Other diagonal
        torch.tensor([[-0.5], [0.3]])  # And something else
    ]


    eps = 1e-6
    for point in test_points:
        point = point.requires_grad_(True)
        
        def f(params):
            return gaussian(point, params, rho_flag=True, mu_flag=True)
        
        # Compute numerical gradient
        numerical_grad = torch.zeros_like(theta)
        
        for i in range(theta.numel()):
            theta_plus = theta.clone().detach()
            theta_plus.data.flatten()[i] += eps
            theta_minus = theta.clone().detach()
            theta_minus.data.flatten()[i] -= eps
            
            numerical_grad.flatten()[i] = (f(theta_plus) - f(theta_minus)).sum() / (2 * eps)
        
        # Compute analytical gradient
        out = f(theta)
        out.sum().backward()
        analytical_grad = theta.grad.clone()
        theta.grad = None  # Clear gradients for next iteration
        
        # Compare gradients
        assert torch.allclose(numerical_grad, analytical_grad, atol=5e-2), \
            f"Gradient mismatch at point {point.T}"
        
        # Additional verification that gradients point in same direction
        # Add cosine similarity check with zero handling
        if torch.norm(numerical_grad) > 1e-10 and torch.norm(analytical_grad) > 1e-10:
            normalized_numerical = numerical_grad / torch.norm(numerical_grad)
            normalized_analytical = analytical_grad / torch.norm(analytical_grad)
            cosine_similarity = torch.sum(normalized_numerical * normalized_analytical)
            assert cosine_similarity > 0.9, \
                f"Gradient directions differ significantly at point {point.T}"
        else:
            # If either gradient is essentially zero, verify both are close to zero
            assert torch.norm(numerical_grad) < 1e-10 and torch.norm(analytical_grad) < 1e-10, \
                f"Only one gradient is zero at point {point.T}"
        
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
        mu_range=[[-1.0, 1.0], [-1.0, 1.0]],
    )
    
    theta = gaussian.initialize()
    assert theta.requires_grad, "Theta should require gradient computation"
    
    # Extract parameters for verification
    rho = theta[:-n_features, :]
    mu = theta[-n_features:, :]
    
    # Verify shapes
    assert rho.shape == (3, 2), "Rho should have shape (3, 2) for 2D gaussian"  # 3 elements for upper triangular in 2D
    assert mu.shape == (2, 2), "Mu should have shape (2, 2) for 2D gaussian"    # 2D means for 2 Gaussians
    
    # Create test points
    x = torch.linspace(-2, 2, 5)
    y = torch.linspace(-2, 2, 5)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([X.flatten(), Y.flatten()], dim=1).T.requires_grad_(True)
    
    # Compute values
    values = gaussian(points, theta)
    
    # Shape tests
    assert values.shape == (25, 2), "Output should have shape (n_points, n_functions)"
    
    # Value range tests
    assert torch.all(values >= 0), "Gaussian values should be non-negative"
    assert torch.all(values <= 1), "Normalized Gaussian values should be <= 1"
    
    # Test gradients as in single gaussian case
    # Test mu gradients only
    theta.grad = None
    values = gaussian(points, theta, rho_flag=False, mu_flag=True)
    values.sum().backward()
    assert theta.grad is not None
    mu_grads = theta.grad[-n_features:, :]
    rho_grads = theta.grad[:-n_features, :]
    assert torch.all(rho_grads == 0), "Rho gradients should be zero when rho_flag is False"
    assert not torch.all(mu_grads == 0), "Mu gradients should be non-zero when mu_flag is True"

    # Test rho gradients only
    theta.grad = None
    values = gaussian(points, theta, rho_flag=True, mu_flag=False)
    values.sum().backward()
    assert theta.grad is not None
    mu_grads = theta.grad[-n_features:, :]
    rho_grads = theta.grad[:-n_features, :]
    assert torch.all(mu_grads == 0), "Mu gradients should be zero when mu_flag is False"
    assert not torch.all(rho_grads == 0), "Rho gradients should be non-zero when rho_flag is True"

    # Test both gradients
    theta.grad = None
    values = gaussian(points, theta, rho_flag=True, mu_flag=True)
    values.sum().backward()
    assert theta.grad is not None
    mu_grads = theta.grad[-n_features:, :]
    rho_grads = theta.grad[:-n_features, :]
    assert not torch.all(mu_grads == 0), "Mu gradients should be non-zero when both flags are True"
    assert not torch.all(rho_grads == 0), "Rho gradients should be non-zero when both flags are True"

if __name__ == "__main__":
    from pytest_helper import print_pytest_instructions
    print_pytest_instructions()    
