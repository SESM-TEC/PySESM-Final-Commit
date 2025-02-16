import torch
import pytest
import logging
from pysesm.models.DictLayer import DictLayer
from pysesm.functions.GaussianFunction import GaussianFunction
import numpy as np
from scipy.stats import multivariate_normal

# For debuggin and understanding >>>
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DEBUG_VISUALIZATION = False

def show_data(X,y,c,marker,label,ax=None):
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    

    # Plot training data
    ax.scatter(X[:, 0], X[:, 1], y.flatten(), 
               c=c, marker=marker, label=label)
    
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    ax.set_zlabel('y')
    ax.legend()

    plt.show(block=False)
    return ax
# End of debugging helpers <<<


def test_dict_layer_find_mu_only():
    """Test dictionary layer's ability to find correct mean with fixed covariance"""
    # Setup
    n_features = 2
    n_functions = 1
    logger = logging.getLogger('test')
    
    # Create uniform grid of points for better coverage
    n_samples = 500
    X = torch.rand(n_samples, 2) * 4 - 2  # Uniform in [-2, 2] x [-2, 2]
    
    # Define target Gaussian parameters
    true_mean = np.array([0.5, -0.3])
    fixed_cov = 0.5*np.eye(2)  # Identity covariance
    
    # Calculate true unnormalized Gaussian values
    gaussian_values = multivariate_normal.pdf(X.numpy(), mean=true_mean, cov=fixed_cov)
    peak_value = multivariate_normal.pdf(true_mean, mean=true_mean, cov=fixed_cov)
    y = torch.tensor(gaussian_values / peak_value, dtype=torch.float32).reshape(-1, 1)
    

    # Debug figure
    if DEBUG_VISUALIZATION:
        ax=show_data(X,y.numpy(),c='0.4',marker='.',label="ground truth")

    # Track parameter evolution
    mu_history = []
    
    def parameter_tracker(info):
        mu_history.append({
            'epoch': info['epoch'],
            'mu': info['mu'].numpy(),
            'mu_grad': info['mu_grad'].numpy(),
            'loss': info['loss']
        })


    # Initialize dictionary layer with fixed covariance (identity)
    dict_layer = DictLayer(
        n_features=n_features,
        n_functions=n_functions,
        psi=GaussianFunction(
            n_features=n_features,
            n_functions=n_functions,
            logger=logger,
            eig_range=[0.5, 0.5],  # Force eigenvalues to 1 (identity covariance)
            mu_range=[[ 0.3,  0.3], 
                      [-0.1, -0.1]]  # Wide range for mean
        ),
        alpha=0.15,  # Learning rate
        
        evaluation_func=lambda d, h: torch.matmul(d, h),
        logger=logger,
        parameter_hook=parameter_tracker,
        momentum=0.15
    )
    
    # Initialize h to [1] since we only have one Gaussian
    h = torch.ones((1, 1), dtype=torch.float32).detach()
    
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

    # DEBUG: >>> plot the learned mean and data
    if DEBUG_VISUALIZATION:

        # Calculate true unnormalized Gaussian values
        adapted_values = multivariate_normal.pdf(X.numpy(), mean=learned_mean, cov=fixed_cov)
        adapted_peak = multivariate_normal.pdf(learned_mean, mean=learned_mean, cov=fixed_cov)
        adapted_y = adapted_values / peak_value
        
        # Debug figure
        show_data(X,adapted_y,c='r',marker='.',label="final adaption",ax=ax)

    # Plot mu trajectory
        mu_points = np.array([entry['mu'].flatten() for entry in mu_history])
        ax.scatter(mu_points[:, 0], mu_points[:, 1], np.zeros_like(mu_points[:, 0]), 
                'b.', label='μ trajectory', alpha=0.5)

        # Plot gradient arrows at regular intervals (every 5 steps to avoid clutter)
        for i in range(0, len(mu_history), 5):
            mu = mu_history[i]['mu'].flatten()
            grad = mu_history[i]['mu_grad'].flatten()
            # Normalize gradient for visualization
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 0:  # Avoid division by zero
                grad = grad / grad_norm * 0.2  # Scale arrow length
                ax.quiver(mu[0], mu[1], 0,  # Starting point (z=0 plane)
                        -grad[0], -grad[1], 0,  # Direction (-grad because we minimize)
                        color='r', alpha=0.5) 


        # <<< End of debug code
    
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
    
    # Create uniform grid of points
    n_samples = 100
    X = torch.rand(n_samples, 2) * 4 - 2  # Uniform in [-2, 2] x [-2, 2]
    
    # Define target Gaussian parameters
    fixed_mean = np.array([0.0, 0.0])
    true_cov = np.array([[2.0, 0.0], [0.0, 0.5]])  # Diagonal covariance
    
    # Calculate true unnormalized Gaussian values
    gaussian_values = multivariate_normal.pdf(X.numpy(), mean=fixed_mean, cov=true_cov)
    peak_value = multivariate_normal.pdf(fixed_mean, mean=fixed_mean, cov=true_cov)
    y = torch.tensor(gaussian_values / peak_value, dtype=torch.float32).reshape(-1, 1)
    
    dict_layer = DictLayer(
        n_features=n_features,
        n_functions=n_functions,
        psi=GaussianFunction(
            n_features=n_features,
            n_functions=n_functions,
            logger=logger,
            eig_range=[0.5, 2.0],  # Allow range for eigenvalues
            mu_range=[[0.0, 0.0], [0.0, 0.0]]  # Fixed mean at origin
        ),
        alpha=0.2,
        momentum=0.1,
        evaluation_func=lambda d, h: torch.matmul(d, h),
        logger=logger
    )
    
    h = torch.ones((1, 1), dtype=torch.float32)
    
    # Train focusing on covariance
    n_epochs = 500
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
    learned_G = torch.matmul(A.T, A)
    
    # Convert to numpy for testing
    learned_G = learned_G.numpy()
    learned_cov = np.linalg.inv(learned_G)

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

def test_dict_layer_find_non_diagonal_covariance():
    """Test dictionary layer's ability to find diagonal covariance with fixed mean"""
    # Setup similar to previous test but with fixed mean and learnable diagonal covariance
    n_features = 2
    n_functions = 1
    logger = logging.getLogger('test')
    
    # Create uniform grid of points
    n_samples = 100
    X = torch.rand(n_samples, 2) * 4 - 2  # Uniform in [-2, 2] x [-2, 2]
    
    # Define target Gaussian parameters
    fixed_mean = np.array([0.0, 0.0])
    true_cov = np.array([[2.0, 0.5], [0.5, 1.0]])  # Diagonal covariance
    
    # Calculate true unnormalized Gaussian values
    gaussian_values = multivariate_normal.pdf(X.numpy(), mean=fixed_mean, cov=true_cov)
    peak_value = multivariate_normal.pdf(fixed_mean, mean=fixed_mean, cov=true_cov)
    y = torch.tensor(gaussian_values / peak_value, dtype=torch.float32).reshape(-1, 1)
    
    dict_layer = DictLayer(
        n_features=n_features,
        n_functions=n_functions,
        psi=GaussianFunction(
            n_features=n_features,
            n_functions=n_functions,
            logger=logger,
            eig_range=[0.5, 2.0],  # Allow range for eigenvalues
            mu_range=[[0.0, 0.0], [0.0, 0.0]]  # Fixed mean at origin
        ),
        alpha=0.2,
        momentum=0.1,
        evaluation_func=lambda d, h: torch.matmul(d, h),
        logger=logger
    )
    
    h = torch.ones((1, 1), dtype=torch.float32)
    
    # Train focusing on covariance
    n_epochs = 500
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
    learned_G = torch.matmul(A.T, A)
    
    # Convert to numpy for testing
    learned_G = learned_G.numpy()
    learned_cov = np.linalg.inv(learned_G)

    # Verify covariance structure matches true covariance
    np.testing.assert_allclose(learned_cov, true_cov, rtol=0.01, atol=0.01), "Learned covariance matrix doesn't match true covariance matrix"
    
    # Verify loss decreased
    assert dict_layer.losses[-1] < dict_layer.losses[0]

if __name__ == "__main__":
    from pytest_helper import print_pytest_instructions
    print_pytest_instructions()
