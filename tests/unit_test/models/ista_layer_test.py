from pysesm.models.ISTALayer import ISTALayer
import torch
import logging
import numpy as np
from scipy.stats import multivariate_normal

def test_ista_perfect_dictionary():
    """Test ISTA optimization with a perfect unnormalized Gaussian dictionary"""
    # Setup
    n_functions = 1
    logger = logging.getLogger('test')
    
    # Create synthetic data from a known 2D Gaussian
    n_samples = 100
    mean = np.array([0.5, -0.3])
    cov = np.array([[1.0, 0.3], [0.3, 0.8]])
    
    # Generate random points
    X = np.random.multivariate_normal(mean, cov, n_samples)
    
    # Get normalized Gaussian values
    gaussian_values = multivariate_normal.pdf(X, mean=mean, cov=cov)
    # Get value at the mean for denormalization
    peak_value = multivariate_normal.pdf(mean, mean=mean, cov=cov)
    # Denormalize by dividing by value at mean - this gives us exp(-0.5(x-μ)'Σ⁻¹(x-μ))
    unnormalized_gaussian = gaussian_values / peak_value
    
    dictionary = torch.tensor(unnormalized_gaussian, dtype=torch.float32).reshape(-1, 1)
    
    # Target y should be exactly dictionary when h=1
    y = dictionary.clone()
    
    # Initialize ISTA layer
    ista = ISTALayer(
        n_functions=n_functions,
        alpha=0.1,
        lambd=0.001,  # Small lambda since we want h≈1
        weight_decay=0.0,
        evaluation_func=lambda d, h: torch.matmul(d, h),
        logger=logger,
        #optimizer=lambda parameters, lr, weight_decay: torch.optim.Adam(
        #    parameters, lr=lr, weight_decay=weight_decay
        #)
    )
    
    # Test initial gradient direction
    ista.h.data = torch.tensor([[0.5]])  # Start h below target
    ista.h.grad = None
    loss = ista.forward(y, dictionary)
    loss.backward()
    initial_grad = ista.h.grad.item()
    assert initial_grad < 0, "Initial gradient should point upward when h is too small"
    
    # Test optimization
    for _ in range(50):
        ista.train_step(y, dictionary)
    
    # Verify h converges close to 1
    final_h = ista.h.item()
    assert abs(final_h - 1.0) < 0.1, f"h should converge to 1, got {final_h}"

def test_ista_sparse_selection():
    """Test ISTA's ability to select correct unnormalized Gaussian when data clearly comes from it"""
    logger = logging.getLogger('test')
    
    # Create synthetic data from a specific 2D Gaussian (similar to one_block_example)
    n_samples = 100
    mean_target = np.array([0.5, -0.3])
    cov_target = np.array([[0.1, 0.03], [0.03, 0.3]])
    
    # Generate actual data points from this Gaussian
    X = np.random.multivariate_normal(mean_target, cov_target, n_samples)
    X_torch = torch.tensor(X, dtype=torch.float32)
    
    # Create dictionary with three elements where first one matches our data
    n_functions = 3
    dictionary = torch.zeros((n_samples, n_functions), dtype=torch.float32)
    
    # First column: the Gaussian that matches our data
    gaussian_values = multivariate_normal.pdf(X, mean=mean_target, cov=cov_target)
    peak_value = multivariate_normal.pdf(mean_target, mean=mean_target, cov=cov_target)
    unnormalized_gaussian = gaussian_values / peak_value
    dictionary[:, 0] = torch.tensor(unnormalized_gaussian, dtype=torch.float32)
    
    # Other columns: clearly different Gaussians
    other_means = [np.array([-1.5, 1.0]), np.array([1.5, 1.0])]
    other_covs = [
        np.array([[0.1, -0.01], [-0.01, 0.1]]),  # Tighter covariance
        np.array([[0.1, 0.0], [0.0, 0.1]])      # Wider covariance
    ]
    
    for i in range(2):
        gaussian_values = multivariate_normal.pdf(X, mean=other_means[i], cov=other_covs[i])
        peak_value = multivariate_normal.pdf(other_means[i], mean=other_means[i], cov=other_covs[i])
        unnormalized_gaussian = gaussian_values / peak_value
        dictionary[:, i+1] = torch.tensor(unnormalized_gaussian, dtype=torch.float32)
    
    # Target y should match first Gaussian only
    y = dictionary[:, 0].clone().detach()
    
    # Initialize ISTA layer with Adam optimizer
    ista = ISTALayer(
        n_functions=n_functions,
        alpha=0.1,  # Learning rate for Adam
        lambd=0.05,  # Lambda for sparsity
        weight_decay=0.0,
        evaluation_func=lambda d, h: torch.matmul(d, h),
        logger=logger
    )
    
    # Initialize h with random values as a column vector
    ista.h.data = torch.rand(n_functions, 1) * 0.5
    
    # Run optimization
    for _ in range(500):
        ista.train_step(y, dictionary)
    
    # Verify results
    h_final = ista.h.detach()
    
    # First component should be close to 1
    assert abs(h_final[0, 0] - 1.0) < 0.2, f"First component should be close to 1, got {h_final[0, 0]}"
    
    # Other components should be close to 0
    assert torch.all(torch.abs(h_final[1:, 0]) < 0.2), \
        f"Other components should be close to 0, got {h_final[1:, 0]}"

def test_ista_gradient_flow():
    """Test proper gradient flow through ISTA layer"""
    logger = logging.getLogger('test')
    
    # Create simple exponential falloff (like unnormalized Gaussian)
    x = torch.linspace(0, 2, 10)
    dictionary = torch.exp(-0.5 * x**2).reshape(-1, 1)  # Pure exp(-0.5x²) values
    y = dictionary.clone()
    
    ista = ISTALayer(
        n_functions=1,
        alpha=0.1,
        lambd=0.001,
        weight_decay=0.0,
        evaluation_func=lambda d, h: torch.matmul(d, h),
        logger=logger
    )
    
    # Test gradient computation
    ista.h.data = torch.tensor([0.5])
    ista.h.grad = None
    loss = ista.forward(y, dictionary)
    loss.backward()
    assert ista.h.grad is not None, "No gradient flowing to h"
    
    # Test gradient flow after shrinkage
    with torch.no_grad():
        h_shrunk = ista.shrinkage()
        ista.h.data.copy_(h_shrunk)
    
    ista.h.grad = None
    loss = ista.forward(y, dictionary)
    loss.backward()
    assert ista.h.grad is not None, "No gradient after shrinkage"

if __name__ == "__main__":
    from pytest_helper import print_pytest_instructions
    print_pytest_instructions()