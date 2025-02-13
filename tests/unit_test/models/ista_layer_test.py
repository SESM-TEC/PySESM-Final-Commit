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
        evaluation_func=lambda d, h: torch.matmul(d, h),
        logger=logger,
        #optimizer=lambda parameters, lr : torch.optim.Adam(
        #    parameters, lr=lr
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
    
     # Create random uniform points in [-2, 2] x [-2, 2]
    n_samples = 100
    X = torch.rand(n_samples, 2) * 4 - 2  # This scales [0,1] to [-2,2]
    
    # Define three different Gaussians
    mean1 = np.array([0.5, -0.3])    # Target Gaussian
    mean2 = np.array([-1.5, 1.0])    # Distractor 1
    mean3 = np.array([1.5, 1.0])     # Distractor 2
    
    cov1 = np.array([[0.1, 0.03], [0.03, 0.3]])
    cov2 = np.array([[0.1, -0.01], [-0.01, 0.1]])
    cov3 = np.array([[0.1, 0.0], [0.0, 0.1]])
    
    means = [mean1, mean2, mean3]
    covs = [cov1, cov2, cov3]
    n_functions = len(means)

    # Create dictionary with three Gaussian components
    dictionary = torch.zeros((n_samples, 3), dtype=torch.float32)
    
    # Evaluate each Gaussian at all points
    for i, (mean, cov) in enumerate(zip(means, covs)):
        values = multivariate_normal.pdf(X.numpy(), mean=mean, cov=cov)
        peak = multivariate_normal.pdf(mean, mean=mean, cov=cov)
        dictionary[:, i] = torch.tensor(values / peak, dtype=torch.float32)
    
    # Target y should match first Gaussian only
    y = dictionary[:, 0].clone().unsqueeze(-1)
    
    # Initialize ISTA layer
    ista = ISTALayer(
        n_functions=n_functions,
        alpha=0.15,
        lambd=0.05,
        evaluation_func=lambda d, h: torch.matmul(d, h),
        logger=logger
    )
    
    # Initialize h with random values as a column vector
    ista.h.data = torch.ones(n_functions, 1) / n_functions
    
    # Run optimization
    for _ in range(100):
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
        evaluation_func=lambda d, h: torch.matmul(d, h),
        logger=logger
    )
    
    # Test gradient computation
    ista.h.data = torch.tensor([[0.5]])
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