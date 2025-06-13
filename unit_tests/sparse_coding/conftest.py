# tests/unit_test/sparse_coding/conftest.py
import pytest
import torch
import logging
import numpy as np

from pysesm.enums.DeviceTargetEnum import DeviceTarget
from pysesm.device_manager.DeviceManager import DeviceManager
from pysesm.sparse_coding.SparseCodingBaseLayer import SparseCodingConfig 

# --- Common Fixtures for Sparse Coding Tests ---

@pytest.fixture(scope="module")
def common_logger():
    """Provides a shared logger instance for sparse coding tests."""
    logger = logging.getLogger('test_sparse_coding_layers')
    logger.setLevel(logging.DEBUG) # Set to INFO or DEBUG to see detailed logs during tests
    if not logger.handlers: # Prevent adding handlers multiple times in pytest
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

@pytest.fixture(scope="module")
def common_device_manager(common_logger):
    """Provides a shared DeviceManager instance for sparse coding tests."""
    # Configure all layers to run on CPU for consistent testing environment
    device_map = {
        DeviceTarget.GLOBAL: "cpu",
        DeviceTarget.SPARSE_CODING_LAYER: "cpu",
        DeviceTarget.DICTIONARY_LAYER: "cpu",
        DeviceTarget.PARTITION_MANAGER: "cpu"
    }
    return DeviceManager(common_logger, default_device="cpu", device_map=device_map)

@pytest.fixture(scope="module")
def common_evaluation_func():
    """Provides a standard evaluation function (matrix multiplication) for D and h."""
    # This function defines how dictionary D and sparse code h combine to form prediction y.
    # For this test, D is (n_samples x n_functions), h is (n_functions x 1), result is (n_samples x 1).
    return lambda d, h: torch.matmul(d, h)

@pytest.fixture
def sparse_coding_data_generator(common_device_manager):
    """
    Generates synthetic data (dictionary D, true h, and target y) for sparse coding tests.
    
    Args:
        n_samples (int): Number of data points.
        n_features (int): Dimensionality of the input space for the dictionary words (not directly used by SC layer, but by Dict layer).
        n_functions (int): Number of dictionary words (columns in D, rows in h).
        sparsity_level (float): Proportion of non-zero elements in the true h (0 to 1).
        noise_level (float): Standard deviation of Gaussian noise added to y.
        
    Returns:
        tuple: (dictionary_D, true_h, target_y)
    """
    def _generator(n_samples: int, n_features: int, n_functions: int, sparsity_level: float = 0.1, noise_level: float = 0.0,
                   random_seed: int = 42):
        # Set seeds for reproducibility of data generation
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        device = common_device_manager.get_device(DeviceTarget.SPARSE_CODING_LAYER)
        
        # Dictionary D (n_samples x n_functions)
        # This is a matrix where each column is a 'word' evaluated at n_samples points.
        # For testing sparse coding, the actual structure of D doesn't matter as much as its dimensions and values.
        dictionary_D = torch.randn(n_samples, n_functions, device=device, dtype=torch.float32)
        
        # Generate true sparse h (n_functions x 1)
        true_h = torch.randn(n_functions, 1, device=device, dtype=torch.float32)
        # Apply sparsity: set a percentage of elements to zero
        num_non_zeros = int(n_functions * sparsity_level)
        if num_non_zeros < n_functions: # Ensure there are zeros to set
            # Randomly select indices to set to zero
            zero_indices = torch.randperm(n_functions, device=device)[num_non_zeros:]
            true_h[zero_indices] = 0.0
        
        # Compute target y = D @ h + noise
        target_y = torch.matmul(dictionary_D, true_h)
        if noise_level > 0:
            target_y += noise_level * torch.randn_like(target_y)
            
        return dictionary_D, true_h, target_y
    return _generator
