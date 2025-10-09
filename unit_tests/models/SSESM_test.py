# unit_tests/models/SSESM_test.py

import logging
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
import torch
import numpy as np

from pysesm.models.SSESM import SSESM, SSESMConfig
from pysesm.blocks.UniformPartitionManager import UniformPartitionConfig, UniformPartitionManager
from pysesm.dictionaries import GaussianDictConfig
from pysesm.sparse_coding import ISTAConfig, ISTALayer
from pysesm.blocks.PartitionBlock import PartitionBlock
from pysesm.base_types import TensorProxy

# --- Logger and Fixtures ---
logger = logging.getLogger("test_ssesm")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

@pytest.fixture(scope="module")
def _device_fixture():
    return "cpu"

@pytest.fixture(scope="module")
def _common_evaluation_func():
    """Provides a simple matrix multiplication evaluation function for SSESM tests."""
    return torch.matmul

@pytest.fixture
def _ssesm_config_fixture(_device_fixture):
    """Provides a common configuration for SSESM tests."""
    n_features = 2
    n_functions = 5

    sparse_coding_config = ISTAConfig(
        epochs=3, lambd=0.01, n_functions=n_functions, device=_device_fixture
    )
    dict_config = GaussianDictConfig(
        epochs=3, alpha=0.1, eig_range=[0.1, 1.0], mu_range=[-1.0, 1.0], device=_device_fixture
    )
    partition_config = UniformPartitionConfig(
        T=torch.tensor([2, 2], dtype=torch.int),
        initial_bounds=np.array([[-2.0, -2.0], [2.0, 2.0]], dtype=np.float32),
        device=_device_fixture
    )
    return SSESMConfig(
        n_features=n_features,
        model_epochs=2,
        permutation_times=2,
        sparse_coding_config=sparse_coding_config,
        dict_config=dict_config,
        partition_config=partition_config,
        seed=42,
        device=_device_fixture
    )

@pytest.fixture
def _sample_ssesm_model(_ssesm_config_fixture):
    """Creates a sample SSESM model instance for testing."""
    return SSESM(config=_ssesm_config_fixture, logger=logger)

@pytest.fixture
def _active_blocks_generator(_ssesm_config_fixture, _common_evaluation_func):
    """Factory to generate a list of active PartitionBlocks with real data and SC layers."""
    def _generator(num_blocks: int, samples_per_block: int, random_seed: int = 42):
        torch.manual_seed(random_seed)
        device = _ssesm_config_fixture.device
        n_features = _ssesm_config_fixture.n_features
        
        active_blocks = []
        for i in range(num_blocks):
            block = PartitionBlock(
                space_origin=torch.tensor([0.0, 0.0], device=device),
                block_index=(0, i),
                block_size=torch.tensor([1.0, 1.0], device=device),
                device=device
            )
            block.normalized_X = TensorProxy(torch.randn(samples_per_block, n_features, device=device))
            block.target = TensorProxy(torch.randn(samples_per_block, 1, device=device))
            block.sparse_coding_layer = ISTALayer(
                config=_ssesm_config_fixture.sparse_coding_config,
                evaluation_func=_common_evaluation_func,
                logger=logger
            )
            active_blocks.append(block)
        return active_blocks
    return _generator

# --- Tests for SSESM Class ---

def test_ssesm_initialization(_sample_ssesm_model):
    """Verify that the SSESM model and its components are initialized correctly."""
    model = _sample_ssesm_model
    assert isinstance(model, SSESM)
    assert model.dictionary_layer is not None
    assert model.partition_manager is not None
    assert model.sparse_coding_layer is None # Should be None at the model level for SSESM
    assert model.permutation_times == model.config.permutation_times

@patch('pysesm.models.SESM.SESM._train_block')
@patch.object(UniformPartitionManager, 'add_points')
@patch.object(UniformPartitionManager, 'init_sparse_coding_per_block')
@patch.object(UniformPartitionManager, 'retrieve_active_blocks')
def test_ssesm_partial_fit_orchestration(
    mock_retrieve_active_blocks,
    mock_init_sc_per_block,
    mock_add_points,
    mock_train_block,
    _sample_ssesm_model,
    _active_blocks_generator
):
    """
    Test the high-level orchestration of `partial_fit` in SSESM.
    Verifies that it retrieves blocks and calls `_train_block` for each block and permutation.
    """
    model = _sample_ssesm_model
    num_blocks = 3
    permutation_times = model.config.permutation_times
    
    # Setup mocks
    mock_active_blocks = _active_blocks_generator(num_blocks, 10)
    mock_retrieve_active_blocks.return_value = mock_active_blocks
    
    X_train = torch.randn(10, model.n_features, device=model.config.device)
    y_train = torch.randn(10, 1, device=model.config.device)

    # Execute
    model.partial_fit(X_train, y_train)

    # Assertions
    mock_add_points.assert_called_once_with(X_train, y_train)
    mock_init_sc_per_block.assert_called_once()
    mock_retrieve_active_blocks.assert_called_once()
    
    # Core logic check: _train_block should be called for each block, for each permutation
    assert mock_train_block.call_count == num_blocks * permutation_times

def test_ssesm_predict_workflow(_sample_ssesm_model, _common_evaluation_func):
    """
    Full integration test of the `predict` workflow without mocks.
    """
    model = _sample_ssesm_model
    device = model.config.device
    n_functions = model.n_functions

    X_test_input = torch.tensor([
        [-1.5, -1.5], [0.5, 1.5], [1.0, -1.0]
    ], device=device)
    y_test_input = torch.randn(X_test_input.shape[0], 1, device=device)

    # 1. Setup model state
    model.partition_manager.add_points(X_test_input, y_test_input)
    model.partition_manager.init_sparse_coding_per_block(
        config=model.config.sparse_coding_config,
        evaluation_func=_common_evaluation_func
    )

    # 2. Get the original TRAINING blocks and modify their state. This simulates a "learned" state.
    training_blocks = model.partition_manager.retrieve_active_blocks()
    assert len(training_blocks) > 0

    for block in training_blocks:   
        block.amplitude = 2.0  # Use a non-trivial amplitude
        # Set h to a predictable value based on block index
        if block.is_active(): # Only modify blocks that were activated by the training data
            block.sparse_coding_layer.h.data.fill_(float(block.block_index[0] + block.block_index[1] + 1))


        
    # 3. Manually calculate the expected prediction
    y_expected = torch.zeros_like(y_test_input)
    active_blocks_for_expected = model.partition_manager.retrieve_inference_blocks(X_test_input)
    with torch.no_grad():
        for block in active_blocks_for_expected:
            X_i = block.normalized_X.get_for_device(model.dictionary_layer.device)
            h_i = block.sparse_coding_layer.h
            
            D_i = model.dictionary_layer.forward(X_i)
            y_pred_normalized = _common_evaluation_func(D_i, h_i)
            y_pred_unnormalized = y_pred_normalized / block.amplitude

            for local_idx, original_pos in enumerate(block.positions):
                y_expected[original_pos] = y_pred_unnormalized[local_idx]
    
    # 4. Get the model's prediction
    y_predicted = model.predict(X_test_input)

    # 5. Assert results match
    assert y_predicted.shape == y_expected.shape
    assert str(y_predicted.device) == device
    assert torch.allclose(y_predicted, y_expected, rtol=1e-5)

# --- Multi-Device Integration Test ---

CUDA_AVAILABLE = torch.cuda.is_available()

@pytest.fixture
def _ssesm_multi_device_config_fixture():
    """Provides a multi-device configuration for SSESM."""
    n_features = 2
    n_functions = 5
    
    dict_config = GaussianDictConfig(epochs=2, alpha=0.1, device='cuda:0')
    sparse_coding_config = ISTAConfig(epochs=2, n_functions=n_functions, device='cpu')
    partition_config = UniformPartitionConfig(
        T=2,
        initial_bounds=np.array([[-2.0, -2.0], [2.0, 2.0]], dtype=np.float32),
        device='cpu'
    )
    
    return SSESMConfig(
        n_features=n_features,
        model_epochs=1,
        permutation_times=1,
        sparse_coding_config=sparse_coding_config,
        dict_config=dict_config,
        partition_config=partition_config,
        seed=42
    )

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available, skipping multi-device test")
def test_ssesm_multi_device_execution_and_predict(_ssesm_multi_device_config_fixture):
    """
    Integration test for SSESM with dictionary on GPU and sparse coding on CPU.
    """
    # 1. Setup model with multi-device config
    config = _ssesm_multi_device_config_fixture
    model = SSESM(config=config, logger=logger)
    
    # Verify initial device placement
    assert str(model.dictionary_layer.device) == 'cuda:0'
    assert str(model.partition_manager.device) == 'cpu'

    # 2. Prepare input data on CPU
    X_train = torch.randn(20, config.n_features, device='cpu')
    y_train = torch.randn(20, 1, device='cpu')
    
    # 3. Execute `partial_fit` and check for errors
    try:
        model.partial_fit(X_train, y_train)
    except Exception as e:
        pytest.fail(f"model.partial_fit failed with a device-related error: {e}")

    # 4. Verify post-training state
    assert str(model.dictionary_layer.theta_params.device) == 'cuda:0'
    active_blocks = model.partition_manager.retrieve_active_blocks()
    assert len(active_blocks) > 0
    # The SC layer within each block should be on the CPU as per its config
    assert str(active_blocks[0].sparse_coding_layer.device) == 'cpu'
    assert str(active_blocks[0].sparse_coding_layer.h.device) == 'cpu'

    # 5. Execute `predict` and check for errors
    X_test = torch.randn(10, config.n_features, device='cpu')
    y_pred = None
    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        pytest.fail(f"model.predict failed with a device-related error: {e}")

    # 6. Verify prediction output
    assert y_pred is not None
    assert y_pred.shape == (X_test.shape[0], 1)
    assert str(y_pred.device) == str(X_test.device) # Output should be on input's device
    assert torch.isfinite(y_pred).all()
    logger.info("SSESM Multi-device test completed successfully.")

if __name__ == "__main__":
    # Ensure pytest instructions are printed if run directly
    try:
        from pytest_helper import print_pytest_instructions
        print_pytest_instructions()
    except ImportError:
        print("Please run this file using pytest. Example: pytest unit_tests/models/SSESM_test.py")
