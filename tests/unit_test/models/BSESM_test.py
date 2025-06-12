'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica

Tests for BSESM Base Class

Provides tests for the batched version of SESM

Authors: The SESM Team 

License: 
'''
import pytest
import torch
import numpy as np
import logging
from unittest.mock import MagicMock, patch

from pysesm.models.BSESM import BSESM, BSESMConfig
from pysesm.blocks.UniformPartitionManager import UniformPartitionConfig, UniformPartitionManager
from pysesm.dictionaries import GaussianDictConfig
from pysesm.sparse_coding import ISTAConfig, ISTALayer
from pysesm.device_manager.DeviceManager import DeviceManager
from pysesm.enums.DeviceTargetEnum import DeviceTarget
from pysesm.blocks.PartitionBlock import PartitionBlock # For mocking blocks

# --- Logger y Fixtures ---
logger = logging.getLogger("test_bsesm")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

@pytest.fixture(scope="module")
def device_manager_fixture():
    # Adjust devices as needed for testing
    device_map = {
        DeviceTarget.GLOBAL: "cpu",
        DeviceTarget.SPARSE_CODING_LAYER: "cpu",
        DeviceTarget.DICTIONARY_LAYER: "cpu",
        DeviceTarget.PARTITION_MANAGER: "cpu"
    }
    return DeviceManager(logger=logger, default_device="cpu", device_map=device_map)

@pytest.fixture
def bsesm_config_fixture():
    # Common configuration for BSESM tests
    n_features = 2
    n_functions = 5

    sparse_coding_config = ISTAConfig(
        epochs=1, lambd=0.01, n_functions=n_functions
    )
    dict_config = GaussianDictConfig(
        epochs=1, alpha=0.1, eig_range=[0.1, 1.0], mu_range=[-1.0, 1.0]
    )
    partition_config = UniformPartitionConfig(
        T=torch.tensor([2, 2], dtype=torch.int),
        initial_bounds=np.array([[-2.0, -2.0], [2.0, 2.0]], dtype=np.float32),
        activity_threshold=0
    )

    return BSESMConfig(
        n_features=n_features,
        model_epochs=1,
        sparse_coding_config=sparse_coding_config,
        dict_config=dict_config,
        partition_config=partition_config,
        seed=42
    )

@pytest.fixture
def sample_bsesm_model(bsesm_config_fixture, device_manager_fixture):
    # Create a BSESM instance
    return BSESM(config=bsesm_config_fixture, logger=logger, device_manager=device_manager_fixture)

# --- Mocking components for isolated testing ---

class MockPartitionBlock(PartitionBlock):
    """A mock PartitionBlock to control X, y, target, and h values."""
    def __init__(self, space_origin, block_index, block_size, device, amplitude=1.0):
        super().__init__(space_origin, block_index, block_size, device)
        self._mock_normalized_X = None
        self._mock_target = None
        self._mock_h = None
        self.amplitude = amplitude # Allow setting amplitude for denormalization tests

    def set_mock_data(self, normalized_X, target, h=None):
        self._mock_normalized_X = normalized_X.to(self.device)
        self._mock_target = target.to(self.device)
        if h is not None:
            self._mock_h = h.to(self.device)
            # Simulate sparse_coding_layer being initialized for prediction
            self.sparse_coding_layer = MagicMock(spec=ISTALayer)
            self.sparse_coding_layer.h = torch.nn.Parameter(self._mock_h)
            self.sparse_coding_layer.device = self.device # Ensure device is set on mock
        else:
            self.sparse_coding_layer = None

    @property
    def normalized_X(self):
        return self._mock_normalized_X

    @property
    def target(self):
        return self._mock_target

    # Override is_active to always be true if mock data is set
    def is_active(self, threshold=0):
        return self._mock_normalized_X is not None and self._mock_normalized_X.shape[0] > 0

    # Mock new_point, append_points, clear_points if they interfere
    def new_point(self, *args): pass
    def append_points(self, *args): pass
    def clear_points(self):
        self._mock_normalized_X = None
        self._mock_target = None
        self._mock_h = None
        self.sparse_coding_layer = None

# --- Tests for BSESM Class ---

def test_bsesm_initialization(sample_bsesm_model):
    """Verify core components are initialized correctly."""
    model = sample_bsesm_model
    assert isinstance(model, BSESM)
    assert hasattr(model, 'global_sparse_coding_layer')
    assert isinstance(model.global_sparse_coding_layer, ISTALayer) # Assuming ISTA is default for ISTAConfig
    assert model.global_sparse_coding_layer.evaluation_func is model._global_evaluation_func
    assert model.global_sparse_coding_layer.device == model.device_manager.get_device(DeviceTarget.SPARSE_CODING_LAYER)
    assert model.dictionary_layer is not None
    assert model.partition_manager is not None
    assert model.n_features == model.config.n_features
    assert model.n_functions == model.config.sparse_coding_config.n_functions

def test_bsesm_aggregate_block_data_padding(sample_bsesm_model, device_manager_fixture):
    """
    Test _aggregate_block_data for correct padding and concatenation
    when blocks have different numbers of points.
    """
    model = sample_bsesm_model
    device = device_manager_fixture.get_device(DeviceTarget.GLOBAL)
    n_features = model.n_features
    n_functions = model.n_functions

    # Create mock blocks with varying number of points
    block00 = MockPartitionBlock(torch.tensor([0.0,0.0]), (0,0), torch.tensor([1.0,1.0]), device)
    block00.set_mock_data(
        normalized_X=torch.tensor([[0.1, 0.1], [0.2, 0.2]], dtype=torch.float32),
        target=torch.tensor([[0.5], [0.6]], dtype=torch.float32)
    )

    block01 = MockPartitionBlock(torch.tensor([0.0,1.0]), (0,1), torch.tensor([1.0,1.0]), device)
    block01.set_mock_data(
        normalized_X=torch.tensor([[0.7, 0.7]], dtype=torch.float32), # Only 1 point
        target=torch.tensor([[0.8]], dtype=torch.float32)
    )

    block10 = MockPartitionBlock(torch.tensor([1.0,0.0]), (1,0), torch.tensor([1.0,1.0]), device)
    block10.set_mock_data(
        normalized_X=torch.tensor([[0.3, 0.3], [0.4, 0.4], [0.5, 0.5]], dtype=torch.float32), # 3 points
        target=torch.tensor([[0.9], [1.0], [1.1]], dtype=torch.float32)
    )
    
    blocks = [block00, block01, block10]

    X_batch_normalized, y_batch_target, max_points_in_block = model._aggregate_block_data(blocks)

    # Expected max_points_in_block should be 3
    assert max_points_in_block == 3

    # Expected shapes: (num_blocks * max_points_in_block, n_features)
    assert X_batch_normalized.shape == (3 * 3, n_features)
    assert y_batch_target.shape == (3 * 3, 1)

    # Verify content and padding
    # Block 00: [[0.1, 0.1], [0.2, 0.2], [0.0, 0.0]] (padded)
    assert torch.allclose(X_batch_normalized[0:2], block00.normalized_X)
    assert torch.allclose(X_batch_normalized[2], torch.zeros(n_features, device=device))
    assert torch.allclose(y_batch_target[0:2], block00.target)
    assert torch.allclose(y_batch_target[2], torch.zeros(1, device=device))

    # Block 01: [[0.7, 0.7], [0.0, 0.0], [0.0, 0.0]] (padded)
    assert torch.allclose(X_batch_normalized[3], block01.normalized_X)
    assert torch.allclose(X_batch_normalized[4:6], torch.zeros(2, n_features, device=device))
    assert torch.allclose(y_batch_target[3], block01.target)
    assert torch.allclose(y_batch_target[4:6], torch.zeros(2, 1, device=device))

    # Block 10: No padding needed
    assert torch.allclose(X_batch_normalized[6:9], block10.normalized_X)
    assert torch.allclose(y_batch_target[6:9], block10.target)

def test_bsesm_global_evaluation_func(sample_bsesm_model, device_manager_fixture):
    """
    Test _global_evaluation_func for correct batched matrix multiplication
    and reshaping.
    """
    model = sample_bsesm_model
    device = device_manager_fixture.get_device(DeviceTarget.GLOBAL)

    num_blocks = 3
    max_points_in_block = 4 # Example padding size
    n_functions = model.n_functions
    n_features = model.n_features # Not directly used in eval_func, but implied by D's setup

    # Create mock dictionary (D) and h_batch
    # D is (N_total_points, N_functions)
    D_total_points = num_blocks * max_points_in_block
    mock_dictionary = torch.randn(D_total_points, n_functions, device=device, dtype=torch.float32)
    
    # h_batch is (N_blocks, N_functions, 1)
    mock_h_batch = torch.randn(num_blocks, n_functions, 1, device=device, dtype=torch.float32)

    # Expected output: (N_total_points, 1)
    # Manually compute expected result by splitting D and multiplying with h_batch
    expected_y_pred_list = []
    for i in range(num_blocks):
        # Extract D slice for this block
        block_D_slice = mock_dictionary[i * max_points_in_block : (i + 1) * max_points_in_block]
        # Extract h for this block
        block_h = mock_h_batch[i] # Shape (N_functions, 1)
        # Perform matmul
        expected_y_pred_list.append(torch.matmul(block_D_slice, block_h))
    
    expected_y_pred = torch.cat(expected_y_pred_list, dim=0)

    # Call the method under test
    y_pred_actual = model._global_evaluation_func(mock_dictionary, mock_h_batch)

    assert y_pred_actual.shape == expected_y_pred.shape
    assert torch.allclose(y_pred_actual, expected_y_pred, atol=1e-6)


@patch('pysesm.blocks.UniformPartitionManager.UniformPartitionManager.add_points')
@patch('pysesm.blocks.UniformPartitionManager.UniformPartitionManager.init_sparse_coding_per_block')
@patch('pysesm.blocks.UniformPartitionManager.UniformPartitionManager.retrieve_active_blocks')
@patch('pysesm.models.BSESM.BSESM._aggregate_block_data')
@patch('pysesm.sparse_coding.ISTALayer.ISTALayer.setup') # Patch for global_sparse_coding_layer.setup
@patch('pysesm.models.SESM.SESM._train_step') # Patch parent's _train_step
def test_bsesm_partial_fit_workflow(
    mock_train_step,
    mock_global_sc_setup,
    mock_aggregate_block_data,
    mock_retrieve_active_blocks,
    mock_init_sc_per_block,
    mock_add_points,
    sample_bsesm_model, bsesm_config_fixture, device_manager_fixture
):
    """
    Test the entire partial_fit workflow by mocking dependencies.
    Focuses on orchestration and data flow.
    """
    model = sample_bsesm_model
    device = device_manager_fixture.get_device(DeviceTarget.GLOBAL)
    n_features = model.n_features
    n_functions = model.n_functions
    model_epochs = bsesm_config_fixture.model_epochs

    # Mock data for inputs to partial_fit
    X_input = torch.randn(10, n_features, device=device, dtype=torch.float32)
    y_input = torch.randn(10, 1, device=device, dtype=torch.float32)

    # --- Mock retrieve_active_blocks and aggregate_block_data outputs ---
    mock_active_blocks = []
    mock_h_data_list = []
    # Simulate 2 active blocks
    for i in range(2):
        mock_block = MagicMock(spec=PartitionBlock)
        mock_block.sparse_coding_layer = MagicMock(spec=ISTALayer)
        # Mock h attribute of sparse_coding_layer as a Parameter
        mock_h_tensor = torch.randn(n_functions, 1, device=device, dtype=torch.float32)
        mock_block.sparse_coding_layer.h = torch.nn.Parameter(mock_h_tensor)
        mock_block.sparse_coding_layer.device = device
        mock_block.block_index = (i, 0) # For logging clarity
        mock_active_blocks.append(mock_block)
        mock_h_data_list.append(mock_h_tensor)

    mock_retrieve_active_blocks.return_value = mock_active_blocks

    mock_X_batch_normalized = torch.randn(20, n_features, device=device, dtype=torch.float32) # (2 blocks * 10 samples)
    mock_y_batch_target = torch.randn(20, 1, device=device, dtype=torch.float32)
    mock_max_points_in_block = 10
    mock_aggregate_block_data.return_value = (mock_X_batch_normalized, mock_y_batch_target, mock_max_points_in_block)

    # --- Mock global_sparse_coding_layer.h to track updates ---
    # The actual Parameter object of global_sparse_coding_layer.h will be updated by _train_step
    # We need to mock its internal .data and return the updated data for distribution
    initial_global_h_tensor = torch.randn(2, n_functions, 1, device=device, dtype=torch.float32)
    model.global_sparse_coding_layer.h = torch.nn.Parameter(initial_global_h_tensor)
    
    # Simulate _train_step updating global_h
    # For a real test, _train_step would perform optimization. Here, we simulate an update.
    # The parent's _train_step (mock_train_step) takes X, y, sparsecoding.
    # It will call sparsecoding.partial_fit, which will update sparsecoding.h.
    # We need to ensure global_sparse_coding_layer.h gets updated.
    
    # We mock _train_step, so we need to simulate its effect on global_sparse_coding_layer.h
    def simulate_train_step_update(X_arg, y_arg, sparsecoding_arg):
        # Simulate that sparsecoding_arg.h is updated
        # It's a torch.nn.Parameter, so its .data attribute can be directly modified.
        sparsecoding_arg.h.data.add_(0.1) # Simple simulated update
        # Also simulate loss appending
        sparsecoding_arg.losses.append(0.5)
        model.dictionary_layer.losses.append(0.3) # Simulate dictionary layer loss

    mock_train_step.side_effect = simulate_train_step_update


    # --- Call partial_fit ---
    model.partial_fit(X_input, y_input)

    # --- Assertions ---
    # 1. Partition manager methods called
    mock_add_points.assert_called_once_with(X_input, y_input)
    mock_init_sc_per_block.assert_called_once()
    mock_retrieve_active_blocks.assert_called_once()
    mock_aggregate_block_data.assert_called_once_with(mock_active_blocks)

    # 2. global_sparse_coding_layer setup with correct h_batch
    expected_h_batch_initial = torch.stack(mock_h_data_list)
    mock_global_sc_setup.assert_called_once()
    # Check setup was called with the correct stacked tensor (by comparing args)
    # The actual tensor passed to setup is a clone, so compare values
    actual_setup_h = mock_global_sc_setup.call_args[0][0]
    assert torch.allclose(actual_setup_h, expected_h_batch_initial)
    
    # 3. _train_step called correct number of times and with correct args
    assert mock_train_step.call_count == model_epochs
    for i in range(model_epochs):
        call_args = mock_train_step.call_args_list[i]
        assert torch.allclose(call_args.kwargs['X'], mock_X_batch_normalized)
        assert torch.allclose(call_args.kwargs['y'], mock_y_batch_target)
        assert call_args.kwargs['sparsecoding'] is model.global_sparse_coding_layer

    # 4. Learned h_batch distributed back to individual blocks
    updated_global_h_data = model.global_sparse_coding_layer.h.data
    for i, mock_block in enumerate(mock_active_blocks):
        assert torch.allclose(mock_block.sparse_coding_layer.h.data, updated_global_h_data[i])


@patch('pysesm.blocks.UniformPartitionManager.UniformPartitionManager.retrieve_test_active_blocks')
@patch('pysesm.models.BSESM.BSESM._aggregate_block_data')
@patch('pysesm.dictionaries.GaussianDictLayer.GaussianDictLayer.forward') # Patch dictionary_layer.forward
@patch('pysesm.models.BSESM.BSESM._global_evaluation_func') # Patch _global_evaluation_func
def test_bsesm_predict_workflow_denormalization(
    mock_global_eval_func,
    mock_dict_forward,
    mock_aggregate_block_data,
    mock_retrieve_test_active_blocks,
    sample_bsesm_model, device_manager_fixture
):
    """
    Test the entire predict workflow, focusing on data aggregation, global prediction,
    and correct denormalization and mapping back to original positions.
    """
    model = sample_bsesm_model
    device = device_manager_fixture.get_device(DeviceTarget.GLOBAL)
    n_features = model.n_features
    n_functions = model.n_functions

    # --- Mock inputs to predict ---
    X_test_input = torch.randn(5, n_features, device=device, dtype=torch.float32)
    y_test_input = torch.randn(5, 1, device=device, dtype=torch.float32) # Original y for positions

    # --- Mock retrieve_test_active_blocks output ---
    # Simulate 2 test blocks, with different numbers of points and amplitudes
    # Total points will be 5, mapping to original positions 0-4
    block1 = MockPartitionBlock(torch.tensor([0.0,0.0]), (0,0), torch.tensor([1.0,1.0]), device, amplitude=0.5) # y_pred / 0.5 = y_unnorm
    block1.set_mock_data(
        normalized_X=torch.tensor([[0.1, 0.1], [0.2, 0.2]], dtype=torch.float32),
        target=torch.tensor([[0.5], [0.6]], dtype=torch.float32),
        h=torch.randn(n_functions, 1, device=device) # Mock h for this block
    )
    block1.positions = [0, 2] # Original positions in X_test_input for these points

    block2 = MockPartitionBlock(torch.tensor([1.0,1.0]), (1,1), torch.tensor([1.0,1.0]), device, amplitude=1.0) # y_pred / 1.0 = y_unnorm
    block2.set_mock_data(
        normalized_X=torch.tensor([[0.7, 0.7], [0.8, 0.8], [0.9, 0.9]], dtype=torch.float32), # 3 points
        target=torch.tensor([[0.8], [0.9], [1.0]], dtype=torch.float32),
        h=torch.randn(n_functions, 1, device=device) # Mock h for this block
    )
    block2.positions = [1, 3, 4] # Original positions in X_test_input for these points
    
    mock_retrieve_test_active_blocks.return_value = [block1, block2]

    # --- Mock _aggregate_block_data output ---
    # This will assume max_points_in_block = 3 (from block2)
    # Total aggregated points: 2 blocks * 3 max_points = 6 total points
    mock_X_aggregated = torch.randn(6, n_features, device=device, dtype=torch.float32)
    mock_y_aggregated = torch.randn(6, 1, device=device, dtype=torch.float32)
    mock_max_points_in_block = 3
    mock_aggregate_block_data.return_value = (mock_X_aggregated, mock_y_aggregated, mock_max_points_in_block)

    # --- Mock dictionary_layer.forward output ---
    # Expected output shape: (total_aggregated_points, n_functions)
    mock_evaluated_D = torch.randn(6, n_functions, device=device, dtype=torch.float32)
    mock_dict_forward.return_value = mock_evaluated_D

    # --- Mock _global_evaluation_func output ---
    # This function takes mock_evaluated_D and h_predict_batch and returns (total_aggregated_points, 1)
    # Simulate raw predictions for each point (before denormalization)
    raw_normalized_preds = torch.tensor([
        [1.0], [1.2], [0.0], # Block 1 (2 actual points, 1 padded)
        [0.8], [0.9], [1.0]  # Block 2 (3 actual points, 0 padded)
    ], device=device, dtype=torch.float32)
    mock_global_eval_func.return_value = raw_normalized_preds

    # --- Expected final predictions after denormalization and re-mapping ---
    # Block 1: points at pos 0, 2
    #   normalized_preds: [1.0], [1.2]
    #   amplitude: 0.5
    #   unnormalized_preds: [1.0/0.5], [1.2/0.5] = [2.0], [2.4]
    # Block 2: points at pos 1, 3, 4
    #   normalized_preds: [0.8], [0.9], [1.0]
    #   amplitude: 1.0
    #   unnormalized_preds: [0.8/1.0], [0.9/1.0], [1.0/1.0] = [0.8], [0.9], [1.0]

    # Expected order in final output y_final_predictions (based on original positions):
    # pos 0: 2.0 (from block1)
    # pos 1: 0.8 (from block2)
    # pos 2: 2.4 (from block1)
    # pos 3: 0.9 (from block2)
    # pos 4: 1.0 (from block2)
    expected_final_preds = torch.tensor([
        [2.0], [0.8], [2.4], [0.9], [1.0]
    ], device=device, dtype=torch.float32)


    # --- Call predict ---
    y_predicted_actual = model.predict(X_test_input, y_test_input)

    # --- Assertions ---
    # 1. retrieve_test_active_blocks called
    mock_retrieve_test_active_blocks.assert_called_once_with(X_test_input, y_test_input)

    # 2. _aggregate_block_data called with test blocks
    mock_aggregate_block_data.assert_called_once_with([block1, block2])
    
    # 3. dictionary_layer.forward called with aggregated X_test
    mock_dict_forward.assert_called_once_with(mock_X_aggregated)

    # 4. _global_evaluation_func called with correct D and h_batch
    # Reconstruct expected h_predict_batch
    expected_h_predict_batch = torch.stack([
        block1.sparse_coding_layer.h.to(device),
        block2.sparse_coding_layer.h.to(device)
    ])
    mock_global_eval_func.assert_called_once_with(mock_evaluated_D, expected_h_predict_batch)

    # 5. Final predictions match expected after denormalization and re-mapping
    assert y_predicted_actual.shape == expected_final_preds.shape
    assert torch.allclose(y_predicted_actual, expected_final_preds, atol=1e-6)
