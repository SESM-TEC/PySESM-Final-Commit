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
from unittest.mock import MagicMock, patch, call # Import 'call' for custom assertions
from typing import Optional

from pysesm.models.BSESM import BSESM, BSESMConfig
from pysesm.blocks.UniformPartitionManager import UniformPartitionConfig, UniformPartitionManager
from pysesm.dictionaries import GaussianDictConfig
from pysesm.sparse_coding import ISTAConfig, ISTALayer
from pysesm.device_manager.DeviceManager import DeviceManager
from pysesm.enums.DeviceTargetEnum import DeviceTarget
from pysesm.blocks.PartitionBlock import PartitionBlock # Keep PartitionBlock import

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

# --- Tests for BSESM Class ---

def test_bsesm_initialization(sample_bsesm_model):
    """Verify core components are initialized correctly."""
    model = sample_bsesm_model
    assert isinstance(model, BSESM)
    assert hasattr(model, 'global_sparse_coding_layer')
    assert isinstance(model.global_sparse_coding_layer, ISTALayer) # Assuming ISTA is default for ISTAConfig
    assert model.global_sparse_coding_layer.device == model.device_manager.get_device(DeviceTarget.SPARSE_CODING_LAYER)
    assert model.dictionary_layer is not None
    assert model.partition_manager is not None
    assert model.n_features == model.config.n_features
    assert model.n_functions == model.config.sparse_coding_config.n_functions

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

    # --- Usar instancias REALES de PartitionBlock para los mocks ---
    dummy_space_origin = torch.tensor([-2.0, -2.0], device=device)
    dummy_block_size = torch.tensor([2.0, 2.0], device=device)

    mock_active_blocks = []
    mock_h_data_list = []
    # Simulate 2 active blocks
    for i in range(2):
        block = PartitionBlock(dummy_space_origin, (i,0), dummy_block_size, device)
        # Poblar el bloque con datos simulados y llamar a sus métodos
        # El número de puntos por bloque aquí no importa mucho, solo que haya alguno.
        block.new_point(torch.randn(n_features, device=device), torch.randn(1, device=device), i)
        block.normalize_points()
        block.calculate_amplitude_and_target() # Esto establece normalized_X y target

        # Adjuntar un mock de sparse_coding_layer (esto es lo que se mockea del SC Layer)
        mock_sc_layer = MagicMock(spec=ISTALayer)
        mock_h_tensor = torch.randn(n_functions, 1, device=device, dtype=torch.float32)
        mock_sc_layer.h = torch.nn.Parameter(mock_h_tensor) # h como Parameter
        mock_sc_layer.device = device
        mock_sc_layer.losses = [] # Initialize losses for the mock SC layer
        block.sparse_coding_layer = mock_sc_layer
        
        mock_active_blocks.append(block)
        mock_h_data_list.append(mock_h_tensor)

    # Simulate the actual output structure of _aggregate_block_data
    mock_X_list_agg = [block.normalized_X for block in mock_active_blocks]
    mock_y_list_agg = [block.target for block in mock_active_blocks]

    mock_retrieve_active_blocks.return_value = mock_active_blocks
    mock_aggregate_block_data.return_value = (mock_X_list_agg, mock_y_list_agg, mock_h_data_list)       

    # --- Mock global_sparse_coding_layer.h to track updates ---
    # The actual Parameter object of global_sparse_coding_layer.h will be updated by _train_step
    # We need to mock its internal .data and return the updated data for distribution
    initial_global_h_tensor = torch.randn(2, n_functions, 1, device=device, dtype=torch.float32)
    model.global_sparse_coding_layer.h = torch.nn.Parameter(initial_global_h_tensor)
    model.global_sparse_coding_layer.losses = [] # Initialize losses for the actual global SC layer
    
    # Simulate _train_step updating global_h
    # Aceptar argumentos por nombre en side_effect
    def simulate_train_step_update(X, y, sparsecoding):
        sparsecoding.h.data.add_(0.1) # Simple simulated update
        sparsecoding.losses.append(0.5) # Update global SC layer's losses
        # Ensure model.dictionary_layer.losses exists and is updated
        # The actual dictionary_layer.partial_fit will append to its own internal losses list,
        # which then gets copied to model.dictionary_layer_losses via SESM._train_step.
        # For this patch, we directly simulate that copy.
        if not hasattr(model.dictionary_layer, 'losses'): # Check if initialized by mock
            model.dictionary_layer.losses = []
        model.dictionary_layer.losses.append(0.3) 

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
        assert call_args.kwargs['X'] is not None # X is now a NestedTensor
        assert call_args.kwargs['y'] is not None # Y is now a concatenated tensor
        assert call_args.kwargs['sparsecoding'] is model.global_sparse_coding_layer

    # 4. Learned h_batch distributed back to individual blocks
    updated_global_h_data = model.global_sparse_coding_layer.h.data
    for i, block in enumerate(mock_active_blocks): # Iterate over real PartitionBlock instances
        assert torch.allclose(block.sparse_coding_layer.h.data, updated_global_h_data[i])


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

    # --- Usar instancias REALES de PartitionBlock para los bloques de prueba ---
    dummy_space_origin = torch.tensor([-2.0, -2.0], device=device)
    dummy_block_size = torch.tensor([2.0, 2.0], device=device)

    # Simulate 2 test blocks, with different numbers of points and amplitudes
    block1 = PartitionBlock(dummy_space_origin, (0,0), dummy_block_size, device)
    # Poblar block1 manualmente con X, y, positions y luego simular sus propiedades
    block1.new_point(torch.randn(n_features, device=device), torch.tensor([1.0], device=device), 0) # point for pos 0
    block1.new_point(torch.randn(n_features, device=device), torch.tensor([1.2], device=device), 2) # point for pos 2
    block1.normalize_points()
    block1.amplitude = 0.5 # Manually set the learned amplitude for testing denormalization
    # Simular la capa SC y su 'h' aprendida
    mock_sc_layer_h1 = MagicMock(spec=ISTALayer)
    mock_sc_layer_h1.h = torch.nn.Parameter(torch.randn(n_functions, 1, device=device))
    mock_sc_layer_h1.device = device
    block1.sparse_coding_layer = mock_sc_layer_h1

    block2 = PartitionBlock(dummy_space_origin, (1,1), dummy_block_size, device)
    block2.new_point(torch.randn(n_features, device=device), torch.tensor([0.8], device=device), 1) # point for pos 1
    block2.new_point(torch.randn(n_features, device=device), torch.tensor([0.9], device=device), 3) # point for pos 3
    block2.new_point(torch.randn(n_features, device=device), torch.tensor([1.0], device=device), 4) # point for pos 4
    block2.normalize_points()
    block2.amplitude = 1.0 # Manually set the learned amplitude
    mock_sc_layer_h2 = MagicMock(spec=ISTALayer)
    mock_sc_layer_h2.h = torch.nn.Parameter(torch.randn(n_functions, 1, device=device))
    mock_sc_layer_h2.device = device
    block2.sparse_coding_layer = mock_sc_layer_h2
    
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
    #   normalized_preds: [1.0], [1.2] (from raw_normalized_preds)
    #   amplitude: 0.5 (from block1)
    #   unnormalized_preds: [1.0/0.5], [1.2/0.5] = [2.0], [2.4]
    # Block 2: points at pos 1, 3, 4
    #   normalized_preds: [0.8], [0.9], [1.0] (from raw_normalized_preds)
    #   amplitude: 1.0 (from block2)
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
    # Reconstruct expected h_predict_batch from the *real* block's attached SC layer
    expected_h_predict_batch = torch.stack([
        block1.sparse_coding_layer.h.to(device),
        block2.sparse_coding_layer.h.to(device)
    ])
    
    # Comparar call_args manualmente para tensores
    mock_global_eval_func.assert_called_once() # First, assert it was called once
    actual_D_arg, actual_h_batch_arg = mock_global_eval_func.call_args[0]
    assert torch.allclose(actual_D_arg, mock_evaluated_D), "Mocked D argument mismatch"
    assert torch.allclose(actual_h_batch_arg, expected_h_predict_batch), "Expected h_predict_batch argument mismatch"


    # 5. Final predictions match expected after denormalization and re-mapping
    assert y_predicted_actual.shape == expected_final_preds.shape
    assert torch.allclose(y_predicted_actual, expected_final_preds, atol=1e-6)

def test_aggregate_block_data_no_padding(sample_bsesm_model, device_manager_fixture):
    """
    Test the refactored _aggregate_block_data to ensure it returns lists without padding.
    """
    model = sample_bsesm_model
    device = device_manager_fixture.get_device(DeviceTarget.GLOBAL)
    n_functions = model.n_functions # Use n_functions from model fixture
    
    # Create mock blocks with different numbers of points
    block1 = MagicMock(spec=PartitionBlock)
    block1.normalized_X = torch.randn(10, 2, device=device)
    block1.target = torch.randn(10, 1, device=device)
    # Create and assign a mock for sparse_coding_layer first
    mock_sc_layer_b1 = MagicMock(spec=ISTALayer)
    mock_sc_layer_b1.h = torch.nn.Parameter(torch.randn(n_functions, 1, device=device))
    block1.sparse_coding_layer = mock_sc_layer_b1

    block2 = MagicMock(spec=PartitionBlock)
    block2.normalized_X = torch.randn(7, 2, device=device)
    block2.target = torch.randn(7, 1, device=device)
    # Create and assign a mock for sparse_coding_layer for block2
    mock_sc_layer_b2 = MagicMock(spec=ISTALayer)
    mock_sc_layer_b2.h = torch.nn.Parameter(torch.randn(n_functions, 1, device=device))
    block2.sparse_coding_layer = mock_sc_layer_b2

    active_blocks = [block1, block2]
    
    # Note: The _aggregate_block_data method does not return max_points_in_block anymore.
    # It returns X_list, y_list, h_initial_list.
    X_list, y_list, h_list = model._aggregate_block_data(active_blocks)

    # Assertions
    assert isinstance(X_list, list) and isinstance(y_list, list) and isinstance(h_list, list)
    assert len(X_list) == 2
    assert torch.allclose(X_list[0], block1.normalized_X)
    assert torch.allclose(X_list[1], block2.normalized_X)
    assert y_list[0].shape == (10, 1)
    assert y_list[1].shape == (7, 1)
    # Assert that h_list contains the Parameter's data (or the Parameter itself, if that's what's copied)
    # The actual implementation copies the Parameter object, so we check for identity.
    assert h_list[0] is block1.sparse_coding_layer.h
    assert h_list[1] is block2.sparse_coding_layer.h

@patch('torch.block_diag')
def test_partial_fit_mega_matrix_workflow(mock_block_diag, sample_bsesm_model):
    """
    Test the partial_fit workflow, focusing on the mega-matrix orchestration.
    """
    model = sample_bsesm_model
    n_functions = model.n_functions # Get n_functions from fixture
    
    # Mock partition manager and its return values
    model.partition_manager = MagicMock()
    
    block1 = MagicMock(spec=PartitionBlock)
    block1.normalized_X = torch.randn(10, 2)
    block1.target = torch.randn(10, 1)
    # Create and assign mock for sparse_coding_layer
    mock_sc_layer_b1 = MagicMock(spec=ISTALayer)
    mock_sc_layer_b1.h = torch.nn.Parameter(torch.randn(n_functions, 1))
    block1.sparse_coding_layer = mock_sc_layer_b1

    block2 = MagicMock(spec=PartitionBlock)
    block2.normalized_X = torch.randn(7, 2)
    block2.target = torch.randn(7, 1)

    # Create and assign mock for sparse_coding_layer
    mock_sc_layer_b2 = MagicMock(spec=ISTALayer)
    mock_sc_layer_b2.h = torch.nn.Parameter(torch.randn(n_functions, 1))
    block2.sparse_coding_layer = mock_sc_layer_b2
    
    active_blocks = [block1, block2]
    model.partition_manager.retrieve_active_blocks.return_value = active_blocks

    # Mock dictionary layer behavior
    model.dictionary_layer = MagicMock()
    model.dictionary_layer.optimizer = MagicMock() # Mock optimizer
    model.dictionary_layer.criterion = MagicMock(return_value=torch.tensor(0.1)) # Mock criterion
    model.dictionary_layer.losses = [] # Initialize losses list
    dict_list = [torch.randn(10, n_functions), torch.randn(7, n_functions)] # Mocked [D_1, D_2]
    model.dictionary_layer.return_value = dict_list # Mock the __call__ behavior of dictionary_layer

    # Mock global sparse coding layer
    model.global_sparse_coding_layer = MagicMock(spec=ISTALayer) # Use spec for more accurate mocking
    # Ensure h is a Parameter on the global SC layer mock
    model.global_sparse_coding_layer.h = torch.nn.Parameter(torch.randn(17, 1)) # (10+7)*1, total h
    model.global_sparse_coding_layer.losses = [] # Initialize losses list
    model.global_sparse_coding_layer.config.n_functions = 17 # Set n_functions for the global SC layer mock

    # --- Call the function to test ---
    model.partial_fit(torch.randn(17, 2), torch.randn(17, 1))

    # --- Assertions ---
    # 1. Dictionary layer was trained with a NestedTensor (this behavior is now internal to BSESM.partial_fit)
    # We check that dictionary_layer was called as a function (its __call__ method) with a nested tensor.
    model.dictionary_layer.assert_called_once()
    actual_X_nested_passed_to_dict_layer = model.dictionary_layer.call_args[0][0]
    assert getattr(actual_X_nested_passed_to_dict_layer, 'is_nested', False)
    
    # 2. Dictionary's optimizer and criterion used
    model.dictionary_layer.optimizer.zero_grad.assert_called_once()
    # The criterion will be called multiple times in the loop
    assert model.dictionary_layer.criterion.call_count == len(active_blocks)
    model.dictionary_layer.optimizer.step.assert_called_once()
    
    # 3. Mega-matrix D_mega was constructed correctly
    mock_block_diag.assert_called_once_with(*dict_list)
    
    # 4. Global sparse coder was called with mega-tensors
    model.global_sparse_coding_layer.partial_fit.assert_called_once()
    sc_fit_call_args = model.global_sparse_coding_layer.partial_fit.call_args
    Y_mega_arg = sc_fit_call_args.kwargs['y']
    D_mega_arg = sc_fit_call_args.kwargs['dictionary']
    
    # Check Y_mega_arg
    expected_Y_mega = torch.cat([block.target for block in active_blocks])
    assert torch.allclose(Y_mega_arg, expected_Y_mega)
    
    # Check D_mega_arg
    expected_D_mega_from_mock = mock_block_diag.return_value # This is what mock_block_diag returns
    assert torch.allclose(D_mega_arg, expected_D_mega_from_mock)
    
    # 5. Optimized H was correctly unpacked and distributed back to blocks
    # H_mega_optimizado is model.global_sparse_coding_layer.h
    H_mega_optimizado = model.global_sparse_coding_layer.h
    
    # These sizes are based on the mock blocks' `normalized_X` batch sizes, which are 10 and 7.
    # The `h_list` was constructed from the initial h values, which are n_functions (5) each.
    # So the total `H_mega_initial` has shape (2*5, 1) = (10, 1).
    # The `global_sparse_coding_layer.h` will be updated to this total size during setup.
    # The `h_split_sizes` should be based on `n_functions` (5) for each block, not the data samples.
    
    # Recalculate expected split based on n_functions
    n_functions_per_block = n_functions # 5
    split_h_optimizado_list = torch.split(H_mega_optimizado, [n_functions_per_block] * len(active_blocks))

    for i, block in enumerate(active_blocks):
        assert torch.allclose(block.sparse_coding_layer.h.data, split_h_optimizado_list[i])

if __name__ == "__main__":
    from pytest_helper import print_pytest_instructions
    print_pytest_instructions()
