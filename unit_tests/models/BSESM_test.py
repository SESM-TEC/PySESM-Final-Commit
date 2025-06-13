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
    # CORRECCION: Comparar las funciones en sí, no la identidad de los objetos
    assert model.global_sparse_coding_layer.evaluation_func == model._global_evaluation_func 
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
    # n_functions = model.n_functions # Not directly used in this test function scope

    # CORRECCION: Crear instancias REALES de PartitionBlock
    # Necesitamos un space_origin y un block_size consistentes con el PartitionBlock.
    # Usaremos valores dummy para que los bloques puedan inicializarse.
    dummy_space_origin = torch.tensor([-2.0, -2.0], device=device)
    dummy_block_size = torch.tensor([2.0, 2.0], device=device) # Example size, doesn't affect padding logic directly

    block00 = PartitionBlock(dummy_space_origin, (0,0), dummy_block_size, device)
    # Poblar el bloque con datos simulados y llamar a sus métodos para que calculen normalized_X y target
    block00.new_point(torch.tensor([-1.9, -1.9], dtype=torch.float32, device=device), torch.tensor([0.5], dtype=torch.float32, device=device), 0)
    block00.new_point(torch.tensor([-1.8, -1.8], dtype=torch.float32, device=device), torch.tensor([0.6], dtype=torch.float32, device=device), 1)
    block00.normalize_points()
    block00.calculate_amplitude_and_target() # Esto establece normalized_X y target

    block01 = PartitionBlock(dummy_space_origin, (0,1), dummy_block_size, device)
    block01.new_point(torch.tensor([-1.0, 0.5], dtype=torch.float32, device=device), torch.tensor([0.8], dtype=torch.float32, device=device), 2) # Only 1 point
    block01.normalize_points()
    block01.calculate_amplitude_and_target()

    block10 = PartitionBlock(dummy_space_origin, (1,0), dummy_block_size, device)
    block10.new_point(torch.tensor([0.1, -1.5], dtype=torch.float32, device=device), torch.tensor([0.9], dtype=torch.float32, device=device), 3)
    block10.new_point(torch.tensor([0.2, -1.4], dtype=torch.float32, device=device), torch.tensor([1.0], dtype=torch.float32, device=device), 4)
    block10.new_point(torch.tensor([0.3, -1.3], dtype=torch.float32, device=device), torch.tensor([1.1], dtype=torch.float32, device=device), 5) # 3 points
    block10.normalize_points()
    block10.calculate_amplitude_and_target()
    
    blocks = [block00, block01, block10]

    X_batch_normalized, y_batch_target, max_points_in_block = model._aggregate_block_data(blocks)

    # Expected max_points_in_block should be 3
    assert max_points_in_block == 3

    # Expected shapes: (num_blocks * max_points_in_block, n_features)
    assert X_batch_normalized.shape == (3 * 3, n_features)
    assert y_batch_target.shape == (3 * 3, 1) # Assuming output_dim=1 from fixture

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
    # n_features = model.n_features # Not directly used in eval_func, but implied by D's setup

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

    # --- CORRECCION: Usar instancias REALES de PartitionBlock ---
    # Necesitamos un space_origin y un block_size consistentes.
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
    model.global_sparse_coding_layer.losses = [] # Initialize losses for the actual global SC layer
    
    # Simulate _train_step updating global_h
    # CORRECCION: Aceptar argumentos por nombre en side_effect
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
        assert torch.allclose(call_args.kwargs['X'], mock_X_batch_normalized)
        assert torch.allclose(call_args.kwargs['y'], mock_y_batch_target)
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

    # --- CORRECCION: Usar instancias REALES de PartitionBlock para los bloques de prueba ---
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
    
    # CORRECCION: Comparar call_args manualmente para tensores
    mock_global_eval_func.assert_called_once() # First, assert it was called once
    actual_D_arg, actual_h_batch_arg = mock_global_eval_func.call_args[0]
    assert torch.allclose(actual_D_arg, mock_evaluated_D), "Mocked D argument mismatch"
    assert torch.allclose(actual_h_batch_arg, expected_h_predict_batch), "Expected h_predict_batch argument mismatch"


    # 5. Final predictions match expected after denormalization and re-mapping
    assert y_predicted_actual.shape == expected_final_preds.shape
    assert torch.allclose(y_predicted_actual, expected_final_preds, atol=1e-6)
