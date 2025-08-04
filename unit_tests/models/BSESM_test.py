# unit_tests/models/BSESM_test.py

import pytest
import torch
import numpy as np
import logging
from unittest.mock import MagicMock, patch, call
from typing import Optional, List, Tuple, Union

from pysesm.models.BSESM import BSESM, BSESMConfig
from pysesm.blocks.UniformPartitionManager import UniformPartitionConfig, UniformPartitionManager
from pysesm.dictionaries import GaussianDictConfig,GaussianDictLayer
from pysesm.sparse_coding import ISTAConfig, ISTALayer
from pysesm.device_manager.DeviceManager import DeviceManager
from pysesm.enums.DeviceTargetEnum import DeviceTarget
from pysesm.blocks.PartitionBlock import PartitionBlock
from pysesm.base_types import TensorBatch


# --- Logger and Fixtures ---
logger = logging.getLogger("test_bsesm")
logger.setLevel(logging.DEBUG) # Set to INFO or DEBUG to see detailed logs during tests
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

@pytest.fixture(scope="module")
def device_manager_fixture():
    # Configure all layers to run on CPU for consistent testing environment
    device_map = {
        DeviceTarget.GLOBAL: "cpu",
        DeviceTarget.SPARSE_CODING_LAYER: "cpu",
        DeviceTarget.DICTIONARY_LAYER: "cpu",
        DeviceTarget.PARTITION_MANAGER: "cpu"
    }
    return DeviceManager(logger=logging.getLogger("test_device_manager_fixture"), default_device="cpu", device_map=device_map)

@pytest.fixture(scope="module")
def common_evaluation_func():
    """
    Provides a standard evaluation function (D @ h) that handles TensorBatch.
    Ensures h is a column vector and output is consistently shaped.
    """
    def _ensure_h_column_vector(h_input: torch.Tensor) -> torch.Tensor:
        if h_input.dim() == 1:
            return h_input.unsqueeze(-1)
        return h_input

    def _perform_matmul_and_shape_output(d: torch.Tensor, h_val: torch.Tensor) -> torch.Tensor:
        res = torch.matmul(d, _ensure_h_column_vector(h_val))
        if res.dim() == 1: # If matmul results in (N_samples,)
            return res.unsqueeze(-1) # Make it (N_samples, 1)
        return res

    def _eval_func_impl(dictionary: TensorBatch, h: TensorBatch) -> TensorBatch:
        if (isinstance(dictionary, torch.Tensor) and getattr(dictionary, "is_nested", False)
              and isinstance(h, torch.Tensor) and getattr(h, "is_nested", False)): # NestedTensor
            results = [_perform_matmul_and_shape_output(d_s, h_s)
                       for d_s, h_s in zip(dictionary.unbind(), h.unbind())]
            return torch.nested.as_nested_tensor(results,
                                                  layout=dictionary.layout,
                                                  device=dictionary.device,
                                                  dtype=results[0].dtype)
        elif isinstance(dictionary, torch.Tensor) and dictionary.dim() <= 2: # Single 2D tensor
            return _perform_matmul_and_shape_output(dictionary, h)
        elif isinstance(dictionary, torch.Tensor) and dictionary.dim() == 3: # 3D tensor
            return torch.vmap(_perform_matmul_and_shape_output)(dictionary, h)
        elif isinstance(dictionary, list) and isinstance(h, list): # List of tensors
            results = [_perform_matmul_and_shape_output(d_s, h_s)
                       for d_s, h_s in zip(dictionary, h)]
            return results
        else:
            raise TypeError("Unsupported TensorBatch types for evaluation_func: "
                            f"D={type(dictionary)}, h={type(h)}")
    return _eval_func_impl

@pytest.fixture
def bsesm_config_fixture():
    # Common configuration for BSESM tests
    n_features = 2
    n_functions = 5

    sparse_coding_config = ISTAConfig(
        epochs=3, lambd=0.01, n_functions=n_functions # Reduced epochs for faster tests
    )
    dict_config = GaussianDictConfig(
        epochs=3, alpha=0.1, eig_range=[0.1, 1.0], mu_range=[-1.0, 1.0] # Reduced epochs
    )
    partition_config = UniformPartitionConfig(
        T=torch.tensor([2, 2], dtype=torch.int),
        initial_bounds=np.array([[-2.0, -2.0], [2.0, 2.0]], dtype=np.float32),
        activity_threshold=0,
        overlap_ratio=0.0 # No overlap for simpler block boundaries in tests
    )
    return BSESMConfig(
        n_features=n_features,
        model_epochs=2, # Main SESM training loop iterations
        sparse_coding_config=sparse_coding_config,
        dict_config=dict_config,
        partition_config=partition_config,
        seed=42,
        log_interval=1
    )

@pytest.fixture
def sample_bsesm_model(bsesm_config_fixture, device_manager_fixture, common_evaluation_func):
    # Create a BSESM instance, ensuring evaluation_func is passed correctly
    model = BSESM(
        config=bsesm_config_fixture,
        logger=logger,
        device_manager=device_manager_fixture,
    )
    return model

@pytest.fixture
def nested_tensor_data_generator(device_manager_fixture, bsesm_config_fixture):
    """
    Generates mock active blocks and corresponding NestedTensor data for tests.
    Returns: (X_nested, y_nested, h_nested, active_blocks)
    """
    def _generator(num_blocks: int, min_samples_per_block: int, max_samples_per_block: int, random_seed: int = 42):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        device = device_manager_fixture.get_device(DeviceTarget.GLOBAL)
        n_features = bsesm_config_fixture.n_features
        n_functions = bsesm_config_fixture.sparse_coding_config.n_functions

        mock_active_blocks = []
        X_list, y_list, h_list = [], [], []

        dummy_space_origin = torch.tensor([-2.0, -2.0], device=device)
        dummy_block_size = torch.tensor([2.0, 2.0], device=device) # For 2x2 partition, each block is 2x2

        for i in range(num_blocks):
            num_samples = torch.randint(min_samples_per_block, max_samples_per_block + 1, (1,)).item()
            
            block = PartitionBlock(dummy_space_origin, (i, 0), dummy_block_size, device) # Simple (i,0) indices
            # Populate block with simulated data
            block.normalized_X = torch.randn(num_samples, n_features, device=device, dtype=torch.float32)
            block.target = torch.randn(num_samples, 1, device=device, dtype=torch.float32)
            block.amplitude = 1.0 # Simple amplitude for testing
            block.positions = list(range(i * min_samples_per_block, i * min_samples_per_block + num_samples)) # Dummy positions

            # Attach a mock sparse_coding_layer with a Parameter 'h'
            mock_sc_layer = MagicMock(spec=ISTALayer)
            # Ensure the 'h' attribute is a real Parameter object as expected by code base
            mock_sc_layer.h = torch.nn.Parameter(torch.randn(n_functions, 1, device=device, dtype=torch.float32))
            mock_sc_layer.device = device # Important for device assertions
            mock_sc_layer.losses = [] # Mock losses list
            mock_sc_layer.config = bsesm_config_fixture.sparse_coding_config # Mock config
            block.sparse_coding_layer = mock_sc_layer
            
            mock_active_blocks.append(block)
            X_list.append(block.normalized_X)
            y_list.append(block.target)
            h_list.append(block.sparse_coding_layer.h.detach().clone()) # Detach h for dictionary training

        # Convert to NestedTensors if lists are not empty
        X_nested = torch.nested.nested_tensor(X_list, layout=torch.jagged, device=device) if X_list else torch.nested.nested_tensor([], layout=torch.jagged, device=device)
        y_nested = torch.nested.nested_tensor(y_list, layout=torch.jagged, device=device) if y_list else torch.nested.nested_tensor([], layout=torch.jagged, device=device)
        h_nested = torch.nested.nested_tensor(h_list, layout=torch.jagged, device=device) if h_list else torch.nested.nested_tensor([], layout=torch.jagged, device=device)

        return X_nested, y_nested, h_nested, mock_active_blocks
    return _generator


# --- Tests for BSESM Class ---

def test_bsesm_initialization(sample_bsesm_model, common_evaluation_func):
    """Verify core components are initialized correctly."""
    model = sample_bsesm_model
    assert isinstance(model, BSESM)
    assert hasattr(model, 'global_sparse_coding_layer')
    assert isinstance(model.global_sparse_coding_layer, ISTALayer)
    assert model.dictionary_layer is not None
    assert model.partition_manager is not None
    assert model.n_features == model.config.n_features
    assert model.n_functions == model.config.sparse_coding_config.n_functions
    

@patch.object(BSESM, '_global_train_step')
@patch.object(UniformPartitionManager, 'add_points')
@patch.object(UniformPartitionManager, 'init_sparse_coding_per_block')
@patch.object(UniformPartitionManager, 'retrieve_active_blocks')
@patch.object(BSESM, '_aggregate_block_data')
def test_bsesm_partial_fit_model_epochs_loop(
    mock_aggregate_block_data,
    mock_retrieve_active_blocks,
    mock_init_sc_per_block,
    mock_add_points,
    mock_global_train_step,
    sample_bsesm_model, bsesm_config_fixture, nested_tensor_data_generator,
    device_manager_fixture
):
    """
    Test that partial_fit orchestrates the main model_epochs loop and calls
    _global_train_step the correct number of times.
    """
    model = sample_bsesm_model
    model_epochs = bsesm_config_fixture.model_epochs
    n_features = bsesm_config_fixture.n_features

    X_input = torch.randn(10, n_features, device=device_manager_fixture.get_device(DeviceTarget.GLOBAL))
    y_input = torch.randn(10, 1, device=device_manager_fixture.get_device(DeviceTarget.GLOBAL))

    # Mock return values for internal steps
    X_nested, y_nested, h_nested, active_blocks = nested_tensor_data_generator(num_blocks=2, min_samples_per_block=5, max_samples_per_block=10)
    mock_retrieve_active_blocks.return_value = active_blocks
    mock_aggregate_block_data.return_value = (X_nested, y_nested, h_nested)
    
    # Simulate internal loss updates for logging check
    model.dictionary_layer.losses = [0.5, 0.4, 0.3] * model_epochs # Ensure enough entries
    model.global_sparse_coding_layer.losses = [0.6, 0.5, 0.4] * model_epochs # Ensure enough entries

    initial_elapsed_time = model.elapsed_time
    initial_partial_fit_count = model.partial_fit_count

    # Call partial_fit
    model.partial_fit(X_input, y_input)

    # Assertions
    mock_add_points.assert_called_once_with(X_input, y_input)
    mock_init_sc_per_block.assert_called_once()
    mock_retrieve_active_blocks.assert_called_once()
    mock_aggregate_block_data.assert_called_once_with(active_blocks)

    # Verify _global_train_step is called correct number of times with correct arguments
    assert mock_global_train_step.call_count == model_epochs
    for i in range(model_epochs):
        mock_global_train_step.assert_any_call(X_nested, y_nested, h_nested, active_blocks, i)
    
    # Verify elapsed_time and partial_fit_count are updated
    assert model.elapsed_time > initial_elapsed_time
    assert model.partial_fit_count == initial_partial_fit_count + 1

    # Verify logging (check info level logs for epoch progress)
    # caplog.records might contain other logs, filter specifically for BSESM epochs
    # Not strictly asserting on specific log messages, but that method was called.
    # For detailed log checks, a separate test with caplog fixture would be better.

    # If sesm_hook is provided, verify it's called
    if model.sesm_hook:
        assert model.sesm_hook.call_count == model_epochs
        for i in range(model_epochs):
            call_args = model.sesm_hook.call_args_list[i]
            hook_info = call_args[0][0] # First arg of first call
            assert hook_info['model_epoch'] == i
            assert 'h_mega' in hook_info
            assert 'dictionary_params' in hook_info
            assert 'h_per_block' in hook_info


@patch.object(BSESM, 'evaluation_func') # Patch BSESM's own evaluation_func
@patch.object(GaussianDictLayer, 'forward') # Patch dictionary_layer.forward
@patch.object(GaussianDictLayer, 'partial_fit') # Patch dictionary_layer.partial_fit
@patch.object(ISTALayer, 'setup') # Patch global_sparse_coding_layer.setup
@patch.object(ISTALayer, 'partial_fit') # Patch global_sparse_coding_layer.partial_fit
def test_bsesm_global_train_step_orchestration(
    mock_global_sc_partial_fit,
    mock_global_sc_setup,
    mock_dict_partial_fit,
    mock_dict_forward,
    mock_eval_func_bsesm, # This is BSESM.evaluation_func
    sample_bsesm_model, nested_tensor_data_generator, device_manager_fixture
):
    """
    Test the _global_train_step method's orchestration of dictionary and sparse coding
    training, including data aggregation, mega-matrix construction, and h distribution.
    """
    model = sample_bsesm_model
    n_functions = model.n_functions
    device = device_manager_fixture.get_device(DeviceTarget.GLOBAL)

    # Prepare mocked data
    num_blocks = 2
    X_nested, y_nested, h_nested_initial, active_blocks = nested_tensor_data_generator(num_blocks=num_blocks, min_samples_per_block=5, max_samples_per_block=10)
    
    # Simulate dictionary_layer.forward output (NestedTensor of D_i matrices)
    mock_dict_evaluated_list = [torch.randn(block.normalized_X.shape[0], n_functions, device=device) for block in active_blocks]
    mock_dict_forward.return_value = torch.nested.nested_tensor(mock_dict_evaluated_list, layout=torch.jagged, device=device)

    # Simulate global_sparse_coding_layer.partial_fit output (optimized H_mega)
    total_h_elements = sum(block.sparse_coding_layer.h.shape[0] for block in active_blocks)
    mock_optimized_h_mega = torch.randn(total_h_elements, 1, device=device)
    mock_global_sc_partial_fit.return_value = None # partial_fit returns None
    
    # We need to manually set the h.data on the mocked global_sparse_coding_layer for the split to work
    model.global_sparse_coding_layer.h.data = mock_optimized_h_mega
    model.global_sparse_coding_layer.losses = [0.1, 0.2, 0.3] # Add some dummy losses

    # Simulate dictionary layer updating its own losses
    model.dictionary_layer.losses = [0.4, 0.5, 0.6]

    # Call the method under test
    model._global_train_step(X_nested, y_nested, h_nested_initial, active_blocks, current_epoch=0)

    # Assertions for dictionary training
    mock_dict_partial_fit.assert_called_once_with(X=X_nested, y=y_nested, h=h_nested_initial)

    # Assertions for sparse coding (mega-matrix) part
    mock_dict_forward.assert_called_once_with(X_nested) # Ensure dictionary is evaluated for SC
    
    # Construct expected Y_mega and D_mega
    expected_Y_mega = torch.cat(y_nested.unbind()) # y_nested is already a NestedTensor
    expected_D_mega = torch.block_diag(*mock_dict_evaluated_list) # Use the mocked list from dict_forward

    mock_global_sc_setup.assert_called_once()
    # Verify setup received the concatenated initial h
    assert torch.allclose(mock_global_sc_setup.call_args[0][0], h_nested_initial.values())
    
    mock_global_sc_partial_fit.assert_called_once()
    # Extract call arguments and assert tensor equality explicitly
    call_args = mock_global_sc_partial_fit.call_args
    actual_y_arg = call_args.kwargs['y']
    actual_dictionary_arg = call_args.kwargs['dictionary']
    assert torch.allclose(actual_y_arg, expected_Y_mega), "Y_mega passed to global SC partial_fit is incorrect"
    assert torch.allclose(actual_dictionary_arg, expected_D_mega), "D_mega passed to global SC partial_fit is incorrect"
    
    # Verify h distribution back to individual blocks
    current_h_split_sizes = [block.sparse_coding_layer.h.shape[0] for block in active_blocks]
    split_mock_optimized_h_mega = torch.split(mock_optimized_h_mega, current_h_split_sizes, dim=0)

    for i, block in enumerate(active_blocks):
        assert torch.allclose(block.sparse_coding_layer.h.data, split_mock_optimized_h_mega[i]), \
            f"H mismatch for block {i}"

    # Verify that the mocked layers' internal loss lists were updated
    assert len(model.dictionary_layer.losses) > 0, "Dictionary layer losses should be populated"
    assert len(model.global_sparse_coding_layer.losses) > 0, "Global SC layer losses should be populated"

    # Ensure BSESM's own evaluation_func is NOT called by _global_train_step directly
    mock_eval_func_bsesm.assert_not_called()

def test_bsesm_aggregate_block_data_behavior(sample_bsesm_model, nested_tensor_data_generator):
    """
    Test the _aggregate_block_data method to ensure it correctly
    creates NestedTensors for X, y, and h from active blocks.
    """
    model = sample_bsesm_model
    device = model.device_manager.get_device(DeviceTarget.GLOBAL)
    n_features = model.n_features
    n_functions = model.n_functions

    # Test with multiple blocks, varying sample counts
    num_blocks = 3
    min_samples = 5
    max_samples = 15
    X_nested_expected, y_nested_expected, h_nested_expected, active_blocks = nested_tensor_data_generator(
        num_blocks=num_blocks, min_samples_per_block=min_samples, max_samples_per_block=max_samples
    )

    X_nested_actual, y_nested_actual, h_nested_actual = model._aggregate_block_data(active_blocks)

    # Assert X_nested_actual matches X_nested_expected
    assert X_nested_actual.is_nested
    assert len(X_nested_actual.unbind()) == num_blocks
    for i in range(num_blocks):
        assert torch.allclose(X_nested_actual.unbind()[i], X_nested_expected.unbind()[i])
        assert X_nested_actual.unbind()[i].shape == active_blocks[i].normalized_X.shape

    # Assert y_nested_actual matches y_nested_expected
    assert y_nested_actual.is_nested
    assert len(y_nested_actual.unbind()) == num_blocks
    for i in range(num_blocks):
        assert torch.allclose(y_nested_actual.unbind()[i], y_nested_expected.unbind()[i])
        assert y_nested_actual.unbind()[i].shape == active_blocks[i].target.shape

    # Assert h_nested_actual matches h_nested_expected (detached copies of block h's)
    assert h_nested_actual.is_nested
    assert len(h_nested_actual.unbind()) == num_blocks
    for i in range(num_blocks):
        # Check that the h tensors are detached and have correct values
        assert not h_nested_actual.unbind()[i].requires_grad
        assert torch.allclose(h_nested_actual.unbind()[i], h_nested_expected.unbind()[i])
        assert h_nested_actual.unbind()[i].shape == (n_functions, 1)

    # Test with no active blocks
    # When no blocks are active, _aggregate_block_data returns NestedTensors
    # that internally hold a single empty tensor, as per PyTorch's current (and sometimes counter-intuitive) behavior.
    X_empty_nested, y_empty_nested, h_empty_nested = model._aggregate_block_data([])
    
    assert X_empty_nested.is_nested # It's still a NestedTensor
    assert len(X_empty_nested.unbind()) == 1 # It unbinds to a list with ONE empty tensor
    assert X_empty_nested.unbind()[0].numel() == 0 # That single tensor is truly empty (0 elements)
    assert X_empty_nested.unbind()[0].shape == (0, n_features) # And has the correct feature dimension
    
    assert y_empty_nested.is_nested
    assert len(y_empty_nested.unbind()) == 1
    assert y_empty_nested.unbind()[0].numel() == 0
    assert y_empty_nested.unbind()[0].shape == (0, 1) # And has the correct target dimension
    
    assert h_empty_nested.is_nested
    assert len(h_empty_nested.unbind()) == 1
    assert h_empty_nested.unbind()[0].numel() == 0
    assert h_empty_nested.unbind()[0].shape == (0, n_functions, 1) # And has the correct function and output dimensions

@pytest.mark.filterwarnings("ignore:The PyTorch API of nested tensors.*:UserWarning")
@patch.object(UniformPartitionManager, 'retrieve_test_active_blocks')
@patch.object(GaussianDictLayer, 'forward') # Patch dictionary_layer.forward
def test_bsesm_predict_workflow(
    mock_retrieve_test_active_blocks,
    mock_dict_forward,
    sample_bsesm_model, device_manager_fixture,
    common_evaluation_func # Use the actual evaluation_func for correctness
):
    """
    Test the entire predict workflow, focusing on data aggregation for prediction,
    dictionary evaluation, evaluation_func usage, denormalization, and re-mapping.
    """
    model = sample_bsesm_model
    # Ensure model uses the common_evaluation_func explicitly
    model.evaluation_func = common_evaluation_func
    device = device_manager_fixture.get_device(DeviceTarget.GLOBAL)
    n_features = model.n_features
    n_functions = model.n_functions

    # Mock inputs to predict
    X_test_input = torch.randn(5, n_features, device=device, dtype=torch.float32)
    y_test_input = torch.randn(5, 1, device=device, dtype=torch.float32) # Original y for positions

    # --- Prepare mock PartitionBlock instances for testing ---
    # These blocks will be "returned" by mock_retrieve_test_active_blocks
    # Setup properties relevant for prediction: normalized_X, amplitude, sparse_coding_layer.h, positions
    
    # Block 1: 2 samples, amplitude 0.5, positions 0, 2
    block1 = MagicMock(spec=PartitionBlock)
    block1.normalized_X = torch.randn(2, n_features, device=device, dtype=torch.float32)
    block1.amplitude = 0.5
    block1.positions = [0, 2]
    # Ensure block.sparse_coding_layer.h is a torch.nn.Parameter for real behavior
    block1.sparse_coding_layer = MagicMock(spec=ISTALayer)
    block1.sparse_coding_layer.h = torch.nn.Parameter(torch.randn(n_functions, 1, device=device, dtype=torch.float32))
    block1.sparse_coding_layer.device = device # Important for device placement checks

    # Block 2: 3 samples, amplitude 1.0, positions 1, 3, 4
    block2 = MagicMock(spec=PartitionBlock)
    block2.normalized_X = torch.randn(3, n_features, device=device, dtype=torch.float32)
    block2.amplitude = 1.0
    block2.positions = [1, 3, 4]
    block2.sparse_coding_layer = MagicMock(spec=ISTALayer)
    block2.sparse_coding_layer.h = torch.nn.Parameter(torch.randn(n_functions, 1, device=device, dtype=torch.float32))
    block2.sparse_coding_layer.device = device

    mock_retrieve_test_active_blocks.return_value = [block1, block2]

    # --- Mock dictionary_layer.forward output ---
    # It should return a NestedTensor where each sub-tensor matches the X for that block
    mock_dict_eval_for_block1 = torch.randn(block1.normalized_X.shape[0], n_functions, device=device, dtype=torch.float32)
    mock_dict_eval_for_block2 = torch.randn(block2.normalized_X.shape[0], n_functions, device=device, dtype=torch.float32)
    mock_dict_forward.return_value = torch.nested.nested_tensor([mock_dict_eval_for_block1, mock_dict_eval_for_block2], layout=torch.jagged, device=device)

    # --- Simulate predicted *normalized* values for each block from evaluation_func ---
    # These are the values *before* denormalization by amplitude
    # Block1: 2 samples
    mock_normalized_preds_block1 = torch.tensor([[0.5], [0.6]], device=device, dtype=torch.float32) # e.g., predictions if amplitude was applied
    # Block2: 3 samples
    mock_normalized_preds_block2 = torch.tensor([[0.7], [0.8], [0.9]], device=device, dtype=torch.float32)

    # Temporarily replace the *real* common_evaluation_func used by the model with a mock
    # to control its output for this specific prediction sequence.
    # This mock will be called multiple times, once per block.
    # We need to set its side_effect to return the specific predictions for each call.
    original_eval_func = model.evaluation_func
    model.evaluation_func = MagicMock(side_effect=[
        mock_normalized_preds_block1, # First call for block1
        mock_normalized_preds_block2  # Second call for block2
    ])

    # --- Call predict ---
    y_predicted_actual = model.predict(X_test_input, y_test_input)

    # --- Assertions ---
    mock_retrieve_test_active_blocks.assert_called_once_with(X_test_input, y_test_input)

    # Verify dictionary_layer.forward is called with the aggregated X_nested from test blocks
    expected_X_nested_for_dict_forward = torch.nested.nested_tensor([block1.normalized_X, block2.normalized_X], layout=torch.jagged, device=device)
    mock_dict_forward.assert_called_once_with(expected_X_nested_for_dict_forward)

    # Verify model.evaluation_func was called for EACH block individually
    # It should be called with the D-matrix for that block (from mock_dict_forward)
    # and the H-vector from that block's sparse_coding_layer.
    assert model.evaluation_func.call_count == 2
    model.evaluation_func.assert_has_calls([
        call(mock_dict_eval_for_block1, block1.sparse_coding_layer.h),
        call(mock_dict_eval_for_block2, block2.sparse_coding_layer.h)
    ])

    # Calculate expected final predictions after denormalization and re-mapping
    # Denormalized predictions for block1: [0.5/0.5, 0.6/0.5] = [1.0, 1.2]
    # Denormalized predictions for block2: [0.7/1.0, 0.8/1.0, 0.9/1.0] = [0.7, 0.8, 0.9]

    expected_final_preds_list = torch.zeros(y_test_input.shape[0], 1, device=device, dtype=torch.float32)
    # Block 1 positions: 0, 2
    expected_final_preds_list[0] = 1.0
    expected_final_preds_list[2] = 1.2
    # Block 2 positions: 1, 3, 4
    expected_final_preds_list[1] = 0.7
    expected_final_preds_list[3] = 0.8
    expected_final_preds_list[4] = 0.9

    assert y_predicted_actual.shape == expected_final_preds_list.shape
    assert torch.allclose(y_predicted_actual, expected_final_preds_list, atol=1e-6)

    # Restore original evaluation_func
    model.evaluation_func = original_eval_func

if __name__ == "__main__":
    # Ensure pytest instructions are printed if run directly
    try:
        from pytest_helper import print_pytest_instructions
        print_pytest_instructions()
    except ImportError:
        print("Please run this file using pytest. Example: pytest unit_tests/models/BSESM_test.py")
