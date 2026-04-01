"""
BSESM Model Tests.

Integration tests for the Batched Sparse-Encoded Surrogate Model, verifying
nested tensor aggregation, global training steps, and multi-device execution.

Copyright (c) 2023-2025, Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
Version: 0.1.3
"""

import logging
from unittest import mock
from unittest.mock import MagicMock, patch


import pytest
import torch
import numpy as np

from pysesm.models.BSESM import BSESM, BSESMConfig, BSESMSolverStrategy
from pysesm.blocks.UniformPartitionManager import UniformPartitionConfig, UniformPartitionManager
from pysesm.dictionaries import GaussianDictConfig,GaussianDictLayer # Will use real GaussianDictLayer
from pysesm.sparse_coding import ISTAConfig, ISTALayer
from pysesm.blocks.PartitionBlock import PartitionBlock
from pysesm.base_types import TensorBatch, TensorProxy


# --- Logger and Fixtures ---
logger = logging.getLogger("test_bsesm")
logger.setLevel(logging.DEBUG) # Set to INFO or DEBUG to see detailed logs during tests
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

@pytest.fixture(scope="module")
def _device_fixture():
    # Configure all layers to run on CPU for consistent testing environment
    return "cpu"

@pytest.fixture(scope="module")
def _common_evaluation_func():
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
        # Check if the inputs are nested tensors and handle empty cases for nested_tensor.as_nested_tensor
        if (isinstance(dictionary, torch.Tensor) and getattr(dictionary, "is_nested", False)
              and isinstance(h, torch.Tensor) and getattr(h, "is_nested", False)):
            results = [_perform_matmul_and_shape_output(d_s, h_s)
                       for d_s, h_s in zip(dictionary.unbind(), h.unbind())]
            # Handle case where results list might be empty (e.g., if input NestedTensor contained only empty tensors)
            if results:
                return torch.nested.as_nested_tensor(results,
                                                      layout=dictionary.layout,
                                                      device=dictionary.device,
                                                      dtype=results[0].dtype)
            else:
                # Return a NestedTensor containing a single empty tensor, mirroring PyTorch's behavior for empty inputs
                # This requires knowing output shape, which is (0, N_functions_per_block, 1) or (0, 1) if already combined by matmul
                # For `evaluation_func` in `SurrogateFunction`, output is (N_samples, N_functions).
                # So if input is empty, output should be empty (0, N_functions).
                # This is tricky for nested tensors, but for now, matching the `_aggregate_block_data`'s empty h shape output for consistency.
                return torch.nested.nested_tensor([torch.empty(0, h.size(-2), device=dictionary.device, dtype=dictionary.dtype)], layout=dictionary.layout, device=dictionary.device)
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

@pytest.fixture(params=[BSESMSolverStrategy.SEQUENTIAL, BSESMSolverStrategy.MEGA_MATRIX])
def _bsesm_config_fixture(request, _device_fixture):
    # Common configuration for BSESM tests
    n_features = 2
    n_functions = 5

    sparse_coding_config = ISTAConfig(
        epochs=3, lambd=0.01, n_functions=n_functions,device=_device_fixture # Reduced epochs for faster tests
    )
    dict_config = GaussianDictConfig(
        epochs=3, alpha=0.1, eig_range=[0.1, 1.0], mu_range=[-1.0, 1.0],device=_device_fixture # Reduced epochs
    )
    partition_config = UniformPartitionConfig(
        T=torch.tensor([2, 2], dtype=torch.int),
        initial_bounds=np.array([[-2.0, -2.0], [2.0, 2.0]], dtype=np.float32),
        activity_threshold=0,
        overlap_ratio=0.0, # No overlap for simpler block boundaries in tests
        device=_device_fixture
    )
    return BSESMConfig(
        n_features=n_features,
        model_epochs=2, # Main SESM training loop iterations
        sparse_coding_config=sparse_coding_config,
        dict_config=dict_config,
        partition_config=partition_config,
        seed=42,
        log_interval=1,
        solver_strategy=request.param,
        device=_device_fixture
    )

@pytest.fixture
def _sample_bsesm_model(_bsesm_config_fixture): # Removed common_evaluation_func here, it's passed in nested_tensor_data_generator
    # Create a BSESM instance, ensuring evaluation_func is passed correctly
    model = BSESM(
        config=_bsesm_config_fixture,
        logger=logger,
    )
    return model

@pytest.fixture
def _nested_tensor_data_generator(_bsesm_config_fixture, _common_evaluation_func):
    """
    Generates active blocks and corresponding NestedTensor data for tests.
    Returns: (X_nested, y_nested, h_nested, active_blocks)
    The sparse_coding_layer within each PartitionBlock is now a real ISTALayer instance.
    """
    def _generator(num_blocks: int, min_samples_per_block: int, max_samples_per_block: int, random_seed: int = 42):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        device = _bsesm_config_fixture.device        
        n_features = _bsesm_config_fixture.n_features
        # n_functions = _bsesm_config_fixture.sparse_coding_config.n_functions

        active_blocks = []
        X_list, y_list, h_list = [], [], []

        dummy_space_origin = torch.tensor([-2.0, -2.0], device=device)
        dummy_block_size = torch.tensor([2.0, 2.0], device=device) # For 2x2 partition, each block is 2x2

        for i in range(num_blocks):
            num_samples = torch.randint(min_samples_per_block, max_samples_per_block + 1, (1,)).item()
            
            block_idx = (i, 0)
            pos = dummy_space_origin + torch.tensor(block_idx, device=device) * dummy_block_size
            block_scope = torch.stack((pos, pos + dummy_block_size))
            
            block = PartitionBlock(block_index=block_idx, block_size=dummy_block_size, block_scope=block_scope, device=device, space_origin=dummy_space_origin)
            block.normalized_X = TensorProxy(torch.randn(num_samples, n_features, device=device, dtype=torch.float32))
            block.target = TensorProxy(torch.randn(num_samples, 1, device=device, dtype=torch.float32))
            block.amplitude = 1.0 # Simple amplitude for testing
            block.positions = list(range(i * min_samples_per_block, i * min_samples_per_block + num_samples)) # Dummy positions

            # Create a real ISTALayer instance for the block
            sparse_coding_layer_instance = ISTALayer(
                config=_bsesm_config_fixture.sparse_coding_config,
                evaluation_func=_common_evaluation_func,
                logger=logging.getLogger(f"test_istalayer_block_{i}")
            )
            # ISTALayer's __init__ calls setup() which initializes h as a Parameter
            block.sparse_coding_layer = sparse_coding_layer_instance
            
            active_blocks.append(block)
            X_list.append(block.normalized_X.get_for_device(device))
            y_list.append(block.target.get_for_device(device))
            h_list.append(block.sparse_coding_layer.h.detach().clone()) # This will now be a real tensor

        # Convert to NestedTensors if lists are not empty
        X_nested = torch.nested.nested_tensor(X_list, layout=torch.jagged, device=device) if X_list else torch.nested.nested_tensor([], layout=torch.jagged, device=device)
        y_nested = torch.nested.nested_tensor(y_list, layout=torch.jagged, device=device) if y_list else torch.nested.nested_tensor([], layout=torch.jagged, device=device)
        h_nested = torch.nested.nested_tensor(h_list, layout=torch.jagged, device=device) if h_list else torch.nested.nested_tensor([], layout=torch.jagged, device=device)

        return X_nested, y_nested, h_nested, active_blocks
    return _generator

# --- Tests for BSESM Class ---

def test_bsesm_initialization(_sample_bsesm_model, _common_evaluation_func):
    """Verify core components are initialized correctly."""
    model = _sample_bsesm_model
    assert isinstance(model, BSESM)
    if model.config.solver_strategy == BSESMSolverStrategy.MEGA_MATRIX:
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
def test_bsesm_partial_fit_model_epochs_loop(
    mock_retrieve_active_blocks,
    mock_init_sc_per_block,
    mock_add_points,
    mock_global_train_step,
    _sample_bsesm_model, _bsesm_config_fixture, _nested_tensor_data_generator
):
    """
    Test that partial_fit orchestrates the main model_epochs loop and calls
    _global_train_step the correct number of times.
    """
    model = _sample_bsesm_model
    model_epochs = _bsesm_config_fixture.model_epochs
    n_features = _bsesm_config_fixture.n_features
    _device = _bsesm_config_fixture.device
    
    X_input = torch.randn(10, n_features, device=_device)
    y_input = torch.randn(10, 1, device=_device)


    # Mock only the block retrieval; the aggregation will be real.
    X_nested, y_nested, h_nested, active_blocks = _nested_tensor_data_generator(2, 5, 10)

    mock_retrieve_active_blocks.return_value = active_blocks


    # Configure mock_global_train_step to return a valid h_nested (e.g., the input one)
    # We need a dummy h_nested for the return value, let's create one.
    _, _, dummy_h_nested, _ = _nested_tensor_data_generator(2, 5, 10)
    mock_global_train_step.return_value = dummy_h_nested   

    # Simulate internal loss updates for logging check
    model.dictionary_layer.losses = [0.5, 0.4, 0.3] * model_epochs # Ensure enough entries
    model._sparse_coding_losses = [0.6, 0.5, 0.4] * model_epochs # Ensure enough entries

    initial_training_time = model.training_time
    initial_partial_fit_count = model.partial_fit_count

    # Call partial_fit
    model.partial_fit(X_input, y_input)

    # Assertions
    mock_add_points.assert_called_once_with(X_input, y_input)
    # The common_evaluation_func needs to be passed to init_sparse_coding_per_block
    mock_init_sc_per_block.assert_called_once_with(
        config=_bsesm_config_fixture.sparse_coding_config,
        evaluation_func=model.evaluation_func # Pass model's evaluation_func
    )
    mock_retrieve_active_blocks.assert_called_once()

    # Verify _global_train_step is called correct number of times with correct arguments
    assert mock_global_train_step.call_count == model_epochs
    for i in range(model_epochs):
        mock_global_train_step.assert_any_call(mock.ANY, mock.ANY, mock.ANY, i, active_blocks)
    
    # Verify training and partial_fit_count are updated
    assert model.training_time > initial_training_time
    assert model.partial_fit_count == initial_partial_fit_count + 1

    # Verify logging (check info level logs for epoch progress)
    # caplog.records might contain other logs, filter specifically for BSESM epochs
    # Not strictly asserting on specific log messages, but that method was called.
    # For detailed log checks, a separate test with caplog fixture would be better.

    # If sesm_hook is provided, verify it's called
    # For this to work, model.sesm_hook must be a MagicMock if you want to assert its calls
    # For simplicity, we are assuming it might be None or a callable.
    # If the user wants to test the hook calls specifically, they would need to mock it in the test.
    # Here, we just check if it was called if it was set up.
    if model.sesm_hook:
        assert isinstance(model.sesm_hook, MagicMock), "sesm_hook must be a MagicMock for assertion"
        assert model.sesm_hook.call_count == model_epochs
        for i in range(model_epochs):
            call_args = model.sesm_hook.call_args_list[i]
            hook_info = call_args[0][0] # First arg of first call
            assert hook_info['model_epoch'] == i
            assert 'h_mega' in hook_info
            assert 'dictionary_params' in hook_info
            assert 'h_per_block' in hook_info


def test_bsesm_global_train_step_orchestration(
    _sample_bsesm_model,
    _nested_tensor_data_generator, 
    _common_evaluation_func # Pass common_evaluation_func
):
    """
    Test the _global_train_step method's orchestration of dictionary and sparse coding
    training, including data aggregation, mega-matrix construction, and h distribution.
    """
    model = _sample_bsesm_model
    strategy = model.config.solver_strategy
    
    n_functions = model.n_functions
    device = model.config.device

    # Prepare mocked data
    num_blocks = 2
    X_nested, y_nested, h_nested_initial, active_blocks = _nested_tensor_data_generator(num_blocks=num_blocks, min_samples_per_block=5, max_samples_per_block=10)

    if strategy == BSESMSolverStrategy.MEGA_MATRIX:
        # Initialize model.global_sparse_coding_layer.h for the test before the call
        total_h_elements = sum(block.sparse_coding_layer.h.shape[0] for block in active_blocks)
        model.global_sparse_coding_layer.config.n_functions = total_h_elements
        model.global_sparse_coding_layer.setup(torch.randn(total_h_elements, 1, device=device))
        initial_h_mega = model.global_sparse_coding_layer.h.data.clone()
    
    X_nested_proxy = TensorProxy(X_nested)
    y_nested_proxy = TensorProxy(y_nested)

    # Call the method under test
    returned_h_nested = model._global_train_step(X_nested_proxy=X_nested_proxy,
                                                 y_nested_proxy=y_nested_proxy,
                                                 h_nested=h_nested_initial,
                                                 epoch=0,
                                                 active_blocks=active_blocks)
    
    # --- Assertions for sparse coding (mega-matrix) part ---
    # The dictionary layer was real, so let's get its actual output to build our expected D_mega
    assert model.dictionary_layer.dictionary is not None, "Real dictionary layer should have produced a dictionary"
    dict_list_from_real_layer = model.dictionary_layer.dictionary.unbind()

    # We can still verify the inputs to the sparse coding layer by mocking it,
    # but it's more robust to check the outcome.
    expected_Y_mega = torch.cat(y_nested.unbind()) # y_nested is already a NestedTensor
    expected_D_mega = torch.block_diag(*dict_list_from_real_layer)
    
    if strategy == BSESMSolverStrategy.MEGA_MATRIX:
        # Verify that the real ISTALayer ran and produced results
        assert len(model.global_sparse_coding_layer.losses) > 0, "Real ISTA layer should have populated its losses"
        assert not torch.allclose(model.global_sparse_coding_layer.h.data, initial_h_mega), "h should have been updated by the real ISTA layer"
    
        # Verify that the returned h_nested is correctly updated with the h from the real layer
        assert returned_h_nested.is_nested
        assert torch.allclose(returned_h_nested.values(), model.global_sparse_coding_layer.h.data)
    else:
        # Sequential checking
        assert len(model._sparse_coding_losses) > 0
        for i, block in enumerate(active_blocks):
            assert len(block.sparse_coding_layer.losses) > 0
            # Should be synchronized back to h_nested
            assert torch.allclose(returned_h_nested.unbind()[i], block.sparse_coding_layer.h.data)    

def test_bsesm_aggregate_block_data_behavior(_sample_bsesm_model, _nested_tensor_data_generator):
    """
    Test the _aggregate_block_data method to ensure it correctly
    creates NestedTensors for X, y, and h from active blocks.
    """
    model = _sample_bsesm_model
    device = model.config.device
    n_features = model.n_features
    n_functions = model.n_functions

    # Test with multiple blocks, varying sample counts
    num_blocks = 3
    min_samples = 5
    max_samples = 15
    X_nested_expected, y_nested_expected, h_nested_expected, active_blocks = _nested_tensor_data_generator(
        num_blocks=num_blocks, min_samples_per_block=min_samples, max_samples_per_block=max_samples
    )

    X_nested_actual, y_nested_actual, h_nested_actual = model._aggregate_block_data(active_blocks, device=device)

    # Assert X_nested_actual matches X_nested_expected
    assert isinstance(X_nested_actual, torch.Tensor) and X_nested_actual.is_nested
    assert len(X_nested_actual.unbind()) == num_blocks
    for i in range(num_blocks):
        assert torch.allclose(X_nested_actual.unbind()[i], X_nested_expected.unbind()[i])
        assert X_nested_actual.unbind()[i].shape == active_blocks[i].normalized_X.get_for_device(device).shape

    # Assert y_nested_actual matches y_nested_expected
    assert isinstance(y_nested_actual, torch.Tensor) and y_nested_actual.is_nested
    assert len(y_nested_actual.unbind()) == num_blocks
    for i in range(num_blocks):
        assert torch.allclose(y_nested_actual.unbind()[i], y_nested_expected.unbind()[i])
        assert y_nested_actual.unbind()[i].shape == active_blocks[i].target.get_for_device(device).shape

    # Assert h_nested_actual matches h_nested_expected (detached copies of block h's)
    assert isinstance(h_nested_actual, torch.Tensor) and h_nested_actual.is_nested
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
    X_empty_actual, y_empty_actual, h_empty_actual = model._aggregate_block_data([], device=device)
    
    assert X_empty_actual.is_nested # It's still a NestedTensor
    assert len(X_empty_actual.unbind()) == 1 # It unbinds to a list with ONE empty tensor
    assert X_empty_actual.unbind()[0].numel() == 0 # That single tensor is truly empty (0 elements)
    assert X_empty_actual.unbind()[0].shape == (0, n_features) # And has the correct feature dimension
    
    assert y_empty_actual.is_nested
    assert len(y_empty_actual.unbind()) == 1
    assert y_empty_actual.unbind()[0].numel() == 0
    # assert y_empty_nested.unbind()[0].shape == (0, 1) # And has the correct target dimension
    
    assert h_empty_actual.is_nested
    assert len(h_empty_actual.unbind()) == 1
    assert h_empty_actual.unbind()[0].numel() == 0
    # assert h_empty_nested.unbind()[0].shape == (0, n_functions, 1) # And has the correct function and output dimensions

@pytest.mark.filterwarnings("ignore:The PyTorch API of nested tensors.*:UserWarning")
def test_bsesm_predict_workflow(_sample_bsesm_model, _common_evaluation_func):
    """
    Test integrador del flujo de predict sin mocks:
    - Usa PartitionManager real para activar bloques sobre X_test_input.
    - Usa Dictionary real (forward) y evaluation_func real.
    - Ajusta h de cada bloque a un patrón conocido y fija amplitudes a 1.0.
    - Reconstruye y_expected con la misma lógica y compara contra model.predict.
    """
    model = _sample_bsesm_model
    model.evaluation_func = _common_evaluation_func

    device = model.config.device
    n_functions = model.n_functions

    # Datos de prueba: 5 puntos que caen en distintos bloques (según partición 2x2 en [-2,2]^2)
    X_test_input = torch.tensor(
        [
            [-1.5, -1.5],  # block (0,0)
            [ 0.5,  0.5],  # block (1,1)
            [-0.5,  1.5],  # block (0,1)
            [ 1.5, -0.5],  # block (1,0)
            [-1.0, -1.0],  # block (0,0) otra muestra
        ],
        device=device, dtype=torch.float32
    )
    # y no se usa para el cálculo, pero ciertas rutas lo requieren por interfaz
    y_test_input = torch.randn(X_test_input.shape[0], 1, device=device, dtype=torch.float32)

    # 1) Preparar estado: puntos y SC por bloque
    model.partition_manager.add_points(X_test_input, y_test_input)
    model.partition_manager.init_sparse_coding_per_block(
        config=model.config.sparse_coding_config,
        evaluation_func=model.evaluation_func
    )

    # 2) Bloques activos reales para este set de test
    test_active_blocks = model.partition_manager.retrieve_inference_blocks(X_test_input)
    assert isinstance(test_active_blocks, (list, tuple)) and len(test_active_blocks) > 0

    # 3) Fijar h de cada bloque a un valor fácil de verificar y amplitud = 1.0
    #    h_i = constante = (i+j+1) en cada entrada, con shape (n_functions, 1)

    # Fija la amplitud en los bloques originales del manager para que predict() la herede
    for b in model.partition_manager.blocks.flatten():
        if hasattr(b, "amplitude"):
            b.amplitude = 1.0

    for block in test_active_blocks: # Iterate over test_active_blocks here
        i, j = block.block_index
        const_val = float(i + j + 1)
        h_param = torch.nn.Parameter(
            torch.full((n_functions, 1), const_val, device=device, dtype=torch.float32)
        )
        # Asegúrate de que el layer existe
        assert hasattr(block, "sparse_coding_layer") and block.sparse_coding_layer is not None
        block.sparse_coding_layer.h = h_param
        # Evitar efectos de (de)normalización dependientes de amplitud
        if hasattr(block, "amplitude"):
            block.amplitude = 1.0

    # 4) Construir y_expected replicando la lógica de predict, pero por bloque
    y_expected = torch.empty_like(y_test_input)
    for block in test_active_blocks: # Iterate over test_active_blocks here
        X_i = block.normalized_X.get_for_device(model.dictionary_layer.device)  # (N_i, n_features)
        h_i = block.sparse_coding_layer.h  # (n_functions, 1)

        # D_i = Dictionary(X_i) usando la implementación real
        D_i = model.dictionary_layer.forward(X_i)  # (N_i, n_functions)

        # y_i = eval_func(D_i, h_i) -> (N_i, 1)
        y_i = model.evaluation_func(D_i, h_i)

        # Si hubiera otros factores de normalización adicionales, aquí deberían aplicarse.
        # Hemos fijado amplitude=1.0 para neutralizar esa parte.

        # Reubicar en el orden original usando positions
        assert hasattr(block, "positions") and len(block.positions) == X_i.shape[0]
        for local_idx, original_pos in enumerate(block.positions):
            y_expected[original_pos] = y_i[local_idx]

    # 5) Llamar a predict (flujo completo real)
    y_predicted_actual = model.predict(X_test_input)

    # 6) Asserts básicos de forma y dispositivo
    assert isinstance(y_predicted_actual, torch.Tensor)
    assert y_predicted_actual.shape == (X_test_input.shape[0], 1)
    assert str(y_predicted_actual.device) == device
    assert y_predicted_actual.dtype == torch.float32
    assert torch.isfinite(y_predicted_actual).all()

    # 7) Comparación numérica con tolerancia
    #    Usamos tolerancia estándar por posibles pequeñas diferencias numéricas internas.
    assert torch.allclose(y_predicted_actual, y_expected, rtol=1e-4, atol=1e-5), \
        f"Predicción no coincide con lo esperado.\nPred:\n{y_predicted_actual}\nExp:\n{y_expected}"
    
# --- Multi-Device Integration Test ---

# Conditional execution marker for CUDA
CUDA_AVAILABLE = torch.cuda.is_available()

@pytest.fixture(params=[BSESMSolverStrategy.SEQUENTIAL, BSESMSolverStrategy.MEGA_MATRIX])
def _bsesm_multi_device_config_fixture(request): 
    """Provides a BSESMConfig specifically for multi-device testing."""
    n_features = 2
    n_functions = 5
    
    # Configure dictionary for CUDA, sparse coding for CPU
    sparse_coding_config = ISTAConfig(
        epochs=2, lambd=0.01, n_functions=n_functions, device='cpu'
    )
    dict_config = GaussianDictConfig(
        epochs=2, alpha=0.1, eig_range=[0.1, 1.0], mu_range=[-1.0, 1.0], device='cuda:0'
    )
    # Partition manager can be on any device, let's use CPU as default
    partition_config = UniformPartitionConfig(
        T=torch.tensor([2, 2], dtype=torch.int),
        initial_bounds=np.array([[-2.0, -2.0], [2.0, 2.0]], dtype=np.float32),
        device='cpu'
    )
    
    return BSESMConfig(
        n_features=n_features,
        model_epochs=1,
        sparse_coding_config=sparse_coding_config,
        dict_config=dict_config,
        partition_config=partition_config,
        seed=42,
        solver_strategy=request.param
        # Global device is not set, letting components use their specific configs
    )

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available, skipping multi-device test")
def test_bsesm_multi_device_execution_and_predict(_bsesm_multi_device_config_fixture):
    """
    Integration test for BSESM with dictionary on GPU and sparse coding on CPU.
    Verifies that `partial_fit` and `predict` run without device mismatch errors.
    """
    # 1. Setup the model with the multi-device configuration
    config = _bsesm_multi_device_config_fixture
    model = BSESM(config=config, logger=logger)

    # Verify initial device placement
    assert str(model.dictionary_layer.device) == 'cuda:0'
    if model.config.solver_strategy == BSESMSolverStrategy.MEGA_MATRIX:
        assert str(model.global_sparse_coding_layer.device) == 'cpu'
    assert str(model.partition_manager.device) == 'cpu'
    
    # 2. Prepare input data on the CPU (a common scenario)
    n_samples = 20
    X_train = torch.randn(n_samples, config.n_features, device='cpu')
    y_train = torch.randn(n_samples, 1, device='cpu')

    # 3. Execute `partial_fit`
    # This is the main stress test. If it completes without a device mismatch error,
    # the core data transfers (proxies, .to(device)) are working.
    try:
        model.partial_fit(X_train, y_train)
    except Exception as e:
        pytest.fail(f"model.partial_fit failed with a device-related error: {e}")

    # 4. Verify post-training state
    # Check that parameters remain on their designated devices
    assert str(model.dictionary_layer.theta_params.device) == 'cuda:0'
    if model.config.solver_strategy == BSESMSolverStrategy.MEGA_MATRIX:
        assert str(model.global_sparse_coding_layer.h.device) == 'cpu'

    # Retrieve an active block and check its internal sparse coding layer's device
    active_blocks = model.partition_manager.retrieve_active_blocks()
    assert len(active_blocks) > 0
    # The SC layer inside the block is created by the partition manager's hook,
    # and its config specifies 'cpu'.
    assert str(active_blocks[0].sparse_coding_layer.device) == 'cpu'
    assert str(active_blocks[0].sparse_coding_layer.h.device) == 'cpu'
    
    # 5. Execute `predict`
    X_test = torch.randn(10, config.n_features, device='cpu')
    y_pred = None
    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        pytest.fail(f"model.predict failed with a device-related error: {e}")

    # 6. Verify prediction output
    assert y_pred is not None
    assert y_pred.shape == (X_test.shape[0], 1)
    # The output should be on the same device as the input tensor X_test
    assert str(y_pred.device) == str(X_test.device)
    assert torch.isfinite(y_pred).all()
    logger.info("Multi-device test completed successfully.")
    
if __name__ == "__main__":
    # Ensure pytest instructions are printed if run directly
    try:
        from pytest_helper import print_pytest_instructions
        print_pytest_instructions()
    except ImportError:
        print("Please run this file using pytest. Example: pytest unit_tests/models/BSESM_test.py")
