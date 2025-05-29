import pytest
import torch
import logging
import numpy as np
from typing import Union, Optional

# Adjust imports based on your new directory structure
from pysesm.blocks.UniformPartitionManager import UniformPartitionManager, UniformPartitionConfig
from pysesm.blocks.PartitionBlock import PartitionBlock
from pysesm.sparse_coding.ISTALayer import ISTALayer, ISTAConfig # Example sparse coding layer for init_sparse_coding_per_block
from pysesm.enums.DeviceTargetEnum import DeviceTarget
from pysesm.device_manager.DeviceManager import DeviceManager

# Configure logger once at the module level
logger = logging.getLogger("test_uniform_partition_manager")
logger.setLevel(logging.DEBUG) # Set a higher level to see debug/info messages if needed
# If you don't see logs in pytest, you might need to run with `pytest --log-cli-level=INFO`
# or ensure a handler is attached to the logger in your test runner config.

# --- Fixtures ---

@pytest.fixture(scope="module")
def common_device_manager():
    """Provides a shared DeviceManager instance for all tests in this module."""
    device_map = {
        DeviceTarget.GLOBAL: "cpu",
        DeviceTarget.SPARSE_CODING_LAYER: "cpu",
        DeviceTarget.DICTIONARY_LAYER: "cpu",
        DeviceTarget.PARTITION_MANAGER: "cpu" # Assuming TargetDevice is an alias for DeviceTarget
    }
    # Using a unique logger for the DeviceManager fixture to avoid conflicts
    return DeviceManager(logging.getLogger("test_device_manager_fixture"), default_device="cpu", device_map=device_map)

@pytest.fixture
def create_manager(common_device_manager):
    """
    Factory fixture to create UniformPartitionManager instances with flexible config.
    Ensures initial_bounds are consistently passed as numpy arrays to the config.
    """
    def _creator(T_val: Union[int, torch.Tensor], initial_bounds_val: Optional[Union[np.ndarray, torch.Tensor]]=None, threshold_val: float=0):
        # Convert torch.Tensor bounds to numpy array for UniformPartitionConfig
        if isinstance(initial_bounds_val, torch.Tensor):
            initial_bounds_np = initial_bounds_val.cpu().numpy()
        else:
            initial_bounds_np = initial_bounds_val # If already numpy or None

        config = UniformPartitionConfig(
            T=T_val,
            initial_bounds=initial_bounds_np,
            threshold=threshold_val
        )
        return UniformPartitionManager(
            config=config,
            logger=logger, # Use the module-level logger for the manager
            device_manager=common_device_manager
        )
    return _creator

# --- Core Functionality Tests (adapted from your original file) ---

def test_uniform_partition_manager_initialization(create_manager):
    """Test initialization of UniformPartitionManager with explicit bounds."""
    # Use int for T to test conversion to tensor
    T = 4
    initial_bounds = torch.tensor([[0, 0], [1, 1]], dtype=torch.float32)

    manager = create_manager(T_val=T, initial_bounds_val=initial_bounds)

    assert manager.T == 4

    assert manager.threshold == 0
    assert manager.blocks is None
    assert manager.block_size is None

    # After _update_block_arrangement (called by add_points), these would be set
    # assert manager.X is None # These attributes are no longer set by manager
    # assert manager.y is None # These attributes are no longer set by manager
def test_find_block(create_manager):
    """
    Test that _find_block correctly locates a block for an INTERIOR point.
    (Boundary points may intentionally overlap and complicate a unique find assertion).
    """
    T = torch.tensor([4, 4], device='cpu')
    initial_bounds = torch.tensor([[0, 0], [1, 1]], dtype=torch.float32)
    manager = create_manager(T_val=T, initial_bounds_val=initial_bounds)

    X_init = torch.tensor([[0.5, 0.5]], device='cpu', dtype=torch.float32)
    y_init = torch.tensor([[0.0]], device='cpu', dtype=torch.float32)
    manager.add_points(X_init, y_init) # Initializes manager.blocks and block_size

    # NEW: Test point clearly in the interior of block (0,0) based on [0, 0.25] boundaries
    # But now with `eps`, block (0,0) is `[-eps, 0.25+eps]`.
    # And block (1,1) is `[0.25-eps, 0.5+eps]`.
    # A point like (0.1, 0.1) is clearly in (0,0).
    test_point = torch.tensor([0.1, 0.1], device='cpu', dtype=torch.float32) # Interior point
    found_block = manager._find_block(test_point)

    assert found_block is not None
    # For (0.1, 0.1) in [0,1] space with 4x4 blocks, block size 0.25.
    # It should fall into the first block (0,0).
    expected_block_index = (0, 0)
    assert found_block.block_index == expected_block_index
    assert torch.allclose(found_block.block_size, torch.tensor([0.25, 0.25], device='cpu'))
    # Assert block_scope lower bound (original_lower - eps)
    assert torch.allclose(found_block.block_scope[0], torch.tensor([0.0, 0.0], device='cpu') - torch.finfo(torch.float32).eps, atol=1e-7)
    


def test_update_block_arrangement_initial_creation(create_manager):
    """Test that _update_block_arrangement correctly initializes blocks (first call)."""
    T = torch.tensor([4, 4], device='cpu')
    initial_bounds = torch.tensor([[0, 0], [1, 1]], dtype=torch.float32)
    manager = create_manager(T_val=T, initial_bounds_val=initial_bounds)

    X_data = torch.tensor([[0.1, 0.2], [0.3, 0.4]], device='cpu', dtype=torch.float32)
    manager._update_block_arrangement(X_data) # This is the direct call being tested

    assert manager.blocks is not None
    assert manager.block_size is not None
    assert torch.allclose(manager.block_size, torch.tensor([0.25, 0.25], device='cpu'))
    assert manager.initial_bounds is not None and manager.initial_bounds.shape == (2, 2)
    assert manager.blocks.shape == (4, 4)
    # Check that blocks are instances of PartitionBlock
    assert isinstance(manager.blocks[0,0], PartitionBlock)


def test_map_points_assigns_and_processes_y(create_manager):
    """Test that _map_points correctly assigns points to blocks and processes y values."""
    T = torch.tensor([2, 2], device='cpu')
    initial_bounds = torch.tensor([[0, 0], [1, 1]], dtype=torch.float32)
    manager = create_manager(T_val=T, initial_bounds_val=initial_bounds)
    
    X = torch.tensor([[0.1, 0.2], [0.6, 0.7], [0.3, 0.3]], device='cpu', dtype=torch.float32)
    y = torch.tensor([[1.0], [2.0], [3.0]], device='cpu', dtype=torch.float32)

    manager._update_block_arrangement(X) # Ensure blocks are initialized
    manager._map_points(X, y)

    total_points_mapped = 0
    mapped_blocks = [block for block in manager.blocks.flat if block.is_active]
    assert len(mapped_blocks) > 0

    for block in mapped_blocks:
        total_points_mapped += len(block.X)
        # Check that X, y are populated
        assert len(block.X) > 0
        assert len(block.y) > 0
        # Check that amplitude and target are calculated
        assert block.amplitude is not None


    assert total_points_mapped == len(X) # All points should be assigned to blocks

def test_add_points_full_workflow(create_manager):
    """
    Test the complete add_points workflow (arrangement, mapping, normalization).
    This is the main entry point for data insertion.
    """
    T = torch.tensor([2, 2], device='cpu')
    initial_bounds = torch.tensor([[0, 0], [1, 1]], dtype=torch.float32)
    manager = create_manager(T_val=T, initial_bounds_val=initial_bounds)

    X = torch.tensor([[0.1, 0.2], [0.6, 0.7], [0.3, 0.3]], device='cpu', dtype=torch.float32)
    y = torch.tensor([[1.0], [2.0], [3.0]], device='cpu', dtype=torch.float32)

    manager.add_points(X, y) # This calls _update_block_arrangement, _map_points, _vectorized_normalization

    assert manager.blocks is not None
    assert manager.block_size is not None
    
    active_blocks = manager.retrieve_active_blocks()
    assert len(active_blocks) > 0
    assert sum(len(b.X) for b in active_blocks) == len(X)

    for block in active_blocks:
        assert block.is_active
        assert block.X is not None and len(block.X) > 0
        assert block.y is not None and len(block.y) > 0
        assert block.normalized_X is not None # Should be set by _vectorized_normalization
        assert block.target is not None # Should be set by _map_points's call to calculate_amplitude_and_target

        # Verify normalization: e.g., (0.1, 0.2) in block (0,0) with size (0.5,0.5) from origin (0,0)
        # Normalized should be (0.1/0.5, 0.2/0.5) = (0.2, 0.4)
        if block.block_index == (0,0) and torch.allclose(block.X[0], torch.tensor([0.1, 0.2], device='cpu')):
            assert torch.allclose(block.normalized_X[0], torch.tensor([0.2, 0.4], device='cpu'), atol=1e-6)

        # Verify amplitude/target
        if block.block_index == (1,1) and torch.allclose(block.y[0], torch.tensor([2.0], device='cpu')):
            # y=2.0 -> amplitude=0.5 -> target=1.0
            assert pytest.approx(block.amplitude, rel=1e-6) == 0.5
            assert torch.allclose(block.target, torch.tensor([[1.0]], device='cpu'))


def test_init_sparse_coding_per_block_initializes_layers(create_manager):
    """Test that init_sparse_coding_per_block correctly initializes sparse coding layers."""
    T = torch.tensor([2, 2], device='cpu')
    initial_bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    manager = create_manager(T_val=T, initial_bounds_val=initial_bounds)

    X = torch.tensor([[0.1, 0.2], [0.6, 0.7]], device='cpu', dtype=torch.float32)
    y = torch.tensor([[1.0], [2.0]], device='cpu', dtype=torch.float32)

    manager.add_points(X, y) # Populates blocks and their data

    # Dummy evaluation function
    def dummy_eval_func(D: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return torch.matmul(D, h) # Simple matmul for testing

    # Example SparseCodingConfig
    sc_config = ISTAConfig(n_functions=5, epochs=50, alpha=0.01, lambd=0.1)

    manager.init_sparse_coding_per_block(config=sc_config, evaluation_func=dummy_eval_func)

    active_blocks = manager.retrieve_active_blocks()
    assert len(active_blocks) > 0 # Should have at least one active block

    for block in active_blocks:
        assert block.sparse_coding_layer is not None
        assert isinstance(block.sparse_coding_layer, ISTALayer) # Assuming ISTALayer is the default for ISTAConfig
        assert block.sparse_coding_layer.config.n_functions == 5
        assert block.sparse_coding_layer.evaluation_func is dummy_eval_func # Check function identity
        assert block.sparse_coding_layer.h is not None
        assert block.sparse_coding_layer.h.shape == (5, 1) # Check h shape based on n_functions

# --- Stress/Edge Case Tests ---

def test_add_points_with_no_initial_bounds(create_manager, caplog):
    """
    Test add_points when initial_bounds is None, forcing manager to infer bounds.
    Stresses the bound inference logic and warning.
    """
    caplog.set_level(logging.WARNING) # Capture expected warning
    manager = create_manager(T_val=torch.tensor([2, 2], device='cpu'), initial_bounds_val=None)
    
    X = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], device='cpu', dtype=torch.float32)
    y = torch.tensor([[1.0], [2.0], [3.0]], device='cpu', dtype=torch.float32)

    manager.add_points(X, y)

    assert manager.blocks is not None
    assert manager.initial_bounds is not None
    assert torch.allclose(manager.initial_bounds[0], torch.tensor([0.0, 0.0], device='cpu'))
    assert torch.allclose(manager.initial_bounds[1], torch.tensor([1.0, 1.0], device='cpu'))
    assert torch.allclose(manager.block_size, torch.tensor([0.5, 0.5], device='cpu'))
    assert manager.blocks.shape == (2, 2)
    assert any("No initial bounds provided, using calculated one" in r.message for r in caplog.records)


def test_add_points_empty_input(create_manager):
    """Test add_points with empty X and y. Should still initialize blocks array."""
    manager = create_manager(T_val=torch.tensor([2, 2], device='cpu'), initial_bounds_val=torch.tensor([[0,0],[1,1]], dtype=torch.float32).cpu().numpy())
    
    X_empty = torch.empty(0, 2, device='cpu', dtype=torch.float32)
    y_empty = torch.empty(0, 1, device='cpu', dtype=torch.float32)

    manager.add_points(X_empty, y_empty)

    assert manager.blocks is not None # Blocks array should still be initialized spatially
    assert manager.blocks.shape == (2, 2)
    assert all(not block.is_active for block in manager.blocks.flat) # No blocks should be active


def test_map_points_out_of_bounds(create_manager, caplog):
    """
    Test mapping points that are outside the defined initial_bounds.
    Should result in warnings and unmapped points.
    """
    caplog.set_level(logging.WARNING) # Capture expected warnings
    manager = create_manager(T_val=torch.tensor([2, 2], device='cpu'), initial_bounds_val=torch.tensor([[0,0],[1,1]], dtype=torch.float32).cpu().numpy())
    
    X_mixed = torch.tensor([
        [0.1, 0.1],  # Inside [0,1]x[0,1]
        [1.5, 0.5],  # Outside (x > 1)
        [-0.2, 0.8], # Outside (x < 0)
        [0.8, 1.2]   # Outside (y > 1)
    ], device='cpu', dtype=torch.float32)
    y_mixed = torch.tensor([[1.0],[2.0],[3.0],[4.0]], device='cpu', dtype=torch.float32)

    manager.add_points(X_mixed, y_mixed)

    active_blocks = manager.retrieve_active_blocks()
    total_mapped_points = sum(len(b.X) for b in active_blocks)

    assert total_mapped_points == 1 # Only (0.1, 0.1) should be mapped
    assert len(caplog.records) == 6 # Expecting 6 warnings for the 3 unmapped points 

def test_add_points_multiple_calls_fixed_bounds(create_manager, caplog):
    """
    Test adding points in multiple calls when initial_bounds are fixed.
    Existing blocks should retain their points, new points should map correctly.
    Verifies state persistence across add_points calls.
    """
    caplog.set_level(logging.WARNING) # Capture potential warnings for unmapped points
    bounds_tensor = torch.tensor([[0, 0], [1, 1]], dtype=torch.float32)
    manager = create_manager(T_val=torch.tensor([2, 2], device='cpu'), initial_bounds_val=bounds_tensor.cpu().numpy())

    X1 = torch.tensor([[0.1, 0.1], [0.6, 0.6]], device='cpu', dtype=torch.float32)
    y1 = torch.tensor([[1.0], [2.0]], device='cpu', dtype=torch.float32)

    manager.add_points(X1, y1)
    
    total_points1 = sum(len(b.X) for b in manager.retrieve_active_blocks())
    assert total_points1 == 2
    
    # Add more points, some to existing blocks, some to new blocks, some out of bounds
    X2 = torch.tensor([[0.15, 0.15], # To existing block (0,0)
                       [0.8, 0.2],   # To block (1,0) - new point to new block for X2
                       [1.1, 0.1]    # Out of bounds
                      ], device='cpu', dtype=torch.float32)
    y2 = torch.tensor([[3.0], [4.0], [5.0]], device='cpu', dtype=torch.float32)

    manager.add_points(X2, y2)
    
    total_points2 = sum(len(b.X) for b in manager.retrieve_active_blocks())
    # Expected: (2 from X1) + (1 from X2 to existing block) + (1 from X2 to new block) = 4
    assert total_points2 == 4
    
    # Check point counts in specific blocks
    block_0_0 = manager.blocks[0, 0] # Contains (0.1,0.1) and (0.15,0.15)
    block_1_1 = manager.blocks[1, 1] # Contains (0.6,0.6)
    block_1_0 = manager.blocks[1, 0] # Contains (0.8,0.2)

    assert len(block_0_0.X) == 2
    assert len(block_1_1.X) == 1
    assert len(block_1_0.X) == 1
    
    # Check for warning messages from the second add_points call for the out-of-bounds point
    assert any("could not be mapped to any block" in r.message for r in caplog.records)


def test_retrieve_test_active_blocks_isolation_old(create_manager):
    """
    Test that retrieve_test_active_blocks creates isolated test blocks
    and correctly assigns original sparse coding layers to them.
    """
    bounds = torch.tensor([[0, 0], [1, 1]], dtype=torch.float32)
    manager = create_manager(T_val=torch.tensor([2, 2], device='cpu'), initial_bounds_val=bounds.cpu().numpy())
    
    X_train = torch.tensor([[0.1, 0.1], [0.6, 0.6]], device='cpu', dtype=torch.float32)
    y_train = torch.tensor([[1.0], [2.0]], device='cpu', dtype=torch.float32)
    manager.add_points(X_train, y_train)

    # Dummy evaluation function for SC layer
    def dummy_eval_func(D: torch.Tensor, h: torch.Tensor) -> torch.Tensor: return torch.matmul(D,h)
    # Example SparseCodingConfig
    ista_config = ISTAConfig(n_functions=2, epochs=50, alpha=0.01, lambd=0.1)
    manager.init_sparse_coding_per_block(config=ista_config, evaluation_func=dummy_eval_func)

    # Store references to original manager's internal state BEFORE retrieve_test_active_blocks
    original_blocks_ref = manager.blocks
    # Get a reference to a specific original sparse coding layer
    original_sc_layer_ref = original_blocks_ref[0, 0].sparse_coding_layer if original_blocks_ref[0,0].is_active else None

    # Test data to be mapped into the test blocks
    X_test = torch.tensor([[0.15, 0.15], [0.25, 0.25], [0.7, 0.7]], device='cpu', dtype=torch.float32)
    y_test = torch.tensor([[3.0], [4.0], [5.0]], device='cpu', dtype=torch.float32)

    test_active_blocks = manager.retrieve_test_active_blocks(X_test, y_test)

    # 1. Assert original manager's blocks array reference is unchanged
    assert manager.blocks is original_blocks_ref
    # 2. Assert contents of original blocks (X, y, etc.) are still the same (not modified by test mapping)
    block_0_0_original = original_blocks_ref[0,0]
    assert len(block_0_0_original.X) == 1 # Only X_train[0] (0.1,0.1)
    assert torch.allclose(block_0_0_original.X[0], X_train[0])
    assert block_0_0_original.normalized_X is not None
    assert block_0_0_original.target is not None

    # 3. Assert test_active_blocks contain new PartitionBlock instances (not identity with originals)
    test_block_0_0 = next((b for b in test_active_blocks if b.block_index == (0,0)), None)
    assert test_block_0_0 is not None
    assert test_block_0_0 is not original_blocks_ref[0,0] # It should be a brand new PartitionBlock instance

    # 4. Assert test blocks contain mapped test data
    assert len(test_block_0_0.X) == 2 # X_test[0] and X_test[1]
    assert torch.allclose(test_block_0_0.X[0], X_test[0])
    assert torch.allclose(test_block_0_0.X[1], X_test[1])
    assert test_block_0_0.normalized_X is not None
    assert test_block_0_0.target is not None

    # 5. Assert test blocks point to the *original* sparse_coding_layer instances
    if original_sc_layer_ref:
        assert test_block_0_0.sparse_coding_layer is original_sc_layer_ref # Identity check: same SC layer object
        assert test_block_0_0.sparse_coding_layer.h is not None # Check h is present
        # Verify that the h reference is the same (not a copy of h)
        if block_0_0_original.sparse_coding_layer:
            assert test_block_0_0.sparse_coding_layer.h is block_0_0_original.sparse_coding_layer.h

    # 6. Test a block for points from X_test that map to a new block
    test_block_1_1 = next((b for b in test_active_blocks if b.block_index == (1,1)), None)
    assert test_block_1_1 is not None
    assert len(test_block_1_1.X) == 1 # X_test[2]
    assert torch.allclose(test_block_1_1.X[0], X_test[2])

def test_retrieve_test_active_blocks_isolation(create_manager):
    """
    Test that retrieve_test_active_blocks creates isolated test blocks
    and correctly assigns original sparse coding layers to them.
    Crucially, it verifies that the sparse coding layers themselves (and their 'h' parameters)
    are *shared instances* between original and test blocks for learned properties.
    """
    bounds = torch.tensor([[0, 0], [1, 1]], dtype=torch.float32)
    manager = create_manager(T_val=torch.tensor([2, 2], device='cpu'), initial_bounds_val=bounds.cpu().numpy())
    
    X_train = torch.tensor([[0.1, 0.1], [0.6, 0.6], [0.15, 0.15]], device='cpu', dtype=torch.float32) # Added a third point for more diversity
    y_train = torch.tensor([[1.0], [2.0], [3.0]], device='cpu', dtype=torch.float32)
    manager.add_points(X_train, y_train)

    def dummy_eval_func(D: torch.Tensor, h: torch.Tensor) -> torch.Tensor: return torch.matmul(D,h)
    ista_config = ISTAConfig(n_functions=2, epochs=50, alpha=0.01, lambd=0.1)
    manager.init_sparse_coding_per_block(config=ista_config, evaluation_func=dummy_eval_func)

    # --- Puntos clave de este test: Simular algún entrenamiento para que h cambie ---
    # Accede directamente a los bloques y simula un entrenamiento para cambiar 'h'
    # Esto es una simulación del proceso de entrenamiento real, no del manager en sí.
    active_blocks_train = manager.retrieve_active_blocks()
    
    # Asignar valores distintivos a 'h' para cada bloque activo
    # block (0,0) will get h_val_A
    # block (1,1) will get h_val_B
    h_val_A = torch.tensor([[0.5], [0.1]], dtype=torch.float32) # Example values for h
    h_val_B = torch.tensor([[-0.2], [0.8]], dtype=torch.float32) # Example values for h
    
    # Busca los bloques activos y asigna valores a sus 'h'
    for block in active_blocks_train:
        if block.block_index == (0,0): # Block containing (0.1,0.1) and (0.15,0.15)
            block.sparse_coding_layer.h.data = h_val_A
            # También simular que las pérdidas cambian
            block.sparse_coding_layer.losses.append(0.123) 
        elif block.block_index == (1,1): # Block containing (0.6,0.6)
            block.sparse_coding_layer.h.data = h_val_B
            block.sparse_coding_layer.losses.append(0.456)
        # Puedes añadir otros bloques si tu X_train los activa

    # Store references to original manager's internal state BEFORE retrieve_test_active_blocks
    original_blocks_ref = manager.blocks
    
    # Get references to specific original sparse coding layers and their 'h'
    original_sc_layer_0_0 = original_blocks_ref[0, 0].sparse_coding_layer
    original_h_0_0_ref = original_sc_layer_0_0.h # Reference to the Parameter object for (0,0)
    original_h_0_0_data_ref = original_h_0_0_ref.data # Reference to the underlying tensor data for (0,0)

    original_sc_layer_1_1 = original_blocks_ref[1, 1].sparse_coding_layer
    original_h_1_1_ref = original_sc_layer_1_1.h # Reference to the Parameter object for (1,1)
    original_h_1_1_data_ref = original_h_1_1_ref.data # Reference to the underlying tensor data for (1,1)


    # Test data to be mapped into the test blocks
    # X_test[0] -> (0,0)
    # X_test[1] -> (0,0)
    # X_test[2] -> (1,1)
    test_X = torch.tensor([[0.15, 0.15], [0.25, 0.25], [0.7, 0.7]], device='cpu', dtype=torch.float32)
    test_y = torch.tensor([[3.0], [4.0], [5.0]], device='cpu', dtype=torch.float32)

    test_active_blocks = manager.retrieve_test_active_blocks(test_X, test_y)

    # 1. Assert original manager's blocks array reference is unchanged
    assert manager.blocks is original_blocks_ref

    # 2. Assert contents of original blocks (X, y, etc.) are still the same (not modified by test mapping)
    # Check block (0,0) in original manager
    assert len(original_blocks_ref[0,0].X) == 2 # (0.1,0.1) and (0.15,0.15) from X_train
    assert torch.allclose(original_blocks_ref[0,0].X[0], X_train[0])
    assert torch.allclose(original_blocks_ref[0,0].X[1], X_train[2]) # original (0.15,0.15)
    assert original_blocks_ref[0,0].normalized_X is not None
    assert original_blocks_ref[0,0].target is not None
    # Check block (1,1) in original manager
    assert len(original_blocks_ref[1,1].X) == 1 # (0.6,0.6) from X_train
    assert torch.allclose(original_blocks_ref[1,1].X[0], X_train[1])
    assert original_blocks_ref[1,1].normalized_X is not None
    assert original_blocks_ref[1,1].target is not None


    # 3. Assert test_active_blocks contain new PartitionBlock instances (not identity with originals)
    test_block_0_0 = next((b for b in test_active_blocks if b.block_index == (0,0)), None)
    test_block_1_1 = next((b for b in test_active_blocks if b.block_index == (1,1)), None) # Block (1,1) now has test data

    assert test_block_0_0 is not None
    assert test_block_1_1 is not None
    assert test_block_0_0 is not original_blocks_ref[0,0] # Should be a new PartitionBlock instance
    assert test_block_1_1 is not original_blocks_ref[1,1] # Should be a new PartitionBlock instance


    # 4. Assert test blocks contain mapped test data (and are cleared of old train data)
    assert len(test_block_0_0.X) == 2 # X_test[0] and X_test[1] mapped to this block
    assert torch.allclose(test_block_0_0.X[0], test_X[0])
    assert torch.allclose(test_block_0_0.X[1], test_X[1])
    assert test_block_0_0.normalized_X is not None
    assert test_block_0_0.target is not None

    assert len(test_block_1_1.X) == 1 # X_test[2] mapped to this block
    assert torch.allclose(test_block_1_1.X[0], test_X[2])
    assert test_block_1_1.normalized_X is not None
    assert test_block_1_1.target is not None


    # 5. Assert test blocks point to the *original* sparse_coding_layer instances
    # Identity check: same SC layer object (shallow copy of SC layer)
    assert test_block_0_0.sparse_coding_layer is original_sc_layer_0_0 
    assert test_block_1_1.sparse_coding_layer is original_sc_layer_1_1

    # And crucially, assert that the 'h' Parameter objects are the same reference
    assert test_block_0_0.sparse_coding_layer.h is original_h_0_0_ref
    assert test_block_1_1.sparse_coding_layer.h is original_h_1_1_ref

    # And that the underlying tensor data for 'h' is also the same reference
    assert test_block_0_0.sparse_coding_layer.h.data is original_h_0_0_data_ref
    assert test_block_1_1.sparse_coding_layer.h.data is original_h_1_1_data_ref

    # 6. Verify that the values of 'h' in the test blocks match the values we set in the original blocks
    assert torch.allclose(test_block_0_0.sparse_coding_layer.h, h_val_A)
    assert torch.allclose(test_block_1_1.sparse_coding_layer.h, h_val_B)

    # 7. Verify that the 'losses' list of the sparse coding layer is also the same object (shared state)
    # This is important for logging and tracking.
    assert test_block_0_0.sparse_coding_layer.losses is original_sc_layer_0_0.losses
    assert test_block_1_1.sparse_coding_layer.losses is original_sc_layer_1_1.losses
    assert test_block_0_0.sparse_coding_layer.losses[-1] == 0.123
    assert test_block_1_1.sparse_coding_layer.losses[-1] == 0.456

    # 8. Sanity check: ensure training data is cleared from test_active_blocks after creation
    # This is managed by _create_test_block_structure within retrieve_test_active_blocks
    # which creates a fresh PartitionBlock and then copies SC layer.
    assert len(test_block_0_0.y) == 2 # test_y points mapped
    assert len(test_block_1_1.y) == 1 # test_y points mapped

    
def test_add_points_with_n_dimensional_input(create_manager):
    """
    Test add_points with a 3D input space (n_features=3).
    Ensures N-dimensional indexing and calculations are correct.
    """
    T_3D = torch.tensor([2, 2, 2], device='cpu')
    initial_bounds_3D_np = torch.tensor([[0,0,0],[1,1,1]], dtype=torch.float32).cpu().numpy()
    manager = create_manager(T_val=T_3D, initial_bounds_val=initial_bounds_3D_np)

    X_3D = torch.tensor([
        [0.1, 0.1, 0.1], # Block (0,0,0) -> Normalized (0.2, 0.2, 0.2)
        [0.6, 0.1, 0.1], # Block (1,0,0) -> Normalized (0.2, 0.2, 0.2)
        [0.1, 0.6, 0.1], # Block (0,1,0) -> Normalized (0.2, 0.2, 0.2)
        [0.6, 0.6, 0.6]  # Block (1,1,1) -> Normalized (0.2, 0.2, 0.2)
    ], device='cpu', dtype=torch.float32)
    y_3D = torch.tensor([[1.0],[2.0],[3.0],[4.0]], device='cpu', dtype=torch.float32)

    manager.add_points(X_3D, y_3D)

    assert manager.blocks.shape == (2, 2, 2)
    assert manager.block_size.shape == (3,)
    assert torch.allclose(manager.block_size, torch.tensor([0.5, 0.5, 0.5], device='cpu'))

    active_blocks = manager.retrieve_active_blocks()
    assert len(active_blocks) == 4 # All 4 points should map to unique blocks
    assert sum(len(b.X) for b in active_blocks) == 4

    # Check specific blocks for content and correct normalization
    # Block (0,0,0)
    block_0_0_0 = manager.blocks[0, 0, 0]
    assert len(block_0_0_0.X) == 1
    assert torch.allclose(block_0_0_0.X[0], X_3D[0])
    assert torch.allclose(block_0_0_0.normalized_X[0], torch.tensor([0.2, 0.2, 0.2], device='cpu'), atol=1e-6)
    
    # Block (1,1,1)
    block_1_1_1 = manager.blocks[1, 1, 1]
    assert len(block_1_1_1.X) == 1
    assert torch.allclose(block_1_1_1.X[0], X_3D[3])
    assert torch.allclose(block_1_1_1.normalized_X[0], torch.tensor([0.2, 0.2, 0.2], device='cpu'), atol=1e-6)


def test_map_points_y_scalar(create_manager):
    """
    Test _map_points with scalar y values (shape [1]) and ensure they are handled correctly.
    """
    manager = create_manager(T_val=torch.tensor([2, 2], device='cpu'), initial_bounds_val=torch.tensor([[0,0],[1,1]], dtype=torch.float32).cpu().numpy())

    X = torch.tensor([[0.1, 0.1]], device='cpu', dtype=torch.float32)
    y_scalar_input = torch.tensor([1.0], device='cpu', dtype=torch.float32) # Single scalar, will be dim 1

    manager._update_block_arrangement(X)
    manager._map_points(X, y_scalar_input)
    manager._prepare_block_targets()

    block = manager.blocks[0,0]
    assert block.is_active
    assert len(block.y) == 1
    assert block.y[0].shape == torch.Size([1]) # y in list maintains original shape
    
    # After calculate_amplitude_and_target, target should be (n_samples, output_dim) -> (1,1)
    assert block.target is not None
    assert block.target.shape == (1, 1)
    assert torch.allclose(block.target, torch.tensor([[1.0]], device='cpu'), atol=1e-6) # Assuming amplitude=1.0

    
if __name__ == "__main__":
    from pytest_helper import print_pytest_instructions
    print_pytest_instructions()
