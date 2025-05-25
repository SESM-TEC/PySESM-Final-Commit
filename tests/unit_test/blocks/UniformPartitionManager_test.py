from pysesm.blocks import UniformPartitionManager, UniformPartitionConfig, PartitionBlock
from pysesm.sparse_coding.ISTALayer import ISTALayer, ISTAConfig, StepSizeMethod
from pysesm.enums.DeviceTargetEnum import DeviceTarget

from pysesm.device_manager.DeviceManager import DeviceManager

import torch
import logging
import numpy as np

# Config for logger
logger = logging.getLogger("test")
logger.setLevel(logging.DEBUG)

# Define device_map
device_map = {
    DeviceTarget.GLOBAL: "cpu",               # Dispositivo global por defecto
    DeviceTarget.SPARSE_CODING_LAYER: "cpu",  # ISTA en CPU
    DeviceTarget.DICTIONARY_LAYER: "cpu",     # Dictionary en CPU
    DeviceTarget.PARTITION_MANAGER: "cpu"     # Partition Manager en CPU
}

# Initializes DeviceManager for all tests
device_manager = DeviceManager(logger, default_device="cpu", device_map=device_map)

def test_uniform_partition_manager_initialization():
    """Test initialization of UniformPartitionManager."""
    T = torch.tensor([4, 4], device='cpu')  # 2**4 = 16
    initial_bounds = torch.tensor([[0, 0], [1, 1]], dtype=torch.float32)

    config = UniformPartitionConfig(T=T,initial_bounds=initial_bounds)
    
    manager = UniformPartitionManager(config=config, logger=logger, device_manager=device_manager)

    assert manager.T.equal(torch.tensor([4, 4], device='cpu'))
    assert np.array_equal(manager.initial_bounds, np.array([[0.0, 0.0], [1.0, 1.0]]))
    assert manager.threshold == 0
    assert manager.blocks is None
    assert manager.block_size is None
    assert manager.X is None
    assert manager.y is None

def test_find_block():
    """Test that _find_block correctly assigns blocks to each point."""
    T = torch.tensor([4, 4], device='cpu')  # 2**4 = 16
    initial_bounds = torch.tensor([[0, 0], [1, 1]], dtype=torch.float32)

    config = UniformPartitionConfig(T=T,initial_bounds=initial_bounds)
    
    manager = UniformPartitionManager(config=config, logger=logger, device_manager=device_manager)

    X = torch.tensor([0.25, 0.2], device='cpu')

    min_values = torch.min(X, dim=0).values  # [0.25, 0.2]
    max_values = torch.max(X, dim=0).values  # [0.25, 0.2]

    bounds = torch.vstack([min_values, max_values]).to('cpu')
    delta = bounds[1] - bounds[0]
    manager.block_size = torch.div(delta, manager.T)

    manager.blocks = np.empty(T.cpu().numpy(), dtype=PartitionBlock)

    for index in np.ndindex(manager.blocks.shape):
        manager.blocks[index] = PartitionBlock(
            space_origin=bounds[0],
            block_index=index,
            block_size=manager.block_size,
            device='cpu'
        )

    block_test = manager._find_block(X)

    for index in np.ndindex(manager.blocks.shape):
        block: PartitionBlock = manager.blocks[index]
        if block.is_point_in_block(X):
            block_true = block

    assert block_test.block_index == block_true.block_index
    assert torch.equal(block_test.block_size, block_true.block_size)
    assert torch.equal(block_test.block_scope, block_true.block_scope)

def test_update_block_arrangement():
    """Test that _update_block_arrangement correctly initializes blocks."""
    T = torch.tensor([4, 4], device='cpu')
    n_functions = 2
    initial_bounds = torch.tensor([[0, 0], [1, 1]], dtype=torch.float32)

    config = UniformPartitionConfig(T=T,initial_bounds=initial_bounds)
    
    manager = UniformPartitionManager(config=config, logger=logger, device_manager=device_manager)

    X = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], device='cpu')

    manager._update_block_arrangement(X)

    assert manager.blocks is not None
    assert manager.block_size is not None
    assert manager.initial_bounds is not None
    assert np.array_equal(manager.blocks.shape, [4, 4])

    # Test with default T
    manager_default = UniformPartitionManager(config=config, logger=logger, device_manager=device_manager)
    manager_default._update_block_arrangement(X)
    assert manager_default.T.equal(torch.tensor([4, 4], device='cpu'))

def test_configure_blocks():
    """Test that _configure_blocks correctly sets up blocks."""
    T = torch.tensor([2, 2], device='cpu')
    n_functions = 2
    initial_bounds = torch.tensor([[0, 0], [1, 1]], dtype=torch.float32)

    config = UniformPartitionConfig(T=T,initial_bounds=initial_bounds)
    
    manager = UniformPartitionManager(config=config, logger=logger, device_manager=device_manager)

    X = torch.tensor([[0.1, 0.2], [0.3, 0.4]], device='cpu')
    y = torch.tensor([[1.0], [2.0]], device='cpu')

    manager._update_block_arrangement(X)
    manager._map_points(X, y)

    for block in manager.blocks.flat:
        if len(block.y) > 0:
            assert block.amplitude is not None
            assert block.target is not None

def test_map_points():
    """Test that _map_points correctly assigns points to blocks."""
    T = torch.tensor([2, 2], device='cpu')
    n_functions = 2
    initial_bounds = torch.tensor([[0, 0], [1, 1]], dtype=torch.float32)

    config = UniformPartitionConfig(T=T,initial_bounds=initial_bounds)    
    manager = UniformPartitionManager(config=config, logger=logger, device_manager=device_manager)
    
    X = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], device='cpu')
    y = torch.tensor([[1.0], [2.0], [3.0]], device='cpu')

    manager._update_block_arrangement(X)
    manager._map_points(X, y)

    total_points = 0
    for block in manager.blocks.flat:
        total_points += len(block.y)

    assert total_points == len(X)  # All points should be assigned to blocks

def test_add_points():
    """Test that add_points correctly processes new points."""
    T = torch.tensor([2, 2], device='cpu')
    n_functions = 2
    initial_bounds = torch.tensor([[0, 0], [1, 1]], dtype=torch.float32)

    config = UniformPartitionConfig(T=T,initial_bounds=initial_bounds)    
    manager = UniformPartitionManager(config=config, logger=logger, device_manager=device_manager)

    X = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], device='cpu')
    y = torch.tensor([[1.0], [2.0], [3.0]], device='cpu')

    manager._update_block_arrangement(X)
    manager._map_points(X, y)

    points_found = 0
    for block in manager.blocks.flat:
        if hasattr(block, 'X') and len(block.X) > 0:  # Solo contar bloques con puntos
            points_found += len(block.X)

    assert points_found > 0, "No points were mapped to blocks"

def test_init_sparse_coding_per_block():
    """Test that init_ista_per_block correctly initializes ISTA layers."""
    T = torch.tensor([2, 2], device='cpu')
    n_functions = 2
    initial_bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], device='cpu')

    config = UniformPartitionConfig(T=T,initial_bounds=initial_bounds)    
    manager = UniformPartitionManager(config=config, logger=logger, device_manager=device_manager)

    X = torch.tensor([[0.1, 0.2], [0.3, 0.4]], device='cpu')
    y = torch.tensor([[1.0], [2.0]], device='cpu')

    manager._update_block_arrangement(X)
    manager._map_points(X, y)

    def dummy_eval_func(D, h):
        return torch.matmul(D,h)

    def dummy_optimizer(params, lr):
        return torch.optim.Adam(params, lr=lr)


    manager.init_sparse_coding_per_block(config=ISTAConfig(n_functions = 2,
                                                           alpha=0.01,
                                                           lambd=0.1),
                                         evaluation_func=dummy_eval_func)

    for block in manager.blocks.flat:
        if hasattr(block, 'X') and len(block.X) > 0:
            assert hasattr(block, 'sparse_coding_layer')
            assert isinstance(block.sparse_coding_layer, ISTALayer)
            assert block.sparse_coding_layer.config.alpha == 0.01
            assert block.sparse_coding_layer.config.lambd == 0.1

def test_uniform_partition_block_assignment():
    """Test that blocks are correctly assigned during partition."""
    T = torch.tensor([2, 2], device='cpu')
    n_functions = 2
    initial_bounds = torch.tensor([[0, 0], [1, 1]], dtype=torch.float32)

    config = UniformPartitionConfig(T=T,initial_bounds=initial_bounds)    
    manager = UniformPartitionManager(config=config, logger=logger, device_manager=device_manager)

    X = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], device='cpu')
    y = torch.tensor([[1.0], [2.0], [3.0]], device='cpu')

    manager._update_block_arrangement(X)
    manager._map_points(X, y)

    assigned_blocks = [block for block in manager.blocks.flat if hasattr(block, 'X') and len(block.X) > 0]
    assert len(assigned_blocks) > 0
    for block in assigned_blocks:
        assert len(block.X) > 0
        assert len(block.y) > 0


if __name__ == "__main__":
    from pytest_helper import print_pytest_instructions
    print_pytest_instructions()
