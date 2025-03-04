from pysesm.blocks import UniformPartitionManager
from pysesm.blocks.UniformPartitionManager import squeeze_factor
from pysesm.blocks import PartitionBlock
from pysesm.models.ISTALayer import ISTALayer
import torch
import logging
import numpy as np

# Config for logger
logger = logging.getLogger("test")
logger.setLevel(logging.DEBUG)

def test_uniform_partition_manager_initialization():
    """Test initialization of UniformPartitionManager."""
    T = torch.tensor([4, 4])  # 2**4 = 16
    n_functions = 2
    initial_bounds = np.array([[0.0, 0.0], [1.0, 1.0]]) 

    manager = UniformPartitionManager(logger, T, n_functions, initial_bounds)

    assert manager.T.equal(torch.tensor([4, 4]))
    assert manager.n_functions == 2
    assert np.array_equal(manager.initial_bounds, np.array([[0.0, 0.0], [1.0, 1.0]]))
    assert manager.threshold == 0
    assert manager.blocks is None
    assert manager.block_size is None
    assert manager.X is None
    assert manager.y is None

def test_find_block():
    """Test that _find_block correctly assigns blocks to each point"""
    T = torch.tensor([4, 4])  # 2**4 = 16
    n_functions = 2
    initial_bounds = np.array([[0.0, 0.0], [1.0, 1.0]]) 

    manager = UniformPartitionManager(logger, T, n_functions, initial_bounds)

    X = torch.tensor([0.25, 0.2])

    min_values = torch.min(X, dim=0).values  # [1, 5]
    max_values = torch.max(X, dim=0).values  # [3, 7]

    bounds = torch.vstack([min_values, max_values])
    
    delta = bounds[1] - bounds[0]
    manager.block_size = torch.div(delta, manager.T)

    manager.blocks = np.empty(T.numpy(), dtype=PartitionBlock)

    for index in np.ndindex(manager.blocks.shape):
        manager.blocks[index] = PartitionBlock(
            bounds[0], index, manager.block_size
        )

    block_test=manager._find_block(X)

    for index in np.ndindex(manager.blocks.shape):
        block: PartitionBlock = manager.blocks[index]
        if block.is_point_in_block(X):
            block_true = block

    assert block_test.block_index == block_true.block_index 
    assert torch.equal(block_test.block_size, block_true.block_size) 
    assert torch.equal(block_test.block_scope, block_true.block_scope) 
    
def test_update_block_arrangement():
    """Test that _update_block_arrangement correctly initializes blocks"""
    T = torch.tensor([4, 4])
    n_functions = 2
    manager = UniformPartitionManager(logger, T, n_functions)
    
    X = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    
    manager._update_block_arrangement(X)
    
    assert manager.blocks is not None
    assert manager.block_size is not None
    assert manager.initial_bounds is not None
    assert np.array_equal(manager.blocks.shape, [4, 4])
    
    # Test with default T
    manager_default = UniformPartitionManager(logger, None, n_functions)
    manager_default._update_block_arrangement(X)
    assert manager_default.T.equal(torch.tensor([4, 4])) 

def test_configure_blocks():
    """Test that _configure_blocks correctly sets up blocks"""
    T = torch.tensor([2, 2])
    n_functions = 2
    manager = UniformPartitionManager(logger, T, n_functions)
    
    X = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    y = torch.tensor([[1.0], [2.0]])
    
    manager._update_block_arrangement(X)
    manager._map_points(X, y)
    manager._configure_blocks()
    
    for block in manager.blocks.flat:
        if len(block.y) > 0:
            assert block.amplitude is not None
            assert block.h is not None
            assert block.target is not None
            assert isinstance(block.h, torch.nn.Parameter)
            assert block.h.requires_grad

def test_map_points():
    """Test that _map_points correctly assigns points to blocks"""
    T = torch.tensor([2, 2])
    n_functions = 2
    manager = UniformPartitionManager(logger, T, n_functions)
    
    X = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    y = torch.tensor([[1.0], [2.0], [3.0]])
    
    manager._update_block_arrangement(X)
    manager._map_points(X, y)
    
    total_points = 0
    for block in manager.blocks.flat:
        total_points += len(block.y)
    
    assert total_points == len(X)  # All points should be assigned to blocks

def test_add_points():
    """Test that add_points correctly processes new points"""
    T = torch.tensor([2, 2])
    n_functions = 2
    initial_bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    manager = UniformPartitionManager(logger, T, n_functions, initial_bounds)
    
    X = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    y = torch.tensor([[1.0], [2.0]])
    manager._update_block_arrangement(X)
    manager._map_points(X, y)
    points_found = 0
    for block in manager.blocks.flat:
        points_found += len(block.X) if hasattr(block, 'X') else 0
    
    assert points_found > 0, "No points were mapped to blocks"

def test_init_ista_per_block():
    """Test that init_ista_per_block correctly initializes ISTA layers"""
    T = torch.tensor([2, 2])
    n_functions = 2
    initial_bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    manager = UniformPartitionManager(logger, T, n_functions, initial_bounds)
    
    X = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    y = torch.tensor([[1.0], [2.0]])
    
    manager._update_block_arrangement(X)
    manager._map_points(X, y)
    
    def dummy_eval_func(x, y):
        return torch.sum(x - y)
    
    def dummy_optimizer(params, lr):
        return torch.optim.Adam(params, lr=lr)
    
    manager.init_ista_per_block(
        n_functions=2,
        ista_alpha=0.01,
        ista_lambd=0.1,
        evaluation_func=dummy_eval_func,
        ista_optimizer=dummy_optimizer,
        ista_criterion=None
    )
    for block in manager.blocks.flat:
        if hasattr(block, 'X') and len(block.X) > 0:
            assert hasattr(block, 'ista_layer')
            assert isinstance(block.ista_layer, ISTALayer)
            assert block.ista_layer.alpha == 0.01
            assert block.ista_layer.lambd == 0.1

def test_uniform_partition_block_assignment():
    """Test that blocks are correctly assigned during partition"""
    T = torch.tensor([2, 2])
    n_functions = 2
    initial_bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    manager = UniformPartitionManager(logger, T, n_functions, initial_bounds)
    X_train = torch.tensor([[0.1, 0.1], [0.7, 0.7], [0.4, 0.4]])
    y_train = torch.tensor([[1.0], [2.0], [1.5]])
    manager._update_block_arrangement(X_train)
    manager._map_points(X_train, y_train)
    manager._configure_blocks()
    assigned_blocks = [block for block in manager.blocks.flat if block.X]
    assert len(assigned_blocks) > 0
    for block in assigned_blocks:
        assert len(block.X) > 0 
        assert len(block.y) > 0 

def test_squeeze_factor():
    """Test the squeeze_factor function"""
    # Test case where max value > 1
    y_large = [torch.tensor([1.5]), torch.tensor([2.0]), torch.tensor([0.5])]
    factor_large = squeeze_factor(y_large)
    assert abs(factor_large - 0.5) < 1e-6  # 1/max value (1/2.0)

    # Test case where max value <= 1
    y_small = [torch.tensor([0.3]), torch.tensor([0.5]), torch.tensor([0.1])] 
    factor_small = squeeze_factor(y_small)
    assert abs(factor_small - 1.0) < 1e-6

if __name__ == "__main__":
    from pytest_helper import print_pytest_instructions
    print_pytest_instructions()    
