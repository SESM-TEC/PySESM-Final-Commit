from pysesm.blocks import UniformPartitionManager
from pysesm.blocks import PartitionBlock
import torch
import logging
import numpy as np
from copy import deepcopy

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
    



#def test_map_points():


if __name__ == "__main__":
    from pytest_helper import print_pytest_instructions
    print_pytest_instructions()    

