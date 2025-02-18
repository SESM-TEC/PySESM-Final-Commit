from pysesm.blocks import UniformPartitionManager
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

#def test_map_points():
#     """Tests whether _find_block correctly assigns blocks to each point"""

if __name__ == "__main__":
    from pytest_helper import print_pytest_instructions
    print_pytest_instructions()    

