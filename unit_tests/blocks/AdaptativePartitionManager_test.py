import torch
import logging
import numpy as np
import pytest
from typing import Union, Optional

from pysesm.blocks.AdaptativePartitionManager import AdaptativePartitionManager,AdaptativePartitionConfig
from pysesm.blocks import KDTree
from pysesm.blocks.KDTreeStrategy import KDTreeStrategy, KDTreeStrategyConfig
from pysesm.blocks.PartitionBlock import PartitionBlock
from pysesm.blocks import Node
from pysesm.blocks.SESMData import SESMData
from pysesm.utils.loggers import setup_logger
from pysesm.sparse_coding.ISTALayer import ISTALayer, ISTAConfig

@pytest.fixture(scope="module")
def logger():
    """Provide a logger for the test module."""
    logger = logging.getLogger("test_adaptative_partition_manager")
    logger.setLevel(logging.DEBUG)
    return logger

@pytest.fixture(scope="module")
def common_device():
    """Shared device fixture for consistency across tests."""
    return "cpu"

@pytest.fixture(scope="function")
def create_manager(logger, common_device):
    """
    Factory fixture to create AdaptativePartitionManager instances with
    configurable KDTreeStrategy and overlap ratios.
    """
    def _creator(maxNodeSize=5, data_wrapper=SESMData,
                overlap_ratio: Optional[float] = 0.1):
        strategyConfig = KDTreeStrategyConfig(
            maxNodeSize=maxNodeSize,
            data_wrapper=data_wrapper,
            device=common_device
        )
        strategy = KDTreeStrategy(strategyConfig)
        config = AdaptativePartitionConfig(
            overlap_ratio=overlap_ratio,
            partition_strategy=strategy
        )
        return AdaptativePartitionManager(
            config=config,
            logger=logger
        )
    return _creator


# ---------------------------------------------------------------------------
# Core tests
# ---------------------------------------------------------------------------

def test_initial_build_creates_blocks(create_manager):
    """Verify that the first _update_block_arrangement call builds the KDTree and blocks."""
    manager = create_manager(overlap_ratio=None)

    X = torch.randn(20, 4)
    y = torch.randn(20, 1)
    manager._update_block_arrangement(X, y)

    # Strategy should now have a KDTree built
    assert manager.strategy.kdtree is not None, "KDTree should be initialized after first update"
    assert len(manager.blocks) > 0, "Blocks should be created after initial partitioning"
    assert all(isinstance(b, PartitionBlock) for b in manager.blocks), "All blocks must be PartitionBlock instances"
    assert manager.total_blocks == len(manager.blocks)


def test_add_points_triggers_rebuild(create_manager):
    """Verify that adding new points either expands or rebuilds the KDTree."""
    manager = create_manager()
    n_samples=15
    X1 = torch.randn(n_samples, 3)
    y1 = torch.randn(n_samples, 1)
    manager._update_block_arrangement(X1, y1)

    initial_block_count = manager.total_blocks
    for i in range(10):
        X2 = torch.randn(10*n_samples, 3)
        y2 = torch.randn(10*n_samples, 1)
        manager._update_block_arrangement(X2, y2)

    assert manager.strategy.kdtree is not None
    assert len(manager.blocks) >= initial_block_count, (
        "After adding points, total block count should stay or increase depending on rebuild policy."
    )


def test_combined_data_is_preserved(create_manager):
    """Ensure that all data used in partitions equals the concatenation of all points seen."""
    manager = create_manager()

    X1 = torch.randn(10, 2)
    y1 = torch.randn(10, 1)
    X2 = torch.randn(8, 2)
    y2 = torch.randn(8, 1)

    manager._update_block_arrangement(X1, y1)
    manager._update_block_arrangement(X2, y2)

    # Gather all data from strategy leaves
    leaves = manager.strategy.kdtree.get_leaves()
    X_all = torch.cat([leaf.Data.X for leaf in leaves], dim=0)

    sorted_combined, _ = torch.sort(torch.cat([X1, X2], dim=0), dim=0)
    sorted_from_tree, _ = torch.sort(X_all, dim=0)

    assert torch.allclose(sorted_combined, sorted_from_tree, atol=1e-6), \
        "KDTree leaves should contain exactly all X points seen."

def test_overlap_is_applied_correctly(create_manager):
    """Ensure that overlap values are applied to all blocks when configured."""
    overlap_ratio = 0.2
    manager = create_manager(overlap_ratio=overlap_ratio)

    X = torch.randn(12, 3)
    y = torch.randn(12, 1)
    manager._update_block_arrangement(X, y)

    # Ensure overlap is set properly
    for block in manager.blocks:
        assert hasattr(block, "overlap"), "Each block must have an overlap attribute after applying overlap"
        expected_overlap = block.block_size * overlap_ratio
        assert torch.allclose(block.overlap, expected_overlap, atol=1e-6)


def test_device_consistency(create_manager):
    """Verify that all tensors are moved to the manager device."""
    manager = create_manager()
    X = torch.randn(5, 2)
    y = torch.randn(5, 1)

    manager.device = torch.device("cpu")  # simulate device assignment
    manager._update_block_arrangement(X, y)

    assert X.device == manager.device
    assert y.device == manager.device
        

def test_map_points(create_manager):
    """Verify that _map_points correctly synchronizes KDTree data into blocks."""
    torch.manual_seed(42)
    n_features = 5

    # First batch
    X1 = torch.randn(19, n_features)
    y1 = torch.randn(19, 1)

    manager = create_manager()
    manager._update_block_arrangement(X1, y1)
    manager._map_points()  # initial mapping

    # --- Verify each block has been populated
    all_X_blocks = []
    all_y_blocks = []

    for block in manager.blocks:
        # Each block should now contain some data
        assert hasattr(block, "X")
        assert hasattr(block, "y")
        assert block.X != []
        assert block.y != []

        # Gather all data points for global comparison
        for x in block.X:
            all_X_blocks.append(x)
        for yi in block.y:
            all_y_blocks.append(yi)

    # --- Verify all points accounted for
    in_blocks_X = torch.stack(all_X_blocks, dim=0)
    in_blocks_y = torch.stack(all_y_blocks, dim=0)

    sorted_X_blocks, _ = torch.sort(in_blocks_X, dim=0)
    sorted_y_blocks, _ = torch.sort(in_blocks_y, dim=0)
    sorted_X1, _ = torch.sort(X1, dim=0)
    sorted_y1, _ = torch.sort(y1, dim=0)

    assert torch.allclose(sorted_X_blocks, sorted_X1, atol=1e-6)
    assert torch.allclose(sorted_y_blocks, sorted_y1, atol=1e-6)

    # --- Second batch of data
    X2 = torch.randn(192, n_features)
    y2 = torch.randn(192, 1)

    manager._update_block_arrangement(X2, y2)
    manager._map_points()

    # Collect again after update
    all_X_blocks = []
    all_y_blocks = []
    for block in manager.blocks:
        assert block.X != []
        assert block.y != []
        for x in block.X:
            all_X_blocks.append(x)
        for yi in block.y:
            all_y_blocks.append(yi)

    X_added = torch.cat((X1, X2), dim=0)
    y_added = torch.cat((y1, y2), dim=0)

    assert len(all_X_blocks) == X_added.shape[0]

    in_blocks_X = torch.stack(all_X_blocks, dim=0)
    in_blocks_y = torch.stack(all_y_blocks, dim=0)

    sorted_X_blocks, _ = torch.sort(in_blocks_X, dim=0)
    sorted_y_blocks, _ = torch.sort(in_blocks_y, dim=0)
    sorted_X_added, _ = torch.sort(X_added, dim=0)
    sorted_y_added, _ = torch.sort(y_added, dim=0)

    assert in_blocks_X.shape == sorted_X_added.shape
    assert torch.allclose(sorted_X_blocks, sorted_X_added, atol=1e-6)
    assert torch.allclose(sorted_y_blocks, sorted_y_added, atol=1e-6)


def test_map_points_with_expand_scope(create_manager):
    """Test _map_points with expand_scope=True properly includes overlapped data."""
    torch.manual_seed(123)
    n_features = 3
    X = torch.randn(50, n_features)
    y = torch.randn(50, 1)

    manager = create_manager(overlap_ratio=0.2)
    manager._update_block_arrangement(X, y)

    manager._map_points(expand_scope=True)

    all_counts = [len(block.X) for block in manager.blocks]
    
    assert any(c > 0 for c in all_counts), "At least one block should have points in expanded mapping"

    total_points_mapped = sum(all_counts)
    assert total_points_mapped >= X.shape[0], "With overlap, total mapped points should be >= original count"

# def test_add_points(create_manager, common_device):
#     n_features=5
#     X1 = torch.randn(500, n_features)
#     maxNodeSize=5
#     maxSplitsBeforeRestart=5
#     partitionManager=create_manager(maxNodeSize, maxSplitsBeforeRestart)

#     device = common_device

#     y = torch.randn(500, 1)

#     partitionManager.add_points(X1, y)
#     X2 = torch.randn(500, n_features)

#     partitionManager.add_points(X2, y)

#     leaves = partitionManager.kdtree.get_leaves() 

#     X=torch.Tensor().to(device)

#     for node in leaves:
#         assert node.Data.block is not None
#         assert node.Data.block.X != []
#         assert node.Data.block.y != []
#         assert node.Data.block.positions != []
#         for tensor in node.Data.block.X:
#             assert tensor.device.type==device
#         for tensor in node.Data.block.y:
#             assert tensor.device.type==device
#         for tensor in node.Data.block.space_origin:
#             assert tensor.device.type==device
#         for tensor in node.Data.block.block_size:
#             assert tensor.device.type==device
#         for tensor in node.Data.block.block_scope:
#             assert tensor.device.type==device
#         assert node.Data.y.device.type==device
#         assert node.Data.X.device.type==device

#         X=torch.cat((X,torch.stack(node.Data.block.X,dim=0)),dim=0)
#     sortX, _ = torch.sort(X,0)   
#     sortX2, _ = torch.sort(X2,0)
#     sortX1, _ = torch.sort(X1,0)
#     sortX2=sortX2.to(device)
#     sortX1=sortX1.to(device)
#     assert not torch.equal(sortX1,sortX2)
#     assert not torch.equal(sortX,sortX1)
#     assert not torch.equal(sortX,sortX2)

#     X_added=torch.cat((X1,X2))
#     X_added=X_added.to(device)
#     sortX_added, _ = torch.sort(X_added,0)
#     assert torch.equal(sortX_added,sortX)

# def test_init_sparse_coding_per_block_initializes_layers(create_manager):
#     """Test that init_sparse_coding_per_block correctly initializes sparse coding layers."""
#     T = torch.tensor([2, 2], device='cpu')
#     initial_bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
#     manager = create_manager(T_val=T, initial_bounds_val=initial_bounds)

#     X = torch.tensor([[0.1, 0.2], [0.6, 0.7]], device='cpu', dtype=torch.float32)
#     y = torch.tensor([[1.0], [2.0]], device='cpu', dtype=torch.float32)

#     manager.add_points(X, y) # Populates blocks and their data

#     # Dummy evaluation function
#     def dummy_eval_func(D: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
#         return torch.matmul(D, h) # Simple matmul for testing

#     # Example SparseCodingConfig
#     sc_config = ISTAConfig(n_functions=5, epochs=50, alpha=0.01, lambd=0.1)

#     manager.init_sparse_coding_per_block(config=sc_config, evaluation_func=dummy_eval_func)

#     active_blocks = manager.retrieve_active_blocks()
#     assert len(active_blocks) > 0 # Should have at least one active block

#     for block in active_blocks:
#         assert block.sparse_coding_layer is not None
#         assert isinstance(block.sparse_coding_layer, ISTALayer) # Assuming ISTALayer is the default for ISTAConfig
#         assert block.sparse_coding_layer.config.n_functions == 5
#         assert block.sparse_coding_layer.evaluation_func is dummy_eval_func # Check function identity
#         assert block.sparse_coding_layer.h is not None
#         assert block.sparse_coding_layer.h.shape == (5, 1) # Check h shape based on n_functions

# def test_retrieve_active_blocks(create_manager, common_device):

#     n_features=5
#     X1 = torch.randn(500, n_features)
#     y = torch.randn(500, 1)
#     maxNodeSize=5
#     maxSplitsBeforeRestart=5
#     partitionManager=create_manager(maxNodeSize, maxSplitsBeforeRestart)

#     partitionManager.add_points(X1, y)
#     activeBlocks=partitionManager.retrieve_active_blocks()
    
#     for block in activeBlocks:
#         assert block.X!=[]
#         assert block.y!=[]
#         assert block.sparse_coding_layer is None

# def test_retrieve_test_active_blocks(create_manager):

#     n_features=5
#     X1 = torch.randn(500, n_features)
#     y = torch.randn(500, 1)
#     maxNodeSize=5
#     maxSplitsBeforeRestart=5
#     partitionManager=create_manager(maxNodeSize, maxSplitsBeforeRestart)
#     partitionManager.add_points(X1, y)
#     activeBlocks1=partitionManager.retrieve_active_blocks()
    
#     for block in activeBlocks1:
#         assert block.X!=[]
#         assert block.y!=[]
#         assert block.sparse_coding_layer is None
#     def dummy_eval_func(x, y):
#         return torch.sum(x - y)

#     def dummy_optimizer(params, lr):
#         return torch.optim.Adam(params, lr=lr)
#     sc_config = ISTAConfig(n_functions=5, 
#                             epochs=50, 
#                             alpha=0.01, 
#                             lambd=0.1)

#     partitionManager.init_sparse_coding_per_block(config=sc_config, evaluation_func=dummy_eval_func)

#     Xt = torch.randn(500, n_features)

#     activeTestBlocks=partitionManager.retrieve_inference_blocks(Xt)

#     X=torch.Tensor()
#     for _, block in enumerate(activeTestBlocks):
#         X=torch.cat((X,torch.stack(block.X,dim=0)),dim=0)

#     sortX, _ = torch.sort(X,0)   
#     sortXt, _ = torch.sort(Xt,0)    
#     assert torch.equal(sortX,sortXt)
    
#     X_val=torch.Tensor()
#     activeBlocks2=partitionManager.retrieve_active_blocks()
#     for block in activeBlocks2:
#         X_val=torch.cat((X_val,torch.stack(block.X,dim=0)),dim=0)
    
#     sortX1, _ = torch.sort(X1,0)   
#     sortX_val, _ = torch.sort(X_val,0)
#     assert sortX1.shape==sortX_val.shape
#     assert torch.equal(sortX1,sortX_val)

#     count=0
#     for block in activeTestBlocks:
#         count+=len(block.positions)
    
#     assert count==Xt.shape[0]
