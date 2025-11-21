import logging 

import pytest

import torch
import numpy as np

from pysesm.blocks.Node import Node
from pysesm.blocks.SESMData import SESMData
from pysesm.blocks.KDTree import KDTree
from pysesm.blocks.AdaptativePartitionManager import AdaptativePartitionManager, AdaptativePartitionConfig
from pysesm.blocks.KDTreeStrategy import KDTreeStrategy, KDTreeStrategyConfig

logger = logging.getLogger("test_uniform_partition_manager")
logger.setLevel(logging.DEBUG)

@pytest.fixture(scope="module")
def common_device():
    """Provides a shared device for all tests in this module."""   
    return "cpu"

@pytest.fixture
def create_KDTree(common_device):
    """
    Factory fixture to create UniformPartitionManager instances with flexible config.
    Ensures initial_bounds are consistently passed as numpy arrays to the config.
    """
    def _creator(data: torch.Tensor, y: torch.Tensor, maxNodeSize: int=5):
        # Convert torch.Tensor bounds to numpy array for UniformPartitionConfig
        return KDTree(
            data=data,
            y=y,
            maxNodeSize=maxNodeSize,
            data_wrapper=SESMData,
            device=common_device
        )
    return _creator

@pytest.fixture
def create_manager(common_device):
    """
    Factory fixture to create UniformPartitionManager instances with flexible config.
    Ensures initial_bounds are consistently passed as numpy arrays to the config.
    """
    def _creator():
        # Convert torch.Tensor bounds to numpy array for UniformPartitionConfig
        strategyConfig = KDTreeStrategyConfig(
            maxNodeSize=5,
            data_wrapper=SESMData,
            device=common_device
        )
        strategy = KDTreeStrategy
        config = AdaptativePartitionConfig(
            partition_strategy=strategy,
            strategy_config=strategyConfig,
            overlap_ratio=None
        )
        return AdaptativePartitionManager(
            config=config,
            logger=logger # Use the module-level logger for the manager            
        )
    return _creator

def test_greatestVarDim(common_device):
    """
    Asserts the greatest variance dimension is calculated correctly in the node
    """
    device = common_device
    x = torch.randn(20, 5).to(device)
    y = torch.randn(20, 1).to(device)    
    node=Node(x,y, SESMData, device)
    dim = node.Data.greatestVarDim()

    variances = x.var(dim=0)
    dim_test = torch.argmax(variances).item()
    assert dim==dim_test

def test_add_points_integrity(create_manager):
    """Checks that all points are preserved through multiple KDTree additions."""
    torch.manual_seed(42)
    n_features = 5

    # Create the manager with KDTreeStrategy inside
    manager = create_manager()
    strategy = manager.strategy

    # --- First batch ---
    X1 = torch.randn(50, n_features)
    y1 = torch.randn(50, 1)
    strategy.build(X1, y1)

    kd = strategy.kdtree
    assert_kdtree_integrity(kd, X1.shape[0])

    # --- Second batch ---
    X2 = torch.randn(150, n_features)
    y2 = torch.randn(150, 1)
    strategy.add_points(X2, y2)

    # After adding both batches, all points must be present
    total_expected = X1.shape[0] + X2.shape[0]
    assert_kdtree_integrity(kd, total_expected)

    # --- Compare concatenated data ---
    all_X, all_y = strategy.get_all_points()
    sorted_X_blocks, _ = torch.sort(all_X, dim=0)
    sorted_y_blocks, _ = torch.sort(all_y, dim=0)
    sorted_X_added, _ = torch.sort(torch.cat((X1, X2), dim=0), dim=0)
    sorted_y_added, _ = torch.sort(torch.cat((y1, y2), dim=0), dim=0)

    assert torch.allclose(sorted_X_blocks, sorted_X_added, atol=1e-6)
    assert torch.allclose(sorted_y_blocks, sorted_y_added, atol=1e-6)

def assert_kdtree_integrity(kdtree: KDTree, expected_points: int):
    """Asserts that the KDTree preserves all points after insertions."""
    leaves = kdtree.get_leaves()
    total_points = 0
    for leaf in leaves:
        assert leaf.Data is not None, "Leaf has no Data object!"
        if leaf.Data.X is None:
            raise AssertionError("Leaf has Data.X = None (data lost)")
        total_points += leaf.Data.X.size(0)
    assert total_points == expected_points, (
        f"KDTree contains {total_points} points, expected {expected_points}"
    )


def test_splitDataInNodes(create_KDTree, common_device):
    """
    Tests the splitDataInNodes function which basically initializes the KDTree
    """
    device = common_device
    torch.manual_seed(42) 
    x = torch.randn(501, 6).to(device)
    y = torch.randn(501, 1).to(device)
    kd=create_KDTree(x,y)
    defaultMaxNodeSize = kd.maxNodeSize

    Data=torch.Tensor().to(kd.device)
    treeNodes=kd.get_leaves()
    for node in treeNodes:
        Data=torch.cat((Data,node.Data.X))

    sortx, _ = torch.sort(Data,0)
    sortData, _ = torch.sort(x,0)

    assert torch.equal(sortData,sortx)
    leaves=kd.get_leaves()
    defaultMaxNodeSize = round(defaultMaxNodeSize/2)
    kd.maxNodeSize=defaultMaxNodeSize
    for leaf in leaves:
        if leaf.Data.X.size()[0] > defaultMaxNodeSize:
            kd._splitDataInNodes(leaf)

    leaves2=kd.get_leaves()
    assert [leaf.Data.X for leaf in leaves]!=[leaf.Data.X for leaf in leaves2]
    

def test_find_node(create_KDTree):
    torch.manual_seed(42) 
    X = torch.randn(191, 6)
    y = torch.randn(191, 1)
    kd=create_KDTree(X,y)
    x=torch.rand(6)

    node=kd._find_node(x)

    node_test=kd.root
    if node_test.Data.X is None:
        if x[node_test.Data.dim].item() >= node_test.Data.split_point:
            node_test= kd._find_node(x, node_test.right)
        elif x[node_test.Data.dim].item() < node_test.Data.split_point:
            node_test = kd._find_node(x, node_test.left)

    assert torch.equal(node.Data.X, node_test.Data.X)   

    return

def test_add_point(create_manager):
    n_features=5
    torch.manual_seed(42) 
    maxNodeSize=5
    maxSplitsBeforeRestart=5
    X = torch.randn(500, n_features)
    y = torch.randn(500, 1)
    partitionManager=create_manager()
    partitionManager._update_block_arrangement(X, y)
    kd=partitionManager.strategy.kdtree
    for _ in range(1):
        x=torch.rand(n_features)
        y=torch.rand(1)
        kd.add_point(x,y)

        x=x.unsqueeze(0)
        y=y.unsqueeze(0)
        X=torch.cat((X,x))
        Data=torch.Tensor()
        treeNodes=kd.get_leaves()
        for node in treeNodes:
            Data=torch.cat((Data,node.Data.X))
        
        sortx, _ = torch.sort(Data,0)
        sortData, _ = torch.sort(X,0)
        assert torch.equal(sortData,sortx)
    leaves=kd.get_leaves()

    for leaf in leaves:                 #If split nodes, check child nodes have blocks defined
        assert leaf.Data.block is not None

    x=torch.rand(n_features)
    y=torch.rand(1)    
    kd.add_point(x,y)
    Data=torch.Tensor()
    treeNodes=kd.get_leaves()
    for node in treeNodes:
        Data=torch.cat((Data,node.Data.X))
    sortx, _ = torch.sort(Data,0)   
    sortData, _ = torch.sort(X,0)

    assert not torch.equal(sortData,sortx)

def test_get_leaves(create_KDTree):
    n_features=5
    torch.manual_seed(42) 
    X = torch.randn(191, n_features)
    y = torch.randn(191, 1)
    kd=create_KDTree(X,y)

    leaves=kd.get_leaves()

    leaves_data=torch.Tensor()

    for node in leaves:
        assert node.Data.X is not None
        leaves_data = torch.cat((leaves_data, node.Data.X))
    
    sortx, _ = torch.sort(leaves_data,0)   
    sortData, _ = torch.sort(X,0)

    assert torch.equal(sortData,sortx)
    
    for _ in range(15):
        x=torch.rand(n_features)
        y=torch.rand(1)

        kd.add_point(x, y)
        x=x.unsqueeze(0)
        X=torch.cat((X,x))

    leaves=kd.get_leaves()
    leaves_data1 = torch.empty(0, n_features) 

    for node in leaves:
        assert node.Data.X is not None
        leaves_data1 = torch.cat((leaves_data1, node.Data.X))
    
    sortx, _ = torch.sort(leaves_data1,0)   
    sortData, _ = torch.sort(X,0)

    assert torch.equal(sortData,sortx)

    x=torch.rand(n_features)
    y=torch.rand(1)
    kd.add_point(x,y)

    leaves=kd.get_leaves()
    leaves_data2 = torch.empty(0, n_features) 

    for node in leaves:
        assert node.Data.X is not None
        leaves_data2 = torch.cat((leaves_data2, node.Data.X))
    
    sortx, _ = torch.sort(leaves_data2,0)   
    sortData, _ = torch.sort(X,0)
    
    assert not torch.equal(sortData,sortx)



    

    
