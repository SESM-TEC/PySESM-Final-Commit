import torch
import logging
import numpy as np
import pytest
from typing import Union, Optional

from pysesm.blocks.AdaptativePartitionManager import AdaptativePartitionManager,AdaptativePartitionConfig
from pysesm.blocks.PartitionBlock import PartitionBlock
from pysesm.blocks import KDTree
from pysesm.blocks import Node
from pysesm.enums.DeviceTargetEnum import DeviceTarget
from pysesm.device_manager.DeviceManager import DeviceManager
from pysesm.utils.loggers import setup_logger
from pysesm.sparse_coding.ISTALayer import ISTALayer, ISTAConfig

logger = logging.getLogger("test_uniform_partition_manager")
logger.setLevel(logging.DEBUG)


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
    def _creator(T_val: Union[int, torch.Tensor],
                 initial_bounds_val: Optional[Union[np.ndarray, torch.Tensor]]=None,
                 threshold_val: float=0):
        # Convert torch.Tensor bounds to numpy array for UniformPartitionConfig
        if isinstance(initial_bounds_val, torch.Tensor):
            initial_bounds_np = initial_bounds_val.cpu().numpy()
        else:
            initial_bounds_np = initial_bounds_val # If already numpy or None

        config = AdaptativePartitionConfig(
            maxNodeSize=5,
            maxSplitsBeforeRestart=5,
            overlap_ratio=None
        )
        return AdaptativePartitionManager(
            config=config,
            logger=logger, # Use the module-level logger for the manager
            device_manager=common_device_manager
        )
    return _creator

def test_update_block_arrangement(create_manager):
    X1 = torch.randn(19, 6)
    y1 = torch.randn(19, 1)
    maxNodeSize=5
    maxSplitsBeforeRestart=5
    partitionManager=create_manager(maxNodeSize, maxSplitsBeforeRestart)
    partitionManager._update_block_arrangement(X1, y1)
    
    for block in partitionManager.blocks:
        assert block is not None
        assert block.X == []
    
    leaves=partitionManager.kdtree.get_leaves()
    X=torch.Tensor()
    for leaf in leaves:
        X=torch.cat((leaf.Data.X, X))
    
    sorted_X, _ =torch.sort(X,0)
    sorted_X1, _ =torch.sort(X1,0)
    
    assert torch.equal(sorted_X, sorted_X1)    

    X2 = torch.randn(19, 6)
    y2 = torch.randn(19, 1)
    partitionManager._update_block_arrangement(X2,y2)
    
    X_added=torch.cat((X1,X2))

    for block in partitionManager.blocks:
        assert block is not None
        assert block.X == []
    
    leaves=partitionManager.kdtree.get_leaves()
    X=torch.Tensor()
    for leaf in leaves:
        X=torch.cat((leaf.Data.X, X))
    
    sorted_X, _ =torch.sort(X,0)
    sorted_X1, _ =torch.sort(X_added,0)
    
    assert torch.equal(sorted_X, sorted_X1)    
        

def test_map_points(create_manager):
    n_features=5
    torch.manual_seed(42) 
    X1 = torch.randn(19, n_features)
    y1 = torch.randn(19, 1)
    Xy=torch.cat((X1,y1),dim=1)
    maxNodeSize=5
    maxSplitsBeforeRestart=5
    partitionManager=create_manager(maxNodeSize, maxSplitsBeforeRestart)
    partitionManager._update_block_arrangement(X1, y1)
    partitionManager._map_points()
    nodes=partitionManager.kdtree.get_leaves()
    in_blocks=[]
    in_blocks_y=[]
    for node in nodes:
        assert node.Data.block.X != []
        assert node.Data.block.y != []
        for x in node.Data.block.X:
            in_blocks.append(x)
        for yi in node.Data.block.y:
            in_blocks_y.append(yi)
    for block in partitionManager.blocks:
        assert block.X !=[]
    in_blocks = torch.stack(in_blocks, dim=0)
    in_blocks_y = torch.stack(in_blocks_y, dim=0)
    in_blocks, _ =torch.sort(in_blocks,0)
    in_blocks_y, _ =torch.sort(in_blocks_y,0)
    sort_X, _ = torch.sort(X1,0)
    sort_y, _ = torch.sort(y1,0)

    assert torch.equal(in_blocks,sort_X)
    assert torch.equal(in_blocks_y,sort_y)
    
    X2 = torch.randn(192, n_features)
    y2 = torch.randn(192, 1)
    Xy2=torch.cat((X2,y2),dim=1)

    partitionManager._update_block_arrangement(X2, y2)
    partitionManager._map_points()
    
    leaves=partitionManager.kdtree.get_leaves()
    in_blocks=[]
    in_blocks_y=[]
    contador=0
    for node in leaves:
        assert node.Data.block.X != []
        assert node.Data.block.y != []
        assert node.Data.block.positions != []
        for x in node.Data.block.X:
            in_blocks.append(x)
            contador+=1
        for yi in node.Data.block.y:
            in_blocks_y.append(yi)
    for block in partitionManager.blocks:
        assert block.X !=[]
    X_added=torch.cat((X1,X2))
    assert len(in_blocks)==X_added.shape[0]
    y_added=torch.cat((y2,y1))
    in_blocks = torch.stack(in_blocks, dim=0)
    in_blocks_y = torch.stack(in_blocks_y, dim=0)
    in_blocks, _ =torch.sort(in_blocks,0)
    in_blocks_y, _ =torch.sort(in_blocks_y,0)
    sort_X, _ = torch.sort(X_added,0)
    sort_y, _ = torch.sort(y_added,0)

    assert in_blocks.shape==sort_X.shape
    assert torch.equal(in_blocks,sort_X)
    assert torch.equal(in_blocks_y,sort_y)

def test_add_points(create_manager, common_device_manager):
    n_features=5
    X1 = torch.randn(500, n_features)
    maxNodeSize=5
    maxSplitsBeforeRestart=5
    partitionManager=create_manager(maxNodeSize, maxSplitsBeforeRestart)

    device = common_device_manager.get_device(DeviceTarget.PARTITION_MANAGER)

    y = torch.randn(500, 1)

    partitionManager.add_points(X1, y)
    X2 = torch.randn(500, n_features)

    partitionManager.add_points(X2, y)

    leaves = partitionManager.kdtree.get_leaves() 

    X=torch.Tensor().to(device)

    for node in leaves:
        assert node.Data.block is not None
        assert node.Data.block.X != []
        assert node.Data.block.y != []
        assert node.Data.block.positions != []
        for tensor in node.Data.block.X:
            assert tensor.device.type==device
        for tensor in node.Data.block.y:
            assert tensor.device.type==device
        for tensor in node.Data.block.space_origin:
            assert tensor.device.type==device
        for tensor in node.Data.block.block_size:
            assert tensor.device.type==device
        for tensor in node.Data.block.block_scope:
            assert tensor.device.type==device
        assert node.Data.y.device.type==device
        assert node.Data.X.device.type==device

        X=torch.cat((X,torch.stack(node.Data.block.X,dim=0)),dim=0)
    sortX, _ = torch.sort(X,0)   
    sortX2, _ = torch.sort(X2,0)
    sortX1, _ = torch.sort(X1,0)
    sortX2=sortX2.to(device)
    sortX1=sortX1.to(device)
    assert not torch.equal(sortX1,sortX2)
    assert not torch.equal(sortX,sortX1)
    assert not torch.equal(sortX,sortX2)

    X_added=torch.cat((X1,X2))
    X_added=X_added.to(device)
    sortX_added, _ = torch.sort(X_added,0)
    assert torch.equal(sortX_added,sortX)

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

def test_retrieve_active_blocks(create_manager, common_device_manager):

    n_features=5
    X1 = torch.randn(500, n_features)
    y = torch.randn(500, 1)
    maxNodeSize=5
    maxSplitsBeforeRestart=5
    partitionManager=create_manager(maxNodeSize, maxSplitsBeforeRestart)

    partitionManager.add_points(X1, y)
    activeBlocks=partitionManager.retrieve_active_blocks()
    
    for block in activeBlocks:
        assert block.X!=[]
        assert block.y!=[]
        assert block.sparse_coding_layer is None

def test_retrieve_test_active_blocks(create_manager):

    n_features=5
    X1 = torch.randn(500, n_features)
    y = torch.randn(500, 1)
    maxNodeSize=5
    maxSplitsBeforeRestart=5
    partitionManager=create_manager(maxNodeSize, maxSplitsBeforeRestart)
    partitionManager.add_points(X1, y)
    activeBlocks1=partitionManager.retrieve_active_blocks()
    
    for block in activeBlocks1:
        assert block.X!=[]
        assert block.y!=[]
        assert block.sparse_coding_layer is None
    def dummy_eval_func(x, y):
        return torch.sum(x - y)

    def dummy_optimizer(params, lr):
        return torch.optim.Adam(params, lr=lr)
    sc_config = ISTAConfig(n_functions=5, 
                            epochs=50, 
                            alpha=0.01, 
                            lambd=0.1)

    partitionManager.init_sparse_coding_per_block(config=sc_config, evaluation_func=dummy_eval_func)

    Xt = torch.randn(500, n_features)
    yt = torch.randn(500)

    activeTestBlocks=partitionManager.retrieve_test_active_blocks(Xt,yt)

    X=torch.Tensor()
    for _, block in enumerate(activeTestBlocks):
        X=torch.cat((X,torch.stack(block.X,dim=0)),dim=0)

    sortX, _ = torch.sort(X,0)   
    sortXt, _ = torch.sort(Xt,0)    
    assert torch.equal(sortX,sortXt)
    
    X_val=torch.Tensor()
    activeBlocks2=partitionManager.retrieve_active_blocks()
    for block in activeBlocks2:
        X_val=torch.cat((X_val,torch.stack(block.X,dim=0)),dim=0)
    
    sortX1, _ = torch.sort(X1,0)   
    sortX_val, _ = torch.sort(X_val,0)
    assert sortX1.shape==sortX_val.shape
    assert torch.equal(sortX1,sortX_val)

    count=0
    for block in activeTestBlocks:
        count+=len(block.positions)
    
    assert count==len(yt)
