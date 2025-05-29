from pysesm.blocks.AdaptativePartitionManager import AdaptativePartitionManager
from pysesm.blocks.PartitionBlock import PartitionBlock
from pysesm.blocks import KDTree
from pysesm.blocks import Node
from pysesm.enums.DeviceTargetEnum import DeviceTarget
from pysesm.device_manager.DeviceManager import DeviceManager
from pysesm.models.ISTALayer import ISTALayer
import torch
import logging
import numpy as np
from pysesm.utils.loggers import setup_logger

def test_update_block_arrangement():
    X1 = torch.randn(192, 6)

    logger=setup_logger()
    device_map = {
        DeviceTarget.GLOBAL: "cpu",               # Dispositivo global por defecto
        DeviceTarget.ISTA_LAYER: "cpu",           # ISTA en GPU 0
        DeviceTarget.DICTIONARY_LAYER: "cpu",     # Dictionary en CPU
        DeviceTarget.PARTITION_MANAGER: "cpu"    # Partition Manager en CPU
    }
    device_manager=DeviceManager(logger,device_map=device_map)
    device = device_manager.get_device(DeviceTarget.PARTITION_MANAGER)

    partitionManager=AdaptativePartitionManager(logger,6, maxNodeSize=5, device_manager=device_manager)
    partitionManager._update_block_arrangement(X1)
    
    for block in partitionManager.blocks:
        assert block is not None
        assert block.X == []
    
    leaves=partitionManager.kdtree.get_leaves()
    X=torch.Tensor()
    for leaf in leaves:
        X=torch.cat((leaf.data, X))
    
    sorted_X, _ =torch.sort(X,0)
    sorted_X1, _ =torch.sort(X1[:,:-1],0)
    
    assert torch.equal(sorted_X, sorted_X1)    

    X2 = torch.randn(192, 6)
    partitionManager._update_block_arrangement(X2)
    
    X_added=torch.cat((X1,X2))

    for block in partitionManager.blocks:
        assert block is not None
        assert block.X == []
    
    leaves=partitionManager.kdtree.get_leaves()
    X=torch.Tensor()
    for leaf in leaves:
        X=torch.cat((leaf.data, X))
    
    sorted_X, _ =torch.sort(X,0)
    sorted_X1, _ =torch.sort(X_added[:,:-1],0)
    
    assert torch.equal(sorted_X, sorted_X1)    
        
def test_configure_blocks():
    """Test that _configure_blocks correctly sets up blocks."""
    T = torch.tensor([2, 2], device='cpu')
    n_functions = 2
    logger=setup_logger()
    manager = AdaptativePartitionManager(logger, n_functions, maxNodeSize=5)

    X = torch.tensor([[0.1, 0.2], [0.3, 0.4]], device='cpu')
    y = torch.tensor([[1.0], [2.0]], device='cpu')

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
    n_features=5
    torch.manual_seed(42) 
    X1 = torch.randn(19, n_features)
    y1 = torch.randn(19, 1)
    Xy=torch.cat((X1,y1),dim=1)
    logger=setup_logger()
    partitionManager=AdaptativePartitionManager(logger,n_features, maxNodeSize=5)
    partitionManager._update_block_arrangement(Xy)
    partitionManager._map_points(X1, y1)
    nodes=partitionManager.kdtree.get_leaves()
    in_blocks=[]
    in_blocks_y=[]
    for node in nodes:
        assert node.block.X != []
        assert node.block.y != []
        for x in node.block.X:
            in_blocks.append(x)
        for yi in node.block.y:
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

    partitionManager._update_block_arrangement(Xy2)
    partitionManager._map_points(X2, y2)
    
    leaves=partitionManager.kdtree.get_leaves()
    in_blocks=[]
    in_blocks_y=[]
    contador=0
    for node in leaves:
        assert node.block.X != []
        assert node.block.y != []
        assert node.block.positions != []
        for x in node.block.X:
            in_blocks.append(x)
            contador+=1
        for yi in node.block.y:
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

def test_add_points():
    logger=setup_logger()
    device_map = {
        DeviceTarget.GLOBAL: "cpu",               # Dispositivo global por defecto
        DeviceTarget.ISTA_LAYER: "cpu",           # ISTA en GPU 0
        DeviceTarget.DICTIONARY_LAYER: "cpu",     # Dictionary en CPU
        DeviceTarget.PARTITION_MANAGER: "cpu"    # Partition Manager en CPU
    }
    device_manager=DeviceManager(logger,device_map=device_map)
    device = device_manager.get_device(DeviceTarget.PARTITION_MANAGER)

    n_features=5
    X1 = torch.randn(500, n_features)

    logger=setup_logger()
    partitionManager=AdaptativePartitionManager(logger,n_features, maxNodeSize=5, device_manager=device_manager)

    y = torch.randn(500, 1)

    partitionManager.add_points(X1, y)
    X2 = torch.randn(500, n_features)

    partitionManager.add_points(X2, y)

    leaves = partitionManager.kdtree.get_leaves() 

    X=torch.Tensor().to(device)

    for node in leaves:
        assert node.block is not None
        assert node.block.X != []
        assert node.block.y != []
        assert node.block.positions != []
        for tensor in node.block.X:
            assert tensor.device.type==device
        for tensor in node.block.y:
            assert tensor.device.type==device
        for tensor in node.block.space_bound:
            assert tensor.device.type==device
        for tensor in node.block.block_size:
            assert tensor.device.type==device
        for tensor in node.block.block_scope:
            assert tensor.device.type==device
        assert node.y.device.type==device
        assert node.data.device.type==device

        X=torch.cat((X,torch.stack(node.block.X,dim=0)),dim=0)
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

def test_init_ista_per_block():
    """Test that init_ista_per_block correctly initializes ISTA layers."""
    T = torch.tensor([2, 2], device='cpu')
    n_functions = 2
    initial_bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], device='cpu')
    logger=setup_logger()
    device_map = {
        DeviceTarget.GLOBAL: "cpu",               # Dispositivo global por defecto
        DeviceTarget.ISTA_LAYER: "cpu",           # ISTA en GPU 0
        DeviceTarget.DICTIONARY_LAYER: "cpu",     # Dictionary en CPU
        DeviceTarget.PARTITION_MANAGER: "cpu"    # Partition Manager en CPU
    }
    device_manager=DeviceManager(logger,device_map=device_map)
    device = device_manager.get_device(DeviceTarget.PARTITION_MANAGER)
    manager = AdaptativePartitionManager(logger, n_functions,maxNodeSize=5, device_manager=device_manager)

    X = torch.tensor([[0.1, 0.2], [0.3, 0.4]], device='cpu')
    y = torch.tensor([[1.0], [2.0]], device='cpu')

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
        ista_optimizer=dummy_optimizer
    )

    for block in manager.blocks.flat:
        if hasattr(block, 'X') and len(block.X) > 0:
            assert hasattr(block, 'ista_layer')
            assert isinstance(block.ista_layer, ISTALayer)
            assert block.ista_layer.alpha == 0.01
            assert block.ista_layer.lambd == 0.1

    for leaf in manager.kdtree.get_leaves():
        assert hasattr(leaf.block, 'ista_layer')
        assert isinstance(block.ista_layer, ISTALayer)
        assert leaf.block.ista_layer is not None
        assert leaf.block.ista_layer.alpha == 0.01
        assert leaf.block.ista_layer.lambd == 0.1

def test_retrieve_active_blocks():
    logger=setup_logger()
    device_map = {
        DeviceTarget.GLOBAL: "cpu",               # Dispositivo global por defecto
        DeviceTarget.ISTA_LAYER: "cpu",           # ISTA en GPU 0
        DeviceTarget.DICTIONARY_LAYER: "cpu",     # Dictionary en CPU
        DeviceTarget.PARTITION_MANAGER: "cpu"    # Partition Manager en CPU
    }
    device_manager=DeviceManager(logger,device_map=device_map)
    device = device_manager.get_device(DeviceTarget.PARTITION_MANAGER)

    n_features=5
    X1 = torch.randn(500, n_features)
    y = torch.randn(500, 1)
    partitionManager=AdaptativePartitionManager(logger,n_features, maxNodeSize=5, device_manager=device_manager)
    partitionManager.add_points(X1, y)
    activeBlocks=partitionManager.retrieve_active_blocks()
    
    for block in activeBlocks:
        assert block.X!=[]
        assert block.y!=[]
        assert not (isinstance(block.ista_layer, ISTALayer))
        assert block.ista_layer is None

def test_retrieve_test_active_blocks():
    logger=setup_logger()
    device_map = {
        DeviceTarget.GLOBAL: "cpu",               # Dispositivo global por defecto
        DeviceTarget.ISTA_LAYER: "cpu",           # ISTA en GPU 0
        DeviceTarget.DICTIONARY_LAYER: "cpu",     # Dictionary en CPU
        DeviceTarget.PARTITION_MANAGER: "cpu"    # Partition Manager en CPU
    }
    device_manager=DeviceManager(logger,device_map=device_map)
    device = device_manager.get_device(DeviceTarget.PARTITION_MANAGER)

    n_features=5
    X1 = torch.randn(500, n_features)
    y = torch.randn(500, 1)
    partitionManager=AdaptativePartitionManager(logger,n_features, maxNodeSize=250, device_manager=device_manager)
    partitionManager.add_points(X1, y)
    activeBlocks1=partitionManager.retrieve_active_blocks()
    
    for block in activeBlocks1:
        assert block.X!=[]
        assert block.y!=[]
        assert not (isinstance(block.ista_layer, ISTALayer))
        assert block.ista_layer is None
    def dummy_eval_func(x, y):
        return torch.sum(x - y)

    def dummy_optimizer(params, lr):
        return torch.optim.Adam(params, lr=lr)
    partitionManager.init_ista_per_block(
        n_functions=2,
        ista_alpha=0.01,
        ista_lambd=0.1,
        evaluation_func=dummy_eval_func,
        ista_optimizer=dummy_optimizer
    )
    Xt = torch.randn(500, n_features)
    yt = torch.randn(500)

    activeTestBlocks=partitionManager.retrieve_test_active_blocks(Xt,yt)

    X=torch.Tensor()
    for i, block in enumerate(activeTestBlocks):
        X=torch.cat((X,torch.stack(block.X,dim=0)),dim=0)
        if (i+1)<len(activeTestBlocks):
            assert not torch.equal(block.h,activeTestBlocks[i+1].h)
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

