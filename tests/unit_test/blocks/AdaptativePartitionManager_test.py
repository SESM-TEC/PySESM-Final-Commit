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
    X1 = torch.randn(19, 6)

    logger=setup_logger()
    device_map = {
        DeviceTarget.GLOBAL: "cpu",               # Dispositivo global por defecto
        DeviceTarget.ISTA_LAYER: "cpu",           # ISTA en GPU 0
        DeviceTarget.DICTIONARY_LAYER: "cpu",     # Dictionary en CPU
        DeviceTarget.PARTITION_MANAGER: "cuda"    # Partition Manager en CPU
    }
    device_manager=DeviceManager(logger,device_map=device_map)
    device = device_manager.get_device(DeviceTarget.PARTITION_MANAGER)

    partitionManager=AdaptativePartitionManager(logger,6,device_manager=device_manager)
    partitionManager._update_block_arrangement(X1)

    for block in partitionManager.blocks:
        assert block is not None
        
    
    X2 = torch.randn(19, 6)
    partitionManager._update_block_arrangement(X2)

    for block in partitionManager.blocks:
        assert block is not None
        
def test_configure_blocks():
    """Test that _configure_blocks correctly sets up blocks."""
    T = torch.tensor([2, 2], device='cpu')
    n_functions = 2
    logger=setup_logger()
    manager = AdaptativePartitionManager(logger, n_functions)

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
    X = torch.randn(6, n_features)
    y = torch.randn(6, 1)
    Xy=torch.cat((X,y),dim=1)
    logger=setup_logger()
    partitionManager=AdaptativePartitionManager(logger,n_features)
    partitionManager._update_block_arrangement(Xy)
    partitionManager._map_points(X, y)
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
    in_blocks = torch.stack(in_blocks, dim=0)
    in_blocks_y = torch.stack(in_blocks_y, dim=0)
    in_blocks, _ =torch.sort(in_blocks,0)
    in_blocks_y, _ =torch.sort(in_blocks_y,0)
    sort_X, _ = torch.sort(X,0)
    sort_y, _ = torch.sort(y,0)

    assert torch.equal(in_blocks,sort_X)
    assert torch.equal(in_blocks_y,sort_y)

def test_add_points():
    logger=setup_logger()
    device_map = {
        DeviceTarget.GLOBAL: "cpu",               # Dispositivo global por defecto
        DeviceTarget.ISTA_LAYER: "cpu",           # ISTA en GPU 0
        DeviceTarget.DICTIONARY_LAYER: "cpu",     # Dictionary en CPU
        DeviceTarget.PARTITION_MANAGER: "cuda"    # Partition Manager en CPU
    }
    device_manager=DeviceManager(logger,device_map=device_map)
    device = device_manager.get_device(DeviceTarget.PARTITION_MANAGER)

    n_features=5
    X1 = torch.randn(19, n_features)

    logger=setup_logger()
    partitionManager=AdaptativePartitionManager(logger,n_features, device_manager=device_manager)

    y = torch.randn(19, 1)

    partitionManager.add_points(X1, y)

    X2 = torch.randn(19, n_features)

    partitionManager.add_points(X2, y)

    leaves = partitionManager.kdtree.get_leaves() 

    X=torch.Tensor().to(device)

    for node in leaves:
        assert node.block is not None
        assert node.block.X != []
        assert node.block.y != []
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

