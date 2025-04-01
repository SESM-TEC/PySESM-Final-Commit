from pysesm.blocks.AdaptativePartitionManager import AdaptativePartitionManager
from pysesm.blocks.PartitionBlock import PartitionBlock
from pysesm.blocks import KDTree
from pysesm.blocks import Node
from pysesm.models.ISTALayer import ISTALayer
import torch
import logging
import numpy as np
from pysesm.utils.loggers import setup_logger

def test_update_block_arrangement():
    X1 = torch.randn(19, 6)

    logger=setup_logger()
    partitionManager=AdaptativePartitionManager(logger,6)
    partitionManager._update_block_arrangement(X1)

    for block in partitionManager.blocks:
        assert block is not None
    
    X2 = torch.randn(19, 6)
    partitionManager._update_block_arrangement(X2)

    for block in partitionManager.blocks:
        assert block is not None

def test_map_points():
    n_features=6
    torch.manual_seed(42) 
    X = torch.randn(191, n_features)
    y=1
    logger=setup_logger()
    partitionManager=AdaptativePartitionManager(logger,n_features)
    partitionManager._update_block_arrangement(X)
    partitionManager._map_points(X, y)
    nodes=partitionManager.kdtree.get_leaves()
    in_blocks=[]
    for node in nodes:
        assert node.block.X != []
        for x in node.block.X:
            in_blocks.append(x)
    in_blocks = torch.stack(in_blocks, dim=0)
    in_blocks, _ =torch.sort(in_blocks,0)
    sort_X, _ = torch.sort(X,0)
    assert torch.equal(in_blocks,sort_X)

def test_add_points():
    X1 = torch.randn(19, 6)

    logger=setup_logger()
    partitionManager=AdaptativePartitionManager(logger,6)
    
    y = torch.randn(19, 1)

    partitionManager.add_points(X1, y)

    X2 = torch.randn(19, 6)

    partitionManager.add_points(X2, y)

    leaves = partitionManager.kdtree.get_leaves() 

    X=torch.Tensor()

    for node in leaves:
        assert node.block is not None
        X=torch.cat((X,torch.stack(node.block.X,dim=0)),dim=0)
    sortX, _ = torch.sort(X,0)   
    sortX2, _ = torch.sort(X2,0)
    sortX1, _ = torch.sort(X1,0)

    assert not torch.equal(sortX1,sortX2)
    assert not torch.equal(sortX,sortX1)
    assert not torch.equal(sortX,sortX2)

    X_added=torch.cat((X1,X2))
    sortX_added, _ = torch.sort(X_added,0)

    assert torch.equal(sortX_added,sortX)

