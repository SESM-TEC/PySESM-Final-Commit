from pysesm.blocks.AdaptativePartitionManager import AdaptativePartitionManager
from pysesm.blocks.PartitionBlock import PartitionBlock
from pysesm.blocks import KDTree
from pysesm.blocks import Node
from pysesm.models.ISTALayer import ISTALayer
import torch
import logging
import numpy as np
from pysesm.utils.loggers import setup_logger



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
        assert node.data is not None
        X=torch.cat((X,node.data),dim=0)
    sortX, _ = torch.sort(X,0)   
    sortX2, _ = torch.sort(X2,0)
    sortX1, _ = torch.sort(X1,0)

    assert not torch.equal(sortX1,sortX2)
    assert not torch.equal(sortX,sortX1)
    assert not torch.equal(sortX,sortX2)

    X_added=torch.cat((X1,X2))
    sortX_added, _ = torch.sort(X_added,0)

    assert torch.equal(sortX_added,sortX)

