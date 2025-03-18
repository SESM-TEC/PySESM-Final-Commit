from pysesm.blocks.AdaptativePartitionManager import AdaptativePartitionManager
from pysesm.blocks.PartitionBlock import PartitionBlock
from pysesm.blocks import KDTree
from pysesm.blocks import Node
from pysesm.models.ISTALayer import ISTALayer
import torch
import logging
import numpy as np
from pysesm.utils.loggers import setup_logger



# def test_add_points():
#     X = torch.randn(191, 6)
#     shape = X.shape  # (2,3)
#     indices = [torch.arange(s) for s in shape]  # Create ranges for rows and cols
#     cartesian_indices = torch.cartesian_prod(*indices)  # Cartesian product
#     for index in cartesian_indices.tolist():  # Convert tensor rows to tuples
#         print(tuple(index)) 
#     logger=setup_logger()
#     partitionManager=AdaptativePartitionManager(logger,1,10)
    
#     y = torch.randn(191, 1)

#     partitionManager.add_points(X, y)

    