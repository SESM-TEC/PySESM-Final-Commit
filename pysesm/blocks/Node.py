'''
Copyright (C) 2025 Tecnológico de Costa Rica

Class for a Node in a kd-tree 

Provides a kd-tree node 

Author: Hender Valdivia
'''

import torch
from pysesm.blocks.KdSESMData import KdSESMData

class Node():
    def __init__(self, Data : torch.Tensor, parent = None):
        """
        This is a node of a tree, it has standard node attributes:
            data: Dataset holding all but the last columns of the given Data
            y:    Labels, corresponding to the last column of the given Data
            right, left: Right and left child nodes
            
        Aditionally, nodes have some attributes needed for its specific use:
        
            split_point (float): Limit value in the greatestVarDim that is used to split the data. 
            dim (int): Dimension where the data has the greatest variance
            bounds (torch.Tensor): Space limits of each dimension.  
        """       
        initial_bounds=None
        if parent is not None:
            initial_bounds=parent.bounds
        self.Data=KdSESMData(Data, initial_bounds)
        self.left=None
        self.right=None 
        self.parent=parent

        
    def __getattr__(self, attribute):
        return getattr(self.Data, attribute)

    