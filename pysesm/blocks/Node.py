'''
Copyright (C) 2025 Tecnológico de Costa Rica

Class for a Node in a kd-tree 

Provides a kd-tree node 

Author: Hender Valdivia
'''

import torch

class Node():
    def __init__(self, Data : torch.Tensor, bounds: torch.tensor=None):
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
        self.data=Data[:,:-1]
        self.y=Data[:,-1:]
        self.left=None
        self.right=None 
        self.split_point=None
        self.block=None
        self.bounds=bounds
        self.dim=self.greatestVarDim(self.data)
    
    def greatestVarDim(self, data : torch.Tensor):
        """Returns the dimension with the greatest variance of the dataset"""
        if data.size(0)>1:
            variances = data.var(dim=0)
            dim = torch.argmax(variances).item()
            return dim
        else:
            return  -1 # Flag that no valid dimension selection was possible
        
