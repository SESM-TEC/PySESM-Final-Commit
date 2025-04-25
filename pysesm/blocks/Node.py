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
        self._updateBounds()
        self.dim=self.greatestVarDim()
    
    def greatestVarDim(self):
        """Returns the dimension with the greatest variance of the dataset"""
        if self.data.size(0)>1:
            variances = self.data.var(dim=0)
            return torch.argmax(variances).item()            
        else:
            return  -1 # Flag that no valid dimension selection was possible
        
    def _updateBounds(self):
        "Update the bounds to fit the internal data"
        upperBounds, _ = torch.max(self.data, dim=0)
        lowerBounds, _ = torch.min(self.data, dim=0)
        self.bounds = torch.stack((upperBounds,lowerBounds),dim=0)        
        
