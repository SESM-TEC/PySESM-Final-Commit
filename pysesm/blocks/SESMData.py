'''
Copyright (C) 2025 Tecnológico de Costa Rica

Class for data needed in the Node of a kd-tree

Author: Hender Valdivia
'''
import torch
from pysesm.blocks.PartitionBlock import PartitionBlock

class SESMData():
    def __init__(self, X, y):
        self.X: torch.Tensor = X
        self.y: torch.Tensor = y
        self.split_point: torch.Tensor = None
        self.block: PartitionBlock = None
        self.test_data: torch.Tensor = None
        self.test_y: torch.Tensor = None
        self.overlap: torch.Tensor = None
        self._updateBounds()
        self.dim : int = self.greatestVarDim()

    def greatestVarDim(self) -> int:
        """Returns the dimension with the greatest variance of the dataset, 
           or -1 if no preferred dimension can be computed."""
        if self.X.size(0)>1:
            variances = self.X.var(dim=0)
            return torch.argmax(variances).item()            
        else:
            return  -1 # Flag that no valid dimension selection was possible
        
    def _updateBounds(self):
        "Update the bounds to fit the internal data"
        upperBounds, _ = torch.max(self.X, dim=0)
        lowerBounds, _ = torch.min(self.X, dim=0)
        self.bounds = torch.stack((lowerBounds, upperBounds),dim=0)   