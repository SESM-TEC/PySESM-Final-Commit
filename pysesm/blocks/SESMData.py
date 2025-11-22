"""
SESMData

Class for data needed in the Node of a kd-tree

Author: Hender Valdivia
Copyright (C) 2025 Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
"""
import torch
from pysesm.blocks.PartitionBlock import PartitionBlock

class SESMData():
    """
    Encapsulates data as needed by the kd-tree to hold a level-wise 
    access to data in SESM.
    """
    def __init__(self, X, y, device):
        self.X: torch.Tensor = X
        self.y: torch.Tensor = y
        self.split_point: torch.Tensor = None
        self.idx: tuple = (0,)
        self.test_data: torch.Tensor = None
        self.test_y: torch.Tensor = None
        self.test_indices: torch.Tensor = None
        self.updateBounds()
        self.dim : int = self.greatestVarDim()
        self.block: PartitionBlock = PartitionBlock(
                self.bounds[0],
                self.idx,
                self.bounds[1]-self.bounds[0],
                device=device)

    def greatestVarDim(self) -> int:
        """Returns the dimension with the greatest variance of the dataset, 
           or -1 if no preferred dimension can be computed."""
        if self.X.size(0)>1:
            variances = self.X.var(dim=0)
            return torch.argmax(variances).item()            
        else:
            return  -1 # Flag that no valid dimension selection was possible
        
    def updateBounds(self):
        "Update the bounds to fit the internal data"
        upperBounds, _ = torch.max(self.X, dim=0)
        lowerBounds, _ = torch.min(self.X, dim=0)
        self.bounds = torch.stack((lowerBounds, upperBounds),dim=0)  

    def empty_data(self):
        self.X = None
        self.y = None
        self.bounds = None
        self.block=None
        self.test_data = None
        self.test_y = None
        self.test_indices = None
