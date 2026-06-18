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
from pysesm.blocks.DataContainer import DataContainer

class SESMData(DataContainer):
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
                block_index=self.idx,
                block_size=self.bounds[1]-self.bounds[0],
                block_scope=self.bounds,
                device=device)

    def size(self) -> int:
        """Returns the number of data points currently held."""
        return self.X.size(0) if self.X is not None else 0

    def append(self, x: torch.Tensor, y: torch.Tensor):
        """Appends a single point (or batch) to the dataset."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)

        # Incremental bounds update (O(1) vs O(N))
        if self.bounds is not None:
            new_min, _ = torch.min(x, dim=0)
            new_max, _ = torch.max(x, dim=0)
            
            updated_min = torch.min(self.bounds[0], new_min)
            updated_max = torch.max(self.bounds[1], new_max)
            self.bounds = torch.stack((updated_min, updated_max), dim=0)
            
            # Update the block's scope reference
            if self.block is not None:
                self.block.block_scope = self.bounds
        else:
            # First points added
            self.X = x
            self.updateBounds()

        self.X = torch.cat((self.X, x))
        self.y = torch.cat((self.y, y))

    def split(self):
        """
        Splits the current data into two new SESMData objects based on the 
        median of the greatest variance dimension.
        """
        self.split_point = torch.median(self.X[:, self.dim]).item()
        
        mask = self.X[:, self.dim] >= self.split_point
        not_mask = ~mask
        
        right_data = self.__class__(self.X[mask], self.y[mask], self.block.device)
        left_data = self.__class__(self.X[not_mask], self.y[not_mask], self.block.device)
        
        return left_data, right_data

    def push_test_data_to_children(self, left_data, right_data):
        """
        Splits and moves the test data residing in this object to the provided children data objects.
        """
        if self.test_data is None:
            return

        mask = self.test_data[:, self.dim] >= self.split_point
        not_mask = ~mask

        right_data.test_data = self.test_data[mask]
        right_data.test_y = self.test_y[mask]
        right_data.test_indices = self.test_indices[mask]

        left_data.test_data = self.test_data[not_mask]
        left_data.test_y = self.test_y[not_mask]
        left_data.test_indices = self.test_indices[not_mask]

        self.clear_payload() # Clear test data from self
        
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

    def clear_payload(self):
        """Clears the data tensors from this node to free memory."""
        self.X = None
        self.y = None
        self.bounds = None
        self.block=None
        self.test_data = None
        self.test_y = None
        self.test_indices = None
