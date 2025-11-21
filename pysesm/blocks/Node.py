'''
Class for a Node in a kd-tree 

Provides a kd-tree node 

Copyright (C) 2025 Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause

Author: Hender Valdivia
'''

from typing import Callable
import torch

class Node():
    def __init__(self, Data: torch.Tensor, y: torch.Tensor, data_wrapper: Callable, device, parent=None):
        """
        Args:
            Data (torch.Tensor): Input dataset.
            parent (Node): Optional parent node.
            data_wrapper (Callable): A class or function to wrap/process the Data (e.g., KdSESMData).
        """
        self.Data: Callable = data_wrapper(Data, y, device)
        self.left: Node = None
        self.right: Node = None 
        self.parent: Node = parent

        
    

    
