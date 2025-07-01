'''
Copyright (C) 2025 Tecnológico de Costa Rica

Class for a Node in a kd-tree 

Provides a kd-tree node 

Author: Hender Valdivia
'''

from typing import Callable
import torch
from pysesm.blocks.SESMData import SESMData

class Node():
    def __init__(self, Data: torch.Tensor, y: torch.Tensor, data_wrapper: Callable, parent=None):
        """
        Args:
            Data (torch.Tensor): Input dataset.
            parent (Node): Optional parent node.
            data_wrapper (Callable): A class or function to wrap/process the Data (e.g., KdSESMData).
        """
        self.Data: Callable = data_wrapper(Data, y)
        self.left: Node = None
        self.right: Node = None 
        self.parent: Node = parent

        
    

    