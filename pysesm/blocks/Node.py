"""
KD-Tree Node.

Provides a Node class for the KD-tree data structure.

Author: Hender Valdivia
Copyright (C) 2025 Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
"""

from typing import Callable
import torch

class Node():
    def __init__(self, data_object, parent=None):
        """
        Args:
            data_object: Instance of the data wrapper (e.g. SESMData).
            parent (Node): Optional parent node.
        """
        self.Data = data_object
        self.left: Node = None
        self.right: Node = None 
        self.parent: Node = parent

        
    

    
