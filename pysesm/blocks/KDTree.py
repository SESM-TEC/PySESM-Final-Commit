"""
KD-Tree implementation.

Provides a kd-tree data structure.

Author: Hender Valdivia
Copyright (C) 2025, Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
"""

from typing import Callable
import torch
from pysesm.blocks.Node import Node

class dummyData():
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X: torch.Tensor=X
        self.y: torch.Tensor=y
        self.split_point: float=None
        self.test_data: torch.Tensor=None
        self.test_y: torch.Tensor=None
        self.test_indices: torch.Tensor=None
        self.dim: int=0

    def empty_data(self):
        self.X = None

class KDTree():
    def __init__(self, data: torch.Tensor, y: torch.Tensor, maxNodeSize: int = 500, data_wrapper: Callable = dummyData,device=None):
        """
        data (Tensor): Tensor holding all data points
        root (Node): Root node of the tree.
        maxNodeSize (int): If a node has more than maxNodeSize points it is split into two nodes.
        data_wrapper (Callable): The object used in the nodes to store the data.
        split (bool): Flag to indicate whether any node has been split  points added after the initial KD-tree build.
        device (string or None): Device where internal tensors will be stored.
        """
        self.device=device
        self.root: Node = Node(data.to(self.device), y, data_wrapper, self.device)
        self.maxNodeSize: int = maxNodeSize
        self.data_wrapper : callable = data_wrapper
        self.split_after_add: bool = False

        self._splitDataInNodes(self.root)
        leaves = self.get_leaves()
        total_points = sum([0 if leaf.Data.X is None else leaf.Data.X.size(0) for leaf in leaves])
        print(f"[KDTree init] leaves={len(leaves)}, total_points={total_points}")
        
    def _splitDataInNodes(self, node: Node) -> None:
        """
        Splits data based on the median of the greatest variance dimension. 
        It stops when the children nodes have maxNodeSize points or less
        """
        
        if node is None or node.Data.X.size(0) <= self.maxNodeSize:
            return      
        # Calculate median only for the dimension we need
        node.Data.split_point = torch.median(node.Data.X[:,node.Data.dim]).item()

        # Create combined data (necessary for mask)
        data=torch.cat((node.Data.X, node.Data.y), dim=1)

        # Create mask of data over threshold only once and reuse its inverse
        mask = data[:, node.Data.dim] >= node.Data.split_point
        not_mask = ~mask
        node.right = Node(data[mask][:,:-1],data[mask][:,-1:], self.data_wrapper, self.device)
        node.left = Node(data[not_mask][:, :-1],data[not_mask][:, -1:], self.data_wrapper, self.device)        

        node.Data.empty_data()
        
        self._splitDataInNodes(node.left)
        self._splitDataInNodes(node.right)

    def _find_node(self, x : torch.Tensor, node = None) -> Node:
        """
        Finds the node where a point should be located based on split points and the greatest variance dimensions of each node.
        Args:
            x (torch.Tensor): one-dimensional tensor
            node (Node): Starting node
        """
        if node is None:
            node=self.root
        if node.Data.X is None:
            if x[node.Data.dim].item() >= node.Data.split_point:
                return self._find_node(x, node.right)
            if x[node.Data.dim].item() < node.Data.split_point:
                return self._find_node(x, node.left)
        else:
            return node

    def add_point(self, x : torch.Tensor, y: torch.Tensor) -> None:
        """
        Adds a point to the KDTree in the corresponding node.
        If the node exceeds maxNodeSize, the node is split.

        x: one-dimensional tensor
        """
        x=x.to(self.device)
        y=y.to(self.device)
        node = self._find_node(x)

        x=x.unsqueeze(0)
        y=y.unsqueeze(0)
        node.Data.X=torch.cat((node.Data.X,x))
        node.Data.y=torch.cat((node.Data.y,y))
        node.Data.updateBounds()

        if not (node.Data.X.size(0) <= self.maxNodeSize):
            self.split_after_add=True
            self._splitDataInNodes(node)
        leaves = self.get_leaves()
        total_points = sum([0 if leaf.Data.X is None else leaf.Data.X.size(0) for leaf in leaves])
        print(f"[KDTree add_point] after add total_leaves={len(leaves)}, total_points={total_points}")

    
    def get_leaves(self,  leaves : list = None, node = None) -> list[Node]:
        """
        Finds leaves of the tree and returns them in a list

        Args:
            leaves (list): A list with some already added leaf nodes.
            node (Node): The starting node, usually the root node.
        """
        if leaves is None:
            leaves=[]
            
        if node is None:
            node = self.root
        
        if node.left is None and node.right is None:
            leaves.append(node)

        if node.left is not None:
            leaves = self.get_leaves(leaves, node.left)
        
        if node.right is not None:
            leaves = self.get_leaves(leaves, node.right)
        return leaves

    def splitDataInNodes_test(self, node : Node = None):
        """
        Splits test data without changing the structure of the kdtree.

        Args:
            node (Node): Starting node, usually the root node
        """
        if node is None:
            node=self.root
        if node.Data.test_data is None or node.Data.X is not None:
            return
        test_Data=torch.cat((node.Data.test_data,node.Data.test_y),dim=1)
        mask = test_Data[:, node.Data.dim] >= node.Data.split_point

        node.right.Data.test_data = test_Data[mask][:,:-1]
        node.right.Data.test_y = test_Data[mask][:,-1:]
        node.right.Data.test_indices = node.Data.test_indices[mask]

        node.left.Data.test_data = test_Data[~mask][:,:-1]
        node.left.Data.test_y = test_Data[~mask][:,-1:]
        node.left.Data.test_indices = node.Data.test_indices[~mask]

        node.Data.test_data = None
        node.Data.test_y = None
        node.Data.test_indices = None
        
        self.splitDataInNodes_test(node.left)
        self.splitDataInNodes_test(node.right)
        
        return


    def _get_global_minimum(self) -> torch.Tensor:
        """
        Returns the global minimum values across all dimensions in the KDTree.
        
        This function traverses all leaf nodes and finds the minimum value
        for each dimension across the entire dataset.
        
        Returns:
            torch.Tensor: A tensor containing the minimum value for each dimension.
                        Shape: (n_features,)
        """
        leaves = self.get_leaves()
        
        if not leaves:
            raise ValueError("KDTree has no leaf nodes")
        
        # Initialize with the first leaf's lower bounds
        first_leaf = leaves[0]
        if first_leaf.Data.bounds is None:
            raise ValueError("Leaf nodes have no bounds information")
        
        global_min = first_leaf.Data.bounds[0].clone()  # Lower bounds of first leaf
        
        # Compare with all other leaves' lower bounds
        for leaf in leaves[1:]:
            if leaf.Data.bounds is not None:
                leaf_min = leaf.Data.bounds[0]  # Lower bounds of this leaf
                global_min = torch.min(global_min, leaf_min)
        
        return global_min.to(self.device) if self.device else global_min

    def _get_global_maximum(self) -> torch.Tensor:
        """
        Returns the global maximum values across all dimensions in the KDTree.
        
        This function traverses all leaf nodes and finds the maximum value
        for each dimension across the entire dataset.
        
        Returns:
            torch.Tensor: A tensor containing the maximum value for each dimension.
                        Shape: (n_features,)
        """
        leaves = self.get_leaves()
        
        if not leaves:
            raise ValueError("KDTree has no leaf nodes")
        
        # Initialize with the first leaf's upper bounds
        first_leaf = leaves[0]
        if first_leaf.Data.bounds is None:
            raise ValueError("Leaf nodes have no bounds information")
        
        global_max = first_leaf.Data.bounds[1].clone()  # Upper bounds of first leaf
        
        # Compare with all other leaves' upper bounds
        for leaf in leaves[1:]:
            if leaf.Data.bounds is not None:
                leaf_max = leaf.Data.bounds[1]  # Upper bounds of this leaf
                global_max = torch.max(global_max, leaf_max)
        
        return global_max.to(self.device) if self.device else global_max

    def get_global_bounds(self) -> torch.Tensor:
        """
        Returns the global bounds (min and max) across all dimensions in the KDTree.
        
        Returns:
            torch.Tensor: A tensor of shape [2, n_features] where:
                        - bounds[0] contains the minimum values for each dimension
                        - bounds[1] contains the maximum values for each dimension
        """
        global_min = self._get_global_minimum()
        global_max = self._get_global_maximum()
        
        return torch.stack((global_min, global_max), dim=0)
