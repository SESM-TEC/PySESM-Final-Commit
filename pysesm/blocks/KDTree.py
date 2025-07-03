'''
Copyright (C) 2025 Tecnológico de Costa Rica

Class for a kd-tree implementation

Provides a kd-tree data structure.

Author: Hender Valdivia
'''
from typing import Callable

from pysesm.blocks.Node import Node
from pysesm.blocks.PartitionBlock import PartitionBlock
from pysesm.blocks.SESMData import SESMData
import torch

class dummyData():
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X: torch.Tensor=X
        self.y: torch.Tensor=y
        self.split_point: float=None
        self.block: PartitionBlock=None
        self.test_data: torch.Tensor=None
        self.test_y: torch.Tensor=None
        self.dim: int=0
        self._updateBounds()
    def _updateBounds(self):
        "Update the bounds to fit the internal data"
        upperBounds, _ = torch.max(self.X, dim=0)
        lowerBounds, _ = torch.min(self.X, dim=0)
        self.bounds = torch.stack((upperBounds,lowerBounds),dim=0)   

class KDTree():
    def __init__(self, data: torch.Tensor, y: torch.Tensor, maxNodeSize: int = 500, data_wrapper: Callable = dummyData,device=None):
        """
        data (Tensor): Tensor holding all data points
        maxNodeSize (int): If a node has more than maxNodeSize points it is split into two nodes.
        device (string or None): Device where internal tensors will be stored.
        """
        self.device=device
        self.root: Node = Node(data.to(self.device), y, data_wrapper)
        self.maxNodeSize: int = maxNodeSize
        self.data_wrapper=data_wrapper

        self._splitDataInNodes(self.root)
        
    def _splitDataInNodes(self, node: Node) -> None:
        """
        Splits data based on the median of the greatest variance dimension. 
        It stops when the children nodes have maxNodeSize points or less
        """
        
        if node is None or node.Data.X.size(0) <= self.maxNodeSize:
            return      
        # Calculate median only for the dimension we need
        node.Data.split_point = torch.median(node.Data.X[:,node.Data.dim]).item()

        # Create combined data (necessary for interface)
        data=torch.cat((node.Data.X, node.Data.y), dim=1)

        # Create mask of data over threshold only once and reuse its inverse
        mask = data[:, node.Data.dim] >= node.Data.split_point
        not_mask = ~mask
        node.right = Node(data[mask][:,:-1],data[mask][:,-1:], self.data_wrapper)
        node.left = Node(data[not_mask][:, :-1],data[not_mask][:, -1:], self.data_wrapper)        

        node.Data.X = None
        node.Data.bounds = None
        node.Data.block=None
        
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
        If the node exceeds maxNodeSize then split it and create the child blocks.

        x: one-dimensional tensor
        """
        x=x.to(self.device)
        y=y.to(self.device)
        node = self._find_node(x)

        x=x.unsqueeze(0)
        y=y.unsqueeze(0)
        node.Data.X=torch.cat((node.Data.X,x))
        node.Data.y=torch.cat((node.Data.y,y))

        if not (node.Data.X.size(0) <= self.maxNodeSize):
            self._splitDataInNodes(node)

            left_bound=node.left.Data.bounds[0]-node.left.Data.bounds[1]
            right_bound=node.right.Data.bounds[0]-node.right.Data.bounds[1]
            node.left.Data.block=PartitionBlock(
                node.left.Data.bounds[1],
                (1,),
                left_bound,
                device=self.device)

            node.right.Data.block=PartitionBlock(
                node.right.Data.bounds[1],
                (2,),
                right_bound,
                device=self.device)
            assert node.right.Data.block is not None
            assert node.left.Data.block is not None
    
    def get_leaves(self,  leaves : list = None, node = None) -> list:
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

    def _splitDataInNodes_test(self, node : Node):
        """
        Splits test data without changing the structure of the kdtree

        Args:
            node (Node): Starting node, usually the root node
        """
        if node.Data.test_data is None or node.Data.X is not None:
            return
        test_Data=torch.cat((node.Data.test_data,node.Data.test_y),dim=1)
        mask = test_Data[:, node.Data.dim] >= node.Data.split_point

        node.right.Data.test_data = test_Data[mask][:,:-1]
        node.right.Data.test_y = test_Data[mask][:,-1:]

        node.left.Data.test_data = test_Data[~mask][:,:-1]
        node.left.Data.test_y = test_Data[~mask][:,-1:]

        node.Data.test_data = None
        node.Data.test_y = None
        
        self._splitDataInNodes_test(node.left)
        self._splitDataInNodes_test(node.right)
        
        return