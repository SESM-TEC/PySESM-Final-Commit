'''
Copyright (C) 2025 Tecnológico de Costa Rica

Class for a kd-tree implementation

Provides a kd-tree data structure.

Author: Hender Valdivia
'''

from pysesm.blocks.Node import Node
from pysesm.blocks.PartitionBlock import PartitionBlock
from typing import Union
import torch

class KDTree():
    def __init__(self, data: torch.Tensor, maxNodeSize: int = 500, device=None):
        """
        data (Tensor): Tensor holding all data points
        maxNodeSize (int): If a node has more than maxNodeSize points it is split into two nodes.
        device (string or None): Device where internal tensors will be stored.
        """
        self.device=device
        self.root = Node(data.to(self.device))
        self.maxNodeSize=maxNodeSize
        self.total_nodes=1
        
        self._splitDataInNodes(self.root)
        
    def _splitDataInNodes(self, node: Node) -> None:
        """
        Splits data based on the median of the greatest variance dimension. 
        It stops when the children nodes have 5 points or less
        """
        if node is None or node.data.size()[0] <= self.maxNodeSize:
            return      

        assert(self.maxNodeSize>0)

        # Calculate median only for the dimension we need
        node.split_point = torch.median(node.data[:,node.dim]).item()

        # Create combined data (necessary for interface)
        data=torch.cat((node.data, node.y), dim=1)

        # Create mask of data over threshold only once and reuse its inverse
        mask = data[:, node.dim] >= node.split_point
        not_mask = ~mask
        
        node.right = Node(data[mask])
        node.left = Node(data[not_mask])        
       
        node.data = None
        node.bounds = None
        node.block=None
        
        self._splitDataInNodes(node.left)
        self._splitDataInNodes(node.right)

    def _find_node(self, x : torch.Tensor, node = None) -> Node:
        """
        Finds the node where a point should be located based on split points and the greatest variance dimensions of each node.

        x: one-dimensional tensor
        """
        if node is None:
            node=self.root
        if node.data is None:
            if x[node.dim].item() >= node.split_point:
                return self._find_node(x, node.right)
            if x[node.dim].item() < node.split_point:
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
        node.data=torch.cat((node.data,x))
        node.y=torch.cat((node.y,y))

        if node.data.size()[0] > self.maxNodeSize:
            self._splitDataInNodes(node)

            left_bound=node.left.bounds[0]-node.left.bounds[1]
            right_bound=node.right.bounds[0]-node.right.bounds[1]
            node.left.block=PartitionBlock(
                node.left.bounds[1],
                (1,),
                left_bound,
                device=self.device)
            
            node.right.block=PartitionBlock(
                node.right.bounds[1],
                (2,),
                right_bound,
                device=self.device)

    def find_block(self, point : torch.Tensor) -> Union[PartitionBlock, None]:
        """
        Finds the node where a given point is. If no node has the given point, returns None
        
        point: One-dimensional tensor
        """
        
        node = self._find_node(point)

        isPointInNode = torch.any(torch.all(node.data == point, dim=1))

        if isPointInNode:
            return node    ##CHECK: THIS MUST RETURN A PARTITIONBLOCK
            
        return None
    
    def get_leaves(self,  leaves : list = None, node = None) -> list:
        """
        Finds leaves of the tree and returns them in a list
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
