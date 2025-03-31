from pysesm.blocks.Node import Node
from pysesm.blocks.PartitionBlock import PartitionBlock
from typing import Union
import torch

class KDTree():
    def __init__(self, data: torch.Tensor, device=None, maxNodeSize: int = 5):
        """
        root (Node): Root node of the tree
        maxNodeSize (int): If a node has more than maxNodeSize points it is split into two nodes.
        """
        self.device=device
        self.root = Node(data)
        self.maxNodeSize=maxNodeSize
        self.total_nodes=1
        
        self._splitDataInNodes(self.root)
        
    def _splitDataInNodes(self, node: Node) -> None:
        """
        Splits data based on the median of the greatest variance dimension. 
        It stops when the children nodes have 5 points or less
        """
        if node is None:
            return

        medians = torch.median(node.data, dim=0).values
        node.split_point = medians[node.dim].item()
        
        mask = node.data[:, node.dim] >= node.split_point
        not_mask = node.data[:, node.dim] < node.split_point
        
        greaterData = node.data[mask].clone()
        lowerData = node.data[not_mask].clone()


        node.right = Node(greaterData)
        node.left = Node(lowerData)        

        self._set_children_bounds(node)
        
        node.data = None
        node.bounds = None
        node.block=None
        if node.left.data.size()[0] > self.maxNodeSize:
            self._splitDataInNodes(node.left)

        if node.right.data.size()[0] > self.maxNodeSize:
            self._splitDataInNodes(node.right)
    
    def _set_children_bounds(self, node) -> None:
        """
        Defines the bounds of child nodes based on parent bounds
        """
        if node.bounds == None:
            upperBounds, _ = torch.max(node.data, dim=0)
            lowerBounds, _ = torch.min(node.data, dim=0)
            bounds = torch.stack((upperBounds,lowerBounds),dim=0)
            node.bounds=bounds

        node.left.bounds=node.bounds.clone()
        node.right.bounds=node.bounds.clone()

        node.left.bounds[0, node.dim]=node.split_point
        node.right.bounds[1, node.dim]=node.split_point

    def _find_node(self, x : torch.Tensor, node = None) -> Node:
        """
        Finds the node where a point should be located based on split points and the greatest variance dimensions of each node.

        x: one-dimensional tensor
        """
        if node == None:
            node=self.root
        if node.data is None:
            if x[node.dim].item() >= node.split_point:
                return self._find_node(x, node.right)
            if x[node.dim].item() < node.split_point:
                return self._find_node(x, node.left)
        else:
            return node

    def add_point(self, x : torch.Tensor) -> None:
        """
        Adds a point to the KDTree in the corresponding node.
        If the node exceeds maxNodeSize then split it and create the child blocks.

        x: one-dimensional tensor
        """

        node = self._find_node(x)

        x=x.unsqueeze(0)
        node.data=torch.cat((node.data,x))

        if node.data.size()[0] > self.maxNodeSize:
            self._splitDataInNodes(node)

            left_bound=node.left.bounds[0]-node.left.bounds[1]
            right_bound=node.right.bounds[0]-node.right.bounds[1]
            node.left.block=PartitionBlock(
                node.left.bounds[1],
                (1,),
                left_bound)
            
            node.right.block=PartitionBlock(
                node.right.bounds[1],
                (2,),
                right_bound)
            print(f"Left block: {node.left.block}")  # Check if the block exists
            print(f"Right block: {node.right.block}")

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
    
    #def get_bounds():

        


    
        
    



