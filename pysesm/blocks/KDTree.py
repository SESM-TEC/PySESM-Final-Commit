from pysesm.blocks.Node import Node

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
        
        self._splitDataInNodes(self.root)
        
    def _splitDataInNodes(self, node: Node):
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
        if node.left.data.size()[0] > self.maxNodeSize:
            self._splitDataInNodes(node.left)

        if node.right.data.size()[0] > self.maxNodeSize:
            self._splitDataInNodes(node.right)
    
    def _set_children_bounds(self, node):
        if node.bounds == None:
            upperBounds, _ = torch.max(node.data, dim=0)
            lowerBounds, _ = torch.min(node.data, dim=0)
            bounds = torch.stack((upperBounds,lowerBounds),dim=0)
            node.bounds=bounds

        node.left.bounds=node.bounds.clone()
        node.right.bounds=node.bounds.clone()

        node.left.bounds[0, node.dim]=node.split_point
        node.right.bounds[1, node.dim]=node.split_point

    def _find_node(self, x, node = None):
        """
        Finds the node where a point should be located based on split points and the greatest variance dimensions of each node.
        """
        if node == None:
            node=self.root

        if node.data is None:
            if x[0, node.dim].item() >= node.split_point:
                return self._find_node(x, node.right)
            if x[0, node.dim].item() < node.split_point:
                return self._find_node(x, node.left)
        else:
            return node

    def add_point(self, x : torch.Tensor):
        """
        Adds a point to the KDTree in the corresponding node.
        If the node exceeds maxNodeSize then split it.
        """

        node = self._find_node(x)
        
        node.data=torch.cat((node.data,x))
        if node.data.size()[0] > self.maxNodeSize:
            self._splitDataInNodes(node)

    def find_block(self, point : torch.Tensor):
        """
        Finds the node where a given point is. If no node has the given point, returns None
        """
        
        node = self._find_node(point)

        isPointInNode = torch.any(torch.all(node.data == point, dim=1))

        if isPointInNode:
            return node
            
        return None
    
    def get_leaves(self,  leaves : list = [], node = None):
        if node is None:
            node = self.root
        
        if node.data is not None:
            leaves.append(node)

        if node.left is not None:
            self.get_leaves(leaves, node.left)
        
        if node.right is not None:
            self.get_leaves(leaves, node.right)

        return leaves

        


    
        
    



