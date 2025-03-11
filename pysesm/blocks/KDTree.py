from pysesm.blocks.Node import Node
import torch

class KDTree():
    def __init__(self, data: torch.Tensor, maxNodeSize: int = 5):
        """
        root (Node): Root node of the tree
        maxNodeSize (int): If a node has more than maxNodeSize points it is split into two nodes.
        """

        self.root = Node(data)
        self.maxNodeSize=maxNodeSize

        self.splitDataInNodes(self.root)
        
        return

    def splitDataInNodes(self, node: Node):
        """
        Splits data based on the median of the greatest variance dimension. 
        It stops when the children nodes have less than 5 points
        """

        medians = torch.median(node.data, dim=0).values
        node.split_point = medians[node.dim].item()
        
        mask = node.data[:, node.dim] >= node.split_point
        not_mask = node.data[:, node.dim] < node.split_point
        
        greaterData = node.data[mask].clone()
        lowerData = node.data[not_mask].clone()

        node.data = None

        node.right = Node(greaterData)
        node.left = Node(lowerData)

        if node.left.data.size()[0] > self.maxNodeSize:
            self.splitDataInNodes(node.left)

        if node.right.data.size()[0] > self.maxNodeSize:
            self.splitDataInNodes(node.right)
        return

    def add_point(self, x : torch.Tensor, node = None):
        """
        Add point to the KDTree in the corresponding node.
        If the node exceeds maxNodeSize then split it.
        """

        if node == None:
            node=self.root

        if node.data is None:
            if x[0, node.dim].item() >= node.split_point:
                self.add_point(x, node.right)
            if x[0, node.dim].item() < node.split_point:
                self.add_point(x, node.left)
        else:
            node.data=torch.cat((node.data,x))
            if node.data.size()[0] > self.maxNodeSize:
                self.splitDataInNodes(node)



