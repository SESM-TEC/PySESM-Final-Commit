from pysesm.blocks.Node import Node
import torch

class KDTree():
    def __init__(self, data: torch.Tensor):
        """
        threshold: Value added to the split point to avoid splitting a data point.
        """
        self.threshold=0.0001
        self.root = Node(data)
        self.splitDataInNodes(self.root)

    def splitDataInNodes(self, node: Node):
        """Splits data based on the median of the greatest variance dimension and adds a threshold to avoid
        splitting at an exact point. It is recursive and stops when the data of the node is less than 5 points"""
        medians = torch.median(node.data, dim=0).values
        node.split_point = medians[node.dim].item()+self.threshold
        mask = node.data[:, node.dim] > node.split_point

        greaterData = node.data[mask]
        lowerData = node.data[~mask]

        node.right = Node(greaterData)
        node.left = Node(lowerData)

        if node.left.data.size()[0] > 5:
            self.splitDataInNodes(node.left)
        if node.right.data.size()[0] > 5:
            self.splitDataInNodes(node.right)

