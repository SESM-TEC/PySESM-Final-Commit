from pysesm.blocks.Node import Node
import torch

class KDTree():
    def __init__(self, data: torch.Tensor):
        self.threshold=0.0001
        dim=self.greatestVarDim(data)
        self.root = Node(data)
        #self.splitDataInNodes(self.root, dim, self.threshold)

    def greatestVarDim(self, data : torch.Tensor):
        variances = data.var(dim=0)
        dim = torch.argmax(variances).item()

        return dim

    def splitDataInNodes(self, node: Node, dim: int, threshold : float):
        """Splits data based on the median of the greatest variance dimension and adds a threshold to avoid
        splitting at an exact point"""
        medians = torch.median(node.data, dim=0).values
        split = medians[dim].item()+threshold
        mask = node.data[:, dim] > split

        greaterData = node.data[mask]
        lowerData = node.data[~mask]

        node.right = Node(greaterData)
        node.left = Node(lowerData)
