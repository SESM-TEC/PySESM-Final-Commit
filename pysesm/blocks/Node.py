import torch

class Node():
    def __init__(self, Data : torch.Tensor):
        """
        This is a node of a tree, it has standard node attributes:
            data: Dataset
            right, left: Right and left child nodes
            
        Aditionally, nodes have some attributes needed for its specific use: 
            split_point (float): Limit value in the greatestVarDim that is used to split the data. 
            dim (int): Dimension where the data has the greatest variance
        """
        self.data=Data
        self.left=None
        self.right=None 
        self.split_point=None
        self.dim=self.greatestVarDim(Data)
    
    def greatestVarDim(self, data : torch.Tensor):
        """Returns the dimension with the greatest variance of the dataset"""
        variances = data.var(dim=0)
        dim = torch.argmax(variances).item()
        return dim