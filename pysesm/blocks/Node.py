import torch

class Node():
    def __init__(self, Data, dim=None, threshold=None):
        self.data=Data
        self.left=None
        self.right=None 
        #self.parent=None
        self.split_point=None
        self.dim=self.greatestVarDim(Data)
    
    def greatestVarDim(self, data : torch.Tensor):
        variances = data.var(dim=0)
        dim = torch.argmax(variances).item()
        return dim
    