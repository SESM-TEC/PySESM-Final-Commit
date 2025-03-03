from pysesm.blocks.Node import Node
from pysesm.blocks.KDTree import KDTree
import torch

def test_KDTree_initialization():
    return

def test_greatestVarDim():
    x = torch.randn(20, 5)  
    kd=KDTree(x)
    dim = kd.greatestVarDim(x)

    variances = x.var(dim=0)

    dim_test = torch.argmax(variances).item()

    assert dim==dim_test

def test_splitDataInNodes():
    x = torch.randn(25, 6) 
    kd=KDTree(x)
    dim = 2
    
    medians = torch.median(x, dim=0).values
    split = medians[dim].item()+kd.threshold

    kd.splitDataInNodes(kd.root, dim, kd.threshold) 

    assert (kd.root.right.data[:, dim] > split).all()
    assert (kd.root.left.data[:, dim] < split).all()
    