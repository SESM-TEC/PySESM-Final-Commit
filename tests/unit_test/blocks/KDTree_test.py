from pysesm.blocks.Node import Node
from pysesm.blocks.KDTree import KDTree
import torch

def test_KDTree_initialization():
    return

# def test_greatestVarDim():
#     x = torch.randn(20, 5)  
#     kd=KDTree(x)
#     dim = kd.greatestVarDim(x)

#     variances = x.var(dim=0)

#     dim_test = torch.argmax(variances).item()

#     assert dim==dim_test

def test_splitDataInNodes():
    torch.manual_seed(42) 
    x = torch.randn(15, 6) 
    kd=KDTree(x)
    dim = 2

    kd.splitDataInNodes(kd.root) 
    preorder_assert(kd.root, kd.threshold)
    
def preorder_assert(node, threshold):

    if node.left is not None:
        print()
        assert (node.left.data[:, node.dim] < node.split_point).all()
        if (node.left.right is not None) and (node.left.left is not None):
            preorder_assert(node.left, threshold)

    if node.right is not None:
        assert (node.right.data[:, node.dim] > node.split_point).all()
        if (node.right.right is not None) and (node.right.left is not None):
            preorder_assert(node.right, threshold)
