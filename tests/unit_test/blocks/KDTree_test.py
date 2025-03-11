from pysesm.blocks.Node import Node
from pysesm.blocks.KDTree import KDTree
import torch

def test_greatestVarDim():
    """
    Asserts the greatest variance dimension is calculated correctly in the node
    """

    x = torch.randn(20, 5)  
    node=Node(x)
    dim = node.greatestVarDim(x)

    variances = x.var(dim=0)

    dim_test = torch.argmax(variances).item()

    assert dim==dim_test
    return


def test_splitDataInNodes():
    """
    Tests the splitDataInNodes function which basically initializes the KDTree
    """
    torch.manual_seed(42) 
    x = torch.randn(191, 6)

    kd=KDTree(x)
    Data=torch.Tensor()
    Data=preorder_assert(kd.root, Data, kd.maxNodeSize)
    
    sortx, _ = torch.sort(Data,0)
    sortData, _ = torch.sort(x,0)

    assert torch.equal(sortData,sortx)

def preorder_assert(node, Data, maxNodeSize):  
    """
    Asserts the nodes that are not in the lowest level have no data,
    Asserts that no node has only a left or only a right child.
    Asserts every node with data has the maximum node size or less.
    Returns the concatenated data from the lowest-level nodes.
    """
    if node.left is not None:
        assert node.right is not None

        if (node.left.right is not None) and (node.left.left is not None):
            Data=preorder_assert(node.left, Data, maxNodeSize)
            
        else:
            Data=torch.cat((Data, node.left.data))
            assert (node.left.data[:, node.dim] < node.split_point).all()
            assert node.left.data.size()[0] <= maxNodeSize

    if node.right is not None:
        assert node.left is not None

        if (node.right.right is not None) and (node.right.left is not None):
            Data=preorder_assert(node.right, Data, maxNodeSize)
        
        else:
            Data=torch.cat((Data, node.right.data))
            assert (node.right.data[:, node.dim] >= node.split_point).all()
            assert node.left.data.size()[0] <= maxNodeSize
    
    return Data

def test_add_point():
    torch.manual_seed(42) 
    X = torch.randn(191, 6)

    kd=KDTree(X)
    for _ in range(5):
        x=torch.rand(1,6)
        kd.add_point(x)

        X=torch.cat((X,x))

        Data=torch.Tensor()
        Data=preorder_assert(kd.root, Data, kd.maxNodeSize)
        
        sortx, _ = torch.sort(Data,0)
        sortData, _ = torch.sort(X,0)

    assert torch.equal(sortData,sortx)