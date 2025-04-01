from pysesm.blocks.Node import Node
from pysesm.blocks.KDTree import KDTree
from pysesm.enums.DeviceTargetEnum import DeviceTarget
from pysesm.device_manager.DeviceManager import DeviceManager
from pysesm.utils.loggers import setup_logger
import torch
import logging 

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
    x = torch.randn(15, 6)
    logger=setup_logger()
    device_map = {
        DeviceTarget.GLOBAL: "cpu",               # Dispositivo global por defecto
        DeviceTarget.ISTA_LAYER: "cpu",           # ISTA en GPU 0
        DeviceTarget.DICTIONARY_LAYER: "cpu",     # Dictionary en CPU
        DeviceTarget.PARTITION_MANAGER: "cpu"    # Partition Manager en CPU
    }
    device_manager=DeviceManager(logger,device_map=device_map)
    device = device_manager.get_device(DeviceTarget.PARTITION_MANAGER)
    kd=KDTree(x,device=device)
    Data=torch.Tensor()
    Data=preorder_assert(kd.root, Data, kd.maxNodeSize)
    
    sortx, _ = torch.sort(Data,0)
    sortData, _ = torch.sort(x,0)

    assert torch.equal(sortData,sortx)
    leaves=kd.get_leaves()
    for leaf in leaves:
        if leaf.data.size()[0] == 5:
            kd._splitDataInNodes(leaf)

    leaves2=kd.get_leaves()
    assert leaves!=leaves2
    

def test_find_node():
    torch.manual_seed(42) 
    X = torch.randn(191, 6)

    kd=KDTree(X)
    x=torch.rand(6)

    node=kd._find_node(x)

    node_test=kd.root
    if node_test.data is None:
        if x[node_test.dim].item() >= node_test.split_point:
            node_test= kd._find_node(x, node_test.right)
        elif x[node_test.dim].item() < node_test.split_point:
            node_test = kd._find_node(x, node_test.left)

    assert torch.equal(node.data, node_test.data)   

    return

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
        x=torch.rand(6)
        kd.add_point(x)

        x=x.unsqueeze(0)
        X=torch.cat((X,x))
        Data=torch.Tensor()
        Data=preorder_assert(kd.root, Data, kd.maxNodeSize)
        
        sortx, _ = torch.sort(Data,0)
        sortData, _ = torch.sort(X,0)
        assert torch.equal(sortData,sortx)
    x=torch.rand(6)
    kd.add_point(x)
    Data=torch.Tensor()
    Data=preorder_assert(kd.root, Data, kd.maxNodeSize)
    sortx, _ = torch.sort(Data,0)   
    sortData, _ = torch.sort(X,0)

    assert not torch.equal(sortData,sortx)

def test_find_block():
    torch.manual_seed(42) 
    X = torch.randn(191, 6)
    x=torch.rand(6)
    kd=KDTree(X)

    node=kd.find_block(x)

    assert node is None

    kd.add_point(x)
    node=kd.find_block(x)

    assert torch.any(torch.all(node.data == x, dim=1))

def test_get_leaves():
    torch.manual_seed(42) 
    X = torch.randn(191, 6)
    kd=KDTree(X)

    leaves=kd.get_leaves()

    leaves_data=torch.Tensor()

    for node in leaves:
        assert node.data is not None
        leaves_data = torch.cat((leaves_data, node.data))
    
    sortx, _ = torch.sort(leaves_data,0)   
    sortData, _ = torch.sort(X,0)

    assert torch.equal(sortData,sortx)
    for _ in range(15):
        x=torch.rand(6)

        kd.add_point(x)
        x=x.unsqueeze(0)
        X=torch.cat((X,x))

    leaves=kd.get_leaves()
    leaves_data1 = torch.empty(0, 6) 

    for node in leaves:
        assert node.data is not None
        leaves_data1 = torch.cat((leaves_data1, node.data))
    
    sortx, _ = torch.sort(leaves_data1,0)   
    sortData, _ = torch.sort(X,0)

    assert torch.equal(sortData,sortx)

    x=torch.rand(6)
    kd.add_point(x)

    leaves=kd.get_leaves()
    leaves_data2 = torch.empty(0, 6) 

    for node in leaves:
        assert node.data is not None
        leaves_data2 = torch.cat((leaves_data2, node.data))
    
    sortx, _ = torch.sort(leaves_data2,0)   
    sortData, _ = torch.sort(X,0)
    
    assert not torch.equal(sortData,sortx)

# def test_set_children_bounds():
#     X = torch.randn(30, 6)
#     kd=KDTree(X)
#     #kd._set_children_bounds(kd.root)

    

    