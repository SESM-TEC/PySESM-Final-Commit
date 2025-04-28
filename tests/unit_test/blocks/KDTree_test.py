from pysesm.blocks.Node import Node
from pysesm.blocks.KDTree import KDTree
from pysesm.blocks.AdaptativePartitionManager import AdaptativePartitionManager
from pysesm.enums.DeviceTargetEnum import DeviceTarget
from pysesm.device_manager.DeviceManager import DeviceManager
from pysesm.utils.loggers import setup_logger
import torch
import logging 

def test_greatestVarDim():
    """
    Asserts the greatest variance dimension is calculated correctly in the node
    """
    logger=setup_logger()
    device_map = {
        DeviceTarget.GLOBAL: "cpu",               # Dispositivo global por defecto
        DeviceTarget.ISTA_LAYER: "cpu",           # ISTA en GPU 0
        DeviceTarget.DICTIONARY_LAYER: "cpu",     # Dictionary en CPU
        DeviceTarget.PARTITION_MANAGER: "cuda"    # Partition Manager en CPU
    }
    device_manager=DeviceManager(logger,device_map=device_map)
    device = device_manager.get_device(DeviceTarget.PARTITION_MANAGER)
    x = torch.randn(20, 5).to(device)
    node=Node(x)
    dim = node.greatestVarDim()

    variances = x[:,:-1].var(dim=0)
    print(variances.device)
    dim_test = torch.argmax(variances).item()
    assert dim==dim_test
    return


def test_splitDataInNodes():
    """
    Tests the splitDataInNodes function which basically initializes the KDTree
    """
    torch.manual_seed(42) 
    logger=setup_logger()
    device_map = {
        DeviceTarget.GLOBAL: "cpu",               # Dispositivo global por defecto
        DeviceTarget.ISTA_LAYER: "cpu",           # ISTA en GPU 0
        DeviceTarget.DICTIONARY_LAYER: "cpu",     # Dictionary en CPU
        DeviceTarget.PARTITION_MANAGER: "cuda"    # Partition Manager en CPU
    }
    device_manager=DeviceManager(logger,device_map=device_map)
    device = device_manager.get_device(DeviceTarget.PARTITION_MANAGER)
    x = torch.randn(15, 6).to(device)
    kd=KDTree(x,device=device)
    defaultMaxNodeSize = kd.maxNodeSize

    Data=torch.Tensor().to(kd.device)
    Data=preorder_assert(kd.root, Data, defaultMaxNodeSize)
    
    sortx, _ = torch.sort(Data,0)
    sortData, _ = torch.sort(x[:,:-1],0)

    assert torch.equal(sortData,sortx)
    leaves=kd.get_leaves()
    kd.maxNodeSize = round(defaultMaxNodeSize/2)
    for leaf in leaves:
        if leaf.data.size()[0] == defaultMaxNodeSize:
            kd._splitDataInNodes(leaf)

    leaves2=kd.get_leaves()
    assert leaves!=leaves2
    

def test_find_node():
    torch.manual_seed(42) 
    X = torch.randn(191, 6)

    kd=KDTree(X)
    x=torch.rand(5)

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
    n_features=5
    torch.manual_seed(42) 
    X = torch.randn(191, n_features+1)
    logger=setup_logger()
    partitionManager=AdaptativePartitionManager(logger,n_features+1)
    partitionManager._update_block_arrangement(X)
    kd=partitionManager.kdtree
    for _ in range(kd.maxNodeSize):
        x=torch.rand(n_features)
        y=torch.rand(1)
        kd.add_point(x,y)

        x=x.unsqueeze(0)
        y=y.unsqueeze(0)
        xy=torch.cat((x,y),dim=1)
        X=torch.cat((X,xy))
        Data=torch.Tensor()
        Data=preorder_assert(kd.root, Data, kd.maxNodeSize)
        
        sortx, _ = torch.sort(Data,0)
        sortData, _ = torch.sort(X,0)
        assert torch.equal(sortData[:,:-1],sortx)
    leaves=kd.get_leaves()

    for leaf in leaves:                 #If split nodes, check child nodes have blocks defined
        assert leaf.block is not None

    x=torch.rand(n_features)
    y=torch.rand(1)    
    kd.add_point(x,y)
    Data=torch.Tensor()
    Data=preorder_assert(kd.root, Data, kd.maxNodeSize)
    sortx, _ = torch.sort(Data,0)   
    sortData, _ = torch.sort(X,0)

    assert not torch.equal(sortData[:,:-1],sortx)

def test_find_block():
    torch.manual_seed(42) 
    X = torch.randn(191, 6)
    x=torch.rand(5)
    y=torch.rand(1)
    kd=KDTree(X)

    node=kd.find_block(x)

    assert node is None

    kd.add_point(x, y)
    node=kd.find_block(x)

    assert torch.any(torch.all(node.data == x, dim=1))

def test_get_leaves():
    n_features=5
    torch.manual_seed(42) 
    X = torch.randn(191, n_features+1)
    kd=KDTree(X)

    leaves=kd.get_leaves()

    leaves_data=torch.Tensor()

    for node in leaves:
        assert node.data is not None
        leaves_data = torch.cat((leaves_data, node.data))
    
    sortx, _ = torch.sort(leaves_data,0)   
    sortData, _ = torch.sort(X,0)

    assert torch.equal(sortData[:, :-1],sortx)
    
    for _ in range(15):
        x=torch.rand(n_features)
        y=torch.rand(1)

        kd.add_point(x, y)
        x=x.unsqueeze(0)
        y=y.unsqueeze(0)
        xy=torch.cat((x,y),dim=1)
        X=torch.cat((X,xy))

    leaves=kd.get_leaves()
    leaves_data1 = torch.empty(0, n_features) 

    for node in leaves:
        assert node.data is not None
        leaves_data1 = torch.cat((leaves_data1, node.data))
    
    sortx, _ = torch.sort(leaves_data1,0)   
    sortData, _ = torch.sort(X,0)

    assert torch.equal(sortData[:, :-1],sortx)

    x=torch.rand(n_features)
    y=torch.rand(1)
    kd.add_point(x,y)

    leaves=kd.get_leaves()
    leaves_data2 = torch.empty(0, n_features) 

    for node in leaves:
        assert node.data is not None
        leaves_data2 = torch.cat((leaves_data2, node.data))
    
    sortx, _ = torch.sort(leaves_data2,0)   
    sortData, _ = torch.sort(X,0)
    
    assert not torch.equal(sortData[:, :-1],sortx)

# def test_set_children_bounds():
#     X = torch.randn(30, 6)
#     kd=KDTree(X)
#     #kd._set_children_bounds(kd.root)

    

    
