from pysesm.blocks.Node import Node
from pysesm.blocks.SESMData import SESMData
from pysesm.blocks.KDTree import KDTree, dummyData
from pysesm.blocks.AdaptativePartitionManager import AdaptativePartitionManager
from pysesm.enums.DeviceTargetEnum import DeviceTarget
from pysesm.device_manager.DeviceManager import DeviceManager
from pysesm.utils.loggers import setup_logger
import torch
import logging 
import pytest

logger = logging.getLogger("test_uniform_partition_manager")
logger.setLevel(logging.DEBUG)

@pytest.fixture(scope="module")
def common_device_manager():
    """Provides a shared DeviceManager instance for all tests in this module."""
    device_map = {
        DeviceTarget.GLOBAL: "cpu",
        DeviceTarget.SPARSE_CODING_LAYER: "cpu",
        DeviceTarget.DICTIONARY_LAYER: "cpu",
        DeviceTarget.PARTITION_MANAGER: "cpu" # Assuming TargetDevice is an alias for DeviceTarget
    }
    # Using a unique logger for the DeviceManager fixture to avoid conflicts
    return DeviceManager(logging.getLogger("test_device_manager_fixture"), default_device="cpu", device_map=device_map)
@pytest.fixture
def create_KDTree(common_device_manager):
    """
    Factory fixture to create UniformPartitionManager instances with flexible config.
    Ensures initial_bounds are consistently passed as numpy arrays to the config.
    """
    def _creator(data: torch.Tensor, y: torch.Tensor, maxNodeSize: int=5):
        # Convert torch.Tensor bounds to numpy array for UniformPartitionConfig
        return KDTree(
            data=data,
            y=y,
            maxNodeSize=maxNodeSize,
            data_wrapper=dummyData,
            device=common_device_manager.get_device(DeviceTarget.PARTITION_MANAGER)
        )
    return _creator


def test_greatestVarDim(common_device_manager):
    """
    Asserts the greatest variance dimension is calculated correctly in the node
    """
    device = common_device_manager.get_device(DeviceTarget.PARTITION_MANAGER)
    x = torch.randn(20, 5).to(device)
    y = torch.randn(20, 1).to(device)    
    node=Node(x,y, SESMData)
    dim = node.Data.greatestVarDim()

    variances = x.var(dim=0)
    dim_test = torch.argmax(variances).item()
    assert dim==dim_test


def test_splitDataInNodes(create_KDTree, common_device_manager):
    """
    Tests the splitDataInNodes function which basically initializes the KDTree
    """
    device = common_device_manager.get_device(DeviceTarget.PARTITION_MANAGER)
    torch.manual_seed(42) 
    x = torch.randn(501, 6).to(device)
    y = torch.randn(501, 1).to(device)
    kd=create_KDTree(x,y)
    defaultMaxNodeSize = kd.maxNodeSize

    Data=torch.Tensor().to(kd.device)
    treeNodes=kd.get_leaves()
    for node in treeNodes:
        Data=torch.cat((Data,node.Data.X))

    sortx, _ = torch.sort(Data,0)
    sortData, _ = torch.sort(x,0)

    assert torch.equal(sortData,sortx)
    leaves=kd.get_leaves()
    defaultMaxNodeSize = round(defaultMaxNodeSize/2)
    kd.maxNodeSize=defaultMaxNodeSize
    for leaf in leaves:
        if leaf.Data.X.size()[0] > defaultMaxNodeSize:
            kd._splitDataInNodes(leaf)

    leaves2=kd.get_leaves()
    assert [leaf.Data.X for leaf in leaves]!=[leaf.Data.X for leaf in leaves2]
    

def test_find_node(create_KDTree):
    torch.manual_seed(42) 
    X = torch.randn(191, 6)
    y = torch.randn(191, 1)
    kd=create_KDTree(X,y)
    x=torch.rand(5)

    node=kd._find_node(x)

    node_test=kd.root
    if node_test.Data.X is None:
        if x[node_test.Data.dim].item() >= node_test.Data.split_point:
            node_test= kd._find_node(x, node_test.Data.right)
        elif x[node_test.Data.dim].item() < node_test.Data.split_point:
            node_test = kd._find_node(x, node_test.Data.left)

    assert torch.equal(node.Data.X, node_test.Data.X)   

    return

def preorder_assert(node, Data, maxNodeSize):  
    """
    Asserts the nodes that are not in the lowest level have no data,
    Asserts that no node has only a left or only a right child.
    Asserts every node with data has the maximum node size or less.
    Returns the concatenated data from the lowest-level nodes.(It could be done using get_leaves())
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
            assert node.right.data.size()[0] <= maxNodeSize
    
    return Data

def test_add_point(create_KDTree):
    n_features=5
    torch.manual_seed(42) 
    X = torch.randn(500, n_features)
    y = torch.randn(500, 1)
    kd=create_KDTree(X,y)
    for _ in range(kd.maxNodeSize):
        x=torch.rand(n_features)
        y=torch.rand(1)
        kd.add_point(x,y)

        x=x.unsqueeze(0)
        y=y.unsqueeze(0)
        X=torch.cat((X,x))
        Data=torch.Tensor()
        treeNodes=kd.get_leaves()
        for node in treeNodes:
            Data=torch.cat((Data,node.Data.X))
        
        sortx, _ = torch.sort(Data,0)
        sortData, _ = torch.sort(X,0)
        assert torch.equal(sortData,sortx)

    x=torch.rand(n_features)
    y=torch.rand(1)    
    kd.add_point(x,y)
    Data=torch.Tensor()
    treeNodes=kd.get_leaves()
    for node in treeNodes:
        Data=torch.cat((Data,node.Data.X))
    sortx, _ = torch.sort(Data,0)   
    sortData, _ = torch.sort(X,0)

    assert not torch.equal(sortData,sortx)

def test_get_leaves(create_KDTree):
    n_features=5
    torch.manual_seed(42) 
    X = torch.randn(191, n_features)
    y = torch.randn(191, 1)
    kd=create_KDTree(X,y)

    leaves=kd.get_leaves()

    leaves_data=torch.Tensor()

    for node in leaves:
        assert node.Data.X is not None
        leaves_data = torch.cat((leaves_data, node.Data.X))
    
    sortx, _ = torch.sort(leaves_data,0)   
    sortData, _ = torch.sort(X,0)

    assert torch.equal(sortData,sortx)
    
    for _ in range(15):
        x=torch.rand(n_features)
        y=torch.rand(1)

        kd.add_point(x, y)
        x=x.unsqueeze(0)
        X=torch.cat((X,x))

    leaves=kd.get_leaves()
    leaves_data1 = torch.empty(0, n_features) 

    for node in leaves:
        assert node.Data.X is not None
        leaves_data1 = torch.cat((leaves_data1, node.Data.X))
    
    sortx, _ = torch.sort(leaves_data1,0)   
    sortData, _ = torch.sort(X,0)

    assert torch.equal(sortData,sortx)

    x=torch.rand(n_features)
    y=torch.rand(1)
    kd.add_point(x,y)

    leaves=kd.get_leaves()
    leaves_data2 = torch.empty(0, n_features) 

    for node in leaves:
        assert node.Data.X is not None
        leaves_data2 = torch.cat((leaves_data2, node.Data.X))
    
    sortx, _ = torch.sort(leaves_data2,0)   
    sortData, _ = torch.sort(X,0)
    
    assert not torch.equal(sortData,sortx)



    

    
