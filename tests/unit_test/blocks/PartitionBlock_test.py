from pysesm.blocks.PartitionBlock import PartitionBlock
import numpy as np
import torch

DEFAULT_BLOCKS_PER_DIM=4

def test_normalize():

    X = torch.tensor([[3, 7], 
                      [1, 5],
                      [2, 3]])

    y = np.array([[1], 
                  [2], 
                  [3]])

    min_values = torch.min(X, dim=0).values  # [1, 5]
    max_values = torch.max(X, dim=0).values  # [3, 7]

    bounds = torch.vstack([min_values, max_values])

    T=torch.tensor([DEFAULT_BLOCKS_PER_DIM for _ in range(X.dim())])
    delta = bounds[1] - bounds[0] 
    block_size = torch.div(delta, T)

    blocks=np.empty(T.numpy(), dtype=PartitionBlock)
    for index in np.ndindex(block_size.shape):
        blocks[index] = PartitionBlock(
        bounds, index, block_size
        )        

        block_index=(0,0)    
        blocks[block_index].new_point(X[0],y[0],0)
        blocks[block_index].normalize()
        
        eps = torch.finfo(torch.float32).eps
        base_edge = bounds + torch.mul(torch.tensor(block_index), block_size)
        block_scope = torch.stack((base_edge - eps, base_edge + block_size + eps))
        
        tensor_X = torch.stack([X[0]])
        min_vals = block_scope[0].to(tensor_X.device)
        sizes = block_size.to(tensor_X.device)
        normalized_X = (tensor_X - min_vals) / sizes

        assert torch.equal(normalized_X,blocks[block_index].normalized_X)