# from pysesm.blocks.PartitionBlock import PartitionBlock
# import numpy as np
# import torch
# from pysesm.

# DEFAULT_BLOCKS_PER_DIM=4

# def test_partition_block_initialization():
#     """Test the initialization of a PartitionBlock."""
#     space_bound = torch.tensor([0.0, 0.0])
#     block_index = (0, 0)
#     block_size = torch.tensor([1.0, 1.0])
#     amplitude = 1
#     h = None
#     ista_layer = None

#     block = PartitionBlock(space_bound, block_index, block_size, amplitude, h, ista_layer)

#     assert block.block_index == block_index
#     assert torch.allclose(block.block_size, block_size)
#     assert block.amplitude == amplitude
#     assert block.h == h
#     assert block.ista_layer == ista_layer
#     assert len(block.X) == 0
#     assert len(block.y) == 0
#     assert len(block.positions) == 0
#     assert block.normalized_X is None

# def test_add_point_to_block():
#     """Test adding a point to the PartitionBlock."""
#     space_bound = torch.tensor([0.0, 0.0])
#     block_index = (0, 0)
#     block_size = torch.tensor([1.0, 1.0])
#     block = PartitionBlock(space_bound, block_index, block_size)

#     point_x = torch.tensor([0.5, 0.5])
#     point_y = 1.0
#     pos = 0

#     block.new_point(point_x, point_y, pos)

#     assert len(block.X) == 1
#     assert torch.allclose(block.X[0], point_x)
#     assert block.y[0] == point_y
#     assert block.positions[0] == pos

# def test_clear_points():
#     """Test clearing all points from the PartitionBlock."""
#     space_bound = torch.tensor([0.0, 0.0])
#     block_index = (0, 0)
#     block_size = torch.tensor([1.0, 1.0])
#     block = PartitionBlock(space_bound, block_index, block_size)

#     point_x = torch.tensor([0.5, 0.5])
#     point_y = 1.0
#     pos = 0

#     block.new_point(point_x, point_y, pos)
#     block.clear_points()

#     assert len(block.X) == 0
#     assert len(block.y) == 0
#     assert len(block.positions) != 0
#     assert block.normalized_X is None

# def test_is_active():
#     """Test the is_active property of the PartitionBlock."""
#     space_bound = torch.tensor([0.0, 0.0])
#     block_index = (0, 0)
#     block_size = torch.tensor([1.0, 1.0])
#     block = PartitionBlock(space_bound, block_index, block_size)

#     assert not block.is_active

#     point_x = torch.tensor([0.5, 0.5])
#     point_y = 1.0
#     pos = 0

#     block.new_point(point_x, point_y, pos)

#     assert block.is_active

# def test_is_point_in_block():
#     """Test if a point is within the block's boundaries."""
#     space_bound = torch.tensor([0.0, 0.0])
#     block_index = (0, 0)
#     block_size = torch.tensor([1.0, 1.0])
#     block = PartitionBlock(space_bound, block_index, block_size)

#     point_inside = torch.tensor([0.5, 0.5])
#     point_outside = torch.tensor([1.5, 1.5])

#     assert block.is_point_in_block(point_inside)
#     assert not block.is_point_in_block(point_outside)

# def test_normalize():
#     """Test the normalization of points within the block."""
#     space_bound = torch.tensor([0.0, 0.0])
#     block_index = (0, 0)
#     block_size = torch.tensor([1.0, 1.0])
#     block = PartitionBlock(space_bound, block_index, block_size)

#     point_x = torch.tensor([0.5, 0.5])
#     point_y = 1.0
#     pos = 0

#     block.new_point(point_x, point_y, pos)
#     block.normalize()

#     assert block.normalized_X is not None
#     assert torch.allclose(block.normalized_X, torch.tensor([[0.5, 0.5]]))

#     X = torch.tensor([[3, 7], 
#                       [1, 5],
#                       [2, 3]])

#     y = np.array([[1], 
#                   [2], 
#                   [3]])

#     min_values = torch.min(X, dim=0).values  # [1, 5]
#     max_values = torch.max(X, dim=0).values  # [3, 7]

#     bounds = torch.vstack([min_values, max_values])

#     T=torch.tensor([DEFAULT_BLOCKS_PER_DIM for _ in range(X.dim())])
#     delta = bounds[1] - bounds[0] 
#     block_size = torch.div(delta, T)

#     blocks=np.empty(T.numpy(), dtype=PartitionBlock)
#     for index in np.ndindex(block_size.shape):
#         blocks[index] = PartitionBlock(
#         bounds, index, block_size
#         )        

#         block_index=(0,0)    
#         blocks[block_index].new_point(X[0],y[0],0)
#         blocks[block_index].normalize()
        
#         eps = torch.finfo(torch.float32).eps
#         base_edge = bounds + torch.mul(torch.tensor(block_index), block_size)
#         block_scope = torch.stack((base_edge - eps, base_edge + block_size + eps))
        
#         tensor_X = torch.stack([X[0]])
#         min_vals = block_scope[0].to(tensor_X.device)
#         sizes = block_size.to(tensor_X.device)
#         normalized_X = (tensor_X - min_vals) / sizes

#         assert torch.equal(normalized_X,blocks[block_index].normalized_X)

# def test_clone_test():
#     """Test the clone_test method of the PartitionBlock."""
#     space_bound = torch.tensor([0.0, 0.0])
#     block_index = (0, 0)
#     block_size = torch.tensor([1.0, 1.0])
#     block = PartitionBlock(space_bound, block_index, block_size)

#     point_x = torch.tensor([0.5, 0.5])
#     point_y = 1.0
#     pos = 0

#     block.new_point(point_x, point_y, pos)
#     cloned_block = block.clone_test()

#     assert cloned_block.block_index == block.block_index
#     assert torch.allclose(cloned_block.block_size, block.block_size)
#     assert torch.allclose(cloned_block.block_scope, block.block_scope)
#     assert cloned_block.h == block.h
#     assert cloned_block.amplitude == block.amplitude
#     assert cloned_block.ista_layer == block.ista_layer
#     assert len(cloned_block.X) == 0
#     assert len(cloned_block.y) == 0
#     assert cloned_block.normalized_X is None
#     assert len(cloned_block.positions) == 0
#     assert len(cloned_block.target) == 0
#     assert len(cloned_block.predicted_output) == 0

# def test_deepcopy():
#     """Test the deepcopy functionality of the PartitionBlock."""
#     import copy

#     space_bound = torch.tensor([0.0, 0.0])
#     block_index = (0, 0)
#     block_size = torch.tensor([1.0, 1.0])
#     block = PartitionBlock(space_bound, block_index, block_size)

#     point_x = torch.tensor([0.5, 0.5])
#     point_y = 1.0
#     pos = 0

#     block.new_point(point_x, point_y, pos)
#     cloned_block = copy.deepcopy(block)

#     assert cloned_block.block_index == block.block_index
#     assert torch.allclose(cloned_block.block_size, block.block_size)
#     assert torch.allclose(cloned_block.block_scope, block.block_scope)
#     assert cloned_block.h == block.h
#     assert cloned_block.amplitude == block.amplitude
#     assert cloned_block.ista_layer == block.ista_layer
#     assert len(cloned_block.X) == 0
#     assert len(cloned_block.y) == 0
#     assert cloned_block.normalized_X is None
#     assert len(cloned_block.positions) == 0
#     assert len(cloned_block.target) == 0
#     assert len(cloned_block.predicted_output) == 0

# def test_normalize_extreme_block_sizes():
#     """Test normalization with very small and very large block sizes."""
#     space_bound = torch.tensor([0.0, 0.0])
#     block_index = (0, 0)
    
#     # Very small block size
#     block_size_small = torch.tensor([1e-6, 1e-6])
#     block_small = PartitionBlock(space_bound, block_index, block_size_small)
    
#     # Add a point at the center of the small block
#     point_small = torch.tensor([0.5e-6, 0.5e-6])
#     block_small.new_point(point_small, 1.0, 0)
    
#     # Normalize the points
#     block_small.normalize()
    
#     # Expected normalized value: (point - min_vals) / sizes
#     min_vals = block_small.block_scope[0].to(point_small.device)
#     sizes = block_small.block_size.to(point_small.device)
#     expected_normalized_small = (point_small - min_vals) / sizes
    
#     # Verify the normalized values
#     assert torch.allclose(
#         block_small.normalized_X, 
#         expected_normalized_small, 
#         rtol=1e-5, atol=1e-8
#     ), f"Normalized values for small block: {block_small.normalized_X}, expected: {expected_normalized_small}"

#     # Very large block size
#     block_size_large = torch.tensor([1e6, 1e6])
#     block_large = PartitionBlock(space_bound, block_index, block_size_large)
    
#     # Add a point at the center of the large block
#     point_large = torch.tensor([0.5e6, 0.5e6])
#     block_large.new_point(point_large, 1.0, 0)
    
#     # Normalize the points
#     block_large.normalize()
    
#     # Expected normalized value: (point - min_vals) / sizes
#     min_vals = block_large.block_scope[0].to(point_large.device)
#     sizes = block_large.block_size.to(point_large.device)
#     expected_normalized_large = (point_large - min_vals) / sizes
    
#     # Verify the normalized values
#     assert torch.allclose(
#         block_large.normalized_X, 
#         expected_normalized_large, 
#         rtol=1e-5, atol=1e-8
#     ), f"Normalized values for large block: {block_large.normalized_X}, expected: {expected_normalized_large}"

# def test_ista_layer_interaction():
#     """Test interaction with ista_layer."""
#     space_bound = torch.tensor([0.0, 0.0])
#     block_index = (0, 0)
#     block_size = torch.tensor([1.0, 1.0])
#     ista_layer = torch.nn.Linear(2, 2)
#     block = PartitionBlock(space_bound, block_index, block_size, ista_layer=ista_layer)

#     # Add a point and check if ista_layer is used
#     block.new_point(torch.tensor([0.5, 0.5]), 1.0, 0)
#     assert isinstance(block.ista_layer, torch.nn.Linear)

# #Maybe change some of the values of space bound or block index for some extreme cases.... 

# if __name__ == "__main__":

#     from pytest_helper import print_pytest_instructions
#     print_pytest_instructions()    
    
#     #pytest.main()