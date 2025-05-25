from pysesm.blocks.PartitionBlock import PartitionBlock
import numpy as np
import torch
import copy
import pytest

DEFAULT_BLOCKS_PER_DIM = 4

def test_partition_block_initialization():
    """Test the initialization of a PartitionBlock."""
    space_origin = torch.tensor([0.0, 0.0], device='cpu')
    block_index = (0, 0)
    block_size = torch.tensor([1.0, 1.0], device='cpu')

    block = PartitionBlock(space_origin, block_index, block_size, device='cpu')

    assert block.block_index == block_index
    assert torch.allclose(block.block_size, block_size)

    assert torch.allclose(block.block_scope[0],space_origin,atol=1.0e-6)
    assert torch.allclose(block.block_scope[1],block_size,atol=1.0e-6)
    
    assert block.amplitude == 1.0
    assert block.sparse_coding_layer == None

    assert len(block.X) == 0
    assert len(block.y) == 0
    
    assert len(block.positions) == 0
    assert block.normalized_X is None
    assert block.target is None

def test_add_point_to_block():
    """Test adding a point to the PartitionBlock."""
    space_origin = torch.tensor([0.0, 0.0], device='cpu')
    block_index = (0, 0)
    block_size = torch.tensor([1.0, 1.0], device='cpu')
    block = PartitionBlock(space_origin, block_index, block_size, device='cpu')

    point_x = torch.tensor([0.5, 0.5], device='cpu')
    point_y = torch.tensor([1.0], device='cpu')
    pos = 0

    block.new_point(point_x, point_y, pos)

    assert len(block.X) == 1
    assert torch.allclose(block.X[0], point_x)
    assert torch.allclose(block.y[0], point_y)
    assert block.positions[0] == pos

def test_clear_points():
    """Test clearing all points from the PartitionBlock."""
    space_origin = torch.tensor([0.0, 0.0], device='cpu')
    block_index = (0, 0)
    block_size = torch.tensor([1.0, 1.0], device='cpu')
    block = PartitionBlock(space_origin, block_index, block_size, device='cpu')

    point_x = torch.tensor([0.5, 0.5], device='cpu')
    point_y = torch.tensor([1.0], device='cpu')
    pos = 0

    block.new_point(point_x, point_y, pos)
    block.clear_points()

    assert len(block.X) == 0
    assert len(block.y) == 0
    assert len(block.positions) == 0
    assert block.normalized_X is None

def test_is_active():
    """Test the is_active property of the PartitionBlock."""
    space_origin = torch.tensor([0.0, 0.0], device='cpu')
    block_index = (0, 0)
    block_size = torch.tensor([1.0, 1.0], device='cpu')
    block = PartitionBlock(space_origin, block_index, block_size, device='cpu')

    assert not block.is_active

    point_x = torch.tensor([0.5, 0.5], device='cpu')
    point_y = torch.tensor([1.0], device='cpu')
    pos = 0

    block.new_point(point_x, point_y, pos)

    assert block.is_active

def test_is_point_in_block():
    """Test if a point is within the block's boundaries."""
    space_origin = torch.tensor([0.0, 0.0], device='cpu')
    block_index = (0, 0)
    block_size = torch.tensor([1.0, 1.0], device='cpu')
    block = PartitionBlock(space_origin, block_index, block_size, device='cpu')

    point_inside = torch.tensor([0.5, 0.5], device='cpu')
    point_outside = torch.tensor([1.5, 1.5], device='cpu')

    assert block.is_point_in_block(point_inside)
    assert not block.is_point_in_block(point_outside)

def test_normalize():
    """Test the normalization of points within the block."""
    space_origin = torch.tensor([0.0, 0.0], device='cpu')
    block_index = (0, 0)
    block_size = torch.tensor([1.0, 1.0], device='cpu')
    block = PartitionBlock(space_origin, block_index, block_size, device='cpu')

    point_x = torch.tensor([0.5, 0.5], device='cpu')
    point_y = torch.tensor([1.0], device='cpu')
    pos = 0

    block.new_point(point_x, point_y, pos)
    block.normalize_points()

    assert block.normalized_X is not None
    assert torch.allclose(block.normalized_X, torch.tensor([[0.5, 0.5]], device='cpu'))


def test_normalize_extreme_block_sizes():
    """Test normalization with very small and very large block sizes."""
    space_origin = torch.tensor([0.0, 0.0], device='cpu')
    block_index = (0, 0)
    
    # Very small block size
    block_size_small = torch.tensor([1e-6, 1e-6], device='cpu')
    block_small = PartitionBlock(space_origin, block_index, block_size_small, device='cpu')
    
    point_small = torch.tensor([0.5e-6, 0.5e-6], device='cpu')
    block_small.new_point(point_small, torch.tensor([1.0], device='cpu'), 0)
    block_small.normalize_points()
    
    min_vals = block_small.block_scope[0].to('cpu')
    sizes = block_small.block_size.to('cpu')
    expected_normalized_small = (point_small - min_vals) / sizes
    
    assert torch.allclose(block_small.normalized_X, expected_normalized_small, rtol=1e-5, atol=1e-8)

    # Very large block size
    block_size_large = torch.tensor([1e6, 1e6], device='cpu')
    block_large = PartitionBlock(space_origin, block_index, block_size_large, device='cpu')
    
    point_large = torch.tensor([0.5e6, 0.5e6], device='cpu')
    block_large.new_point(point_large, torch.tensor([1.0], device='cpu'), 0)
    block_large.normalize_points()
    
    min_vals = block_large.block_scope[0].to('cpu')
    sizes = block_large.block_size.to('cpu')
    expected_normalized_large = (point_large - min_vals) / sizes
    
    assert torch.allclose(block_large.normalized_X, expected_normalized_large, rtol=1e-5, atol=1e-8)

if __name__ == "__main__":
    from pytest_helper import print_pytest_instructions
    print_pytest_instructions()    
    #pytest.main()
