"""
Partition Block Tests.

Unit tests for the PartitionBlock class, verifying point addition, normalization,
amplitude calculation, and target scaling.

Copyright (c) 2023-2025, Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
"""

import copy
import pytest

import numpy as np
import torch

from pysesm.blocks.PartitionBlock import PartitionBlock

DEFAULT_BLOCKS_PER_DIM = 4

def test_partition_block_initialization():
    """Test the initialization of a PartitionBlock."""
 
    device='cpu'

    space_origin = torch.tensor([0.0, 0.0], device=device)
    block_index = (0, 0)
    block_size = torch.tensor([1.0, 1.0], device=device)

    block = PartitionBlock(space_origin, block_index, block_size, device=device)

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
 
    device='cpu'

    space_origin = torch.tensor([0.0, 0.0], device=device)
    block_index = (0, 0)
    block_size = torch.tensor([1.0, 1.0], device=device)
    block = PartitionBlock(space_origin, block_index, block_size, device=device)

    point_x = torch.tensor([0.5, 0.5], device=device)
    point_y = torch.tensor([1.0], device=device)
    pos = 0

    block.new_point(point_x, point_y, pos)

    assert len(block.X) == 1
    assert torch.allclose(block.X[0], point_x)
    assert torch.allclose(block.y[0], point_y)
    assert block.positions[0] == pos

def test_clear_points():
    """Test clearing all points from the PartitionBlock."""

    device='cpu'

    space_origin = torch.tensor([0.0, 0.0], device=device)
    block_index = (0, 0)
    block_size = torch.tensor([1.0, 1.0], device=device)
    block = PartitionBlock(space_origin, block_index, block_size, device=device)

    point_x = torch.tensor([0.5, 0.5], device=device)
    point_y = torch.tensor([1.0], device=device)
    pos = 0

    block.new_point(point_x, point_y, pos)
    block.clear_points()

    assert len(block.X) == 0
    assert len(block.y) == 0
    assert len(block.positions) == 0
    assert block.normalized_X is None

def test_is_active():
    """Test the is_active property of the PartitionBlock."""

    device='cpu'

    space_origin = torch.tensor([0.0, 0.0], device=device)
    block_index = (0, 0)
    block_size = torch.tensor([1.0, 1.0], device=device)
    block = PartitionBlock(space_origin, block_index, block_size, device=device)

    assert not block.is_active()

    point_x = torch.tensor([0.5, 0.5], device=device)
    point_y = torch.tensor([1.0], device=device)
    pos = 0

    block.new_point(point_x, point_y, pos)

    assert block.is_active()

def test_normalize():
    """Test the normalization of points within the block."""

    device='cpu'

    space_origin = torch.tensor([0.0, 0.0], device=device)
    block_index = (0, 0)
    block_size = torch.tensor([1.0, 1.0], device=device)
    block = PartitionBlock(space_origin, block_index, block_size, device=device)

    point_x = torch.tensor([0.5, 0.5], device=device)
    point_y = torch.tensor([1.0], device=device)
    pos = 0

    block.new_point(point_x, point_y, pos)
    block.normalize_points()

    assert block.normalized_X is not None
    assert torch.allclose(block.normalized_X.get_for_device(device), torch.tensor([[0.5, 0.5]], device=device))


def test_normalize_extreme_block_sizes():
    """Test normalization with very small and very large block sizes."""

    device='cpu'

    space_origin = torch.tensor([0.0, 0.0], device=device)
    block_index = (0, 0)
    
    # Very small block size
    block_size_small = torch.tensor([1e-6, 1e-6], device=device)
    block_small = PartitionBlock(space_origin, block_index, block_size_small, device=device)
    
    point_small = torch.tensor([0.5e-6, 0.5e-6], device=device)
    block_small.new_point(point_small, torch.tensor([1.0], device=device), 0)
    block_small.normalize_points()
    
    min_vals = block_small.block_scope[0].to(device)
    sizes = block_small.block_size.to(device)
    expected_normalized_small = (point_small - min_vals) / sizes
    
    assert torch.allclose(block_small.normalized_X.get_for_device(device), expected_normalized_small, rtol=1e-5, atol=1e-8)

    # Very large block size
    block_size_large = torch.tensor([1e6, 1e6], device=device)
    block_large = PartitionBlock(space_origin, block_index, block_size_large, device=device)
    
    point_large = torch.tensor([0.5e6, 0.5e6], device=device)
    block_large.new_point(point_large, torch.tensor([1.0], device=device), 0)
    block_large.normalize_points()
    
    min_vals = block_large.block_scope[0].to(device)
    sizes = block_large.block_size.to(device)
    expected_normalized_large = (point_large - min_vals) / sizes
    
    assert torch.allclose(block_large.normalized_X.get_for_device(device), expected_normalized_large, rtol=1e-5, atol=1e-8)


def test_partition_block_calculate_amplitude_and_target():
    """
    Test that PartitionBlock.calculate_amplitude_and_target correctly
    computes amplitude and scales target values.
    """

    cpu_device=torch.device("cpu");
    
    # 1. Setup a dummy PartitionBlock
    # These spatial parameters don't affect amplitude/target, but are required for init.
    space_origin = torch.tensor([0.0, 0.0], device=cpu_device)
    block_index = (0, 0)
    block_size = torch.tensor([1.0, 1.0], device=cpu_device)

    # Create an instance of PartitionBlock
    block = PartitionBlock(
        space_origin=space_origin,
        block_index=block_index,
        block_size=block_size,
        device=cpu_device
    )

    # --- Test Case 1: Max absolute y value > 1 ---
    # Add points with y values > 1 (e.g., max_abs = 2.0)
    block.new_point(torch.tensor([0.1, 0.1], device=cpu_device), torch.tensor([1.5], device=cpu_device), 0)
    block.new_point(torch.tensor([0.2, 0.2], device=cpu_device), torch.tensor([-2.0], device=cpu_device), 1) # Max abs is 2.0
    block.new_point(torch.tensor([0.3, 0.3], device=cpu_device), torch.tensor([0.5], device=cpu_device), 2)

    # Call the method under test
    block.calculate_amplitude_and_target()

    # Assertions for amplitude
    expected_amplitude = 1.0 / 2.0 # 1 / max_abs(y)
    assert pytest.approx(block.amplitude, rel=1e-6) == expected_amplitude

    # Assertions for target (scaled y values)
    # Original y values: [1.5, -2.0, 0.5]
    # Expected target values: [1.5 * 0.5, -2.0 * 0.5, 0.5 * 0.5] = [0.75, -1.0, 0.25]
    expected_target_values = torch.tensor([[0.75], [-1.0], [0.25]], device=cpu_device) # Uniques to -1 for 2D
    assert torch.allclose(block.target.get_for_device(cpu_device), expected_target_values, atol=1e-6)

    # --- Test Case 2: Max absolute y value <= 1 ---
    # Clear previous points for the next test case
    block.clear_points()

    # Add points with y values <= 1 (e.g., max_abs = 0.5)
    block.new_point(torch.tensor([0.4, 0.4], device=cpu_device), torch.tensor([0.3], device=cpu_device), 3)
    block.new_point(torch.tensor([0.5, 0.5], device=cpu_device), torch.tensor([-0.5], device=cpu_device), 4) # Max abs is 0.5
    block.new_point(torch.tensor([0.6, 0.6], device=cpu_device), torch.tensor([0.1], device=cpu_device), 5)

    # Call the method again
    block.calculate_amplitude_and_target()

    # Assertions for amplitude
    expected_amplitude = 1.0 # Max abs <= 1, so amplitude is 1.0
    assert pytest.approx(block.amplitude, rel=1e-6) == expected_amplitude

    # Assertions for target (scaled y values - no change as amplitude is 1.0)
    expected_target_values_2 = torch.tensor([[0.3], [-0.5], [0.1]], device=cpu_device)
    assert torch.allclose(block.target.get_for_device(cpu_device), expected_target_values_2, atol=1e-6)

    # --- Test Case 3: Empty block ---
    block.clear_points()
    block.calculate_amplitude_and_target()
    assert block.amplitude == 1.0 # Default amplitude for empty
    assert block.target is None # Target should be None

    # --- Test Case 4: Single scalar y value in list ---
    block.clear_points()
    block.new_point(torch.tensor([0.1, 0.1], device=cpu_device), torch.tensor(2.5, device=cpu_device), 0)
    block.calculate_amplitude_and_target()
    assert pytest.approx(block.amplitude, rel=1e-6) == 1.0 / 2.5
    assert torch.allclose(block.target.get_for_device(cpu_device), torch.tensor([[1.0]], device=cpu_device), atol=1e-6)


    # --- Test Case 5: Single multi-dim y value in list ---
    block.clear_points()
    block.new_point(torch.tensor([0.1, 0.1], device=cpu_device), torch.tensor([1.0, 3.0], device=cpu_device), 0)
    block.calculate_amplitude_and_target()
    assert pytest.approx(block.amplitude, rel=1e-6) == 1.0 / 3.0
    assert torch.allclose(block.target.get_for_device(cpu_device), torch.tensor([[1.0/3.0, 1.0]], device=cpu_device), atol=1e-6)


# En tests/unit_test/blocks/PartitionBlock_test.py

def test_append_points_to_block():
    """Test appending multiple points to the PartitionBlock."""

    device='cpu'

    space_origin = torch.tensor([0.0, 0.0], device=device)
    block_index = (0, 0)
    block_size = torch.tensor([1.0, 1.0], device=device)
    block = PartitionBlock(space_origin, block_index, block_size, device=device)

    points_x = torch.tensor([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]], device=device)
    points_y = torch.tensor([[1.0], [2.0], [3.0]], device=device)
    positions = [0, 1, 2]

    block.append_points(points_x, points_y, positions)

    assert len(block.X) == 3
    assert torch.allclose(torch.stack(block.X), points_x)
    assert torch.allclose(torch.stack(block.y), points_y)
    assert block.positions == positions

    # Test appending more points
    more_points_x = torch.tensor([[0.2, 0.2], [0.8, 0.8]], device=device)
    more_points_y = torch.tensor([[4.0], [5.0]], device=device)
    more_positions = [3, 4]

    block.append_points(more_points_x, more_points_y, more_positions)

    assert len(block.X) == 5
    assert torch.allclose(torch.stack(block.X), torch.cat((points_x, more_points_x)))
    assert torch.allclose(torch.stack(block.y), torch.cat((points_y, more_points_y)))
    assert block.positions == [0, 1, 2, 3, 4]

def test_append_points_dimension_mismatch_raises_error():
    """
    Test that append_points raises ValueError if input dimensions mismatch.
    """

    device='cpu'

    space_origin = torch.tensor([0.0, 0.0], device=device)
    block_index = (0, 0)
    block_size = torch.tensor([1.0, 1.0], device=device)
    block = PartitionBlock(space_origin, block_index, block_size, device=device)

    # Mismatch: points_x (3 samples), points_y (2 samples)
    points_x_mismatch = torch.tensor([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]], device=device)
    points_y_mismatch = torch.tensor([[1.0], [2.0]], device=device)
    positions_mismatch = [0, 1, 2]

    with pytest.raises(ValueError, match="Dimension mismatch"):
        block.append_points(points_x_mismatch, points_y_mismatch, positions_mismatch)

    # Mismatch: points_x (3 samples), positions (2 samples)
    points_x_mismatch2 = torch.tensor([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]], device=device)
    points_y_mismatch2 = torch.tensor([[1.0], [2.0], [3.0]], device=device)
    positions_mismatch2 = [0, 1]

    with pytest.raises(ValueError, match="Dimension mismatch"):
        block.append_points(points_x_mismatch2, points_y_mismatch2, positions_mismatch2)
    
if __name__ == "__main__":
    from pytest_helper import print_pytest_instructions
    print_pytest_instructions()
