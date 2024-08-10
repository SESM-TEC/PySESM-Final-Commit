import pytest
import numpy as np
import torch
from collections import defaultdict
from pysesm.models.ISTALayer import ISTALayer
from pysesm.base_functions.sub_block_partition import *

def test_subblock_initialization():
    # Test for initializing SubBlock with default values
    sub_block = SubBlock()
    assert sub_block.amplitude == 1
    assert sub_block.ista_layer is None
    assert sub_block._X == []
    assert sub_block.output_values == []

def test_add_point():
    # Test for adding a point to SubBlock
    sub_block = SubBlock()
    point = np.array([0.5, 0.5])
    y = 10

    sub_block.add_point(point, y)
    
    assert len(sub_block._X) == 1
    assert len(sub_block.output_values) == 1
    assert np.array_equal(sub_block._X[0], point)
    assert sub_block.output_values[0] == y

def test_get_sub_block_vertices():
    # Test for obtaining the vertices of a sub-block
    grid_size = 4
    row = 1
    col = 2

    expected_vertices = np.array([[0.5, 0.25], [0.75, 0.25], [0.5, 0.5], [0.75, 0.5]])
    vertices = get_sub_block_vertices(grid_size, row, col)

    assert np.allclose(vertices, expected_vertices)

def test_locate_samples_in_sub_blocks():
    # Test for locating samples in sub-blocks
    x_n = np.array([[0.1, 0.2], [0.5, 0.6], [0.9, 0.8]])
    y = np.array([1, 2, 3])
    t = np.array([[0, 0], [1, 1], [2, 2]])
    T = 3

    sub_blocks = locate_samples_in_sub_blocks(x_n, y, t, T)
    
    assert len(sub_blocks) == T * T
    assert sub_blocks[0].output_values == [1]
    assert sub_blocks[4].output_values == [2]
    assert sub_blocks[8].output_values == [3]

def test_data_mapping():
    # Test for mapping the data
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    T = 2

    t, x_n = data_mapping(X, T)

    expected_t = np.array([[0, 0], [1, 1]])
    expected_x_n = torch.tensor([[0.0, 0.0], [0.0, 0.0]])

    assert np.array_equal(t, expected_t)
    assert torch.allclose(x_n, expected_x_n)

def test_squeze_factor():
    # Test for calculating the squeeze factor
    Y = [0.5, 2.0, 0.8]
    expected_factor = 0.5
    factor = squeze_factor(Y)

    assert factor == expected_factor

    Y = [0.5, 0.8]
    expected_factor = 1.0
    factor = squeze_factor(Y)

    assert factor == expected_factor

def test_generate_list_of_subblock():
    # Test for generating the list of sub-blocks with expected squeeze factor
    sub_blocks = np.empty(4, dtype=object)
    sub_blocks[0] = SubBlock()
    sub_blocks[0].output_values = [2.0, 3.0]

    sub_blocks[1] = SubBlock()
    sub_blocks[1].output_values = []

    sub_blocks[2] = SubBlock()
    sub_blocks[2].output_values = [0.5]

    sub_blocks[3] = SubBlock()
    sub_blocks[3].output_values = [1.0, 0.8]

    l_functions = [lambda x: x]
    SEED = 42

    result = generate_list_of_subblock(sub_blocks, l_functions, SEED)

    assert len(result) == 3
    assert result[0].amplitude == 0.3333333333333333
    assert result[0].ista_layer is not None
    assert result[0].output_values == [0.6666666666666666, 1.0]

    assert result[1].amplitude == 1.0
    assert result[1].ista_layer is not None
    assert result[1].output_values == [0.5]

def test_predict_on_test_set():
    # Test for predicting on the test set
    X_test = torch.tensor([[0.1, 0.1], [0.9, 0.9]])
    model = MockModel()  # Assuming you have a mock model
    T = 2

    train_sb = [SubBlock() for _ in range(T * T)]
    train_sb[0].ista_layer = MockISTALayer()
    train_sb[0].amplitude = 0.5

    train_sb[3].ista_layer = MockISTALayer()
    train_sb[3].amplitude = 1.0

    predictions = predict_on_test_set(X_test, model, T, train_sb)

    assert len(predictions) == len(X_test)
    assert predictions[0] == pytest.approx(2.0)  # Adjust as necessary
    assert predictions[1] == pytest.approx(1.0)  # Adjust as necessary

def test_count_unique_combinations():
    # Test for counting unique combinations
    T = [(0, 1), (1, 1), (0, 1), (2, 2)]
    expected_count = {(0, 1): 2, (1, 1): 1, (2, 2): 1}

    count = count_unique_combinations(T)

    assert count == expected_count


# Mock classes for the tests
class MockModel:
    def predict(self, X_sub_block, ista_layer):
        return torch.tensor([4.0] * len(X_sub_block), dtype=torch.float32)


class MockISTALayer:
    def __init__(self):
        self.h = torch.tensor([1.0])

print("Todo salio bien")
