import pytest
import torch
import math
from pysesm.utils.linalg import *


def test_to_triu_matrix():
    """
    Test the to_triu_matrix function to ensure correct
    transformation into an upper triangular matrix.
    """
    # Test case: Vector of length 6 for a 3x3 matrix
    non_zeros = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32)
    expected_matrix = torch.tensor(
        [[1, 2, 3], [0, 4, 5], [0, 0, 6]], dtype=torch.float32
    )

    result_matrix = to_triu_matrix(non_zeros)
    assert torch.equal(
        result_matrix, expected_matrix
    ), "Upper triangular matrix conversion failed"

    # Test edge case: Vector of length 0 for a 0x0 matrix
    non_zeros_empty = torch.tensor([], dtype=torch.float32)
    expected_matrix_empty = torch.zeros((0, 0))

    result_matrix_empty = to_triu_matrix(non_zeros_empty)
    assert torch.equal(
        result_matrix_empty, expected_matrix_empty
    ), "Empty vector conversion failed"


def test_gram_schmidt():
    """
    Test the gram_schmidt function to ensure orthonormalization of vectors.
    """
    # Test case: Simple orthonormalization
    Q = torch.tensor([[1.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    result_Q = gram_schmidt(Q.clone())

    # Expected result is an orthonormal basis
    expected_Q = torch.tensor(
        [[math.sqrt(2) / 2, math.sqrt(2) / 2], [math.sqrt(2) / 2, -math.sqrt(2) / 2]],
        dtype=torch.float32,
    )
    assert torch.allclose(result_Q, expected_Q, atol=1e-5), "Orthonormalization failed"

    # Test case: Already orthonormal matrix
    Q_ortho = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    result_Q_ortho = gram_schmidt(Q_ortho.clone())

    assert torch.allclose(
        result_Q_ortho, Q_ortho, atol=1e-5
    ), "Orthonormalization of orthonormal matrix failed"


def test_generate_random_vectors():
    """
    Test the generate_random_vectors function for correct random vector generation.
    """
    features = 3
    max_val = 5.0
    min_val = 1.0
    random_vectors = generate_random_vectors(features, max_val, min_val)

    # Check the shape of the generated vectors
    assert random_vectors.shape == (
        features,
        features,
    ), "Generated vectors have incorrect shape"

    # Ensure values are within the specified range
    assert (random_vectors >= min_val).all() and (
        random_vectors <= max_val
    ).all(), "Values are out of the specified range"


def test_get_upper_triangle():
    """
    Test the get_upper_triangle function for correct extraction of upper triangular elements.
    """
    A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
    expected_upper = torch.tensor([1, 2, 3, 5, 6, 9], dtype=torch.float32)

    result_upper = get_upper_triangle(A)
    assert torch.equal(
        result_upper, expected_upper
    ), "Upper triangular extraction failed"

    # Test with non-square matrix
    B = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
    expected_upper_non_square = torch.tensor([1, 2, 4], dtype=torch.float32)

    result_upper_non_square = get_upper_triangle(B)
    assert torch.equal(
        result_upper_non_square, expected_upper_non_square
    ), "Upper triangular extraction for non-square matrix failed"


def test_reshape_upper_triangle():
    """
    Test the reshape_upper_triangle function for correct reshaping.
    """
    upper_triangle = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32)
    n = 3
    expected_reshaped = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

    result_reshaped = reshape_upper_triangle(upper_triangle, n)
    assert torch.equal(
        result_reshaped, expected_reshaped
    ), "Reshape upper triangle failed"

    # Test with padding requirement
    upper_triangle_padding = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    expected_reshaped_padding = torch.tensor(
        [[1, 2, 3], [4, 0, 0]], dtype=torch.float32
    )

    result_reshaped_padding = reshape_upper_triangle(upper_triangle_padding, n)
    assert torch.equal(
        result_reshaped_padding, expected_reshaped_padding
    ), "Reshape upper triangle with padding failed"

    # Test with zero elements
    upper_triangle_empty = torch.tensor([], dtype=torch.float32)
    expected_reshaped_empty = torch.zeros((0, n))

    result_reshaped_empty = reshape_upper_triangle(upper_triangle_empty, n)
    assert torch.equal(
        result_reshaped_empty, expected_reshaped_empty
    ), "Reshape upper triangle with zero elements failed"

if __name__ == "__main__":
    from pytest_helper import print_pytest_instructions
    print_pytest_instructions()    
