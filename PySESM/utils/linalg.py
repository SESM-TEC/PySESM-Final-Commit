import numpy as np
import torch
import math

def to_triu_matrix(non_zeros):
    """ Fill an upper triangular matrix with the content of the column
    vector non_zeros.  If the matrix has dimensions n x n, then the
    provided vector must have n(n+1)/2 elements.
    non_zeros must be a torch tensor of floats, as a column vector
    The matrix is filled rowwise, so, if the vector is [1 2 3 4 5 6]'
    then the final matrix will be
    |1 2 3|
    |0 4 5|
    |0 0 6|
    """

    # Find the dimension of the matrix.  If N is the dimension of the
    # given vector, the solution of n²+n-2N = 0 is the size of the matrix,
    # which is given by (sqrt(1+8N)-1)/2, and we round it up.
    matrix_n = math.ceil(0.5*(math.sqrt(8*non_zeros.size()[0] + 1) - 1))

    matrix = torch.zeros((matrix_n, matrix_n))

    matrix_triu_indices = torch.triu_indices(row=matrix_n, col=matrix_n, offset=0)

    matrix[matrix_triu_indices[0], matrix_triu_indices[1]] = non_zeros

    return matrix

def gram_schmidt(Q):
    for i in range(Q.shape[1]):
        for j in range(i):
            Q[:, i] -= torch.dot(Q[:, i], Q[:, j]) * Q[:, j]
        Q[:, i] /= torch.norm(Q[:, i])
    return Q

def generate_random_vectors(features, max_val, min_val):
    # Generates a random vector of uniform values between 0 and 1 and then scales it with a factor
    return torch.rand(features, features) * (max_val - min_val) + min_val

def get_upper_triangle(A):
    n = A.shape[0]
    indices = torch.triu_indices(n, n, offset=0)
    upper_triangle = A[indices[0], indices[1]]
    print("Upper: ", upper_triangle.shape)
    return upper_triangle

def reshape_upper_triangle(upper_triangle, n):
    # Calculate the number of rows needed
    num_rows = (len(upper_triangle) + n - 1) // n

    # Pad with zeros if necessary
    padding_size = n * num_rows - len(upper_triangle)
    upper_triangle_padded = torch.cat([upper_triangle, torch.zeros(padding_size)])

    # Reshape to 2D tensor
    reshaped_tensor = upper_triangle_padded.view(num_rows, n)

    return reshaped_tensor
