import numpy as np
import torch
import math

def to_triu_matrix(non_zeros):
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
    return torch.rand(features, features)

def get_upper_triangle(A):
    n = A.shape[0]
    indices = torch.triu_indices(n, n, offset=0)
    upper_triangle = A[indices[0], indices[1]]
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

def whiten_matrix(A):
    # Step 1: Center the columns of A
    mean_A = torch.mean(A, dim=0, keepdim=True)
    centered_A = A - mean_A

    # Step 2: Whitening using SVD
    U, S, Vt = torch.svd(centered_A)
    whitened_A = torch.mm(U, Vt)

    # Step 3: Scale the resulting matrix
    scale_factor = torch.sqrt(torch.tensor(A.shape[1]).float())
    whitened_A_scaled = whitened_A / scale_factor

    return whitened_A_scaled

def reshape_and_whiten(Rho, n):
    upper_triangle = get_upper_triangle(Rho)

    # Reshape the whitened upper triangle
    reshaped_upper_triangle = reshape_upper_triangle(upper_triangle, n)

    # Apply whitening to the upper triangle
    whitened_reshaped = whiten_matrix(reshaped_upper_triangle)

    return whitened_reshaped
