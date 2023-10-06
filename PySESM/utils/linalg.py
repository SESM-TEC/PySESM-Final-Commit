import numpy as np
import torch
import math

def to_triu_matrix(non_zeros):
    matrix_n = math.ceil(0.5*(math.sqrt(8*non_zeros.size()[0] + 1) - 1))

    matrix = torch.zeros((matrix_n, matrix_n))

    matrix_triu_indices = torch.triu_indices(row=matrix_n, col=matrix_n, offset=0)

    matrix[matrix_triu_indices[0], matrix_triu_indices[1]] = non_zeros

    return matrix
