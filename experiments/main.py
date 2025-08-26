import numpy as np
import torch

from test_all import test_all
from train_all import train_all

from pysesm.utils_dataset.generate_dataset import generate_custom_function_dataset


#CREAR DATASET
def f(x, y):
    pi = np.pi
    return torch.sin(pi*x)/pi*x - torch.sin(pi*y)/pi*y

train_data, xtrain, ytrain, test_data, xtest, ytest = generate_custom_function_dataset(
    n_samples=200,
    function=f,
    mesh_divisions=50
)

train_all(train_data, test_data)
test_all(train_data, test_data)

