from model import SVR

import numpy as np
import torch

from pysesm.utils_dataset.generate_dataset import generate_custom_function_dataset


# CREAR DATASET
def f(x, y):
    pi = np.pi
    return torch.sin(pi*x)/pi*x - torch.sin(pi*y)/pi*y
n_samples = 100
mesh_divisions = 50
train_data, xtrain, ytrain, test_data, xtest, ytest = generate_custom_function_dataset(
    n_samples=n_samples,
    function=f,
    mesh_divisions=mesh_divisions
)

# ENTRENAMIENTO
model = SVR(kernel='rbf', C=100, gamma=.1, epsilon=.1)
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)

# GUARDAR MODELO
path = r"C:\Users\Lenovo Yoga\Desktop\SEMESTRE_II_2025\ASISTENCIA\PySESM\experiments\SVR\svr_model.pth"
model.save(path)


