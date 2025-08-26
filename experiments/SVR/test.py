from model import SVR

import matplotlib.pyplot as plt
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

model = SVR(kernel='rbf', C=100, gamma=.1, epsilon=.1)
path = r"C:\Users\Lenovo Yoga\Desktop\SEMESTRE_II_2025\ASISTENCIA\PySESM\experiments\SVR\svr_model.pth"
model.load(path)
ypred = model.predict(xtest)


# VISUALIZACION
fig = plt.figure(figsize=(10, 5))
fig.suptitle("Regression with SVR", fontsize=16)

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(test_data["X"], test_data["Y"], test_data["Z"], c=test_data["Z"], s=10)
ax1.set_title("Ground truth")
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(test_data["X"], test_data["Y"], ypred, c=ypred, s=10)
ax2.set_title("Predictions")
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

plt.show()

print(f"Dataset de entrenamiento: {xtrain.shape} puntos")
print(f"Dataset de prueba: {xtest.shape} puntos")