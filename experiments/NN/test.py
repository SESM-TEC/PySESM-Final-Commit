
from pysesm.utils_dataset.generate_dataset import generate_custom_function_dataset
from model import RegressionNN  

import matplotlib.pyplot as plt


#CREAR DATASET
def f(x, y):
    return x**2 - y**2
train_data, xtrain, ytrain, test_data, xtest, ytest = generate_custom_function_dataset(
    n_samples=200,
    function=f,
    mesh_divisions=50
)

# PREDICCION
model = RegressionNN()
model_path = r"C:\Users\Lenovo Yoga\Desktop\SEMESTRE_II_2025\ASISTENCIA\PySESM\experiments\NN\regression_nn_model.pth"
model.load_model(model_path)
ypred = model.forward(xtest)

# VISUALIZACION
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(test_data["X"], test_data["Y"], test_data["Z"], c='k', s=10, label='Ground truth')
ax.scatter(test_data["X"], test_data["Y"], ypred.detach().cpu().numpy().squeeze(), c='b', s=10, label='Predictions')
ax.set_title("Conjunto de Datos de Función 'Sinc' (2D)")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()

print(f"Dataset de entrenamiento: {xtrain.shape} puntos")
print(f"Dataset de prueba: {xtest.shape} puntos")