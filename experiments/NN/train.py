from pysesm.utils_dataset.generate_dataset import generate_custom_function_dataset
from model import RegressionNN

import torch
import torch.nn as nn
import torch.optim as optim

# CREAR DATASET
def f(x, y):
    return x**2 - y**2

n_samples = 100
mesh_divisions = 50
train_data, xtrain, ytrain, test_data, xtest, ytest = generate_custom_function_dataset(
    n_samples=n_samples,
    function=f,
    mesh_divisions=mesh_divisions
)

# CREAR MODELO
model = RegressionNN(input_dim=2, hidden_dim=16)

# ENTRENAMIENTO
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 100
print("Iniciando el entrenamiento (sin DataLoader)...")

for epoch in range(num_epochs):
    model.train()
    # 1. Vaciar los gradientes
    optimizer.zero_grad()
    # 2. Forward pass: pasar el tensor completo
    # Asegúrate de que ytrain tenga la forma correcta [n_samples, 1]
    predictions = model(xtrain)
    # 3. Calcular la pérdida
    loss = criterion(predictions, ytrain.unsqueeze(1))
    # 4. Backward pass y actualización de pesos
    loss.backward()
    optimizer.step()
    # Evaluación (opcional pero recomendado)
    model.eval()
    with torch.no_grad():
        test_loss = criterion(model(xtest), ytest.unsqueeze(1))
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Pérdida de entrenamiento: {loss.item():.4f}, "
              f"Pérdida de prueba: {test_loss.item():.4f}")

print("\n¡Entrenamiento finalizado!")
print(f"Pérdida de prueba final: {test_loss.item():.4f}")

# GUARDAR MODELO
model_path = r"C:\Users\Lenovo Yoga\Desktop\SEMESTRE_II_2025\ASISTENCIA\PySESM\experiments\NN\regression_nn_model.pth"
model.save_model(model_path)