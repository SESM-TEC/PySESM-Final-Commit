from .model import NN

import torch
import torch.nn as nn
import torch.optim as optim


def train_nn(train_data: dict, 
             test_data: dict, 
             nn_config: dict):

    epochs = nn_config["epochs"]
    lr = nn_config["lr"]

    # CREAR MODELO
    model = NN(hidden_dim = nn_config["hidden_dim"])

    # PREPARAR DATOS
    xtrain, ytrain, xtest, ytest = model.prepare_dataset(train_data, test_data)

    # ENTRENAMIENTO
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print("Iniciando el entrenamiento...")

    for epoch in range(epochs):
        model.train()

        # 1. Vaciar los gradientes
        optimizer.zero_grad()

        # 2. Forward pass: pasar el tensor completo
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
            print(f"Epoch [{epoch+1}/{epochs}], "
                f"mse_train: {loss.item():.4f}, "
                f"mse_val: {test_loss.item():.4f}")

    # GUARDAR MODELO
    path = "nn_model.pth"
    model.save(path)