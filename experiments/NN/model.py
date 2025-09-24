import torch.nn as nn
import torch
import torch.optim as optim
import time
from torch.utils.data import TensorDataset, DataLoader

class NN(nn.Module):
    def __init__(self, epochs, lr, hidden_dim, input_d, batch_size=32):
        super().__init__()
 
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size # Añadido batch_size
        self.layers = nn.Sequential(
            nn.Linear(input_d, hidden_dim),
            nn.Tanh(), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(), 
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)
    
    def train_nn(self, xtrain, ytrain, xtest, ytest):

        # Convertir datos a TensorDataset
        train_dataset = TensorDataset(xtrain, ytrain.unsqueeze(1))
        # Crear DataLoader
        train_loader = DataLoader(dataset=train_dataset, 
                                  batch_size=self.batch_size, 
                                  shuffle=True)
        
        # ENTRENAMIENTO
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        print("\n Training NN...")

        start_time = time.time()
        for epoch in range(self.epochs):
            self.train()
            
            # Iterar sobre el DataLoader para entrenar en lotes
            for batch_xtrain, batch_ytrain in train_loader:
                # 1. Vaciar los gradientes
                optimizer.zero_grad()

                # 2. Forward pass: pasar el lote actual
                predictions = self(batch_xtrain)

                # 3. Calcular la pérdida
                loss = criterion(predictions, batch_ytrain)

                # 4. Backward pass y actualización de pesos
                loss.backward()
                optimizer.step()
            
            # Evaluación (opcional pero recomendado)
            self.eval()
            with torch.no_grad():
                test_loss = criterion(self(xtest), ytest.unsqueeze(1))
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], "
                      f"mse_train: {loss.item():.6f}, "
                      f"mse_val: {test_loss.item():.6f}")
        end_time = time.time()

        return end_time - start_time

    def test(self, xtest):
        print("\n Testing NN...")
        # Evaluación
        predictions = self.layers(xtest)
        # Conversion a numpy
        predictions = predictions.detach().cpu().numpy().squeeze()
        return predictions