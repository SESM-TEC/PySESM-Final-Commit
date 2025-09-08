"""ANN para regresion"""
import torch.nn as nn
import torch
import torch.optim as optim
import time
from sklearn.preprocessing import StandardScaler


class NN(nn.Module):
    def __init__(self, epochs, lr, hidden_dim, input_d):
        super().__init__()
        self.scaler = StandardScaler()
  
        self.epochs = epochs
        self.lr = lr
        self.layers = nn.Sequential(
            nn.Linear(input_d, hidden_dim),
            nn.Tanh(), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(), 
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Convertir el tensor de PyTorch a un array de NumPy
        x_np = x.detach().cpu().numpy()
        
        # 2. Normalizar el array de NumPy, capturando el resultado
        x_normalized_np = self.scaler.transform(x_np)
        
        # 3. Convertir el array de NumPy normalizado de nuevo a un tensor de PyTorch
        x_normalized_tensor = torch.from_numpy(x_normalized_np).float()
        
        # 4. Pasar el tensor normalizado a la red neuronal para la predicción
        predictions = self.layers(x_normalized_tensor)
        
        # 5. El resto del código es correcto para la salida
        predictions = predictions.detach().cpu().numpy().squeeze()
        return predictions
        
    def save(self, path: str):
        torch.save(self.state_dict(), path)
        print(f"Model saved as {path} \n")

    def load(self, path: str = 'nn_model.pth') -> 'NN':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_state_dict(torch.load(path, map_location=device))
        print(f"Model loaded {path}")
        return self



    def train_for_experiment(self, xtrain, ytrain, xtest, ytest):
        xtrain = self.scaler.fit_transform(xtrain)
        xtest = self.scaler.transform(xtest)
        # Convertir los arrays de NumPy a tensores de PyTorch
        xtrain = torch.from_numpy(xtrain).float()
        xtest = torch.from_numpy(xtest).float()
        

        # ENTRENAMIENTO
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        print("\n Training NN...")

        start_time = time.time()
        for epoch in range(self.epochs):
            self.train()

            # 1. Vaciar los gradientes
            optimizer.zero_grad()

            # 2. Forward pass: pasar el tensor completo
            predictions = self(xtrain)

            # 3. Calcular la pérdida
            loss = criterion(predictions, ytrain.unsqueeze(1))

            # 4. Backward pass y actualización de pesos
            loss.backward()
            optimizer.step()
            
            # Evaluación (opcional pero recomendado)
            self.eval()
            with torch.no_grad():
                test_loss = criterion(self(xtest), ytest.unsqueeze(1))
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], "
                    f"mse_train: {loss.item():.4f}, "
                    f"mse_val: {test_loss.item():.4f}")
        end_time = time.time()


        # GUARDAR MODELO
        path = "nn_model.pth"
        self.save(path)

        return end_time - start_time

    def test(self, xtest):
        model_path = "nn_model.pth"
        self.load(model_path)
        # PREDICTIONS
        ypred = self.predict(xtest)
        return ypred