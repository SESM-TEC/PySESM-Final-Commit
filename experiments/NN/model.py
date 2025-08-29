"""ANN para regresion"""
import torch.nn as nn
import torch
import torch.optim as optim
from sklearn.preprocessing import StandardScaler


class NN(nn.Module):
    def __init__(self, nn_config: dict = None):
        super().__init__()
        self.scaler = StandardScaler()

        self.epochs = nn_config["epochs"]
        self.lr = nn_config["lr"]
        hidden_dim = nn_config["hidden_dim"]

        self.layers = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Sigmoid(), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(), 
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        predictions = self.layers(x)
        predictions = predictions.detach().cpu().numpy().squeeze()
        return predictions
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)
        print(f"Model saved as {path}")

    def load(self, path: str = 'nn_model.pth') -> 'NN':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_state_dict(torch.load(path, map_location=device))
        print(f"Model loaded {path}")
        return self


    def normalize(self, xtrain, xtest) -> torch.Tensor:
        xtrain_np = xtrain.numpy()
        xtest_np = xtest.numpy()
        xtrain_norm = self.scaler.fit_transform(xtrain_np)
        xtest_norm = self.scaler.transform(xtest_np)
        return torch.tensor(xtrain_norm, dtype=torch.float32), torch.tensor(xtest_norm, dtype=torch.float32)


    def train_for_experiment(self, xtrain, ytrain, xtest, ytest):
        # ENTRENAMIENTO
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        print("Iniciando el entrenamiento...")

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
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], "
                    f"mse_train: {loss.item():.4f}, "
                    f"mse_val: {test_loss.item():.4f}")

        # GUARDAR MODELO
        path = "nn_model.pth"
        self.save(path)

    def test(self, xtest):
        model_path = "nn_model.pth"
        self.load(model_path)
        # PREDICTIONS
        ypred = self.predict(xtest)
        return ypred