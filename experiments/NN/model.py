"""ANN para regresion"""
import torch.nn as nn
import torch
import torch.optim as optim


class NN(nn.Module):
    def __init__(self, nn_config, input_dim: int = 2, hidden_dim: int = 16, output_dim: int = 1 ):
        super().__init__()
        self.nn_config=nn_config
        hidden_dim=nn_config["hidden_dim"]
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(), 
            nn.Linear(hidden_dim, output_dim)
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

    def load(self, path: str) -> 'NN':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_state_dict(torch.load(path, map_location=device))
        print(f"Model loaded {path}")
        return self

    def train_for_experiment(self, xtrain, ytrain, xtest, ytest):

        epochs = self.nn_config["epochs"]
        lr = self.nn_config["lr"]

        # ENTRENAMIENTO
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        print("Iniciando el entrenamiento...")

        for epoch in range(epochs):
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
                print(f"Epoch [{epoch+1}/{epochs}], "
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