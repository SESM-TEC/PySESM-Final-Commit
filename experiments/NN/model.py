"""ANN para regresion"""
import torch.nn as nn
import torch
import torch.optim as optim
import time


class NN(nn.Module):
    def __init__(self, epochs, lr, hidden_dim, input_d):
        super().__init__()
  
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
    

        
    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str = 'nn_model.pth') -> 'NN':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_state_dict(torch.load(path, map_location=device))
        return self



    def train_for_experiment(self, xtrain, ytrain, xtest, ytest):
        self.Xmean = torch.mean(xtrain, dim=0)
        self.Xstd = torch.std(xtrain, dim=0)
        self.ymean = torch.mean(ytrain, dim=0)
        self.ystd = torch.std(ytrain, dim=0)

        xtrain = (xtrain - self.Xmean) / self.Xstd
        xtest = (xtest - self.Xmean) / self.Xstd

        ytrain = (ytrain - self.ymean) / self.ystd
        ytest = (ytest - self.ymean) / self.ystd

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
        print("\n Testing NN...")
        model_path = "nn_model.pth"
        self.load(model_path)

        xtest = (xtest - self.Xmean) / self.Xstd  # Normalizar usando los mismos parámetros que en el entrenamiento
        
        # 4. Pasar el tensor normalizado a la red neuronal para la predicción
        predictions = self.layers(xtest)
        predictions = predictions * self.ystd + self.ymean  # Desnormalizar las predicciones

        # 5. El resto del código es correcto para la salida
        predictions = predictions.detach().cpu().numpy().squeeze()
        return predictions
        