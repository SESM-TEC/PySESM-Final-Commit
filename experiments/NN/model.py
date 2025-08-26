"""ANN para regresion"""
import numpy as np
import torch.nn as nn
import torch


class NN(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 16, output_dim: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> np.ndarray:
        predictions = self.layers(x)
        predictions = predictions.detach().cpu().numpy().squeeze()
        return predictions
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)
        print(f"Modelo guardado en {path}")

    def load(self, path: str) -> 'NN':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_state_dict(torch.load(path, map_location=device))
        return self
