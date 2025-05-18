import torch

#2D
def paraboloid(x: torch.Tensor, y: torch.Tensor, a=1.0, b=1.0, c=0.0) -> torch.Tensor:
    return a * x**2 + b * y**2 + c

def sinusoidal(x: torch.Tensor, y: torch.Tensor, a=1.0, freq=1.0, phase=0.0) -> torch.Tensor:
    return a * torch.sin(freq * (x + y) + phase)

def exponential(x: torch.Tensor, y: torch.Tensor, a=1.0, b=1.0, offset=0.0) -> torch.Tensor:
    return a * torch.exp(-b * (x**2 + y**2)) + offset

def ripple(x: torch.Tensor, y: torch.Tensor, a=1.0, freq=3.0) -> torch.Tensor:
    r = torch.sqrt(x**2 + y**2)
    return a * torch.sin(freq * r) / (r + 1e-5)  # evitar división por cero



#ND
#a deberia de ser una lista o index para las N-dimensiones
def nd_paraboloid(X: torch.Tensor, a: float = 1.0, c: float = 0.0) -> torch.Tensor:
    # Suma ponderada de cuadrados: f(x) = a * sum(x_i^2) + c
    return a * torch.sum(X**2, dim=1) + c

def nd_exponential(X: torch.Tensor, a: float = 1.0, b: float = 1.0, offset: float = 0.0) -> torch.Tensor:
    #Función exponencial N-dimensional: f(X) = a * exp(-b * sum(x_i^2)) + offset
    return a * torch.exp(-b * torch.sum(X**2, dim=-1)) + offset