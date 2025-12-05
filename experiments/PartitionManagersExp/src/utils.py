"""Utility functions for plotting and tensor/array conversion used by
PartitionManagers experiments.

This module provides helpers to convert PyTorch tensors (or lists) to
NumPy arrays and to save a simple scatter plot comparing ground-truth vs
predicted values.
"""

import matplotlib.pyplot as plt
import torch
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error

def to_numpy(data):
    """Convertir ``data`` a un ndarray de NumPy.

    - Si ``data`` es un ``torch.Tensor`` se mueve a CPU y se desconecta del
      grafo para obtener un array.
    - Si ``data`` es una ``list`` se convierte con ``np.array``.
    - En cualquier otro caso se devuelve ``data`` tal cual.

    Args:
        data: ``torch.Tensor``, ``list`` u objeto ya en formato NumPy.

    Returns:
        Un objeto NumPy (o el valor original si no aplica conversión).
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, list):
        return np.array(data)
    return data

def plot_multi_method_comparison(X_test, y_test, predictions_dict, X_train, y_train, dim, title, outpath):
    """
    Grafica comparación visual: Ground Truth vs Método 1 vs Método 2.
    
    Args:
        predictions_dict: Dict { "nombre_metodo": y_pred_tensor, ... }
    """
    
    # Filtro: Solo graficamos Dim 2 (Superficie)
    # (Se podría extender a dim 3, pero se vuelve difícil de ver en 3 paneles)
    if dim != 2:
        return

    # Convertir datos base a numpy
    Xt = to_numpy(X_test)
    Yt = to_numpy(y_test).flatten()
    Xtr = to_numpy(X_train)
    Ytr = to_numpy(y_train).flatten()

    methods = list(predictions_dict.keys())
    n_methods = len(methods)
    
    # Configuración de la figura: 1 fila, 1 (GT) + n_methods columnas
    cols = 1 + n_methods
    fig = plt.figure(figsize=(6 * cols, 6))

    # --- SUBPLOT 1: GROUND TRUTH ---
    ax1 = fig.add_subplot(1, cols, 1, projection='3d')
    # Puntos reales (Gris)
    ax1.scatter(Xt[:, 0], Xt[:, 1], Yt, c='0.4', marker='.', s=15, alpha=0.2, label='Ground Truth')
    # Puntos de entrenamiento (Rojos)
    ax1.scatter(Xtr[:, 0], Xtr[:, 1], Ytr, c='r', marker='x', s=40, label='Train Data')
    
    ax1.set_title(f"Ground Truth\n{title}")
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.set_zlabel('Y')
    ax1.view_init(elev=30, azim=-60)

    # --- SUBPLOTS MÉTODOS ---
    # Colores para diferenciar métodos: Azul, Verde, Purpura...
    colors = ['b', 'g', 'm', 'c']
    
    for i, method_name in enumerate(methods):
        y_pred_tensor = predictions_dict[method_name]
        Yp = to_numpy(y_pred_tensor).flatten()
        
        # Calcular MSE localmente para el título
        mse_val = np.mean((Yt - Yp)**2)
        
        ax = fig.add_subplot(1, cols, i + 2, projection='3d')
        
        # Predicción
        col = colors[i % len(colors)]
        ax.scatter(Xt[:, 0], Xt[:, 1], Yp, c=col, marker='.', s=15, alpha=0.2, label='Predicción')
        
        # Referencia Entrenamiento (para ver si pasaron por los puntos)
        ax.scatter(Xtr[:, 0], Xtr[:, 1], Ytr, c='r', marker='x', s=40)
        
        ax.set_title(f"Modelo: {method_name.upper()}\nMSE: {mse_val:.5f}")
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('Y')
        ax.view_init(elev=30, azim=-60)

    plt.tight_layout()
    plt.savefig(outpath, dpi=100) # dpi moderado para no hacer archivos gigantes
    plt.close()
