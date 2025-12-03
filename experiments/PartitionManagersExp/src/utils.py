"""Utility functions for plotting and tensor/array conversion used by
PartitionManagers experiments.

This module provides helpers to convert PyTorch tensors (or lists) to
NumPy arrays and to save a simple scatter plot comparing ground-truth vs
predicted values.
"""

import matplotlib.pyplot as plt
import torch
import numpy as np


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


def plot_predictions(y_true, y_pred, title, outpath):
    """Guardar un scatter plot de ``y_true`` vs ``y_pred`` en ``outpath``.

    Convierte internamente tensores a NumPy, dibuja la línea ideal
    ``y_true == y_pred`` y guarda la figura a disco.

    Args:
        y_true: Valores reales (tensor/list/ndarray).
        y_pred: Valores predichos (tensor/list/ndarray).
        title: Título de la figura.
        outpath: Ruta de salida para el PNG.
    """
    y_true_np = to_numpy(y_true)
    y_pred_np = to_numpy(y_pred)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true_np, y_pred_np, alpha=0.6, s=15, edgecolors='k', linewidth=0.5)

    # Línea ideal
    min_val = min(y_true_np.min(), y_pred_np.min())
    max_val = max(y_true_np.max(), y_pred_np.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction")
    plt.title(title)
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close()
