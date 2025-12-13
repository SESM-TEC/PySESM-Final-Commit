"""Utility functions for plotting and tensor/array conversion used by
PartitionManagers experiments.

This module provides helpers to convert Pytorch tensors (or lists) to
NumPy arrays and to save a simple scatter plot comparing ground-truth vs
predicted values.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

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

def plot_multi_method_comparison(
    dataset,
    predictions_dict,
    dim,
    title,
    outpath,
):  # pylint: disable=too-many-locals
    """Comparación visual 3D entre Ground Truth y métodos.

    Crea una figura con subplots: uno para los datos reales y uno por cada
    método en ``predictions_dict``. Cada subplot muestra los puntos de prueba
    y de entrenamiento, y en el caso de los métodos se reporta el MSE local.

    Parameters
    ----------
    dataset : dict
        Diccionario con los datos base: 'x_test', 'y_test', 'x_train', 'y_train'.

    predictions_dict : dict[str, array-like or torch.Tensor]
        Diccionario con las predicciones por método. Llaves son nombres de
        método y valores son tensores/arrays con las predicciones sobre
        ``x_test``.

    dim : int
        Dimensionalidad del problema. Actualmente solo se grafica si ``dim == 2``.

    title : str
        Título base para los subplots.

    outpath : str
        Ruta de salida del archivo PNG que se genera.
    """

    # Filtro: Solo graficamos Dim 2 (Superficie)
    if dim != 2:
        return

    x_test = dataset["x_test"]
    y_test = dataset["y_test"]
    x_train = dataset["x_train"]
    y_train = dataset["y_train"]

    # Convertir datos base a numpy
    xt = to_numpy(x_test)
    yt = to_numpy(y_test).flatten()
    xtr = to_numpy(x_train)
    ytr = to_numpy(y_train).flatten()

    # --- NUEVO: Calcular límites globales basados en Ground Truth y Train ---
    # Esto asegura que todos los plots tengan la misma escala vertical.
    z_min = min(np.min(yt), np.min(ytr))
    z_max = max(np.max(yt), np.max(ytr))

    # Opcional: Agregar un pequeño margen (padding) del 5% para que los puntos no toquen los bordes
    z_range = z_max - z_min
    z_min -= z_range * 0.05
    z_max += z_range * 0.05

    methods = list(predictions_dict.keys())
    n_methods = len(methods)

    # Configuración de la figura: 1 fila, 1 (GT) + n_methods columnas
    cols = 1 + n_methods
    fig = plt.figure(figsize=(6 * cols, 6))

    # --- SUBPLOT 1: GROUND TRUTH ---
    ax1 = fig.add_subplot(1, cols, 1, projection='3d')
    # Puntos reales (Gris)
    ax1.scatter(
        xt[:, 0],
        xt[:, 1],
        yt,
        c='0.4',
        marker='.',
        s=15,
        alpha=0.2,
        label='Ground Truth'
    )

    # Puntos de entrenamiento (Rojos)
    ax1.scatter(
        xtr[:, 0],
        xtr[:, 1],
        ytr,
        c='r',
        marker='x',
        s=40,
        label='Train Data'
    )

    ax1.set_title(f"Ground Truth\n{title}")
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.set_zlabel('Y')

    # APLICAR ESCALA
    ax1.set_zlim(z_min, z_max)

    ax1.view_init(elev=30, azim=-60)

    # --- SUBPLOTS MÉTODOS ---
    # Colores para diferenciar métodos: Azul, Verde, Purpura...
    colors = ['b', 'g', 'm', 'c']

    for i, method_name in enumerate(methods):
        y_pred_tensor = predictions_dict[method_name]
        yp = to_numpy(y_pred_tensor).flatten()

        # Calcular MSE localmente para el título
        mse_val = np.mean((yt - yp)**2)

        ax = fig.add_subplot(1, cols, i + 2, projection='3d')

        # Predicción
        col = colors[i % len(colors)]
        ax.scatter(
            xt[:, 0],
            xt[:, 1],
            yp,
            c=col,
            marker='.',
            s=15,
            alpha=0.2,
            label='Predicción'
        )

        # Referencia Entrenamiento (para ver si pasaron por los puntos)
        ax.scatter(xtr[:, 0], xtr[:, 1], ytr, c='r', marker='x', s=40)

        ax.set_title(f"Modelo: {method_name.upper()}\nMSE: {mse_val:.5f}")
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('Y')

        # APLICAR ESCALA (La misma del GT)
        ax.set_zlim(z_min, z_max)

        ax.view_init(elev=30, azim=-60)

    plt.tight_layout()
    plt.savefig(outpath, dpi=100)
    plt.close()
