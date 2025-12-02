import matplotlib.pyplot as plt
import torch
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def to_numpy(data):
    """Convierte Tensores o Listas a Numpy array de forma segura (CPU)."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, list):
        return np.array(data)
    return data

def plot_surface_comparison(X_test, y_test, y_pred, X_train, y_train, dim, title, outpath):
    """
    Grafica la comparación visual solo si dim es 2.
    Para dim=3, se hace un intento de visualización volumétrica pero se advierte.
    """
    
    # Filtro: Solo graficamos Dim 2 (Superficie) y Dim 3 (Volumen)
    if dim not in [2, 3]:
        return

    # Convertir todo a numpy
    Xt = to_numpy(X_test)
    Yt = to_numpy(y_test).flatten()
    Yp = to_numpy(y_pred).flatten()
    Xtr = to_numpy(X_train)
    Ytr = to_numpy(y_train).flatten()

    # Configuración de la figura
    fig = plt.figure(figsize=(16, 7))

    # ==========================================
    # CASO 2D: Superficie z = f(x, y)
    # ESTILO: Puntos Grises (Real) vs Azules (Pred), SIN escala de color
    # ==========================================
    if dim == 2:
        # --- SUBPLOT 1: GROUND TRUTH ---
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Realidad: Puntos pequeños en GRIS (0.4)
        ax1.scatter(Xt[:, 0], Xt[:, 1], Yt, c='0.4', marker='.', s=1, alpha=0.2, label='Ground Truth')
        
        # Entrenamiento: Cruces ROJAS
        ax1.scatter(Xtr[:, 0], Xtr[:, 1], Ytr, c='r', marker='x', s=30, label='Train Data')
        
        ax1.set_title(f"Realidad (Ground Truth)\n{title}")
        ax1.set_xlabel('X1')
        ax1.set_ylabel('X2')
        ax1.set_zlabel('Y')
        # Ajustamos vista inicial para que no se vea distorsionado
        ax1.view_init(elev=30, azim=-60)

        # --- SUBPLOT 2: PREDICCIÓN ---
        ax2 = fig.add_subplot(122, projection='3d')
        
        # Predicción: Puntos pequeños en AZUL (o el color que prefiera)
        ax2.scatter(Xt[:, 0], Xt[:, 1], Yp, c='b', marker='.', s=1, alpha=0.2, label='Predicción')
        
        # Referencia Entrenamiento
        ax2.scatter(Xtr[:, 0], Xtr[:, 1], Ytr, c='r', marker='x', s=30, label='Train Data')
        
        mse_val = np.mean((Yt - Yp)**2)
        ax2.set_title(f"Predicción Modelo\nMSE: {mse_val:.5f}")
        ax2.set_xlabel('X1')
        ax2.set_ylabel('X2')
        ax2.set_zlabel('Y')
        ax2.view_init(elev=30, azim=-60)

    # ==========================================
    # CASO 3D: Volumen w = f(x, y, z)
    # ESTILO: Aquí SI es necesaria la escala de color (4ta dimensión)
    # ==========================================
    elif dim == 3:
        # --- SUBPLOT 1: GROUND TRUTH ---
        ax1 = fig.add_subplot(121, projection='3d')
        p1 = ax1.scatter(Xt[:, 0], Xt[:, 1], Xt[:, 2], c=Yt, cmap='viridis', marker='.', s=2, alpha=0.3)
        fig.colorbar(p1, ax=ax1, shrink=0.5, label='Valor Real')
        
        ax1.set_title(f"Volumen Real (Color=Valor)\n{title}")
        ax1.set_xlabel('X1')
        ax1.set_ylabel('X2')
        ax1.set_zlabel('X3')

        # --- SUBPLOT 2: PREDICCIÓN ---
        ax2 = fig.add_subplot(122, projection='3d')
        p2 = ax2.scatter(Xt[:, 0], Xt[:, 1], Xt[:, 2], c=Yp, cmap='viridis', marker='.', s=2, alpha=0.3)
        fig.colorbar(p2, ax=ax2, shrink=0.5, label='Valor Predicho')
        
        ax2.set_title("Volumen Predicho")
        ax2.set_xlabel('X1')
        ax2.set_ylabel('X2')
        ax2.set_zlabel('X3')

    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close()