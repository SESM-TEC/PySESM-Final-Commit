import csv
import os

import numpy as np
import pandas as pd
import torch

import plotly.express as px

import matplotlib.pyplot as plt

from pysesm.models.SESM import SESM

def plot_surface(test_dataset: dict,
                 X_train: torch.Tensor,
                 y_train: torch.Tensor,
                 y_pred: torch.Tensor,
                 model: SESM, 
                 *,
                 hypset: int,
                 dfngroup: str = "sesm"):
    """
    Plots multiple subplots including loss curves, sampled data, original function, and surrogate model surface.

    Args:
    - test_dataset (dict): A dictionary containing the test dataset, with ground-truth "Z".
    - X_train (torch.Tensor): The input data points for training.
    - y_train (torch.Tensor): The target values for training.
    - y_pred (torch.Tensor): The predicted values (surface) from the surrogate model.
    - model(SESM): trained model
    - hypset(int): The hyperparameter set identifier.
    - dfngroup(str): Descriptive name for the function group.
    """
    
    # Matplotlib subplots (no interactivas)
    fig = plt.figure(figsize=(15, 10))

    # Total epochs = 6 [2 * ( 3 permutaciones )] * 16 bloques
    ax1 = fig.add_subplot(231)
    ax1.scatter(range(len(model.sparse_coding_layer_losses)), model.sparse_coding_layer_losses)
    ax1.set_xlabel("Total epochs")
    ax1.set_ylabel("ISTA loss")
    ax1.set_title("ISTA Loss (Model epochs)")

    ax2 = fig.add_subplot(232)
    ax2.scatter(range(len(model.dictionary_layer_losses)), model.dictionary_layer_losses)
    ax2.set_xlabel("Total epochs")
    ax2.set_ylabel("Dictionary loss")
    ax2.set_title("Dictionary Loss (Model epochs)")

    # Modificación para mostrar puntos por bloque con colores diferentes
    ax3 = fig.add_subplot(233)
    
    # Verificar si el modelo tiene partition_manager (para SSESM)
    if hasattr(model, 'partition_manager'):
        # Obtener bloques activos
        active_blocks = model.partition_manager.retrieve_active_blocks()
        
        # Generar colores únicos para cada bloque
        colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(active_blocks)))

        # Iterar sobre los bloques y plotear sus puntos
        for i, block in enumerate(active_blocks):
            if len(block.X) > 0:  # Solo si el bloque tiene puntos
                # Usar las coordenadas originales (sin normalizar)
                block_X = torch.stack(block.X).detach().cpu().numpy()
                ax3.scatter(block_X[:, 0], block_X[:, 1], 
                            c=[colors[i]], 
                            label=f'Block {i}',
                            alpha=0.7)
        
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.set_title("Sampled positions by block")
    else:
        # Fallback para modelos sin partition_manager
        ax3.scatter(X_train[:, 0], X_train[:, 1])
        ax3.set_title("Sampled positions")
    
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")

    # Ajuste de los límites de los ejes
    ax3.set_xlim([min(X_train[:, 0]), max(X_train[:, 0])])
    ax3.set_ylim([min(X_train[:, 1]), max(X_train[:, 1])])

    # Guessing that the data is in a square mesh
    N = int(np.sqrt(test_dataset["X"].shape[0]))

    # Matplotlib para "Original Function" (estática)
    X = test_dataset["X"].reshape(N, N)
    Y = test_dataset["Y"].reshape(N, N)
    Z_test = test_dataset["Z"].reshape(N, N)

    ax4 = fig.add_subplot(234, projection="3d")
    ax4.scatter(X,Y,Z_test,c='0.4',marker='.',label='Test')
    ax4.scatter(X_train[:,0],X_train[:,1], y_train,c='r',marker='x',label='Train')
    ax4.set_xlabel('x_1')
    ax4.set_ylabel('x_2')
    ax4.set_zlabel('y')
    ax4.legend()

    ax4.set_title(f"Ground truth - {hypset}/{dfngroup}")

    # Matplotlib para "Surrogate Model" (estática)
    Z_pred = y_pred.detach().clone().reshape(N, N).numpy()
    ax5 = fig.add_subplot(235, projection="3d")
    ax5.scatter(X, Y, Z_pred, c='0.4',marker='.',label='Predicted')
    ax5.scatter(X_train[:,0],X_train[:,1], y_train,c='r',marker='x',label='Train')
    ax5.set_xlabel('x_1')
    ax5.set_ylabel('x_2')
    ax5.set_zlabel('y')
    ax5.legend()

    ax5.set_title(f"Surrogate Model - {hypset}/{dfngroup}")

    plt.tight_layout()
    return fig

def save_surface(test_dataset: dict,
                 X_train: torch.Tensor,
                 y_train: torch.Tensor,
                 y_pred: torch.Tensor,
                 folder_name: str,
                 model: SESM,
                 *,
                 hypset: int,
                 dfngroup: str = "sesm"):
    """
    Saves the plot results as subplots including loss curves, sampled data, original function, and surrogate model surface.

    Args:
    - train_dataset (dict): A dictionary containing the training dataset.
    - test_dataset (dict): A dictionary containing the test dataset.
    - X_train (torch.Tensor): The input data points for training.
    - y_train (torch.Tensor): The target values for training.
    - Z (torch.Tensor): The predicted values (surface) from the surrogate model.
    - model(SESM): trained model
    - folder_name (str): Destination of created files
    - hypset(int): The hyperparameter set identifier.
    """
    # Asegurarse de que el directorio exista
    os.makedirs(f"{folder_name}/plots", exist_ok=True)

    # Matplotlib subplots (no interactivas)
    fig = plot_surface(test_dataset=test_dataset,
                       X_train=X_train,
                       y_train=y_train,
                       y_pred=y_pred,
                       model=model,
                       hypset=hypset,
                       dfngroup=dfngroup)

    plt.savefig(f"{folder_name}/plots/{dfngroup}.{1}_static.png")
    plt.close(fig)



def plot_stats(directory, num_files):
    """
    Plot statistics for loss values from multiple CSV files.

    Args:
        directory (str): The directory containing CSV files.
        num_files (int): The number of CSV files to process.

    Note:
        Assumes that each CSV file contains loss values for the same number of epochs.

    """
    fig = px.scatter(title="Loss analysis")
    m_epochs_losses = []

    for i in range(num_files):
        file_path = f"{directory}/stats/{i}.csv"

        df = pd.read_csv(file_path)
        m_epochs_losses.append(df)

    merged_losses = pd.concat(m_epochs_losses, axis=1)

    # Compute mean, std, min, and max for each row
    summary_df = pd.DataFrame(
        {
            "Mean": merged_losses.mean(axis=1),
            "Std": merged_losses.std(axis=1),
            "Min": merged_losses.min(axis=1),
            "Max": merged_losses.max(axis=1),
        }
    )

    summary_df.to_csv(f"{directory}/stats/processed.csv", index=False)

    fig.add_scatter(
        x=summary_df.index,
        y=summary_df["Mean"],
        mode="lines+markers",
        error_y=dict(type="data", array=summary_df["Std"]),
        name="Mean",
    )

    fig.add_scatter(x=summary_df.index, y=summary_df["Max"], mode="markers", name="Max")

    fig.add_scatter(x=summary_df.index, y=summary_df["Min"], mode="markers", name="Min")

    fig.update_layout(
        xaxis_title="m_epochs",
        yaxis_title="Loss",
        legend_title="Legend",
        showlegend=True,
    )
    fig.show()
    fig.write_html(f"interactive_plot-{directory}.html")


def save_results(data, fngroup):
    # Compute Mean and Std for executio
    times = [item[1] for item in data]
    mse_values = [item[2] for item in data]

    average_time = np.mean(times)
    std_time = np.std(times)
    average_mse = np.mean(mse_values)
    std_mse = np.std(mse_values)

    with open(f"results_{fngroup}.csv", mode="w", newline="", encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Repetion", "Time (min)", "MSE"])
        writer.writerows(data)
        writer.writerow(["Mean", average_time, average_mse])
        writer.writerow(["Std", std_time, std_mse])
