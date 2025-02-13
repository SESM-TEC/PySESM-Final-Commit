import csv
import numpy as np
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os

from pysesm.models.SESM.SESM import SESM

def plot_surface(test_dataset, X_train, y_train, Z, model: SESM, hypset):
    """
    Plots multiple subplots including loss curves, sampled data, original function, and surrogate model surface.

    Args:
    - train_dataset (dict): A dictionary containing the training dataset.
    - test_dataset (dict): A dictionary containing the test dataset.
    - X_train (torch.Tensor): The input data points for training.
    - y_train (torch.Tensor): The target values for training.
    - Z (torch.Tensor): The predicted values (surface) from the surrogate model.
    - model(SESM): trained model
    - hypset(int): The hyperparameter set identifier.
    """
    
    # Matplotlib subplots (no interactivas)
    fig = plt.figure(figsize=(15, 10))

    # Total epochs = 6 [2 * ( 3 permutaciones )] * 16 bloques
    ax1 = fig.add_subplot(231)
    ax1.scatter(range(len(model.losses_ISTA)), model.losses_ISTA)
    ax1.set_xlabel("Total epochs")
    ax1.set_ylabel("losses_ISTA")
    ax1.set_title("ISTA Loss (Model epochs)")

    ax5 = fig.add_subplot(232)
    ax5.scatter(range(len(model.losses_Dictionary)), model.losses_Dictionary)
    ax5.set_xlabel("Total epochs")
    ax5.set_ylabel("losses_Dictionary")
    ax5.set_title("Dictionary Loss (Model epochs)")

    ax3 = fig.add_subplot(233)
    ax3.scatter(X_train[:, 0], X_train[:, 1])
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_title("Sampled positions")

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

    ax4.set_title(f"Ground truth - {hypset}/{model.dfngroup}")

    # Matplotlib para "Surrogate Model" (estática)
    Z_pred = Z.clone().reshape(N, N).detach().numpy()
    ax2 = fig.add_subplot(235, projection="3d")
    ax2.scatter(X, Y, Z_pred, c='0.4',marker='.',label='Predicted')
    ax2.scatter(X_train[:,0],X_train[:,1], y_train,c='r',marker='x',label='Train')
    ax2.set_xlabel('x_1')
    ax2.set_ylabel('x_2')
    ax2.set_zlabel('y')
    ax2.legend()

    ax2.set_title(f"Surrogate Model - {hypset}/{model.dfngroup}")

    plt.tight_layout()
    return fig

def save_surface(test_dataset, X_train, y_train, Z, folder_name, model: SESM, hypset):
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
    os.makedirs(f"{folder_name}/stats", exist_ok=True)

    # Matplotlib subplots (no interactivas)
    fig = plot_surface(test_dataset,X_train,y_train,Z,model,hypset)

    plt.savefig(f"{folder_name}/plots/{model.dfngroup}.{1}_static.png")
    plt.close(fig)

    # Plotly para la gráfica interactiva de la "Original Function"
    fig_original = go.Figure(
        data=[go.Surface(z=Z_test, x=X, y=Y, colorscale="Viridis")]
    )
    fig_original.add_scatter3d(
        x=X_train[:, 0],
        y=X_train[:, 1],
        z=y_train,
        mode="markers",
        marker=dict(size=5, color="red"),
    )
    fig_original.update_layout(
        title=f"Original Function - {hypset}/{model.dfngroup}",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
    )
    fig_original.write_html(
        f"{folder_name}/plots/{model.dfngroup}.{1}_original.html"
    )

    # Plotly para la gráfica interactiva del "Surrogate Model"
    fig_surrogate = go.Figure(
        data=[go.Surface(z=Z_pred, x=X, y=Y, colorscale="Viridis")]
    )
    fig_surrogate.add_scatter3d(
        x=X_train[:, 0],
        y=X_train[:, 1],
        z=y_train,
        mode="markers",
        marker=dict(size=5, color="red"),
    )
    fig_surrogate.update_layout(
        title=f"Surrogate Model - {hypset}/{model.dfngroup}",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
    )
    fig_surrogate.write_html(
        f"{folder_name}/plots/{model.dfngroup}.{1}_surrogate.html"
    )


def plot_stats(directory, num_files):
    """
    Plot statistics for loss values from multiple CSV files.

    Args:
        directory (str): The directory containing CSV files.
        num_files (int): The number of CSV files to process.

    Returns:
        None: Displays an interactive plot and saves an HTML file.

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

    with open(f"results_{fngroup}.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Repetion", "Time (min)", "MSE"])
        writer.writerows(data)
        writer.writerow(["Mean", average_time, average_mse])
        writer.writerow(["Std", std_time, std_mse])
