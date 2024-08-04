import csv
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

def plot_surface(test_dataset, X_train, y_train, Z, hypset, fngroup, iteration, losses_ISTA, losses_Dictionary):
    """
    Plots multiple subplots including loss curves, sampled data, original function, and surrogate model surface.

    Args:
    - test_dataset (dict): A dictionary containing the test dataset.
    - X_train (torch.Tensor): The input data points for training.
    - y_train (torch.Tensor): The target values for training.
    - Z (torch.Tensor): The predicted values (surface) from the surrogate model.
    - hypset (str): The hyperparameter set identifier.
    - fngroup (str): The function group identifier.
    - iteration (int): The iteration number of the experiment.
    - losses_ISTA (list): List of ISTA losses over epochs.
    - losses_Dictionary (list): List of Dictionary losses over epochs.
    """
    fig = plt.figure(figsize=(15, 10))
    X = test_dataset["X"].reshape(50, 50)  # Remodelar X a una matriz 2D
    Y = test_dataset["Y"].reshape(50, 50)  # Remodelar Y a una matriz 2D
    Z = Z.clone().reshape(50, 50).detach().numpy()

    x_samples_train = X_train[:, 0]
    y_samples_train = X_train[:, 1]
    z_samples_train = y_train

    #Total epochs = 6 [2 * ( 3 permutaciones )] * 16 bloques
    ax1 = fig.add_subplot(231)
    ax1.scatter(range(len(losses_ISTA)), losses_ISTA)
    ax1.set_xlabel('Total epochs')
    ax1.set_ylabel('losses_ISTA')
    ax1.set_title('losses_ISTA vs Total epochs')

    ax5 = fig.add_subplot(232)
    ax5.scatter(range(len(losses_Dictionary)), losses_Dictionary)
    ax5.set_xlabel('Total epochs')
    ax5.set_ylabel('losses_Dictionary')
    ax5.set_title('losses_Dictionary vs Total epochs')

    ax3 = fig.add_subplot(233)
    ax3.scatter(x_samples_train, y_samples_train)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('Sampled Data')

    # Ajuste de los límites de los ejes
    ax3.set_xlim([min(x_samples_train), max(x_samples_train)])
    ax3.set_ylim([min(y_samples_train), max(y_samples_train)])

    ax4 = fig.add_subplot(234, projection='3d')
    ax4.plot_surface(X, Y, test_dataset["Z"].reshape(50, 50), cmap='viridis', alpha=0.9)
    ax4.scatter(x_samples_train, y_samples_train, z_samples_train, c='red')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.set_title('Original Function')

    # Ajuste de los límites de los ejes
    ax4.set_xlim([min(x_samples_train), max(x_samples_train)])
    ax4.set_ylim([min(y_samples_train), max(y_samples_train)])
    ax4.set_zlim([min(z_samples_train), max(z_samples_train)])

    ax2 = fig.add_subplot(212, projection='3d')
    ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9)
    ax2.scatter(x_samples_train, y_samples_train, z_samples_train, c='red')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Surrogate Model')

    # Ajuste de los límites de los ejes
    ax2.set_xlim([min(x_samples_train), max(x_samples_train)])
    ax2.set_ylim([min(y_samples_train), max(y_samples_train)])
    ax2.set_zlim([min(z_samples_train), max(z_samples_train)])

    filename = f"results_{hypset}/plots/{fngroup}.{iteration}.png"

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)


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
    fig = px.scatter(title='Loss analysis')
    m_epochs_losses = []

    for i in range(num_files):
        file_path = f"{directory}/stats/{i}.csv"

        df = pd.read_csv(file_path)
        m_epochs_losses.append(df)

    merged_losses = pd.concat(m_epochs_losses, axis=1)

    # Compute mean, std, min, and max for each row
    summary_df = pd.DataFrame({
        'Mean': merged_losses.mean(axis=1),
        'Std': merged_losses.std(axis=1),
        'Min': merged_losses.min(axis=1),
        'Max': merged_losses.max(axis=1)
    })

    summary_df.to_csv(f'{directory}/stats/processed.csv', index=False)

    fig.add_scatter(
        x=summary_df.index,
        y=summary_df['Mean'],
        mode='lines+markers',
        error_y=dict(type='data', array=summary_df['Std']),
        name='Mean'
    )

    fig.add_scatter(
        x=summary_df.index,
        y=summary_df['Max'],
        mode='markers',
        name='Max'
    )

    fig.add_scatter(
        x=summary_df.index,
        y=summary_df['Min'],
        mode='markers',
        name='Min'
    )

    fig.update_layout(
        xaxis_title='m_epochs',
        yaxis_title='Loss',
        legend_title='Legend',
        showlegend=True
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

