import csv
import numpy as np
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

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
    # Gráfica interactiva de las pérdidas
    fig_loss_ISTA = px.scatter(x=range(len(losses_ISTA)), y=losses_ISTA, labels={'x':'Total epochs', 'y':'losses_ISTA'}, title='losses_ISTA vs Total epochs')
    fig_loss_Dictionary = px.scatter(x=range(len(losses_Dictionary)), y=losses_Dictionary, labels={'x':'Total epochs', 'y':'losses_Dictionary'}, title='losses_Dictionary vs Total epochs')
    
    # Datos de entrenamiento
    x_samples_train = X_train[:, 0].detach().numpy()
    y_samples_train = X_train[:, 1].detach().numpy()
    z_samples_train = y_train.detach().numpy()

    # Remodelar los datos de entrada
    X = test_dataset["X"].reshape(50, 50)  # Remodelar X a una matriz 2D
    Y = test_dataset["Y"].reshape(50, 50)  # Remodelar Y a una matriz 2D
    Z = Z.clone().reshape(50, 50).detach().numpy()

    # Gráfica interactiva de la función original
    fig_original_function = go.Figure(data=[go.Surface(z=test_dataset["Z"].reshape(50, 50), x=X, y=Y)])
    fig_original_function.update_traces(contours_z=dict(show=True, usecolormap=True))
    fig_original_function.add_scatter3d(x=x_samples_train, y=y_samples_train, z=z_samples_train, mode='markers', marker=dict(color='red'))
    fig_original_function.update_layout(title="Original Function", scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))

    # Gráfica interactiva del modelo sustituto
    fig_surrogate_model = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
    fig_surrogate_model.update_traces(contours_z=dict(show=True, usecolormap=True))
    fig_surrogate_model.add_scatter3d(x=x_samples_train, y=y_samples_train, z=z_samples_train, mode='markers', marker=dict(color='red'))
    fig_surrogate_model.update_layout(title="Surrogate Model", scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))

    # Mostrar las gráficas
    fig_loss_ISTA.show()
    fig_loss_Dictionary.show()
    fig_original_function.show()
    fig_surrogate_model.show()

    # Guardar las gráficas en archivos PNG
    fig_original_function.write_image(f"results_{hypset}/plots/{fngroup}.{iteration}_original_function.png")
    fig_surrogate_model.write_image(f"results_{hypset}/plots/{fngroup}.{iteration}_surrogate_model.png")


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

