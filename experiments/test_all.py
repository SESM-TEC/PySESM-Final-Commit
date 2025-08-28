from SVR.test import test_svr
from NN.test import test_nn

import wandb
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import mean_squared_error, mean_absolute_error



def test_all(train_data, test_data, plot_flag=False):

    svr_pred = test_svr(train_data, test_data)
    nn_pred = test_nn(train_data, test_data)
    y_true = test_data["Z"]


    metrics = {
        "SVR_MSE": mean_squared_error(y_true, svr_pred),
        "SVR_MAE": mean_absolute_error(y_true, svr_pred),
        "NN_MSE": mean_squared_error(y_true, nn_pred),
        "NN_MAE": mean_absolute_error(y_true, nn_pred)
    }
    if plot_flag:
        fig = comparative_plot(svr_pred, nn_pred, test_data)
        wandb.log({"comparative_plot": wandb.Image(fig)})
    return metrics




def comparative_plot(svr_pred, nn_pred, test_data):
    """
    Visualiza en 3D la superficie de los datos de prueba, SVR y NN.
    """
    fig, axes = plt.subplots(
        nrows=1, ncols=3, figsize=(6, 2), 
        subplot_kw={'projection': '3d'}, dpi=300
    )

    titles = ["Ground truth", "SVR predictions", "NN predictions"]
    predictions = [test_data["Z"], svr_pred, nn_pred]

    for ax, title, pred in zip(axes, titles, predictions):
        ax.plot_trisurf(
            test_data["X"], test_data["Y"], pred,
            cmap='viridis',
            shade=True,
            alpha=1,
            antialiased=False
        )
        ax.set_title(title, fontsize=10, pad = -10)
        ax.set_xlabel('X', fontsize=5, labelpad=-12)
        ax.set_ylabel('Y', fontsize=5, labelpad=-12)
        ax.set_zlabel('Z', fontsize=4, labelpad=-14)

        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_tick_params(labelsize=4, pad=-5)

        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.zaxis.set_major_locator(MultipleLocator(1))
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)
    plt.subplots_adjust(wspace=0.1)
    plt.tight_layout() 
    plt.show()
    return fig