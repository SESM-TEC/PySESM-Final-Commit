from SVR.model import SVR
from NN.model import NN

import wandb
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
from pysesm.models.SSESM import SSESM

def prepare_dataset(train_data: dict = None, test_data: dict = None):
        
    xtrain = torch.stack([train_data["X"], train_data["Y"]], dim=1)
    ytrain = train_data["Z"]
    
    xtest = torch.stack([test_data["X"], test_data["Y"]], dim=1)
    ytest = test_data["Z"]

    return xtrain, ytrain, xtest, ytest

def train_all(
        train_data, 
        test_data, 
        svr_config: dict, 
        nn_config: dict):
    # ENTRENAMIENTO
    xtrain, ytrain, xtest, ytest = prepare_dataset(train_data, test_data)

    SVR_model = SVR(kernel = svr_config["kernel"], 
                    C = svr_config["C"], 
                    gamma = svr_config["gamma"], 
                    epsilon = svr_config["epsilon"])
    SVR_model.train(xtrain, ytrain)

    nn_model = NN(nn_config)
    nn_model.train_for_experiment(xtrain, ytrain, xtest, ytest)

def test_all(train_data, test_data, SESM_model: SSESM, nn_config, plot_flag=False):
    
    xtrain, ytrain, xtest, ytest = prepare_dataset(train_data, test_data)
    
    SVR_model=SVR()
    nn_model=NN(nn_config)
    SESM_model.partial_fit(xtrain, ytrain)

    svr_pred = SVR_model.test(xtest)
    nn_pred = nn_model.test(xtest)
    SESM_pred, _, SESM_mse = SESM_model.performance_stats(xtest, ytest)

    y_true = test_data["Z"]
    
    metrics = {
        "SVR_MSE": mean_squared_error(y_true, svr_pred),
        "SVR_MAE": mean_absolute_error(y_true, svr_pred),
        "NN_MSE": mean_squared_error(y_true, nn_pred),
        "NN_MAE": mean_absolute_error(y_true, nn_pred),
        "SESM_MSE": SESM_mse,
        "SESM_MAE":mean_absolute_error(y_true, SESM_pred)
    }
    if plot_flag:
        SESM_pred=SESM_pred.detach().cpu().numpy().squeeze()
        fig = comparative_plot(svr_pred, nn_pred, SESM_pred, test_data)
        wandb.log({"comparative_plot": wandb.Image(fig)})
    return metrics




def comparative_plot(svr_pred, nn_pred, SESM_pred, test_data):
    """
    Visualiza en 3D la superficie de los datos de prueba, SVR, NN y SESM.
    """
    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(8, 8), 
        subplot_kw={'projection': '3d'}, dpi=100
    )
    
    # Flatten axes for easier iteration
    axes = axes.flatten()

    titles = ["Ground truth", "SVR predictions", "NN predictions", "SESM predictions"]
    predictions = [test_data["Z"], svr_pred, nn_pred, SESM_pred]

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
    
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.tight_layout() 
    plt.show()
    return fig