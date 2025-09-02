from SVR.model import SVR
from NN.model import NN
from PCE.model import PCE

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from scipy.interpolate import griddata
import torch
import wandb
from pysesm.models.SSESM import SSESM
from pysesm.utils.loggers import setup_logger
import logging

def prepare_dataset(train_data: dict = None, test_data: dict = None):
        
    xtrain = torch.stack([train_data["X"], train_data["Y"]], dim=1)
    ytrain = train_data["Z"]
    
    xtest = torch.stack([test_data["X"], test_data["Y"]], dim=1)
    ytest = test_data["Z"]
    return xtrain, ytrain, xtest, ytest

def comparative_plot(svr_pred, nn_pred, SESM_pred, pce_pred, train_data, test_data):
    """
    Visualiza en 3D la superficie de los datos de prueba, SVR, NN y SESM.
    """
    # 1. Ajustar la grilla para incluir un quinto plot
    fig, axes = plt.subplots(
        nrows=2, ncols=3, figsize=(12, 8), 
        subplot_kw={'projection': '3d'}, dpi=100
    )
    
    # Reajustar la proyección del quinto eje
    axes[1,2].remove() # Eliminar el último subplot 3D de la grilla
    ax5 = fig.add_subplot(2, 3, 6) # Agregar un nuevo subplot 2D en su lugar

    # Flatten axes for easier iteration
    axes_3d = fig.get_axes()[:5]
    
    titles = ["Ground truth", "SVR predictions", "NN predictions", "SESM predictions", "PCE predictions"]
    predictions = [test_data["Z"], svr_pred, nn_pred, SESM_pred, pce_pred]

    for ax, title, pred in zip(axes_3d, titles, predictions):

        ax.scatter(test_data["X"], test_data["Y"], pred, s=.5, c="0.4", alpha = .6, marker=".", label="Predicted")
        ax.scatter(train_data["X"], train_data["Y"], train_data["Z"], s=2, c='r', marker = 'x', label="Train")
        ax.legend()
        ax.set_title(title, fontsize=10, pad = -10)
        ax.set_xlabel('X', fontsize=5, labelpad=-12)
        ax.set_ylabel('Y', fontsize=5, labelpad=-12)
        ax.set_zlabel('Z', fontsize=4, labelpad=-14)

        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_tick_params(labelsize=4, pad=-5) # Numeros de los ejes

        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.zaxis.set_major_locator(MultipleLocator(1))
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)
    
    # Preparar los datos para el plot de contornos
    points = np.column_stack((test_data["X"], test_data["Y"]))
    values = test_data["Z"]
    
    xi = np.linspace(min(test_data["X"]), max(test_data["X"]), 100)
    yi = np.linspace(min(test_data["Y"]), max(test_data["Y"]), 100)
    
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata(points, values, (xi, yi), method='linear')

    # 3. Agregar el quinto plot en 2D
    ax5.set_title( f"{len(train_data['X'])} Samples of training", fontsize=10, pad = 0)
    img = ax5.imshow(zi, extent=[xi.min(), xi.max(), yi.min(), yi.max()], origin='lower', cmap='viridis')
    ax5.scatter(train_data["X"], train_data["Y"], s=1, alpha=1, c='red')
    ax5.set_xlabel('X', fontsize=5)
    ax5.set_ylabel('Z', fontsize=5)
    ax5.set_aspect('equal')
    ax5.grid(False)

    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.tight_layout() 
    plt.show()
    return fig



class EXPERIMENT:
    def __init__(self, svr_config: dict, nn_config: dict, experiment1: dict, pce_config: dict):

        logger = setup_logger(level=logging.DEBUG)

        self.SESM_model=SSESM(**experiment1, logger=logger)
        self.SVR_model = SVR(**svr_config)
        self.nn_model = NN(**nn_config)
        self.PCE=PCE(**pce_config)


    def train_all(
            self,
            train_data, 
            test_data):
        # ENTRENAMIENTO
        xtrain, ytrain, xtest, ytest = prepare_dataset(train_data, test_data)

        self.SVR_model.train(xtrain, ytrain)
        self.nn_model.train_for_experiment(xtrain, ytrain, xtest, ytest)
        self.SESM_model.partial_fit(xtrain, ytrain)
        self.PCE.train(xtrain, ytrain)


    def test_all(self, train_data, test_data, plot_flag=False):

        _, _, xtest, ytest = prepare_dataset(train_data, test_data)
        

        svr_pred = self.SVR_model.test(xtest)
        nn_pred = self.nn_model.test(xtest)
        SESM_pred, _, SESM_mse = self.SESM_model.performance_stats(xtest, ytest)
        pce_pred=self.PCE.test(xtest)

        metrics = {
            "SVR_MSE": mean_squared_error(ytest, svr_pred),
            "SVR_MAE": mean_absolute_error(ytest, svr_pred),
            "NN_MSE": mean_squared_error(ytest, nn_pred),
            "NN_MAE": mean_absolute_error(ytest, nn_pred),
            "SESM_MSE": SESM_mse,
            "SESM_MAE":mean_absolute_error(ytest, SESM_pred),
            "PCE_MSE": mean_squared_error(ytest, pce_pred),
            "PCE_MAE": mean_absolute_error(ytest, pce_pred)
        }
        if plot_flag:
            SESM_pred=SESM_pred.detach().cpu().numpy().squeeze()
            print("AQUIII", nn_pred.size, pce_pred.size)
            fig = comparative_plot(svr_pred, nn_pred, SESM_pred, pce_pred, train_data, test_data)
            wandb.log({"comparative_plot": wandb.Image(fig)})
        return metrics






