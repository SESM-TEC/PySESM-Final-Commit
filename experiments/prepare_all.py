from SVR.model import SVR
from NN.model import NN

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import mean_squared_error, mean_absolute_error
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


def comparative_plot( svr_pred, nn_pred, SESM_pred, train_data, test_data):
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

        ax.scatter(test_data["X"], test_data["Y"], pred, s=.5, c=test_data["Z"], alpha = .6)
        ax.scatter(train_data["X"], train_data["Y"], train_data["Z"], s=2, c='r', marker = 'x')
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
    
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.tight_layout() 
    plt.show()
    return fig




class EXPERIMENT:
    def __init__(self, svr_config: dict, nn_config: dict, experiment1: dict):
        self.svr_config = svr_config
        self.nn_config = nn_config
        self.experiment1 = experiment1
        logger = setup_logger(level=logging.DEBUG)

        self.SESM_model=SSESM(**self.experiment1, logger=logger)
        self.SVR_model = SVR(svr_config)
        self.nn_model = NN(nn_config)



    def train_all(
            self,
            train_data, 
            test_data):
        # ENTRENAMIENTO
        xtrain, ytrain, xtest, ytest = prepare_dataset(train_data, test_data)

        self.SVR_model.train(xtrain, ytrain)
        self.nn_model.train_for_experiment(xtrain, ytrain, xtest, ytest)
        self.SESM_model.partial_fit(xtrain, ytrain)



    def test_all(self, train_data, test_data, plot_flag=False):

        _, _, xtest, ytest = prepare_dataset(train_data, test_data)
        
        svr_pred = self.SVR_model.test(xtest)
        nn_pred = self.nn_model.test(xtest)
        SESM_pred, _, SESM_mse = self.SESM_model.performance_stats(xtest, ytest)

        metrics = {
            "SVR_MSE": mean_squared_error(ytest, svr_pred),
            "SVR_MAE": mean_absolute_error(ytest, svr_pred),
            "NN_MSE": mean_squared_error(ytest, nn_pred),
            "NN_MAE": mean_absolute_error(ytest, nn_pred),
            "SESM_MSE": SESM_mse,
            "SESM_MAE":mean_absolute_error(ytest, SESM_pred)
        }
        if plot_flag:
            SESM_pred=SESM_pred.detach().cpu().numpy().squeeze()
            fig = comparative_plot(svr_pred, nn_pred, SESM_pred, train_data, test_data)
            wandb.log({"comparative_plot": wandb.Image(fig)})
        return metrics



