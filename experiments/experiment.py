from SVR.model import SVR
from NN.model import NN
from PF.model import PF

from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from pysesm.models.SSESM import SSESM
from pysesm.utils.loggers import setup_logger
import logging

def prepare_dataset(train_data: dict = None, test_data: dict = None):
    xtrain = train_data["X"]  
    ytrain = train_data["Z"]

    xtest = test_data["X"]
    ytest = test_data["Z"]

    return xtrain, ytrain, xtest, ytest


class EXPERIMENT:
    def __init__(self, svr_config: dict, nn_config: dict, experiment1: dict, pce_config: dict):

        logger = setup_logger(level=logging.DEBUG)

        self.SESM_model=SSESM(**experiment1, logger=logger)
        self.SVR_model = SVR(**svr_config)
        self.nn_model = NN(**nn_config)
        self.PF=PF(**pce_config)




    def train_all(
            self,
            train_data, 
            test_data):
        # ENTRENAMIENTO
        xtrain, ytrain, xtest, ytest = prepare_dataset(train_data, test_data)

        svr_time = self.SVR_model.train(xtrain, ytrain)
        nn_time = self.nn_model.train_for_experiment(xtrain, ytrain, xtest, ytest)
        pf_time = self.PF.train(xtrain, ytrain)
        sesm_time = self.SESM_model.partial_fit(xtrain, ytrain)

        times = {
            "svr_time": svr_time,
            "nn_time": nn_time,
            "pf_time": pf_time,
            "sesm_time": sesm_time
        }
        return times
        


    def test_all(self, train_data, test_data, plot_flag=False):

        _, _, xtest, ytest = prepare_dataset(train_data, test_data)
        
 
        svr_pred = self.SVR_model.test(xtest)
        nn_pred = self.nn_model.test(xtest)
        SESM_pred, _, SESM_mse = self.SESM_model.performance_stats(xtest, ytest)
        pf_pred=self.PF.test(xtest)

        metrics = {
            "SVR_MSE": mean_squared_error(ytest, svr_pred),
            "SVR_MAE": mean_absolute_error(ytest, svr_pred),
            "NN_MSE": mean_squared_error(ytest, nn_pred),
            "NN_MAE": mean_absolute_error(ytest, nn_pred),
            "SESM_MSE": SESM_mse,
            "SESM_MAE":mean_absolute_error(ytest, SESM_pred),
            "PF_MSE": mean_squared_error(ytest, pf_pred),
            "PF_MAE": mean_absolute_error(ytest, pf_pred)
        }
        
        return metrics
    
    def save_metrics(self, metrics, times, n_samples, function):
        """ 
        Esta funcion recibe 2 diccionarios y un vector

        metrics (dict): Contiene las metricas obtenidas con una funcion en especifico,
        la clave seria la dimension, en cada dimension se encuentran todos los errores registrados
        
        times (dict): la clave seria la dimension, en cada dimension se registran los tiempos
        de entrenamiento de cada modelo con distintos tamaños de dataset

        n_samples (list): contiene los tamaños de los datasets en 1 dimension, se usa para calcular 
        el tamaño de dataset de otras dimensiones
        """
        joblib.dump(metrics, "./plots/all_metrics"+str(function.__name__)+".joblib")
        joblib.dump(times, "./plots/all_times"+str(function.__name__)+".joblib")
        joblib.dump(n_samples, "./plots/n_samples.joblib")






