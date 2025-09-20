from SVR.model import SVR
from NN.model import NN
from PF.model import PF

from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import torch
from pysesm.models.SSESM import SSESM
from pysesm.utils.loggers import setup_logger
import logging





class EXPERIMENT:
    def __init__(self, svr_config: dict, nn_config: dict, sesm_config: dict, pce_config: dict):

        logger = setup_logger(level=logging.DEBUG)

        self.SESM_model=SSESM(**sesm_config, logger=logger)
        self.SVR_model = SVR(**svr_config)
        self.nn_model = NN(**nn_config)
        self.PF=PF(**pce_config)

        

    def normalize_dataset(self, train_data: dict = None, test_data: dict = None):
        xtrain = train_data["X"]  
        ytrain = train_data["Z"]

        xtest = test_data["X"]
        ytest = test_data["Z"]

        meanx = torch.mean(xtrain, dim=0)
        stdx = torch.std(xtrain, dim=0)
        meany = torch.mean(ytrain)
        stdy = torch.std(ytrain)

        xtrain = (xtrain - meanx) / stdx
        ytrain = (ytrain - meany) / stdy
        xtest = (xtest - meanx) / stdx
        ytest = (ytest - meany) / stdy

        return xtrain, ytrain, xtest, ytest



    def train_all(
            self,
            train_data, 
            test_data):
        # ENTRENAMIENTO
        xtrain, ytrain, xtest, ytest = self.normalize_dataset(train_data, test_data)

        svr_time = self.SVR_model.train(xtrain, ytrain)
        nn_time = self.nn_model.train_nn(xtrain, ytrain, xtest, ytest)
        pf_time = self.PF.train(xtrain, ytrain)
        sesm_time = self.SESM_model.partial_fit(xtrain, ytrain)

        times = {
            "svr_time": svr_time,
            "nn_time": nn_time,
            "pf_time": pf_time,
            "sesm_time": sesm_time
        }

        print("\n Training times (s):")
        for key, value in times.items():
            print(f"{key}: {value} ")

        return times
        


    def test_all(self, train_data, test_data):

        _, _, xtest, ytest = self.normalize_dataset(train_data, test_data)
        
 
        svr_pred = self.SVR_model.test(xtest)
        nn_pred = self.nn_model.test(xtest)
        SESM_pred, _, SESM_mse = self.SESM_model.performance_stats(xtest, ytest)
        pf_pred = self.PF.test(xtest)

        metrics = {
            "SESM_MSE": SESM_mse,
            "SVR_MSE": mean_squared_error(ytest, svr_pred),
            "NN_MSE": mean_squared_error(ytest, nn_pred),
            "PF_MSE": mean_squared_error(ytest, pf_pred),

            "SESM_MAE":mean_absolute_error(ytest, SESM_pred),
            "SVR_MAE": mean_absolute_error(ytest, svr_pred),
            "NN_MAE": mean_absolute_error(ytest, nn_pred),
            "PF_MAE": mean_absolute_error(ytest, pf_pred)
        }

        print("\n Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")
      
        return metrics
    


    # DE MOMENTO NO SE USA
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






