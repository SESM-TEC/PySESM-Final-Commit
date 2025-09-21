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
    def __init__(
            self, 
            svr_config: dict, 
            nn_config: dict, 
            sesm_config: object, 
            pf_config: dict,
            ):

        logger = setup_logger(level=logging.DEBUG)

        self.SESM_model=SSESM(config=sesm_config, logger=logger)
        self.SVR_model = SVR(**svr_config)
        self.nn_model = NN(**nn_config)
        self.PF=PF(**pf_config)


        

    def setup_dataset(self, train_data: dict = None, test_data: dict = None):
        xtrain = train_data["X"]  
        ytrain = train_data["Z"]

        xtest = test_data["X"]
        ytest = test_data["Z"]

        # Store original, unnormalized data for pysesm
        self.xtrain_orig = xtrain.clone()
        self.ytrain_orig = ytrain.clone()
        self.xtest_orig = xtest.clone()
        self.ytest_orig = ytest.clone()       

        meanx = torch.mean(xtrain, dim=0)
        stdx = torch.std(xtrain, dim=0)
        meany = torch.mean(ytrain)
        stdy = torch.std(ytrain)

        self.xtrain = (xtrain - meanx) / stdx
        self.ytrain = (ytrain - meany) / stdy
        self.xtest = (xtest - meanx) / stdx
        self.ytest = (ytest - meany) / stdy




    def train_all(self):
        # ENTRENAMIENTO
        times = {}
        times['svr_time'] = self.SVR_model.train(self.xtrain, self.ytrain)
        times['nn_time'] = self.nn_model.train_nn(self.xtrain, self.ytrain, self.xtest, self.ytest)
        times['pf_time'] = self.PF.train(self.xtrain, self.ytrain)
        
        self.SESM_model.partial_fit(self.xtrain_orig, self.ytrain_orig)
        times['sesm_time'] = self.SESM_model.training_time

        print("\n Training times (s):")
        for key, value in times.items():
            print(f"{key}: {value} ")

        return times
        


    def test_all(self):
         # TESTING
        svr_pred = self.SVR_model.test(self.xtest)
        nn_pred = self.nn_model.test(self.xtest)
        SESM_pred, _, SESM_mse = self.SESM_model.performance_stats(self.xtest_orig, self.ytest_orig)
        pf_pred = self.PF.test(self.xtest)

        metrics = {
            "SESM_MSE": SESM_mse,
            "SVR_MSE": mean_squared_error(self.ytest, svr_pred),
            "NN_MSE":  mean_squared_error(self.ytest, nn_pred),
            "PF_MSE":  mean_squared_error(self.ytest, pf_pred),

            "SESM_MAE":mean_absolute_error(self.ytest_orig, SESM_pred),
            "SVR_MAE": mean_absolute_error(self.ytest, svr_pred),
            "NN_MAE":  mean_absolute_error(self.ytest, nn_pred),
            "PF_MAE":  mean_absolute_error(self.ytest, pf_pred)
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






