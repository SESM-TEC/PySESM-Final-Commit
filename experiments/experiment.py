from SVR.model import SVR
from NN.model import NN
from PF.model import PF

from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import logging
import joblib

from pysesm.models.SSESM import SSESM
from pysesm.utils.loggers import setup_logger





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

        self.metrics = {}

        # Para guardar las configuraciones y luego serializarlas
        self.sparse_coding_config = sesm_config.sparse_coding_config
        self.dict_config = sesm_config.dict_config
        self.ssesm_config = sesm_config
        self.svr_config = svr_config
        self.nn_config = nn_config
        self.pf_config = pf_config


        

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
        self.SESM_model.partial_fit(self.xtrain_orig, self.ytrain_orig)
        self.metrics['TIME_SESM'] = self.SESM_model.training_time
        self.metrics['TIME_SVR'] = self.SVR_model.train(self.xtrain, self.ytrain)
        self.metrics['TIME_NN'] = self.nn_model.train_nn(self.xtrain, self.ytrain, self.xtest, self.ytest)
        self.metrics['TIME_PF'] = self.PF.train(self.xtrain, self.ytrain)
        
        return self.metrics



    def test_all(self):
         # TESTING
        svr_pred = self.SVR_model.test(self.xtest)
        nn_pred = self.nn_model.test(self.xtest)
        SESM_pred, _, SESM_mse = self.SESM_model.performance_stats(self.xtest_orig, self.ytest_orig)
        pf_pred = self.PF.test(self.xtest)

        self.metrics.update({
            "MSE_SESM": SESM_mse,
            "MSE_SVR": mean_squared_error(self.ytest, svr_pred),
            "MSE_NN":  mean_squared_error(self.ytest, nn_pred),
            "MSE_PF":  mean_squared_error(self.ytest, pf_pred),

            "MAE_SESM": mean_absolute_error(self.ytest_orig, SESM_pred),
            "MAE_SVR": mean_absolute_error(self.ytest, svr_pred),
            "MAE_NN":  mean_absolute_error(self.ytest, nn_pred),
            "MAE_PF":  mean_absolute_error(self.ytest, pf_pred)
        })

        print("\n Metrics:")
        for key, value in self.metrics.items():
            print(f"{key}: {value}")

    def save_configs(self):
        # Crear una copia de los diccionarios de configuración y eliminar los elementos problemáticos
        sparse_coding_config_to_save = self.sparse_coding_config.__dict__.copy()
        del sparse_coding_config_to_save["criterion"]

        dict_config_to_save = self.dict_config.__dict__.copy()
        del dict_config_to_save["criterion"]
        del dict_config_to_save["optimizer_factory"]

        # Crear una versión serializable de ssesm_config
        ssesm_config_to_save = self.ssesm_config.__dict__.copy()
        ssesm_config_to_save["sparse_coding_config"] = sparse_coding_config_to_save
        ssesm_config_to_save["dict_config"] = dict_config_to_save

        # GUARDAR CONFIGURACIONES DE LOS MODELOS
        models_config_data = {
            "SVR (Support Vector Regressor)": self.svr_config,
            "PF (Polynomial Features)": self.pf_config,
            "NN (Neural Network)": self.nn_config,
            "SESM (Sparse Encoding Surrogate Model)": ssesm_config_to_save
        }
        # ...
        joblib.dump(models_config_data, "./plots/config/config_models.joblib")
        
        





