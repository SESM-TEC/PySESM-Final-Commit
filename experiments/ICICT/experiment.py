"""Módulo de orquestación de experimentos ICICT.

Este módulo define la clase `EXPERIMENT` que coordina la creación de
datasets, el entrenamiento y evaluación de los modelos (SESM, SVR, NN, PF),
y la serialización de configuraciones y métricas.

Uso típico:
    experiment = EXPERIMENT(svr_cfg, nn_cfg, ssesm_cfg, pf_cfg)
    experiment.setup_dataset(train_data, test_data)
    experiment.train_all()
    experiment.test_all()
"""

import logging

# third-party
import joblib
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

# pysesm package
from pysesm.models.SSESM import SSESM
from pysesm.utils.loggers import setup_logger

# local experiment models
from SVR.model import SVR
from NN.model import NN
from PF.model import PF




class EXPERIMENT:
    """Orquesta un experimento completo con los modelos disponibles.

    La clase gestiona la inicialización de modelos, normalización de datos,
    entrenamiento conjunto y evaluación, además de guardar configuraciones
    para reproducibilidad.

    Atributos principales:
        sesm_model: instancia de `pysesm.models.SSESM`.
        svr_model: instancia del wrapper `SVR`.
        nn_model: instancia del wrapper `NN`.
        PF: instancia del wrapper `PF`.
        metrics: diccionario donde se almacenan métricas y tiempos.
    """

    def __init__(
        self,
        svr_config: dict,
        nn_config: dict,
        sesm_config: object,
        pf_config: dict,
    ):
        """Inicializa los modelos del experimento.

        Args:
            svr_config: configuración para el wrapper SVR.
            nn_config: configuración para la red neuronal.
            sesm_config: objeto de configuración para SSESM.
            pf_config: configuración para Polynomial Features.
        """

        logger = setup_logger(level=logging.DEBUG)

        self.sesm_model = SSESM(config=sesm_config, logger=logger)
        self.svr_model = SVR(**svr_config)
        self.nn_model = NN(**nn_config)
        self.pf_model = PF(**pf_config)

        self.metrics = {}

        # Para guardar las configuraciones y luego serializarlas
        self.sparse_coding_config = sesm_config.sparse_coding_config
        self.dict_config = sesm_config.dict_config
        self.ssesm_config = sesm_config
        self.svr_config = svr_config
        self.nn_config = nn_config
        self.pf_config = pf_config

        # Inicializaciones anticipadas para evitar atributos creados fuera de __init__
        # (evita W0201 de pylint). Serán sobrescritos por `setup_dataset`.
        self.xtrain_orig = None
        self.ytrain_orig = None
        self.xtest_orig = None
        self.ytest_orig = None

        self.meanx = None
        self.stdx = None
        self.meany = None
        self.stdy = None

        self.xtrain = None
        self.ytrain = None
        self.xtest = None
        self.ytest = None




    def setup_dataset(self, train_data: dict = None, test_data: dict = None):
        """Prepara y normaliza los datasets de entrenamiento y test.

        Guarda copias sin normalizar en `*_orig` y crea versiones normalizadas
        `xtrain`, `ytrain`, `xtest`, `ytest` usando media y desviación estándar
        calculadas sobre el conjunto de entrenamiento.

        Args:
            train_data: diccionario que debe contener la llave "X" y "Z".
            test_data: diccionario que debe contener la llave "X" y "Z".
        """

        xtrain = train_data["X"]
        ytrain = train_data["Z"]

        xtest = test_data["X"]
        ytest = test_data["Z"]

        # Store original, unnormalized data for pysesm
        self.xtrain_orig = xtrain.clone()
        self.ytrain_orig = ytrain.clone()
        self.xtest_orig = xtest.clone()
        self.ytest_orig = ytest.clone()

        self.meanx = torch.mean(xtrain, dim=0)
        self.stdx = torch.std(xtrain, dim=0)
        self.meany = torch.mean(ytrain)
        self.stdy = torch.std(ytrain)

        self.xtrain = (xtrain - self.meanx) / self.stdx
        self.ytrain = (ytrain - self.meany) / self.stdy
        self.xtest = (xtest - self.meanx) / self.stdx
        self.ytest = (ytest - self.meany) / self.stdy




    def train_all(self):
        """Entrena todos los modelos y registra los tiempos.

        Ejecuta el ajuste de SSESM, SVR, NN y PF y almacena los tiempos en
        `self.metrics` con claves `TIME_*`.

        Returns:
            dict: el diccionario `self.metrics` actualizado con los tiempos.
        """

        # ENTRENAMIENTO
        self.sesm_model.partial_fit(self.xtrain_orig, self.ytrain_orig)
        self.metrics["TIME_SESM"] = self.sesm_model.training_time
        self.metrics["TIME_SVR"] = self.svr_model.train(self.xtrain, self.ytrain)
        self.metrics["TIME_NN"] = self.nn_model.train_nn(
            self.xtrain, self.ytrain, self.xtest, self.ytest
        )
        self.metrics["TIME_PF"] = self.pf_model.train(self.xtrain, self.ytrain)

        return self.metrics



    def test_all(self):
        """Evalúa todos los modelos y actualiza las métricas.

        - Obtiene estadísticas de SSESM en el espacio original.
        - Ejecuta inferencia de PF, NN y SVR en el espacio normalizado y
          desnormaliza las predicciones antes de calcular las métricas.
        - Actualiza `self.metrics` con MSE y MAE para cada modelo.
        """

        # TESTING
        # SESM evalua el error en el espacio de entrada original
        sesm_pred, _, sesm_mse = self.sesm_model.performance_stats(
            self.xtest_orig, self.ytest_orig
        )

        # El resto de modelos en el espacio normalizado
        pf_pred = self.pf_model.test(self.xtest)
        nn_pred = self.nn_model.test(self.xtest)
        svr_pred = self.svr_model.test(self.xtest)

        # Se desnormalizan las predicciones de los modelos
        pf_pred = pf_pred * self.stdy + self.meany
        nn_pred = nn_pred * self.stdy + self.meany
        svr_pred = svr_pred * self.stdy + self.meany

        self.metrics.update(
            {
                "MSE_SESM": sesm_mse,
                "MSE_SVR": mean_squared_error(self.ytest_orig, svr_pred),
                "MSE_NN": mean_squared_error(self.ytest_orig, nn_pred),
                "MSE_PF": mean_squared_error(self.ytest_orig, pf_pred),

                "MAE_SESM": mean_absolute_error(self.ytest_orig, sesm_pred),
                "MAE_SVR": mean_absolute_error(self.ytest_orig, svr_pred),
                "MAE_NN": mean_absolute_error(self.ytest_orig, nn_pred),
                "MAE_PF": mean_absolute_error(self.ytest_orig, pf_pred),
            }
        )

        print("\n Metrics:")
        for key, value in self.metrics.items():
            print(f"{key}: {value}")



    def save_configs(self):
        """Serializa y guarda las configuraciones de los modelos.

        Elimina elementos no serializables (por ejemplo funciones o criterios)
        antes de guardar con `joblib.dump` en `./plots/config/config_models.joblib`.
        """

        # Se copian los diccionarios de configuración
        # Se eliminan los elementos problemáticos
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
