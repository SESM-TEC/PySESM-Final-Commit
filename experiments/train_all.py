from SVR.train import train_svr
from NN.train import train_nn


def train_all(
        train_data, 
        test_data, 
        svr_config: dict, 
        nn_config: dict):
    # ENTRENAMIENTO
     
    
    train_svr(train_data, test_data, svr_config)
    train_nn(train_data, test_data, nn_config)