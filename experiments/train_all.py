from SVR.train import train_svr
from NN.train import train_nn


def train_all(train_data, test_data):
    # ENTRENAMIENTO
    train_svr(train_data, test_data, kernel='rbf', C=1000, gamma=.1, epsilon=.1)
    train_nn(train_data, test_data, epochs=500, lr=0.01, hidden_dim=16)