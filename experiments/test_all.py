from SVR.test import test_svr
from NN.test import test_nn

import matplotlib.pyplot as plt


def test_all(train_data, test_data):
    svr_pred = test_svr(train_data, test_data, kernel='rbf', C=1000, gamma=.1, epsilon=.1)
    nn_pred = test_nn(train_data, test_data)
    comparative_plot(svr_pred, nn_pred, test_data)

def comparative_plot(svr_pred, nn_pred, test_data):
    # VISUALIZACION
    fig = plt.figure(figsize=(10, 3))
    #fig.suptitle("Regression", fontsize=16)

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(test_data["X"], test_data["Y"], test_data["Z"], c=test_data["Z"], s=10)
    ax1.set_title("Ground truth")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(test_data["X"], test_data["Y"], svr_pred, c=svr_pred, s=10)
    ax2.set_title("SVR predictions")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    ax2 = fig.add_subplot(133, projection='3d')
    ax2.scatter(test_data["X"], test_data["Y"], nn_pred, c=nn_pred, s=10)
    ax2.set_title("NN predictions")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    plt.show()