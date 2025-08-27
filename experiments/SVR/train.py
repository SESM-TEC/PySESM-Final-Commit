from .model import SVR



def train_svr(train_data, test_data, kernel='rbf', C=100, gamma=.1, epsilon=.1):

    model = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)

    xtrain, ytrain, _, _ = model.prepare_dataset(train_data, test_data)
    
    model.fit(xtrain, ytrain)

    # GUARDAR MODELO
    path = r"./SVR/svr_model.pth"
    model.save(path)
    




