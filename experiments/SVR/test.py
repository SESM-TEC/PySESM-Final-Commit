from .model import SVR

def test_svr(train_data: dict,
             test_data: dict, 
             kernel: str = 'rbf', 
             C: float = 100, 
             gamma: float = .1, 
             epsilon: float = .1):

    model = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)
    _, _, xtest, _ = model.prepare_dataset(train_data , test_data)

    path = r"./SVR/svr_model.pth"
    model.load(path)

    ypred = model.predict(xtest)

    return ypred
