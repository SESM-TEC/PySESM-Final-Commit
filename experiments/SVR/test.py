from .model import SVR

def test_svr(train_data: dict,
             test_data: dict):

    model = SVR()
    _, _, xtest, _ = model.prepare_dataset(train_data , test_data)

    path = "svr_model.pth"
    model.load(path)

    ypred = model.predict(xtest)

    return ypred
