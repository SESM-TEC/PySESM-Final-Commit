from .model import NN  



def test_nn(train_data, test_data):
    # MODEL CLASS
    model = NN()
    # PREPARE DATASET
    _, _, xtest, _ = model.prepare_dataset(train_data, test_data)
    # LOAD MODEL
    model_path = r"C:\Users\Lenovo Yoga\Desktop\SEMESTRE_II_2025\ASISTENCIA\PySESM\experiments\NN\nn_model.pth"
    model.load(model_path)
    # PREDICTIONS
    ypred = model.predict(xtest)
    return ypred


