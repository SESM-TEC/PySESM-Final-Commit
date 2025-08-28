from .model import SVR



def train_svr(train_data, test_data, svr_config: dict):
                        
    model = SVR(kernel = svr_config["kernel"], 
                C = svr_config["C"], 
                gamma = svr_config["gamma"], 
                epsilon = svr_config["epsilon"])

    xtrain, ytrain, _, _ = model.prepare_dataset(train_data, test_data)
    
    model.fit(xtrain, ytrain)

    # GUARDAR MODELO
    path = "svr_model.pth"
    model.save(path)
    




