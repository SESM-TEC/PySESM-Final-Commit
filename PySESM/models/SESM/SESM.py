import torch
from tqdm import tqdm
from PySESM.models.DictLayer import DictLayer
import time
from PySESM.models.ISTALayer import ISTALayer

class SESM_Model(torch.nn.Module):
    """
    A custom PyTorch module for implementing a surrogate model that uses the SESM architecture.

    This layer is designed for use in surrogate modeling and function approximation tasks.

    Args:
        n_functions (int): The number of functions or basis functions.

    Attributes:
        n_samples (int): The number of samples taken from the original function.
        n_features (int): The number of input features or dimensions.
        n_functions (int): The number of functions or basis functions.
        psi (callable): The function used for generating the model's dictionary.
        losses (list): A list of the losses computed during training.

    Methods:
        fit(X, y, model_epochs, ista_epochs, ista_alpha, ista_lambd, dictionary_epochs, dictionary_alpha):
            Trains the model by learning a sparse vector and a dictionary that represent the original function.
            Args:
                X (Tensor): Input data of shape (n_samples, n_features).
                y (Tensor): Target data of shape (n_samples,).
                model_epochs (int): Number of training epochs for the model.
                ista_epochs (int): Number of training epochs for the ISTA layer.
                ista_alpha (float): Learning rate for the ISTA layer.
                ista_lambd (float): Regularization parameter for the ISTA layer.
                dictionary_epochs (int): Number of training epochs for the dictionary layer.
                dictionary_alpha (float): Learning rate for the dictionary layer.
                
        predict(x):
            Predicts the value of a function using the learned sparse vector and dictionary.
            Args:
                X (Tensor): Input data of shape (n_samples, n_features).
            Returns:
                Tensor: The predicted values for each sample of the objective function.
    """
    def __init__(self, n_samples, n_features, n_functions, psi):
        super().__init__()

        self.ista_layer = ISTALayer(n_functions)
        self.dictionary_layer = DictLayer(n_samples, n_features, n_functions, psi) 
        
        self.losses = []
        self.time = 0
        
        
    def fit(self, X, y, model_epochs, ista_epochs, ista_alpha, ista_lambd, dictionary_epochs, dictionary_alpha):
        for epoch in tqdm(range(model_epochs), desc='Training model'):
            epoch_start_time = time.time()
            
            self.ista_layer.fit(
                y=y,
                epochs=ista_epochs,
                dictionary=self.dictionary_layer.dictionary,
                alpha=ista_alpha,
                lambd=ista_lambd
            )
            
            self.dictionary_layer.fit(
                X=X,
                y=y,
                epochs=dictionary_epochs,
                h=self.ista_layer.h,
                alpha=dictionary_alpha
            )
            
            epoch_end_time = time.time()
        
            self.time = self.time + (epoch_end_time - epoch_start_time)
        
            self.losses.append(self.dictionary_layer.losses[-1])
            print(f'Epoch {epoch+1} Loss: {self.losses[-1]}\n')
         
                    
    def predict(self, X):
        with torch.no_grad():
           self.dictionary_layer.forward(X)
            
        dictionary = self.dictionary_layer.dictionary.double()
        h = self.ista_layer.h.double()
            
        return dictionary @ h
