import torch
import numpy as np
import time
import matplotlib.pyplot as plt

from PySESM.models.DictLayer import DictLayer
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
        
        self.n_samples = n_samples
        self.n_features = n_features

        self.ista_layer = ISTALayer(n_functions)
        self.dictionary_layer = DictLayer(n_samples, n_features, n_functions, psi) 
        
        self.losses = []
        self.loss_stats = {
            "loss_mean" : [],
            "loss_std"  : [],
            "loss_max"  : [],
            "loss_min"  : [],
        }
        self.time = 0
        
        
    def fit(self, X, y, model_epochs, ista_epochs, ista_alpha, ista_lambd, dictionary_epochs, dictionary_alpha):
        for epoch in range(model_epochs):
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
            self.loss_analysis(dictionary_epochs)
            print(f'Epoch {epoch+1} Loss: {self.losses[-1]}\n')
         
                    
    def predict(self, X):
        with torch.no_grad():
           self.dictionary_layer.forward(X)
            
        dictionary = self.dictionary_layer.dictionary.double()
        h = self.ista_layer.h.double()
            
        return dictionary @ h


    def loss_analysis(self, dict_epochs):
        current_loss  = self.dictionary_layer.losses[-dict_epochs:]
        self.loss_stats["loss_mean"].append(np.mean(current_loss))
        self.loss_stats["loss_std"].append(np.std(current_loss))
        self.loss_stats["loss_max"].append(np.max(current_loss))
        self.loss_stats["loss_min"].append(np.min(current_loss))
        
        
    def plot_function(self):
        n_plots = 4
        plot_elevs = [30, 60, 90, 30]
        plot_azims = [30, 60, 90, 120]

        samples = torch.tensor([])

        for i in range(self.n_features):
            feature = torch.linspace(-2, 2, self.n_samples)
            samples = torch.cat([samples, feature.unsqueeze(1)], dim=1)
            
        X = torch.meshgrid(samples[:, 0], samples[:, 1])

        xy_grid = torch.stack([X[0].ravel(), X[1].ravel()], dim=1)

        y = self.predict(xy_grid).reshape(self.n_samples, self.n_samples)

        fig = plt.figure(figsize=(8, 8))

        for i in range(n_plots):
            ax = fig.add_subplot(2, 2, i+1, projection='3d')
            ax.plot_surface(X[0].numpy(), X[1].numpy(), y.detach().numpy(), cmap='plasma')
            # ax.scatter(X[:, 0], X[:, 1], y_pred, color='red', label='Predicted')
            ax.view_init(elev=plot_elevs[i], azim=plot_azims[i])
                
        plt.tight_layout()
        plt.show()
        
    
    def plot_loss(self):
        plt.plot(self.losses)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.show()