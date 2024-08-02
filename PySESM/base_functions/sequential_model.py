import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from PySESM.base_functions.gaussian_parameter_optimization import GaussianFunctions
from PySESM.models.SESM.SESM import SESM_Model
from PySESM.base_functions.sub_block_partition import predict_on_test_set, data_mapping

class ModeloSecuencial:
    def __init__(self, hyperparams, fngroup, iter, train_dataset, test_dataset, debug=True):
        """
        Initialize the ModeloSecuencial class.

        Args:
        - hyperparams (dict): Dictionary containing hyperparameters for model training and configuration.
        - fngroup (str): Identifier for the function group.
        - iter (int): The iteration number of the experiment.
        - train_dataset (dict): Dictionary containing the training dataset.
        - test_dataset (dict): Dictionary containing the test dataset.
        - debug (bool, optional): Flag to enable or disable debug mode. Default is True.
        """
        self.hyperparams = hyperparams
        self.fngroup = fngroup
        self.iter = iter
        self.debug = debug
        self.model = self.build_model()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    #Problema de las variables globales 
    def build_model(self, logger, SEED):
        """
        Build and configure the SESM (Sparse Evolutionary Structural Modeling) model.

        Returns:
        - SESM_Model: An instance of the SESM model.
        """
        n_samples = self.hyperparams["n_samples"]
        n_features = self.hyperparams["n_features"]
        l_functions = self.hyperparams["l_functions"]
        eig_range = self.hyperparams["eig_range"]
        mu_range = self.hyperparams["mu_range"]
        vector_range = self.hyperparams["vector_range"]
        model_epochs = self.hyperparams["m_epochs"]
        ista_epochs = self.hyperparams["h_epochs"]
        rho_epochs = self.hyperparams["rho_epochs"]
        mu_epochs = self.hyperparams["mu_epochs"]
        ista_alpha = self.hyperparams["ista_alpha"]
        ista_lambd = self.hyperparams["ista_lambd"]
        dictionary_alpha = self.hyperparams["dictionary_alpha"]
        weight_decay = self.hyperparams["weight_decay"]
        
        gaussian_function = GaussianFunctions(n_features=n_features, n_functions=l_functions, logger=logger, eig_range=eig_range, mu_range=mu_range, vector_range=vector_range, seed=SEED)
        model = SESM_Model(
            n_samples=n_samples,
            psi=gaussian_function,
            seed=SEED,
            model_epochs=model_epochs,
            ista_epochs=ista_epochs,
            ista_alpha=ista_alpha,
            ista_lambd=ista_lambd,
            mu_epochs=mu_epochs,
            rho_epochs=rho_epochs,
            dictionary_alpha=dictionary_alpha,
            weight_decay=weight_decay
        )
        return model

    def train_model(self, X, y):
        """
        Train the SESM model using the specified training parameters.

        Args:
        - X (torch.Tensor): Input features.
        - y (torch.Tensor): Target values.

        Returns:
        None
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

    def train_regular(self, X, y):
        """
        Train the SESM model using standard training.

        Args:
        - X (array-like): Input features.
        - y (array-like): Target values.

        Returns:
        None
        """
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        self.train_model(X, y)

    def train_sequential(self, list_sub_blocks, T):
        """
        Train the SESM model using a sequential approach with sub-blocks.

        Args:
        - list_sub_blocks (list): List of sub-blocks used for training.
        - T (int): The scaling factor for normalization.

        Returns:
        None
        """
        permutation_times = self.hyperparams["permutation_times"]
        for _ in range(permutation_times):
            selected_indexes = np.random.permutation(T**2)
            permuted_list_sub_blocks = [list_sub_blocks[i] for i in selected_indexes]
            for block in permuted_list_sub_blocks:
                y = torch.tensor(block.output_values, dtype=torch.float32)
                X = torch.tensor(np.array(block._X), dtype=torch.float32)
                self.model.ista_layer = block.ista_layer
                self.train_model(X, y)

    def predict_regular(self, X_test):
        """
        Predict values using the standard model.

        Args:
        - X_test (torch.Tensor): Input features for prediction.

        Returns:
        - torch.Tensor: Predicted values.
        """
        print("type(X_test): ", type(X_test))
        y = self.model.predict(X_test, self.model.ista_layer)
        return y

    def predict_sequential(self, X_test, list_sub_blocks):
        """
        Predict values using the sequential model with sub-blocks.

        Args:
        - X_test (torch.Tensor): Input features for prediction.
        - list_sub_blocks (list): List of sub-blocks used for prediction.

        Returns:
        - torch.Tensor: Predicted values.
        """
        T = self.hyperparams["T"]
        #t_test, x_n_test = data_mapping(X_test, T)
        y = predict_on_test_set(X_test, self.model, T, list_sub_blocks)
        return y

    def evaluate_regular(self, X_test, y_test):
        """
        Evaluate the standard model.

        Args:
        - X_test (torch.Tensor): Input features for evaluation.
        - y_test (torch.Tensor): True target values for evaluation.

        Returns:
        - tuple: A tuple containing:
          - Predicted values (torch.Tensor).
          - Training time (float).
          - Mean squared error (float).
        """
        y = self.predict_regular(X_test)
        time = self.model.time / 60
        mse = mean_squared_error(y.clone().detach(), y_test)
        return y, time, mse

    def evaluate_sequential(self, X_test, y_test, list_sub_blocks):
        """
        Evaluate the sequential model with sub-blocks.

        Args:
        - X_test (torch.Tensor): Input features for evaluation.
        - y_test (torch.Tensor): True target values for evaluation.
        - list_sub_blocks (list): List of sub-blocks used for evaluation.

        Returns:
        - tuple: A tuple containing:
          - Predicted values (torch.Tensor).
          - Training time (float).
          - Mean squared error (float).
        """
        y = self.predict_sequential(X_test, list_sub_blocks)
        time = self.model.time / 60
        mse = mean_squared_error(y.clone().detach(), y_test)
        return y, time, mse
