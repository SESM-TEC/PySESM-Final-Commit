'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica

Trivial example with one single block trying to represent three gaussians

Authors: The SESM Team 

License: 
'''


import logging
import torch

from pysesm.enums import SurrogateFunctionEnum
from pysesm.models import BSESM, SSESM, SESM
from pysesm.utils.loggers import setup_logger
from pysesm.utils.generate_dataset import generate_gaussian_dataset, generate_one_gaussian_dataset
from pysesm.utils.plot_and_save_stats import plot_surface
from pysesm.enums.DeviceTargetEnum import DeviceTarget
from pysesm.enums.HookTypeEnum import HookType
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class KLDivLossWrapper(torch.nn.Module):
    def __init__(self, reduction='mean', log_input=False):
        super(KLDivLossWrapper, self).__init__()
        self.kl_loss = torch.nn.KLDivLoss(reduction=reduction)
        
    def forward(self, inputs, targets):
        # Step 1: Ensure non-negativity (if your data can be negative)
        inputs = torch.nn.functional.relu(inputs) + 1e-8  # Small constant for numerical stability
        targets = torch.nn.functional.relu(targets) + 1e-8
        
        # Step 2: Normalize to make them proper distributions
        # Option 1: Normalize across all elements
        inputs_normalized = inputs / torch.sum(inputs)
        targets_normalized = targets / torch.sum(targets)
        
        # Option 2: If batched data, normalize each sample independently
        # inputs_normalized = inputs / torch.sum(inputs, dim=1, keepdim=True)
        # targets_normalized = targets / torch.sum(targets, dim=1, keepdim=True)
        
        # Step 3: Log-space transformation (since log_input=False by default)
        log_inputs = torch.log(inputs_normalized)
        
        # Step 4: Apply KL divergence
        loss = self.kl_loss(log_inputs, targets_normalized)
        
        return loss

class CrossEntropyLossWrapper(torch.nn.Module):
    """
    Custom Cross-Entropy loss implementation based on the Octave code.
    This implementation normalizes both inputs and targets to make them proper
    probability distributions before calculating cross-entropy.
    """
    def __init__(self, reduction='mean', epsilon=1e-10):
        super(CrossEntropyLossWrapper, self).__init__()
        self.reduction = reduction
        self.epsilon = epsilon
        
    def forward(self, inputs, targets):
        # Ensure non-negativity
        inputs = torch.nn.functional.relu(inputs) + self.epsilon
        targets = torch.nn.functional.relu(targets) + self.epsilon
        
        # Normalize to make them proper distributions
        inputs_normalized = inputs / torch.sum(inputs)
        targets_normalized = targets / torch.sum(targets)
        
        # Cross-entropy = -sum(P * log(Q))
        # where P is targets and Q is inputs
        cross_entropy = -torch.sum(targets_normalized * torch.log(inputs_normalized + self.epsilon))
        
        return cross_entropy


class JensenShannonLossWrapper(torch.nn.Module):
    """
    Custom Jensen-Shannon divergence implementation based on the Octave code.
    JS divergence is a symmetrized and smoothed version of the KL divergence.
    
    JS(P||Q) = 0.5 * (KL(P||M) + KL(Q||M)) where M = 0.5 * (P + Q)
    """
    def __init__(self, reduction='mean', epsilon=1e-10):
        super(JensenShannonLossWrapper, self).__init__()
        self.reduction = reduction
        self.epsilon = epsilon
        
    def forward(self, inputs, targets):
        # Ensure non-negativity
        inputs = torch.nn.functional.relu(inputs) + self.epsilon
        targets = torch.nn.functional.relu(targets) + self.epsilon
        
        # Normalize to make them proper distributions
        inputs_normalized = inputs / torch.sum(inputs)
        targets_normalized = targets / torch.sum(targets)
        
        # Compute the average distribution M
        M = 0.5 * (inputs_normalized + targets_normalized)
        
        # Compute KL(targets || M)
        ratio1 = (targets_normalized + self.epsilon) / (M + self.epsilon)
        kl1 = torch.sum(targets_normalized * torch.log(ratio1))
        
        # Compute KL(inputs || M)
        ratio2 = (inputs_normalized + self.epsilon) / (M + self.epsilon)
        kl2 = torch.sum(inputs_normalized * torch.log(ratio2))
        
        # JS = 0.5 * (KL(P||M) + KL(Q||M))
        js_divergence = 0.5 * (kl1 + kl2)
        
        return js_divergence

    
# SESM CONFIGURATION
experiment = {
    "hyp_set": 1,
    "n_samples": 500,
    "n_features": 2,
    "n_functions": 10,
    "eig_range": [0.05, 0.2],
    "mu_range": [-2.0, 2.0],
    "ista_alpha": 0.15,
    "ista_lambd": 0.005,
    "dictionary_alpha": 0.1,
    "dictionary_optimizer": lambda params, lr: torch.optim.SGD(params, lr=lr, momentum=0.1),
    ##"dictionary_criterion": torch.nn.MSELoss(),
    ##"dictionary_criterion": KLDivLossWrapper(),
    "dictionary_criterion": JensenShannonLossWrapper(), 
    "ista_optimizer": lambda params, lr: torch.optim.SGD(params, lr=lr, momentum=0.1),
    "ista_criterion": torch.nn.MSELoss(),
    "rho_epochs": 10,
    "mu_epochs": 10,
    "model_epochs": 10,
    "dict_epochs": 10,
    "ista_epochs": 10,
    "psi": SurrogateFunctionEnum.GAUSSIAN,
    "T": 1,
    "initial_bounds": torch.tensor([[-2, -2], [2, 2]], dtype=torch.float32),
    "permutation_times": 1,
    "seed": 45,
    "dfngroup": 1,
    "iter": 0,
    "debug": True,
    "device_map": {
        DeviceTarget.GLOBAL: "cpu",               # Dispositivo global por defecto
        DeviceTarget.ISTA_LAYER: "cpu",           # ISTA en GPU 0
        DeviceTarget.DICTIONARY_LAYER: "cpu",     # Dictionary en CPU
        DeviceTarget.PARTITION_MANAGER: "cpu"    # Partition Manager en CPU
    },
    "use_wandb": True,
    "active_hooks": [HookType.ISTALAYER],
    "project_name": "sesm-test"
}

def show_data(X,y,c,marker,label,ax=None):
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    

    # Plot training data
    ax.scatter(X[:, 0], X[:, 1], y, 
               c=c, marker=marker, label=label)
    
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    ax.set_zlabel('y')
    ax.legend()

    plt.show(block=False)
    return ax

# DATA GENERATION
trainDataset, X_train, y_train, testDataset, X_test, y_test = generate_gaussian_dataset(experiment)

# ax = show_data(X_train,y_train,'r','x','Training')
# show_data(X_test,y_test,'0.4','.','Test',ax)

# RESULTS FOLDER NAME CREATION
folder_name = f"results_one_block_{experiment['hyp_set']}"

# LOGGER INSTANCE
logger = setup_logger()

# INSTANTIATE THE MODELS
ssesm_model = SSESM(**experiment,logger=logger)
#bsesm_model = BSESM(**experiment,logger=logger)

try:
    # TRAIN AND TEST THE ALL MODELS
    for model in [ssesm_model]: # bsesm_model
        logging.info("Training model {}".format(model.__class__.__name__))
        model_folder = f"{folder_name}_{model.__class__.__name__}"
        model.partial_fit(X_train, y_train)
        y_predicted, time, mse_value = model.performance_stats(X_test, y_test)

        logging.info("Model: {}, MSE Value = {:.6f}, time ={:.6f}".format(model.__class__.__name__, mse_value, time))

        plot_surface(testDataset, X_train, y_train, y_predicted, model, experiment["hyp_set"])

    plt.show(block=True)
except KeyboardInterrupt:
    print("\nShutting down...")
    plt.close('all')
    exit(0)
