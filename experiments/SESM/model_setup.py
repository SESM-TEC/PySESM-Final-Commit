"""
This script configures a model of SESM required by the experiments.
"""
import logging
import torch

from LossWrappers import KLDivLossWrapper, JensenShannonLossWrapper, CrossEntropyLossWrapper

from pysesm.models.SSESM import SSESM, SSESMConfig
from pysesm.sparse_coding import ISTALayer, ISTAConfig, StepSizeMethod
from pysesm.dictionaries import GaussianDictLayer, GaussianDictConfig
from pysesm.blocks.UniformPartitionManager import UniformPartitionConfig
from pysesm.utils.loggers import setup_logger
from pysesm.utils.loggers import setup_logger
from pysesm.utils_dataset.generate_dataset import generate_gaussian_dataset
from pysesm.utils.plot_and_save_stats import plot_surface
from pysesm.utils.metric_loggers import *
from pysesm.enums.DeviceTargetEnum import DeviceTarget
from pysesm.device_manager.DeviceManager import DeviceManager

logger = setup_logger(level=logging.DEBUG)

# SESM CONFIGURATION
n_functions = 10
n_features = 2

device_map = {
    DeviceTarget.GLOBAL: "cpu",
    DeviceTarget.SPARSE_CODING_LAYER: "cpu",
    DeviceTarget.DICTIONARY_LAYER: "cpu",
    DeviceTarget.PARTITION_MANAGER: "cpu"
}

sparse_coding_config = ISTAConfig(
    epochs=50,
    alpha=0.10,
    lambd=0.00001,
    step_size_method=StepSizeMethod.FROBENIUS,  # POWER_ITERATION,
    power_iterations=10,
    n_functions=n_functions,
    criterion=torch.nn.MSELoss()
)

dict_config = GaussianDictConfig(
    epochs = 4,
    alpha = 0.01,
    # criterion = torch.nn.MSELoss(),
    # criterion = KLDivLossWrapper(),
    criterion = JensenShannonLossWrapper(),
    optimizer_factory = lambda params, lr: torch.optim.SGD(params, lr=lr, momentum=0.1),
    mu_epochs = 10,
    rho_epochs = 10,
    split_mu_rho = True,
    eig_range = [0.05, 0.2],
    mu_range = [-2.0, 2.0],
)

partition_config = UniformPartitionConfig(
    T=1,
    initial_bounds = torch.tensor([[-2, -2], [2, 2]], dtype=torch.float32),
    activity_threshold=0,
    overlap_ratio=0.25
)

ssesm_config = SSESMConfig(
    n_features = n_features,
    model_epochs = 7500,
    sparse_coding_config = sparse_coding_config,
    dict_config = dict_config,
    partition_config = partition_config,
    log_interval=100,
    permutation_times=1
)

experiment = {
    "config": ssesm_config,
    "hyp_set": 1,
    "n_samples": 500,
    "seed": 45,
    "iter": 0,
    "device_map": {
        DeviceTarget.GLOBAL: "cpu",               # Dispositivo global por defecto
        DeviceTarget.SPARSE_CODING_LAYER: "cpu",  # ISTA en GPU 0
        DeviceTarget.DICTIONARY_LAYER: "cpu",     # Dictionary en CPU
        DeviceTarget.PARTITION_MANAGER: "cpu"     # Partition Manager en CPU
    }
}

model = SSESM(**experiment,logger=logger)