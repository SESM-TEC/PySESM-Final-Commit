# PySESM

**PySESM** (Python Sparse-Encoded Surrogate Model) is a
high-performance library for surrogate modeling and function
approximation, developed at **Tecnológico de Costa Rica**.

It excels at representing complex, high-dimensional functions using a
**"divide and conquer"** strategy. The core architecture partitions
the input space into manageable blocks (using Uniform or
Adaptive/KD-Tree strategies), where each block is handled by a local
sparse representation sharing a global, learnable dictionary of basis
functions (e.g., Gaussians).

This approach allows the model to learn a rich, shared representation
of the function's features while using sparse, localized codes to
efficiently capture specific behaviors in different regions.

## Installation

PySESM requires **Python 3.12+**.

### 1. Prepare Environment

We recommend using conda or micromamba to manage the environment:
```bash
conda create -n "sesm" python=3.12
conda activate sesm
```

### 2. Install PyTorch

Install PyTorch according to your hardware.

* **For CPU only:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

* **For GPU (CUDA):**
```bash
pip install torch torchvision
```

*(Check [pytorch.org](https://pytorch.org) for the command specific to
your CUDA version).*

### 3. Install PySESM

You can install the library in three modes depending on your needs:

**A. Core Installation (Library only)**

Installs only the necessary dependencies (numpy, torch, scipy) to run the model.

```bash
pip install -e .
```

**B. Visualization Support (Recommended for Examples)**

Includes libraries for plotting, metrics logging, and video generation
(matplotlib, plotly, wandb, imageio).

```bash
pip install -e ".[viz]"
```

**C. Full Development (Tests & Dev tools)**

Includes everything above plus testing frameworks (pytest).

```bash
pip install -e ".[dev]"
```

## Quick Start

PySESM uses a modular configuration system. You need to configure
three main components: the **Partition Manager** (how space is
divided), the **Dictionary** (the basis functions), and the **Sparse
Coding** algorithm (how to solve for coefficients).

Here is a minimal example of how to set up and train a Sequential SESM
(**SSESM**) model:

```python
import torch
import logging
from pysesm.models.SSESM import SSESM, SSESMConfig
from pysesm.blocks import UniformPartitionConfig
from pysesm.dictionaries import GaussianDictConfig
from pysesm.sparse_coding import ISTAConfig, StepSizeMethod
from pysesm.utils.loggers import setup_logger

# 1. Setup Logger
logger = setup_logger(level=logging.INFO)

# 2. Define Configurations
n_features = 2

# A. Partition: Divide space into a 2x2 grid
partition_config = UniformPartitionConfig(
    T=2, 
    initial_bounds=torch.tensor([[-2, -2], [2, 2]], dtype=torch.float32)
)

# B. Dictionary: Gaussian Basis Functions
dict_config = GaussianDictConfig(
    epochs=50,
    alpha=0.01,
    mu_epochs=10,
    rho_epochs=10
)

# C. Sparse Coding: ISTA Algorithm
sparse_coding_config = ISTAConfig(
    epochs=100,
    alpha=0.15,
    lambd=0.005,
    n_functions=50,  # Dictionary size
    step_size_method=StepSizeMethod.FROBENIUS
)

# 3. Main Model Config
ssesm_config = SSESMConfig(
    n_features=n_features,
    model_epochs=200,
    partition_config=partition_config,
    dict_config=dict_config,
    sparse_coding_config=sparse_coding_config,
    log_interval=20
)

# 4. Instantiate and Train
# Assuming X_train (inputs) and y_train (targets) are torch.Tensors
model = SSESM(config=ssesm_config, seed=42, logger=logger)

# model.partial_fit(X_train, y_train)
# preds = model.predict(X_test)
```

Check the `examples/` folder for complete scripts, including data
generation and visualization hooks.

## Directory Structure

* **pysesm/**: The core source code of the library.
  * **blocks/**: Logic for space partitioning (Uniform, KD-Tree).
  * **dictionaries/**: Learnable dictionary layers (e.g., Gaussian).
  * **sparse_coding/**: Algorithms to find sparse representations (ISTA, FISTA, ADMM).
  * **models/**: Main model architectures (SESM, SSESM, BSESM).
  * **utils_dataset/**: Tools for generating synthetic datasets.
* **examples/**: Scripts demonstrating basic usage, visualization hooks, and hyperparameter sweeps (WandB).
* **unit_tests/**: Comprehensive test suite for all components.
* **pyproject.toml**: Project configuration and dependencies.

## License

This project is licensed under the **BSD 3-Clause License**. See the
LICENSE file for details.

Copyright (c) 2023-2025, Tecnológico de Costa Rica. All rights reserved.
