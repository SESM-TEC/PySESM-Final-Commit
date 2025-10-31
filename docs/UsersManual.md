Of course! Based on the provided source code and examples, here is a comprehensive User's Manual for the `pysesm` library in Markdown format.

---

# PySESM User's Manual

## 1. Introduction

Welcome to the User's Manual for **PySESM**, a Python library for building Sparse-Encoded Surrogate Models. This framework is designed for complex function approximation and surrogate modeling tasks, leveraging sparse encoding, dictionary learning, and a unique block-based spatial partitioning approach.

### Core Concepts

PySESM is built on three main pillars:

1.  **Space Partitioning:** The input data space is divided into smaller, manageable regions called "blocks." This allows the model to focus on learning local features of a function, making it highly scalable and effective for complex, non-stationary functions.
2.  **Dictionary Learning:** The model learns a global dictionary of basis functions (e.g., Gaussian functions). These functions, or "dictionary words," serve as the fundamental building blocks for approximating the target function. The dictionary is shared across all blocks.
3.  **Sparse Coding:** For each block, the model finds a sparse vector `h` that represents the optimal linear combination of dictionary words to approximate the function within that block's local region. The goal is to use as few dictionary words as possible, hence "sparse."

The core idea is to approximate a target function `y` as `y ≈ D @ h`, where `D` is the dictionary and `h` is the sparse code.

## 2. Library Architecture

The power of `pysesm` lies in its modular and configuration-driven design. You can easily swap out components for partitioning, sparse coding, and dictionary learning to tailor the model to your specific problem.

 <!-- Placeholder for a potential architecture diagram -->

### The Configuration-Driven Design

Every aspect of a `pysesm` model is controlled by a hierarchy of configuration objects (Python dataclasses). This makes experiments reproducible, readable, and easy to modify.

The main configuration object, `SSESMConfig` or `BSESMConfig`, contains sub-configurations for each major component:

```python
# A typical configuration structure
ssesm_config = SSESMConfig(
    n_features=2,
    model_epochs=500,
    # 1. Configuration for the space partitioner
    partition_config=UniformPartitionConfig(...),
    # 2. Configuration for the dictionary
    dict_config=GaussianDictConfig(...),
    # 3. Configuration for the sparse coding algorithm
    sparse_coding_config=ISTAConfig(...),
    ...
)
```

### Key Components

*   **Models (`pysesm.models`):** These are the main entry points for the user.
    *   `SESM`: The abstract base class for all models.
    *   `SSESM` (Sequential SESM): A training strategy where blocks are processed sequentially. The global dictionary is updated after each block is trained. This is robust and flexible.
    *   `BSESM` (Batched SESM): A training strategy where all active blocks are trained on in a single batch step. This can be more computationally efficient.

*   **Partition Managers (`pysesm.blocks`):** This component is responsible for dividing the input space.
    *   `UniformPartitionManager`: Divides the space into a uniform grid. You specify the number of blocks per dimension (`T`).
    *   `AdaptativePartitionManager`: Uses a KD-Tree to dynamically partition the space based on data density, creating blocks of varying sizes where needed.

*   **Dictionary Layers (`pysesm.dictionaries`):** This component manages the dictionary of basis functions (`D`).
    *   `DictBaseLayer`: The abstract base class.
    *   `GaussianDictLayer`: The primary implementation, which uses a dictionary of Gaussian functions. It learns the means (`mu`) and covariance-related parameters (`rho`) of each Gaussian.

*   **Sparse Coding Layers (`pysesm.sparse_coding`):** This component is responsible for finding the sparse activation vector (`h`) for each block.
    *   `SparseCodingBaseLayer`: The abstract base class.
    *   `ISTALayer`: Implements the Iterative Shrinkage-Thresholding Algorithm. A classic, fundamental choice.
    *   `FISTALayer`: Implements the Fast Iterative Shrinkage-Thresholding Algorithm, which often converges faster than ISTA by using a momentum term.
    *   `ADMMLayer`: Implements the Alternating Direction Method of Multipliers, a powerful and often more robust solver.

*   **Factories (`pysesm.factories`):** The library uses a factory pattern to instantiate components based on the provided configuration objects. This is handled internally but is a key part of the flexible design.

## 3. Getting Started: A Basic Example

Let's walk through a complete example of approximating a 2D function composed of three Gaussian distributions using a single block. This is based on `one_block_example.py`.

### Step 1: Setup Logger

It's always good practice to set up a logger to see the model's progress.

```python
import logging
from pysesm.utils.loggers import setup_logger

logger = setup_logger(level=logging.INFO)```

### Step 2: Define Model Configurations

This is the most important step. We define the configuration for each component of our model.

```python
import torch
from pysesm.models.SSESM import SSESMConfig
from pysesm.blocks import UniformPartitionConfig
from pysesm.dictionaries import GaussianDictConfig, GaussianDictLayer
from pysesm.sparse_coding import ISTAConfig, StepSizeMethod

# --- Overall Model Parameters ---
n_features = 2  # Our data is 2-dimensional (x, y)
n_functions = 100 # We want our dictionary to have 100 Gaussian functions

# 1. Partition Configuration: A single block covering the space [-2, 2] on each axis.
partition_config = UniformPartitionConfig(
    T=1, # T=1 creates a single block
    initial_bounds=torch.tensor([[-2, -2], [2, 2]], dtype=torch.float32),
)

# 2. Dictionary Configuration: A dictionary of Gaussian functions.
dict_config = GaussianDictConfig(
    epochs=100,
    alpha=0.01, # Learning rate for dictionary updates
    criterion=torch.nn.MSELoss(),
    optimizer_factory=lambda params, lr: torch.optim.AdamW(params, lr=lr),
    # Use electrostatic regularization to encourage dictionary words to spread out
    regularization_func=GaussianDictLayer.electrostatic_regularization,
    regularization_gamma=0.001,
)

# 3. Sparse Coding Configuration: Use the ISTA algorithm.
sparse_coding_config = ISTAConfig(
    epochs=150,
    alpha=0.15, # Step size for ISTA
    lambd=0.005, # Sparsity penalty (higher means more sparse `h` vectors)
    step_size_method=StepSizeMethod.FROBENIUS,
    n_functions=n_functions,
    criterion=torch.nn.MSELoss(),
)

# 4. Main SSESM Configuration: Combine all the pieces.
ssesm_config = SSESMConfig(
    n_features=n_features,
    model_epochs=1500, # Total training epochs for the model
    partition_config=partition_config,
    dict_config=dict_config,
    sparse_coding_config=sparse_coding_config,
    log_interval=50, # Log progress every 50 epochs
)
```

### Step 3: Generate Data

PySESM provides utility functions to generate sample datasets. Here, we create a function composed of three non-diagonal Gaussians.

```python
from pysesm.utils_dataset.generate_dataset import generate_gaussian_dataset
from pysesm.utils_dataset.gaussian_covariance_density import generate_nondiag_covariance_matrices

# Generate some interesting, non-diagonal covariance matrices
sigma1, sigma2, sigma3 = generate_nondiag_covariance_matrices()

# Create training and test datasets
(trainDataset, X_train, y_train,
 testDataset, X_test, y_test,
 gt_mu, gt_sigma) = generate_gaussian_dataset(
    n_samples=500,
    variances=[sigma1, sigma2, sigma3]
)
```

### Step 4: Instantiate and Train the Model

With the configuration and data ready, we can instantiate the `SSESM` model and start the training by calling `partial_fit`.

```python
from pysesm.models.SSESM import SSESM

# Define the experiment parameters
experiment = {
    "config": ssesm_config,
    "seed": 45
}

# Instantiate the model
model = SSESM(**experiment, logger=logger)

# Train the model
logging.info("Training model...")
model.partial_fit(X_train, y_train)
```

### Step 5: Evaluate the Model

After training, you can evaluate the model's performance on the test set.

```python
y_predicted, time, mse_value = model.performance_stats(X_test, y_test)

logging.info(
    f"Model: {model.__class__.__name__}, MSE Value = {mse_value:.6f}, time = {time:.2f} min"
)
```

## 4. Advanced Usage

### Multi-Block Partitioning

To handle more complex functions, you can easily partition the space into a grid. The only change required is in the `UniformPartitionConfig`.

```python
# From multi_block_example.py
# Create a 2x2 grid of blocks (4 total)
partition_config = UniformPartitionConfig(
    T=2, # Or T=torch.tensor([2, 2]) for a 2x2 grid
    initial_bounds=torch.tensor([[-2, -2], [2, 2]], dtype=torch.float32),
    overlap_ratio=0.25 # Add 25% overlap between blocks for smoother transitions
)
```
The rest of the training pipeline remains the same. The `SSESM` model will automatically iterate through the four blocks during training.

### Higher-Dimensional Problems

PySESM is not limited to 2D. It can approximate N-dimensional functions.

```python
# From n_dimensions_example.py

# --- Key Parameters ---
n_features = 4  # Number of input dimensions
n_functions = 50
n_samples = 2000

# --- Define N-dimensional Bounds ---
domain_limits = (-2.0, 2.0)
initial_bounds_list = [[domain_limits[0]] * n_features, [domain_limits[1]] * n_features]
initial_bounds_tensor = torch.tensor(initial_bounds_list, dtype=torch.float32)

# --- Create a 2x2x2x2 Grid (16 blocks) ---
blocks_per_dim = 2
t_list = [blocks_per_dim] * n_features
t_tensor = torch.tensor(t_list)

partition_config = UniformPartitionConfig(
    T=t_tensor,
    initial_bounds=initial_bounds_tensor,
)
```
The model will then operate on input tensors of shape `(n_samples, 4)`.

### Training Visualization

You can monitor the training process in real-time by attaching a `VisualizerHook`. This hook generates frames of the dictionary's state at each logging interval and can compile them into a video.

```python
# From visualization.py and any example script
import matplotlib.pyplot as plt
from visualization import VisualizerHook # Assumes visualization.py is in the same directory

# ... (model and data setup)

# 1. Create a matplotlib figure and axis
fig_hook, ax_hook = plt.subplots(figsize=(10, 8))
plt.ion() # Turn on interactive mode

# 2. Instantiate the model
model = SSESM(**experiment, logger=logger)

# 3. Create and attach the visualization hook
visual_hook = VisualizerHook(model, ax_hook, X_train, gt_mu, gt_sigma, plot_limits=((-5, 5), (-5, 5)))
model.sesm_hook = visual_hook

# 4. Train the model (the hook will be called automatically)
try:
    model.partial_fit(X_train, y_train)
finally:
    # 5. Create a video from the saved frames after training
    visual_hook.create_video(video_name="training_evolution.mp4")

plt.show(block=True)
```

### Hyperparameter Tuning with Weights & Biases

The configuration-driven design of `pysesm` makes it perfect for hyperparameter optimization. The `wandb_sweep_example.py` shows how to integrate with `wandb` for a Bayesian sweep.

**Key Steps:**

1.  **Define a `sweep_config` dictionary:** Specify the search method (`bayes`), the metric to optimize (`mse_value`), and the parameters to search over with their distributions.

    ```python
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'mse_value', 'goal': 'minimize'},
        'parameters': {
            'n_functions': {'distribution': 'q_uniform', 'min': 10, 'max': 80},
            'dict_alpha': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-2},
            'sc_lambd': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-2},
            # ... other parameters
        }
    }
    ```

2.  **Create a `train` function:** This function will be called by the `wandb` agent for each run. Inside this function:
    *   Initialize `wandb` (`wandb.init()`).
    *   Access the run's hyperparameters from `wandb.config`.
    *   Build the `pysesm` configuration objects dynamically using these hyperparameters.
    *   Instantiate and train the model.
    *   Evaluate the model and log the result (`wandb.log({"mse_value": mse_value})`).

3.  **Start the sweep agent:**

    ```python
    import wandb

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="pysesm-hyperparameter-optimization")

    # Start the agent to run the `train` function multiple times
    wandb.agent(sweep_id, function=train, count=200)
    ```

## 5. API Reference (Core Classes)

*   `pysesm.models.SSESM(config, logger, **kwargs)`
    *   `.partial_fit(X, y)`: Trains the model on the provided data. This is the main training method.
    *   `.predict(X)`: Generates predictions for new input data `X`.
    *   `.performance_stats(X, y)`: Evaluates the model, returning predictions, training time, and MSE.

*   `pysesm.models.SSESMConfig`: Main configuration dataclass.
*   `pysesm.blocks.UniformPartitionConfig`: Configuration for grid-based partitioning.
*   `pysesm.blocks.AdaptativePartitionConfig`: Configuration for data-driven partitioning.
*   `pysesm.dictionaries.GaussianDictConfig`: Configuration for the Gaussian dictionary.
*   `pysesm.sparse_coding.ISTAConfig`: Configuration for the ISTA solver.
*   `pysesm.sparse_coding.FISTAConfig`: Configuration for the FISTA solver.
*   `pysesm.sparse_coding.ADMMConfig`: Configuration for the ADMM solver.
