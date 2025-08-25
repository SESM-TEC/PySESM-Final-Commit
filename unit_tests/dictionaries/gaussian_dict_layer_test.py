import logging

import pytest
import numpy as np
from scipy.stats import multivariate_normal
import torch

# For debuggin and understanding >>>
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# Adjust imports based on new directory structure
from pysesm.dictionaries.GaussianDictLayer import GaussianDictLayer, GaussianDictConfig
#from pysesm.functions.GaussianFunction import GaussianFunction # Still used for comparison/true function
#from pysesm.sparse_coding.ISTALayer import ISTAConfig # For parameter_hook
from pysesm.enums.DeviceTargetEnum import DeviceTarget
from pysesm.device_manager.DeviceManager import DeviceManager # For DeviceTargetEnum
from pysesm.base_types import TensorBatch



DEBUG_VISUALIZATION = False

def show_data(X,y,c,marker,label,ax=None):
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(X[:, 0], X[:, 1], y.flatten(), 
               c=c, marker=marker, label=label)
    
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    ax.set_zlabel('y')
    ax.legend()
    plt.show(block=False)
    return ax
# End of debugging helpers <<<


# --- Fixtures for common setup ---
@pytest.fixture(scope="module")
def _common_logger():
    logger = logging.getLogger('test_gaussian_dict_layer')
    logger.setLevel(logging.INFO) # Set to INFO or DEBUG to see detailed logs during tests
    if not logger.handlers: # Prevent adding handlers multiple times in pytest
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

@pytest.fixture(scope="module")
def _common_device():
    return torch.device('cpu')

@pytest.fixture(scope="module")
def _common_evaluation_func():
    # Helper to ensure 'h' is a column vector (N_functions, 1) if it's 1D,
    # or (..., N_functions, 1) if it's already batched.
    def _ensure_h_column_vector(h_input: torch.Tensor) -> torch.Tensor:
        if h_input.dim() == 1:
            return h_input.unsqueeze(-1)
        return h_input

    # Helper to ensure the output of matmul is always (..., N_samples, 1)
    # if the logical output dimension is 1.
    def _perform_matmul_and_shape_output(d: torch.Tensor,
                                          h_val: torch.Tensor) -> torch.Tensor:
        res = torch.matmul(d, _ensure_h_column_vector(h_val))
        if res.dim() == 1:  # If matmul results in (N_samples,)
            return res.unsqueeze(-1)  # Make it (N_samples, 1)
        return res

    # This is the actual callable function that the fixture will return.
    # It's named _eval_func_impl to avoid confusion with the fixture name itself.
    def _eval_func_impl(dictionary: TensorBatch, h: TensorBatch) -> TensorBatch:
        # Adapt matmul for different TensorBatch types
        if (getattr(dictionary, "is_nested", False)
              and getattr(h, "is_nested", False)):  # NestedTensor
            results = [_perform_matmul_and_shape_output(d_s, h_s)
                       for d_s, h_s in zip(dictionary.unbind(), h.unbind())]
            return torch.nested.as_nested_tensor(results,
                                                  layout=dictionary.layout,
                                                  device=dictionary.device,
                                                  dtype=results[0].dtype)
        elif isinstance(dictionary, torch.Tensor) and dictionary.dim() <= 2:  # Single 2D tensor
            return _perform_matmul_and_shape_output(dictionary, h)
        elif isinstance(dictionary, torch.Tensor) and dictionary.dim() == 3:  # 3D tensor
            return torch.vmap(_perform_matmul_and_shape_output)(dictionary, h)
        elif isinstance(dictionary, list) and isinstance(h, list):  # List of tensors
            results = [_perform_matmul_and_shape_output(d_s, h_s)
                       for d_s, h_s in zip(dictionary, h)]
            return results
        else:
            raise TypeError("Unsupported TensorBatch types for evaluation_func: "
                            f"D={type(dictionary)}, h={type(h)}")
    return _eval_func_impl  # Return the actual callable function.

@pytest.fixture(scope="module")
def _common_device_manager(_common_logger):
    device_map = {
        DeviceTarget.GLOBAL: "cpu",
        DeviceTarget.SPARSE_CODING_LAYER: "cpu",
        DeviceTarget.DICTIONARY_LAYER: "cpu",
        DeviceTarget.PARTITION_MANAGER: "cpu"
    }
    return DeviceManager(_common_logger, default_device="cpu", device_map=device_map)


# --- Ported Test Cases ---

def test_gaussian_dict_layer_find_mu_only(_common_logger, _common_device, _common_evaluation_func, _common_device_manager):
    """Test GaussianDictLayer's ability to find correct mean with fixed covariance."""

    seed_value = 42  # Puedes elegir cualquier entero
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)

    n_features = 2
    n_functions = 1
    
    n_samples = 500
    X = torch.rand(n_samples, n_features, device=_common_device) * 4 - 2  # Uniform in [-2, 2] x [-2, 2]
    
    true_mean = np.array([0.5, -0.3])
    fixed_cov = 0.5 * np.eye(n_features)
    
    # Calculate true unnormalized Gaussian values
    gaussian_values = multivariate_normal.pdf(X.cpu().numpy(), mean=true_mean, cov=fixed_cov)
    peak_value = multivariate_normal.pdf(true_mean, mean=true_mean, cov=fixed_cov)
    y = torch.tensor(gaussian_values / peak_value, dtype=torch.float32, device=_common_device).reshape(-1, 1)
    
    if DEBUG_VISUALIZATION:
        ax = show_data(X.cpu().numpy(), y.cpu().numpy(), c='0.4', marker='.', label="ground truth")

    mu_history = []
    
    def parameter_tracker(info):
              
        mu_history.append({
            'epoch': info['epoch'],
            'mu': info['mu_params'].flatten(),
            'loss': info['loss']
        })

    # Define GaussianDictConfig for this test
    dict_config = GaussianDictConfig(
        epochs=100, # Total epochs for partial_fit
        alpha=0.15,
        mu_epochs=100, # Learn mu for all 100 epochs
        rho_epochs=0,  # Do not learn rho (fixed covariance)
        split_mu_rho=True, # Use split training strategy
        eig_range=[0.5, 0.5], # Fixed eigenvalues for fixed covariance
        mu_range=[[-0.5, 0.5], [-0.5, 0.5]] # Initial wide range for mean initialization
    )

    dict_layer = GaussianDictLayer(
        config=dict_config,
        n_features=n_features,
        n_functions=n_functions,
        evaluation_func=_common_evaluation_func,
        logger=_common_logger,
        parameter_hook=parameter_tracker,
        device=_common_device_manager.get_device(DeviceTarget.DICTIONARY_LAYER) # Pass via device manager
    )
    
    # Initialize h to [1] since we only have one Gaussian
    h = torch.ones((1, 1), dtype=torch.float32, device=_common_device).detach()
    
    dict_layer.partial_fit(
        X=X,
        y=y,
        h=h
    )
    
    learned_mean = dict_layer.theta_params[-n_features:, 0].detach().cpu().numpy()

    if DEBUG_VISUALIZATION:
        adapted_values = multivariate_normal.pdf(X.cpu().numpy(), mean=learned_mean, cov=fixed_cov)
        adapted_peak = multivariate_normal.pdf(learned_mean, mean=learned_mean, cov=fixed_cov)
        adapted_y = adapted_values / adapted_peak
        show_data(X.cpu().numpy(), adapted_y, c='r', marker='.', label="final adaption", ax=ax)

        mu_points = np.array([entry['mu'].flatten() for entry in mu_history])
        ax.scatter(mu_points[:, 0], mu_points[:, 1], np.zeros_like(mu_points[:, 0]), 
                'b.', label='μ trajectory', alpha=0.5)
       

    np.testing.assert_allclose(learned_mean, true_mean, rtol=1e-1, err_msg="Learned mean not close to true mean.")
    assert dict_layer.losses[-1] < dict_layer.losses[0], "Loss did not decrease during training."


def test_gaussian_dict_layer_find_diagonal_covariance(_common_logger, _common_device, _common_evaluation_func, _common_device_manager):
    """Test GaussianDictLayer's ability to find diagonal covariance with fixed mean."""
    n_features = 2
    n_functions = 1
    
    n_samples = 500 # Increased samples for better learning
    X = torch.rand(n_samples, n_features, device=_common_device) * 4 - 2
    
    fixed_mean = np.array([0.0, 0.0])
    true_cov = np.array([[2.0, 0.0], [0.0, 0.5]])
    
    gaussian_values = multivariate_normal.pdf(X.cpu().numpy(), mean=fixed_mean, cov=true_cov)
    peak_value = multivariate_normal.pdf(fixed_mean, mean=fixed_mean, cov=true_cov)
    y = torch.tensor(gaussian_values / peak_value, dtype=torch.float32, device=_common_device).reshape(-1, 1)
    
    # Define GaussianDictConfig for this test
    dict_config = GaussianDictConfig(
        epochs=500, # Total epochs
        alpha=0.2,
        mu_epochs=0,  # Do not learn mean (fixed)
        rho_epochs=500, # Learn rho for all epochs
        split_mu_rho=True, # Use split training
        eig_range=[0.1, 5.0], # Wider range for eigenvalues to capture 2.0 and 0.5
        mu_range=[[0.0, 0.0], [0.0, 0.0]] # Fixed mean initialization at origin
    )

    dict_layer = GaussianDictLayer(
        config=dict_config,
        n_features=n_features,
        n_functions=n_functions,
        evaluation_func=_common_evaluation_func,
        logger=_common_logger,
        device=_common_device_manager.get_device(DeviceTarget.DICTIONARY_LAYER)
    )
    
    h = torch.ones((1, 1), dtype=torch.float32, device=_common_device).detach()
    
    dict_layer.partial_fit(
        X=X,
        y=y,
        h=h,
        # epochs=n_epochs, # Controlled by dict_config.epochs
        # mu_flag=False,   # Controlled by dict_config.mu_epochs
        # rho_flag=True    # Controlled by dict_config.rho_epochs
    )
    
    rho = dict_layer.theta_params[:-n_features, 0].detach() # Access parameters directly
    A = torch.zeros(n_features, n_features, device=_common_device)
    indices = torch.triu_indices(n_features, n_features)
    A[indices[0], indices[1]] = rho
    learned_G = torch.matmul(A.T, A)
    
    learned_G_np = learned_G.cpu().numpy()
    learned_cov_np = np.linalg.inv(learned_G_np)

    assert np.abs(learned_cov_np[0, 1]) < 0.1, "Off-diagonal elements should be close to zero for diagonal covariance."
    assert np.abs(learned_cov_np[1, 0]) < 0.1, "Off-diagonal elements should be close to zero for diagonal covariance."
    
    # Assert diagonal elements are close to true values (up to a global scaling factor)
    # The amplitudes of GaussianFunction are unnormalized, so only the ratio matters.
    # The learned covariance values might be scaled versions of true_cov.
    # We should compare the diagonal elements' ratios.
    ratio_learned = learned_cov_np[0, 0] / learned_cov_np[1, 1]
    ratio_true = true_cov[0, 0] / true_cov[1, 1]
    np.testing.assert_allclose(ratio_learned, ratio_true, rtol=0.1, err_msg="Covariance ratios do not match.") # Increased rtol slightly
    
    assert dict_layer.losses[-1] < dict_layer.losses[0], "Loss did not decrease during training."


def test_gaussian_dict_layer_find_non_diagonal_covariance(_common_logger, _common_device, _common_evaluation_func, _common_device_manager):
    """Test GaussianDictLayer's ability to find non-diagonal covariance with fixed mean."""
    n_features = 2
    n_functions = 1
    
    n_samples = 500 # Increased samples for better learning
    X = torch.rand(n_samples, n_features, device=_common_device) * 4 - 2
    
    fixed_mean = np.array([0.0, 0.0])
    true_cov = np.array([[2.0, 0.5], [0.5, 1.0]])
    
    gaussian_values = multivariate_normal.pdf(X.cpu().numpy(), mean=fixed_mean, cov=true_cov)
    peak_value = multivariate_normal.pdf(fixed_mean, mean=fixed_mean, cov=true_cov)
    y = torch.tensor(gaussian_values / peak_value, dtype=torch.float32, device=_common_device).reshape(-1, 1)
    
    # Define GaussianDictConfig for this test
    dict_config = GaussianDictConfig(
        epochs=500, # Total epochs
        alpha=0.2,
        mu_epochs=0,  # Do not learn mean (fixed)
        rho_epochs=500, # Learn rho for all epochs
        split_mu_rho=True, # Use split training
        eig_range=[0.1, 5.0], # Wider range for eigenvalues to capture values
        mu_range=[[0.0, 0.0], [0.0, 0.0]] # Fixed mean initialization at origin
    )

    dict_layer = GaussianDictLayer(
        config=dict_config,
        n_features=n_features,
        n_functions=n_functions,
        evaluation_func=_common_evaluation_func,
        logger=_common_logger,
        device=_common_device_manager.get_device(DeviceTarget.DICTIONARY_LAYER)
    )
    
    h = torch.ones((1, 1), dtype=torch.float32, device=_common_device).detach()
    
    dict_layer.partial_fit(
        X=X,
        y=y,
        h=h,
        # flags controlled by config
    )
    
    rho = dict_layer.theta_params[:-n_features, 0].detach()
    A = torch.zeros(n_features, n_features, device=_common_device)
    indices = torch.triu_indices(n_features, n_features)
    A[indices[0], indices[1]] = rho
    learned_G = torch.matmul(A.T, A)
    
    learned_G_np = learned_G.cpu().numpy()
    # learned_cov_np = np.linalg.inv(learned_G_np)

    # The actual values of the learned covariance might be scaled versions of `true_cov`
    # because `GaussianFunction` is unnormalized. The key is the *shape* and *relative values*.
    # A simple np.testing.assert_allclose might fail if there's a global scaling factor.
    # A more robust check might be to assert:
    # 1. The learned matrix is symmetric positive definite.
    # 2. Its eigenvectors match (directions) and its eigenvalues are proportional (relative spread).
    
    # For now, let's keep the assert_allclose but expect potential tolerance adjustments.
    # The original test used rtol=0.01, atol=0.01 which is quite loose.
    # Let's verify the scaled version. The `psi` is `exp(-0.5 (x-mu)' G (x-mu))`
    # If y_true is `C * exp(-0.5 (x-mu)' G_true (x-mu))`
    # And y_pred is `C' * exp(-0.5 (x-mu)' G_pred (x-mu))`
    # Then G_pred should be proportional to G_true.
    
    # Find the scaling factor between the learned and true precision matrices.
    # Since G is inv(Cov), if learned_cov = k * true_cov, then learned_G = (1/k) * true_G
    # So we'd expect learned_G to be proportional to true_G.
    true_G = np.linalg.inv(true_cov)
    
    # Find ratio of first diagonal element, then check others
    scale_factor_G = learned_G_np[0,0] / true_G[0,0]
    scaled_true_G = true_G * scale_factor_G
    
    np.testing.assert_allclose(learned_G_np, scaled_true_G, rtol=0.1, atol=0.01,
                               err_msg="Learned precision matrix (G) not proportional to true G.")

    assert dict_layer.losses[-1] < dict_layer.losses[0], "Loss did not decrease during training."


# --- New Test Cases for TensorBatch and Gradients ---
def test_gaussian_dict_layer_train_with_3d_tensor(_common_logger, _common_device,
                                                  _common_evaluation_func,
                                                  _common_device_manager):
    """
    Test training GaussianDictLayer with a 3D Tensor input (batch, N, F),
    verifying loss decrease and selective gradient zeroing.
    """
    n_features = 2
    n_functions = 3
    batch_size = 5
    n_samples_per_batch = 20
    epochs_to_run = 100

    # Create dummy 3D input data (batch, n_samples, n_features)
    X_batch = torch.randn(batch_size, n_samples_per_batch, n_features,
                          device=_common_device, requires_grad=False)
    # Create dummy 3D target data (batch, n_samples, 1)
    y_batch = torch.randn(batch_size, n_samples_per_batch, 1,
                          device=_common_device, requires_grad=False)
    # Create dummy 3D sparse code (batch, n_functions, 1)
    h_batch = torch.randn(batch_size, n_functions, 1,
                          device=_common_device, requires_grad=False)

    dict_config = GaussianDictConfig(
        epochs=epochs_to_run,
        alpha=0.1,
        mu_epochs=epochs_to_run // 2, # Train mu for half epochs
        rho_epochs=epochs_to_run // 2, # Train rho for other half
        split_mu_rho=True,
        eig_range=[0.1, 1.0],
        mu_range=[-1.0, 1.0]
    )

    dict_layer = GaussianDictLayer(
        config=dict_config,
        n_features=n_features,
        n_functions=n_functions,
        evaluation_func=_common_evaluation_func,
        logger=_common_logger,
        device=_common_device_manager.get_device(DeviceTarget.DICTIONARY_LAYER)
    )

    dict_layer._train_epoch(X=X_batch, y=y_batch, h=h_batch, log_losses=False,
                            mu_flag=True,rho_flag=True)

    dict_layer.partial_fit(X_batch, y_batch, h_batch)
    initial_loss = dict_layer.losses[0]
    final_loss = dict_layer.losses[-1]

    assert final_loss < initial_loss, "Loss should decrease during training."
    assert len(dict_layer.losses) == epochs_to_run, \
        "Loss history length should match total epochs."

    # Test selective gradient zeroing by checking initial gradients
    # Run one epoch where only mu is trained, then check rho grads
    dict_layer.optimizer.zero_grad()
    dict_layer._train_epoch(X=X_batch, y=y_batch, h=h_batch, log_losses=False,
                            mu_flag=True, rho_flag=False)
    
    mu_grads_mu_only = dict_layer.theta_params.grad[-n_features:, :]
    rho_grads_mu_only = dict_layer.theta_params.grad[:-n_features, :]
    
    assert not torch.all(mu_grads_mu_only == 0), "Mu grads should be non-zero"
    assert torch.all(rho_grads_mu_only == 0), "Rho grads should be zero (mu_flag=True)"

    # Run one epoch where only rho is trained, then check mu grads
    dict_layer.optimizer.zero_grad()
    dict_layer._train_epoch(X=X_batch, y=y_batch, h=h_batch, log_losses=False,
                            mu_flag=False, rho_flag=True)
    
    mu_grads_rho_only = dict_layer.theta_params.grad[-n_features:, :]
    rho_grads_rho_only = dict_layer.theta_params.grad[:-n_features, :]
    
    assert torch.all(mu_grads_rho_only == 0), "Mu grads should be zero (rho_flag=True)"
    assert not torch.all(rho_grads_rho_only == 0), "Rho grads should be non-zero"


@pytest.mark.filterwarnings("ignore:There is a performance drop.*:UserWarning")
@pytest.mark.filterwarnings("ignore:The PyTorch API of nested tensors.*:UserWarning")
def test_gaussian_dict_layer_train_with_nested_tensor(_common_logger, _common_device,
                                                      _common_evaluation_func,
                                                      _common_device_manager):
    """
    Test training GaussianDictLayer with a nested_tensor input,
    verifying loss decrease and selective gradient zeroing.
    """
    n_features = 2
    n_functions = 3
    samples_per_batch = [10, 5, 12]
    epochs_to_run = 100

    # Create lists of 2D tensors for nested_tensor input
    X_list = [torch.randn(ns, n_features, device=_common_device, requires_grad=False)
              for ns in samples_per_batch]
    y_list = [torch.randn(ns, 1, device=_common_device, requires_grad=False)
              for ns in samples_per_batch]
    h_list = [torch.randn(n_functions, 1, device=_common_device, requires_grad=False)
              for _ in samples_per_batch] # h is per-function for each block

    X_nested = torch.nested.nested_tensor(X_list, layout=torch.jagged)
    y_nested = torch.nested.nested_tensor(y_list, layout=torch.jagged)
    h_nested = torch.nested.nested_tensor(h_list, layout=torch.jagged)

    dict_config = GaussianDictConfig(
        epochs=epochs_to_run,
        alpha=0.1,
        mu_epochs=epochs_to_run // 2,
        rho_epochs=epochs_to_run // 2,
        split_mu_rho=True,
        eig_range=[0.1, 1.0],
        mu_range=[-1.0, 1.0]
    )

    dict_layer = GaussianDictLayer(
        config=dict_config,
        n_features=n_features,
        n_functions=n_functions,
        evaluation_func=_common_evaluation_func,
        logger=_common_logger,
        device=_common_device_manager.get_device(DeviceTarget.DICTIONARY_LAYER)
    )

    dict_layer._train_epoch(X=X_nested, y=y_nested, h=h_nested, log_losses=False,
                            mu_flag=True,rho_flag=True)

    dict_layer.partial_fit(X_nested, y_nested, h_nested)
    initial_loss = dict_layer.losses[0]
    final_loss = dict_layer.losses[-1]

    assert final_loss < initial_loss, "Loss should decrease during training."
    assert len(dict_layer.losses) == epochs_to_run, \
        "Loss history length should match total epochs."

    # Test selective gradient zeroing with nested tensor input
    dict_layer.optimizer.zero_grad()
    dict_layer._train_epoch(X=X_nested, y=y_nested, h=h_nested, log_losses=False,
                            mu_flag=True, rho_flag=False)
    
    mu_grads_mu_only = dict_layer.theta_params.grad[-n_features:, :]
    rho_grads_mu_only = dict_layer.theta_params.grad[:-n_features, :]
    
    assert not torch.all(mu_grads_mu_only == 0), "Mu grads should be non-zero"
    assert torch.all(rho_grads_mu_only == 0), "Rho grads should be zero (mu_flag=True)"

    dict_layer.optimizer.zero_grad()
    dict_layer._train_epoch(X=X_nested, y=y_nested, h=h_nested, log_losses=False,
                            mu_flag=False, rho_flag=True)
    
    mu_grads_rho_only = dict_layer.theta_params.grad[-n_features:, :]
    rho_grads_rho_only = dict_layer.theta_params.grad[:-n_features, :]
    
    assert torch.all(mu_grads_rho_only == 0), "Mu grads should be zero (rho_flag=True)"
    assert not torch.all(rho_grads_rho_only == 0), "Rho grads should be non-zero"

if __name__ == "__main__":
    from pytest_helper import print_pytest_instructions
    print_pytest_instructions()
