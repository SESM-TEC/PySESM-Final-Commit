from pysesm.functions.GaussianFunction import GaussianFunction
import torch
import logging
import numpy as np
import pytest
from scipy.stats import multivariate_normal
import time
from typing import Any, Union, List

# --- Fixtures ---

@pytest.fixture(scope="module")
def module_logger() -> logging.Logger:
    """Provides a shared logger for all tests in this module."""
    logger = logging.getLogger("test_gaussian_function_batch")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

@pytest.fixture(scope="module")
def gaussian_function_fixture(module_logger: logging.Logger) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Provides an initialized instance of GaussianFunction."""
    return GaussianFunction(
        n_features=2,
        n_functions=5,
        logger=module_logger
    )


def test_initialize_parameter_properties():
    """
    Verifica que initialize() genera parámetros 'mu' y 'rho' cuyas
    propiedades estadísticas respetan los rangos 'mu_range' y 'eig_range'.
    """
    # 1. Arrange: Configurar parámetros no triviales
    n_features = 3
    n_functions = 10
    logger = logging.getLogger('test_initialize')
    
    # Rangos diferentes por dimensión para una prueba robusta
    mu_range_list = [[-1.0, 1.0], [10.0, 20.0], [100.0, 100.0]]
    eig_range_list = [[0.1, 0.5], [2.0, 3.0], [10.0, 10.0]]

    gaussian = GaussianFunction(
        n_features=n_features,
        n_functions=n_functions,
        logger=logger,
        mu_range=mu_range_list,
        eig_range=eig_range_list
    )

    # 2. Act: Llamar al método que queremos probar
    theta = gaussian.initialize()

    # 3. Assert: Deconstruir theta y verificar las propiedades

    # -- Verificación de Mu --
    mu_params = theta[-n_features:, :] # Shape: (n_features, n_functions)
    assert mu_params.shape == (n_features, n_functions)

    mu_range_tensor = torch.tensor(mu_range_list)
    for i in range(n_features):
        min_val, max_val = mu_range_tensor[i, 0], mu_range_tensor[i, 1]
        feature_mus = mu_params[i, :]
        assert torch.all(feature_mus >= min_val), f"Mu en la dimensión {i} está por debajo del mínimo"
        assert torch.all(feature_mus <= max_val), f"Mu en la dimensión {i} está por encima del máximo"

    # -- Verificación de los Eigenvalores de la Covarianza (a partir de Rho) --
    num_rho_params = n_features * (n_features + 1) // 2
    rho_params = theta[:num_rho_params, :] # Shape: (num_rho_params, n_functions)
    
    eig_range_tensor = torch.tensor(eig_range_list)
    min_eig_vals = eig_range_tensor[:, 0]
    max_eig_vals = eig_range_tensor[:, 1]

    # Iterar sobre cada una de las n_functions (cada palabra del diccionario)
    for j in range(n_functions):
        rho_j = rho_params[:, j]
        
        # Reconstruir la matriz triangular superior A
        A_j = torch.zeros(n_features, n_features)
        indices = torch.triu_indices(n_features, n_features)
        # La forma en que se reconstruye A depende de cómo se almacenó en __call__
        # Basado en el código actual de __call__, parece ser A[:, indices[1], indices[0]] = rho.T
        # lo que significa que el llenado es por columnas de la matriz triangular.
        # Vamos a reconstruirlo de la misma forma que lo haría to_triu_matrix
        A_j[indices[0], indices[1]] = rho_j

        # Reconstruir la matriz de precisión G y la de covarianza Sigma
        # Nota: el `initialize` usa A de Cholesky(G) tal que G = A.T @ A.
        # Sin embargo, el __call__ usa una transposición diferente. Se debe ser consistente.
        # Asumamos la lógica de `initialize`: G = A.T @ A
        G_j = torch.matmul(A_j.T, A_j)
        Sigma_j = torch.linalg.inv(G_j)

        # Calcular los eigenvalores de la matriz de covarianza
        eigvals_j = torch.linalg.eigvalsh(Sigma_j)
        
        # Verificar que los eigenvalores están en el rango correcto.
        # Los eigenvalores no tienen un orden garantizado, así que los ordenamos.
        sorted_eigvals = torch.sort(eigvals_j).values
        
        assert torch.all( (sorted_eigvals - min_eig_vals) >= -0.01), \
            f"Eigenvalores para la función {j} están por debajo del rango. Obtenido: {sorted_eigvals}, Esperado >: {min_eig_vals}"
        assert torch.all( (max_eig_vals - sorted_eigvals) >= -0.01), \
            f"Eigenvalores para la función {j} están por encima del rango. Obtenido: {sorted_eigvals}, Esperado <: {max_eig_vals}"


def test_gaussian_fixed_parameters():
    """Test that GaussianFunction produces correct values with fixed parameters"""
    n_features = 2
    n_functions = 1
    logger = logging.getLogger('test')
    
    # Create uniform points just like in dict_layer_test
    n_samples = 100
    X = torch.rand(n_samples, 2) * 4 - 2  # Uniform in [-2, 2] x [-2, 2]
    
    # Define the exact parameters we're using in dict_layer_test
    true_mean = np.array([0.5, -0.3])
    fixed_cov = 0.5 * np.eye(2)  # Identity covariance
    
    # Calculate expected values using scipy
    expected_values = multivariate_normal.pdf(X.numpy(), mean=true_mean, cov=fixed_cov)
    peak_value = multivariate_normal.pdf(true_mean, mean=true_mean, cov=fixed_cov)
    expected_y = torch.tensor(expected_values / peak_value, dtype=torch.float32).reshape(-1, 1)
    
    # Create GaussianFunction with fixed parameters
    gaussian = GaussianFunction(
        n_features=n_features,
        n_functions=n_functions,
        logger=logger,
        eig_range=[0.5, 0.5],  # Fixed eigenvalue
        mu_range=[[0.5, 0.5], [-0.3, -0.3]]  # Fixed mean
    )
    
    # Initialize and get parameters
    theta = gaussian.initialize()
    
    # Evaluate function
    values = gaussian(X, theta)
    
    # Values should match expected_y
    assert torch.allclose(values, expected_y, rtol=1e-4, atol=1e-4), \
        f"Maximum difference: {(values - expected_y).abs().max()}"


def test_single_gaussian_identity():
    """Test a single Gaussian with zero mean and identity covariance"""
    # Setup
    n_features = 2
    n_functions = 1
    logger = logging.getLogger('test')

    rndseed = 63
    torch.manual_seed(rndseed)  # PyTorch seed
    np.random.seed(rndseed)     # NumPy seed
    
    gaussian = GaussianFunction(
        n_features=n_features,
        n_functions=n_functions,
        logger=logger,
        eig_range=[1.0, 1.0],  # Force eigenvalue of 1
        mu_range=[[0.0, 0.0],[0.0, 0.0]],   # Force mean at zero
    )
    
    # Initialize and verify parameters
    theta = gaussian.initialize()
    assert theta.requires_grad,"Theta should require gradient computation"
    
    # Create test points in a grid
    x = torch.linspace(-2, 2, 5)
    y = torch.linspace(-2, 2, 5)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([X.flatten(), Y.flatten()], dim=1).requires_grad_(True)  # (25, 2)
    
    # Compute Gaussian function values
    values = gaussian(points, theta)
    
    # Expected values for standard normal distribution
    mean = np.zeros(2)
    cov = np.eye(2)
    expected_values = torch.tensor(
        ( multivariate_normal.pdf(points.detach().numpy(), mean=mean, cov=cov) /
          multivariate_normal.pdf(mean, mean=mean, cov=cov) ),
          dtype = values.dtype
    ).unsqueeze(1)
    
    max_diff = torch.max(torch.abs(values - expected_values)).item()

    assert max_diff < 1e-5, "Distribution not similar enough"


    # Check both implementations
    torch_result = torch.allclose(values, expected_values, rtol=1e-4, atol=1e-4)

    # Test forward pass
    assert torch_result, "Distribuitions not similar enough"
    
    # Test mu gradients only
    theta.grad = None
    values = gaussian(points, theta, rho_flag=False, mu_flag=True)
    values.sum().backward()
    assert theta.grad is not None
    mu_grads = theta.grad[-n_features:, :]
    rho_grads = theta.grad[:-n_features, :]
    assert torch.all(rho_grads == 0), "Rho gradients should be zero when rho_flag is False"
    assert not torch.all(mu_grads == 0), "Mu gradients should be non-zero when mu_flag is True"

    # Test rho gradients only
    theta.grad = None
    values = gaussian(points, theta, rho_flag=True, mu_flag=False)
    values.sum().backward()
    assert theta.grad is not None
    mu_grads = theta.grad[-n_features:, :]
    rho_grads = theta.grad[:-n_features, :]
    assert torch.all(mu_grads == 0), "Mu gradients should be zero when mu_flag is False"
    assert not torch.all(rho_grads == 0), "Rho gradients should be non-zero when rho_flag is True"

    # Test both gradients
    theta.grad = None
    values = gaussian(points, theta, rho_flag=True, mu_flag=True)
    values.sum().backward()
    assert theta.grad is not None
    mu_grads = theta.grad[-n_features:, :]
    rho_grads = theta.grad[:-n_features, :]
    assert not torch.all(mu_grads == 0), "Mu gradients should be non-zero when both flags are True"
    assert not torch.all(rho_grads == 0), "Rho gradients should be non-zero when both flags are True"

def test_single_gaussian_gradient():
    """Test gradients using finite differences for a single Gaussian"""
    n_features = 2
    n_functions = 1
    logger = logging.getLogger('test')
    
    gaussian = GaussianFunction(
        n_features=n_features,
        n_functions=n_functions,
        logger=logger,
        eig_range=[1.0, 1.0],
        mu_range=[[0.0, 0.0], [0.0, 0.0]],
    )
    
    theta = gaussian.initialize()
    assert theta.requires_grad, "Theta should require gradient computation"
    
    # Test points at different locations
    test_points = [
        torch.tensor([[0.0, 0.0]]),  # At mean
        torch.tensor([[1.0, 0.0]]),  # Along x axis
        torch.tensor([[0.0, 1.0]]),  # Along y axis
        torch.tensor([[1.0, 1.0]]),  # Diagonal
        torch.tensor([[-1.0, -1.0]]), # Other diagonal
        torch.tensor([[-0.5, 0.3]])  # And something else
    ]


    eps = 1e-6
    for point in test_points:
        point = point.requires_grad_(True)
        
        def f(params):
            return gaussian(point, params, rho_flag=True, mu_flag=True)
        
        # Compute numerical gradient
        numerical_grad = torch.zeros_like(theta)
        
        for i in range(theta.numel()):
            theta_plus = theta.clone().detach()
            theta_plus.data.flatten()[i] += eps
            theta_minus = theta.clone().detach()
            theta_minus.data.flatten()[i] -= eps
            
            numerical_grad.flatten()[i] = (f(theta_plus) - f(theta_minus)).sum() / (2 * eps)
        
        # Compute analytical gradient
        out = f(theta)
        out.sum().backward()
        analytical_grad = theta.grad.clone()
        theta.grad = None  # Clear gradients for next iteration
        
        # Compare gradients
        assert torch.allclose(numerical_grad, analytical_grad, atol=5e-2), \
            f"Gradient mismatch at point {point}"
        
        # Additional verification that gradients point in same direction
        # Add cosine similarity check with zero handling
        if torch.norm(numerical_grad) > 1e-10 and torch.norm(analytical_grad) > 1e-10:
            normalized_numerical = numerical_grad / torch.norm(numerical_grad)
            normalized_analytical = analytical_grad / torch.norm(analytical_grad)
            cosine_similarity = torch.sum(normalized_numerical * normalized_analytical)
            assert cosine_similarity > 0.9, \
                f"Gradient directions differ significantly at point {point}"
        else:
            # If either gradient is essentially zero, verify both are close to zero
            assert torch.norm(numerical_grad) < 1e-10 and torch.norm(analytical_grad) < 1e-10, \
                f"Only one gradient is zero at point {point}"
        
def test_two_gaussians():
    """Test two Gaussians with different means and covariances"""
    n_features = 2
    n_functions = 2
    logger = logging.getLogger('test')
    
    gaussian = GaussianFunction(
        n_features=n_features,
        n_functions=n_functions,
        logger=logger,
        eig_range=[0.5, 1.0],
        mu_range=[[-1.0, 1.0], [-1.0, 1.0]],
    )
    
    theta = gaussian.initialize()
    assert theta.requires_grad, "Theta should require gradient computation"
    
    # Extract parameters for verification
    rho = theta[:-n_features, :]
    mu = theta[-n_features:, :]
    
    # Verify shapes
    assert rho.shape == (3, 2), "Rho should have shape (3, 2) for 2D gaussian"  # 3 elements for upper triangular in 2D
    assert mu.shape == (2, 2), "Mu should have shape (2, 2) for 2D gaussian"    # 2D means for 2 Gaussians
    
    # Create test points
    x = torch.linspace(-2, 2, 5)
    y = torch.linspace(-2, 2, 5)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([X.flatten(), Y.flatten()], dim=1).requires_grad_(True)
    
    # Compute values
    values = gaussian(points, theta)
    
    # Shape tests
    assert values.shape == (25, 2), "Output should have shape (n_points, n_functions)"
    
    # Value range tests
    assert torch.all(values >= 0), "Gaussian values should be non-negative"
    assert torch.all(values <= 1), "Normalized Gaussian values should be <= 1"
    
    # Test gradients as in single gaussian case
    # Test mu gradients only
    theta.grad = None
    values = gaussian(points, theta, rho_flag=False, mu_flag=True)
    values.sum().backward()
    assert theta.grad is not None
    mu_grads = theta.grad[-n_features:, :]
    rho_grads = theta.grad[:-n_features, :]
    assert torch.all(rho_grads == 0), "Rho gradients should be zero when rho_flag is False"
    assert not torch.all(mu_grads == 0), "Mu gradients should be non-zero when mu_flag is True"

    # Test rho gradients only
    theta.grad = None
    values = gaussian(points, theta, rho_flag=True, mu_flag=False)
    values.sum().backward()
    assert theta.grad is not None
    mu_grads = theta.grad[-n_features:, :]
    rho_grads = theta.grad[:-n_features, :]
    assert torch.all(mu_grads == 0), "Mu gradients should be zero when mu_flag is False"
    assert not torch.all(rho_grads == 0), "Rho gradients should be non-zero when rho_flag is True"

    # Test both gradients
    theta.grad = None
    values = gaussian(points, theta, rho_flag=True, mu_flag=True)
    values.sum().backward()
    assert theta.grad is not None
    mu_grads = theta.grad[-n_features:, :]
    rho_grads = theta.grad[:-n_features, :]
    assert not torch.all(mu_grads == 0), "Mu gradients should be non-zero when both flags are True"
    assert not torch.all(rho_grads == 0), "Rho gradients should be non-zero when both flags are True"


def test_gaussian_exponent():
    """Test just the quadratic form (x-μ)'Σ⁻¹(x-μ) calculation"""
    # Setup with identity precision matrix and zero mean
    n_features = 3
    n_functions = 1
    logger = logging.getLogger('test')
    
    gaussian = GaussianFunction(
        n_features=n_features,
        n_functions=n_functions,
        logger=logger,
        eig_range=[1.0, 1.0],  # Force eigenvalue of 1
        mu_range=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]  # Force mean at zero
    )
    
    # Create identity precision matrix (A = I)
    theta = gaussian.initialize()
    identity_rho = torch.eye(3)[torch.triu_indices(3,3)[0], torch.triu_indices(3,3)[1]]

    assert torch.allclose(identity_rho,theta.data[:-n_features, 0],atol=1e-5), "initialization is not making sense with rho"

    theta.data[:-n_features, 0] = identity_rho
    theta.data[-n_features:, 0] = 0.0  # zero mean
    
    # Test single point [1,0,0]
    point = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    
    # The exponent should be -0.5 for this point
    values = gaussian(point, theta)
    expected = torch.tensor([[torch.exp(torch.tensor(-0.5))]], dtype=values.dtype)

    assert torch.allclose(values, expected, rtol=1e-4, atol=1e-4)

def test_gaussian_exponent_non_diagonal():
    """Test quadratic form with non-diagonal precision matrix"""
    n_features = 3
    n_functions = 1
    logger = logging.getLogger('test')
    
    gaussian = GaussianFunction(
        n_features=n_features,
        n_functions=n_functions,
        logger=logger,
        eig_range=[1.0, 1.0],
        mu_range=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    )
    
    # Create a precision matrix with known structure
    # Let's use A (upper triangular):
    # [[1  1  0]
    #  [0  1  1]
    #  [0  0  1]]
    # This means G = A'A will be:
    # [[1  1  0]
    #  [1  2  1]
    #  [0  1  2]]
    
    theta = gaussian.initialize()
    # Set upper triangular elements: [1,1,0,1,1,1]
    rho = torch.tensor([1.0, 0.1, -0.3, 0.5, -0.2, 0.7])
    theta.data[:-n_features, 0] = rho
    theta.data[-n_features:, 0] = 0.0  # zero mean
    
    # First test point 
    point = torch.tensor([[-1.0, 0.7, 0.5]], requires_grad=True)
    
    values = gaussian(point, theta)
    expected = torch.exp(torch.tensor(-0.6757)).view(1, 1)
    assert torch.allclose(values, expected, rtol=1e-4, atol=1e-4)
    
    # Second test point 
    point = torch.tensor([[0.8577, 0.1606, -0.4884]], requires_grad=True)

    values = gaussian(point, theta)
    expected = torch.exp(torch.tensor(-0.5947650466)).view(1, 1)  # 
    assert torch.allclose(values, expected, rtol=1e-4, atol=1e-4)


def test_complex_3d_gaussian():
    """
    Test a single 3D Gaussian with a non-diagonal precision matrix.
    This test verifies correct handling of precision matrix (inverse covariance),
    its Cholesky decomposition, and triangular matrices in higher dimensions.
    """
    from pysesm.functions.GaussianFunction import GaussianFunction
    import torch
    import logging
    import numpy as np
    from scipy.stats import multivariate_normal

    # Setup
    n_features = 3
    n_functions = 1
    logger = logging.getLogger('test')
    
    # Initialize with constrained parameters
    gaussian = GaussianFunction(
        n_features=n_features,
        n_functions=n_functions,
        logger=logger,
        eig_range=[0.5, 1.0],  # Allow some variation in eigenvalues
        mu_range=[[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]  # 3D means
    )
    
    # Initialize and force specific parameters for a known case
    theta = gaussian.initialize()
    
    # Create a known covariance matrix (will be inverted)
    # This matrix will be positive definite with off-diagonal elements
    true_covariance = torch.tensor([
        [2.0, 0.5, 0.3],
        [0.5, 1.5, -0.2],
        [0.3, -0.2, 1.0]
    ], dtype=torch.float32)
    
    # Compute precision matrix (inverse of covariance)
    precision_matrix = torch.linalg.inv(true_covariance)
    
    # Compute Cholesky decomposition of precision matrix
    # G = Σ⁻¹ = A'A where A is upper triangular
    A = torch.linalg.cholesky(precision_matrix).T  # Note: we transpose to get upper triangular
    
    # Set known mean
    true_mean = torch.tensor([[0.5], [-0.3], [0.7]])
    
    # Manually set theta parameters
    # For 3D, we need 6 parameters for the triangular matrix (n*(n+1)/2)
    # and 3 for the mean, total 9 parameters
    indices = torch.triu_indices(3, 3)
    A_flat = A[indices[0], indices[1]] 
    
    theta.data[:-n_features, 0] = A_flat  # Set Cholesky factors
    theta.data[-n_features:, 0] = true_mean.squeeze()  # Set mean
    
    # Create test points in a 3D grid
    x = torch.linspace(-2, 2, 4)
    points = torch.cartesian_prod(x, x, x).requires_grad_(True)  # (64, 3)
    
    # Compute Gaussian function values
    values = gaussian(points, theta)
    
    # Compute expected values using scipy for verification
    expected_values = torch.tensor(
        multivariate_normal.pdf(
            points.detach().numpy(),
            mean=true_mean.squeeze().numpy(),
            cov=true_covariance.numpy()  # Note: scipy wants covariance, not precision
        ) / multivariate_normal.pdf(
            true_mean.squeeze().numpy(),
            mean=true_mean.squeeze().numpy(),
            cov=true_covariance.numpy()
        ),
        dtype=values.dtype
    ).unsqueeze(1)
    
    # Test forward pass
    assert torch.allclose(values, expected_values, rtol=1e-4, atol=1e-4), \
        "3D Gaussian values don't match expected values"
    
    # Verify reconstruction of precision matrix
    reconstructed_precision = torch.matmul(A.T, A)
    assert torch.allclose(reconstructed_precision, precision_matrix, rtol=1e-4, atol=1e-4), \
        "Reconstructed precision matrix doesn't match original"
    
    # Test gradients using finite differences
    eps = 1e-6
    
    # Test at specific points of interest
    test_points = [
        true_mean.T,  # At mean
        true_mean.T + torch.tensor([[1.0, 0.0, 0.0]]),  # Offset in x
        true_mean.T + torch.tensor([[0.0, 1.0, 0.0]]),  # Offset in y
        true_mean.T + torch.tensor([[0.0, 0.0, 1.0]]),  # Offset in z
        true_mean.T + torch.tensor([[1.0, 1.0, 1.0]]),  # Diagonal offset
        true_mean.T + torch.tensor([[-1.0, -1.0, -1.0]])  # Negative diagonal offset
    ]
    
    for point in test_points:
        point = point.requires_grad_(True)
        
        def f(params):
            return gaussian(point, params, rho_flag=True, mu_flag=True)
        
        # Compute numerical gradient
        numerical_grad = torch.zeros_like(theta)
        
        for i in range(theta.numel()):
            theta_plus = theta.clone().detach()
            theta_plus.data.flatten()[i] += eps
            theta_minus = theta.clone().detach()
            theta_minus.data.flatten()[i] -= eps
            
            numerical_grad.flatten()[i] = (f(theta_plus) - f(theta_minus)).sum() / (2 * eps)
        
        # Compute analytical gradient
        out = f(theta)
        out.sum().backward()
        analytical_grad = theta.grad.clone()
        theta.grad = None
        
        # Compare gradients with higher tolerance due to 3D complexity
        assert torch.allclose(numerical_grad, analytical_grad, atol=1e-1), \
            f"Gradient mismatch at point {point}"
        
        # Verify gradient directions
        if torch.norm(numerical_grad) > 1e-10 and torch.norm(analytical_grad) > 1e-10:
            normalized_numerical = numerical_grad / torch.norm(numerical_grad)
            normalized_analytical = analytical_grad / torch.norm(analytical_grad)
            cosine_similarity = torch.sum(normalized_numerical * normalized_analytical)
            assert cosine_similarity > 0.9, \
                f"Gradient directions differ significantly at point {point}"
        else:
            assert torch.norm(numerical_grad) < 1e-10 and torch.norm(analytical_grad) < 1e-10, \
                f"Only one gradient is zero at point {point}"

    # Test gradient flow control
    # Test mu gradients only
    theta.grad = None
    values = gaussian(points, theta, rho_flag=False, mu_flag=True)
    values.sum().backward()
    
    mu_grads = theta.grad[-n_features:, :]
    rho_grads = theta.grad[:-n_features, :]
    assert torch.all(rho_grads == 0), "Rho gradients should be zero when rho_flag is False"
    assert not torch.all(mu_grads == 0), "Mu gradients should be non-zero when mu_flag is True"

    # Test rho gradients only
    theta.grad = None
    values = gaussian(points, theta, rho_flag=True, mu_flag=False)
    values.sum().backward()
    
    mu_grads = theta.grad[-n_features:, :]
    rho_grads = theta.grad[:-n_features, :]
    assert torch.all(mu_grads == 0), "Mu gradients should be zero when mu_flag is False"
    assert not torch.all(rho_grads == 0), "Rho gradients should be non-zero when rho_flag is True"


def test_gaussian_function_batch_evaluation_and_gradients():
    """
    Test GaussianFunction with 3D (batch) input X.
    Verifies output shape, values, and gradient flow for mu and rho parameters.
    """
    n_features = 2
    n_functions = 3  # Test with multiple functions in the dictionary
    logger = logging.getLogger('test_gaussian_batch')

    gaussian = GaussianFunction(
        n_features=n_features,
        n_functions=n_functions,
        logger=logger,
        eig_range=[0.5, 1.5],  # Allow some variation in eigenvalues
        mu_range=[[-1.0, 1.0], [-1.0, 1.0]], # Allow variation in means
    )
    
    # Initialize parameters (theta) and ensure they require gradients
    theta = gaussian.initialize().requires_grad_(True)
    
    # Generate a batch of input data
    n_batches = 4
    n_samples_per_batch = 10
    X_raw = torch.randn(n_batches, n_samples_per_batch, n_features, dtype=torch.float32,requires_grad=True)

    X_batch = torch.nested.as_nested_tensor([t for t in torch.unbind(X_raw, dim=0)],layout=torch.jagged)

    # --- Test Forward Pass (Output Shape and Values) ---
    output_D = gaussian(X_batch, theta)
    
    # We must unbind the nested tensor to check the shapes of its components.
    assert gaussian._is_nested(output_D), "Output should be a nested_tensor"
    output_tensors_list = output_D.unbind()

    # Check the number of tensors in the batch (the outer dimension)
    assert len(output_tensors_list) == n_batches, \
        f"Expected {n_batches} tensors in the batch, but got {len(output_tensors_list)}"

    # Check the shape of EACH tensor within the batch
    for i, tensor in enumerate(output_tensors_list):
        expected_shape = (n_samples_per_batch, n_functions)
        assert tensor.shape == expected_shape, \
            f"Shape mismatch for tensor {i} in the batch. Expected {expected_shape}, got {tensor.shape}"

    
    # Values should be between 0 and 1
    assert torch.all(output_D >= 0.0), "Output D should be non-negative"
    assert torch.all(output_D <= 1.0), "Output D should be <= 1.0"

    # --- Test Gradients for Theta (mu and rho) ---
    # Case 1: Both mu and rho gradients enabled
    theta.grad = None
    output_D_for_grad = gaussian(X_batch, theta, rho_flag=True, mu_flag=True)
    
    # Simulate a loss that depends on output_D
    #mock_loss = sum(torch.sum(t ** 2) for t in output_D_for_grad.unbind()) # Simple loss for backprop
    mock_loss = torch.sum(output_D_for_grad**2)
    mock_loss.backward()
    
    assert theta.grad is not None, "Theta gradients should not be None"
    
    # Check that both mu and rho parts of theta received gradients
    num_rho_params_per_func = n_features * (n_features + 1) // 2
    rho_grads = theta.grad[:num_rho_params_per_func, :]
    mu_grads = theta.grad[-n_features:, :]

    assert not torch.all(rho_grads == 0), "Rho gradients should be non-zero when enabled"
    assert not torch.all(mu_grads == 0), "Mu gradients should be non-zero when enabled"

    # Case 2: Only mu gradients enabled
    theta.grad = None
    output_D_mu_only = gaussian(X_batch, theta, rho_flag=False, mu_flag=True)
    torch.sum(output_D_mu_only**2).backward()

    rho_grads_mu_only = theta.grad[:num_rho_params_per_func, :]
    mu_grads_mu_only = theta.grad[-n_features:, :]
    
    assert torch.all(rho_grads_mu_only == 0), "Rho gradients should be zero when rho_flag is False"
    assert not torch.all(mu_grads_mu_only == 0), "Mu gradients should be non-zero when mu_flag is True"

    # Case 3: Only rho gradients enabled
    theta.grad = None
    output_D_rho_only = gaussian(X_batch, theta, rho_flag=True, mu_flag=False)
    torch.sum(output_D_rho_only**2).backward()

    rho_grads_rho_only = theta.grad[:num_rho_params_per_func, :]
    mu_grads_rho_only = theta.grad[-n_features:, :]

    assert not torch.all(rho_grads_rho_only == 0), "Rho gradients should be non-zero when rho_flag is True"
    assert torch.all(mu_grads_rho_only == 0), "Mu gradients should be zero when mu_flag is False"

    # --- Test Gradients for X ---
    X_raw.grad = None # Clear previous gradients on X
    output_D_x_grad = gaussian(X_batch, theta, rho_flag=True, mu_flag=True)
    loss=torch.sum(output_D_x_grad**2)
    loss.backward()
    
    assert X_raw.grad is not None, "X gradients should not be None"
    assert not torch.all(X_raw.grad == 0), "X gradients should be non-zero"
    assert X_raw.grad.shape == X_raw.shape, "X gradients shape mismatch"


def test_gaussian_function_batch_vs_individual_consistency():
    """
    Verifies that processing a single sample as a batch (1, N_samples, N_features)
    yields the same result as processing it individually (N_samples, N_features).
    """
    n_features = 2
    n_functions = 5
    logger = logging.getLogger('test_gaussian_consistency')

    gaussian = GaussianFunction(
        n_features=n_features,
        n_functions=n_functions,
        logger=logger,
        eig_range=[0.8, 1.2],
        mu_range=[[0.0, 0.0], [0.0, 0.0]],
    )
    
    theta = gaussian.initialize()
    
    # Create a single set of 5 samples
    X_tensor = torch.randn(1, 5, n_features, dtype=torch.float32)
    X_single_batch = torch.nested.as_nested_tensor([t for t in torch.unbind(X_tensor, dim=0)],layout=torch.jagged)

    # Process as a batch (n_batches=1)
    output_batch = gaussian(X_single_batch, theta)
    
    # Process the same 5 samples individually (flattened)
    X_individual = X_tensor.squeeze(0) # (5, n_features)
    output_individual = gaussian(X_individual, theta)

    # Output from batch should be (1, 5, n_functions)
    # Output from individual should be (5, n_functions)
    # So, squeeze the batch output for comparison
    assert torch.allclose(torch.unbind(output_batch,dim=0)[0], output_individual, atol=1e-7), \
        "Batch evaluation (single batch) does not match individual evaluation"
    
# --- Tests for Correctness of Inherited __call__ ---

def test_call_with_nested_tensor_is_consistent(gaussian_function_fixture: torch.Tensor | List[torch.Tensor]):
    """
    Tests that calling the function with a NestedTensor produces the same result
    as evaluating each constituent tensor individually.
    """
    g_func = gaussian_function_fixture
    theta = g_func.initialize()

    tensors_list = [torch.randn(10, 2), torch.randn(5, 2), torch.randn(20, 2)]
    nested_tensor_input = torch.nested.nested_tensor(tensors_list, layout=torch.jagged)

    # 1. Evaluate using the inherited __call__ with NestedTensor
    nested_output = g_func(nested_tensor_input, theta)
    
    # 2. Evaluate each tensor individually using the specific `evaluate` method
    manual_loop_output = [g_func.evaluate(t, theta) for t in tensors_list]

    # 3. Compare the results
    assert g_func._is_nested(nested_output)
    unbound_nested_output = nested_output.unbind()
    
    assert len(unbound_nested_output) == len(manual_loop_output)
    for i in range(len(tensors_list)):
        assert torch.allclose(unbound_nested_output[i], manual_loop_output[i], atol=1e-7)
        assert unbound_nested_output[i].shape[0] == tensors_list[i].shape[0]
        assert unbound_nested_output[i].shape[1] == g_func.n_functions

def test_call_with_list_of_tensors_is_consistent(gaussian_function_fixture: torch.Tensor | List[torch.Tensor]):
    """
    Tests that calling the function with a list of tensors produces the same result
    as evaluating each tensor individually.
    """
    g_func = gaussian_function_fixture
    theta = g_func.initialize()

    tensors_list = [torch.randn(12, 2), torch.randn(3, 2), torch.randn(15, 2)]

    # 1. Evaluate using the inherited __call__ with the list
    list_output = g_func(tensors_list, theta)
    
    # 2. Evaluate each tensor individually
    manual_loop_output = [g_func.evaluate(t, theta) for t in tensors_list]

    # 3. Compare the results
    assert isinstance(list_output, list)
    assert len(list_output) == len(manual_loop_output)
    for i in range(len(tensors_list)):
        assert torch.allclose(list_output[i], manual_loop_output[i], atol=1e-7)

def test_call_with_nested_tensor_gradients(gaussian_function_fixture: torch.Tensor | List[torch.Tensor]):
    """
    Tests that gradients flow correctly through a batched evaluation
    using a NestedTensor. This replaces the old gradient test for 3D tensors.
    """
    g_func = gaussian_function_fixture
    theta = g_func.initialize().requires_grad_(True)

    tensors_list = [torch.randn(8, 2), torch.randn(4, 2)]
    nested_tensor_input = torch.nested.nested_tensor(tensors_list, layout=torch.jagged,requires_grad=True)
    
    # Evaluate with gradients enabled for all parameters
    nested_output = g_func(nested_tensor_input, theta, rho_flag=True, mu_flag=True)

    # The output of a NestedTensor evaluation is another NestedTensor.
    # To compute a scalar loss for backpropagation, we can sum over all elements of all tensors.
    scalar_loss = torch.sum(nested_output**2)
    scalar_loss.backward()

    # Assert that Theta has received gradients
    assert theta.grad is not None
    assert not torch.all(theta.grad == 0)


    # 1. Check the .grad attribute of the LEAF tensor, which is `nested_tensor_input`.
    assert nested_tensor_input.grad is not None, "The .grad of the leaf NestedTensor should be populated."

    # 2. The .grad attribute is itself a NestedTensor. Unbind it to check the components.
    unbound_grads = nested_tensor_input.grad.unbind()

    # 3. Assert that each component of the gradient exists and is non-zero.
    assert unbound_grads[0] is not None
    assert unbound_grads[1] is not None

    assert not torch.all(nested_tensor_input.grad == 0)

# # --- Test for Performance Demonstration of @torch.compile ---
# def test_torch_compile_performance_on_nested_tensor(gaussian_function_fixture: GaussianFunction, module_logger: logging.Logger):
#     """
#     Demonstrates the performance gain from @torch.compile on a NestedTensor.
#     This test uses a larger workload to make the benefits of compilation apparent.
#     """
#     g_func = gaussian_function_fixture
#     theta = g_func.initialize()

#     # --- CORRECCIÓN 1: Aumentar la carga de trabajo significativamente ---
#     num_tensors = 5000  
#     tensor_size_range = (100, 500) # Tamaños de tensor más grandes
#     tensors_list = [torch.randn(torch.randint(*tensor_size_range, (1,)).item(), 2) for _ in range(num_tensors)]
#     nested_tensor_input = torch.nested.nested_tensor(tensors_list, layout=torch.jagged)

#     # --- 1. Time the uncompiled, manual Python loop ---
#     start_time_loop = time.perf_counter()
#     for tensor in tensors_list:
#         _ = g_func.evaluate(X=tensor, Theta=theta)
#     end_time_loop = time.perf_counter()
#     python_loop_time = end_time_loop - start_time_loop
#     module_logger.info(f"Performance Test: Manual Python loop took {python_loop_time:.6f} seconds.")

#     # --- 2. Time the compiled __call__ method ---
#     # --- CORRECCIÓN 2: Usar el modo 'reduce-overhead' para JIT más rápido ---
#     # Esto compila la función localmente para el test.
#     compiled_call = torch.compile(g_func.__call__, mode="reduce-overhead")

#     # Warm-up call for JIT compilation.
#     module_logger.info("Performance Test: Running warm-up call for torch.compile...")
#     _ = compiled_call(X=nested_tensor_input, Theta=theta)
#     module_logger.info("Performance Test: Warm-up complete.")

#     # Now, time the second, optimized call
#     start_time_compiled = time.perf_counter()
#     _ = compiled_call(X=nested_tensor_input, Theta=theta)
#     end_time_compiled = time.perf_counter()
#     compiled_call_time = end_time_compiled - start_time_compiled
#     module_logger.info(f"Performance Test: Compiled __call__ took {compiled_call_time:.6f} seconds.")

#     # Assert that the compiled version is significantly faster.
#     speedup_factor = python_loop_time / compiled_call_time
#     module_logger.info(f"Performance Test: Achieved a speedup factor of {speedup_factor:.2f}x.")
    
#     # La aserción ahora debería pasar con una carga de trabajo más realista.
#     assert compiled_call_time < python_loop_time * 0.5, \
#         "The @torch.compile version was not significantly faster than the manual Python loop."
    
if __name__ == "__main__":
    from pytest_helper import print_pytest_instructions
    print_pytest_instructions()    
