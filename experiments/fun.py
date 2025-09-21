import torch

# Funcion con más oscilaciones
def zhou_function(X: torch.Tensor) -> torch.Tensor:
    """
    Zhou (1998) function.

    Args:
        X (torch.Tensor): Input tensor of shape (n_samples, n_dimensions).
                        Each row is a point in the search space (values typically in [0,1]).

    Returns:
        torch.Tensor: Function values of shape (n_samples,).
    """
    d = X.shape[1]

    # Shift and scale
    X_a = 10 * (X - 1.0/3.0)
    X_b = 10 * (X - 2.0/3.0)

    # Norms squared
    norm_a2 = torch.sum(X_a**2, dim=1)
    norm_b2 = torch.sum(X_b**2, dim=1)

    # Gaussian components
    coeff = (2 * torch.pi) ** (-d / 2)
    phi1 = coeff * torch.exp(-0.5 * norm_a2)
    phi2 = coeff * torch.exp(-0.5 * norm_b2)

    # Final result
    return (10.0**d) / 2.0 * (phi1 + phi2)

#Funcion con algunos valles o montañas "suaves"
def rosenbrock_rescaled_function(X: torch.Tensor) -> torch.Tensor:
    """
    Rescaled Rosenbrock function.

    Args:
        X (torch.Tensor): Input tensor of shape (n_samples, 4).
                        Each row is a point in the search space, with values in [0,1].

    Returns:
        torch.Tensor: Function values of shape (n_samples,).
    """
    # Rescale inputs: [0,1] -> [-5,10]
    Xbar = 15 * X - 5  # shape (n_samples, 4)

    # Rosenbrock sum
    sum_terms = torch.sum(
        100.0 * (Xbar[:, 1:] - Xbar[:, :-1]**2)**2 + (1 - Xbar[:, :-1])**2,
        dim=1
    )

    # Rescale output
    return (sum_terms - 3.827e5) / 3.755e5

#Funcion tipo parábola
def zakharov_function(X: torch.Tensor) -> torch.Tensor:
    """
    Zakharov function.

    Args:
        X (torch.Tensor): Input tensor of shape (n_samples, n_dimensions).
                        Each row is a point in the search space.

    Returns:
        torch.Tensor: Function values of shape (n_samples,).
    """
    # indices for 1..d (broadcasted to match X)
    d = X.shape[1]
    ii = torch.arange(1, d + 1, dtype=X.dtype, device=X.device).unsqueeze(0)

    # sum1 = sum(x_i^2)
    sum1 = torch.sum(X**2, dim=1)

    # sum2 = sum(0.5 * i * x_i)
    sum2 = torch.sum(0.5 * ii * X, dim=1)

    # final function value
    return sum1 + sum2**2 + sum2**4




def styblinski_tang_function(X: torch.Tensor) -> torch.Tensor:
    """
    Styblinski-Tang function.

    Args:
        X (torch.Tensor): Input tensor of shape (n_samples, n_dimensions).
                        Each row is a point in the search space.

    Returns:
        torch.Tensor: Function values of shape (n_samples,).
    """
    return 0.5 * torch.sum(X**4 - 16 * X**2 + 5 * X, dim=1)