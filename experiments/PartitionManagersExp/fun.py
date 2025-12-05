"""Benchmark functions used by ICICT experiments.

This module provides three multivariate test functions used to generate
datasets for experiments:

- `function_zhou`: multi-modal Gaussian-like function (useful for oscillatory tests).
- `function_zakharov`: smooth paraboloid-like function.
- `function_styblinski_tang`: non-linear polynomial benchmark.

Each function accepts a PyTorch tensor `x` of shape `(n_samples, n_dimensions)`
and returns a 1-D tensor of shape `(n_samples,)` containing the function values.
Implementations preserve the input device and dtype and are intended to be
called with inputs scaled according to the experiment configuration.
"""
import torch

# Funcion con más oscilaciones
def function_zhou(x: torch.Tensor) -> torch.Tensor:
    """
    Zhou (1998) function.

    Args:
        x (torch.Tensor): Input tensor of shape (n_samples, n_dimensions).
                        Each row is a point in the search space (values typically in [0,1]).

    Returns:
        torch.Tensor: Function values of shape (n_samples,).
    """
    d = x.shape[1]

    # Shift and scale
    x_a = 10 * (x - 1.0/3.0)
    x_b = 10 * (x - 2.0/3.0)

    # Norms squared
    norm_a2 = torch.sum(x_a**2, dim=1)
    norm_b2 = torch.sum(x_b**2, dim=1)

    # Gaussian components
    coeff = (2 * torch.pi) ** (-d / 2)
    phi1 = coeff * torch.exp(-0.5 * norm_a2)
    phi2 = coeff * torch.exp(-0.5 * norm_b2)

    # Final result
    return (10.0**d) / 2.0 * (phi1 + phi2)

#Funcion tipo parábola
def function_zakharov(x: torch.Tensor) -> torch.Tensor:
    """
    Zakharov function.

    Args:
        x (torch.Tensor): Input tensor of shape (n_samples, n_dimensions).
                        Each row is a point in the search space.

    Returns:
        torch.Tensor: Function values of shape (n_samples,).
    """
    # indices for 1..d (broadcasted to match x)
    d = x.shape[1]
    ii = torch.arange(1, d + 1, dtype=x.dtype, device=x.device).unsqueeze(0)

    # sum1 = sum(x_i^2)
    sum1 = torch.sum(x**2, dim=1)

    # sum2 = sum(0.5 * i * x_i)
    sum2 = torch.sum(0.5 * ii * x, dim=1)

    # final function value
    return sum1 + sum2**2 + sum2**4




def function_styblinski_tang(x: torch.Tensor) -> torch.Tensor:
    """
    Styblinski-Tang function.

    Args:
        x (torch.Tensor): Input tensor of shape (n_samples, n_dimensions).
                        Each row is a point in the search space.

    Returns:
        torch.Tensor: Function values of shape (n_samples,).
    """
    return 0.5 * torch.sum(x**4 - 16 * x**2 + 5 * x, dim=1)
