"""
Mathematical Distribution Functions.

Provides a collection of 2D and N-dimensional mathematical functions (e.g.,
paraboloids, exponentials, sinusoids) used as ground truth targets for
generating synthetic datasets and testing surrogate models.

Copyright (c) 2023-2025, Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
"""

import torch

# --- 2D Functions ---
# These functions take separate x and y tensors, typically from a meshgrid.

def paraboloid(x: torch.Tensor, y: torch.Tensor, a: float = 1.0, b: float = 1.0, c: float = 0.0) -> torch.Tensor:
    """
    Computes a 2D elliptic paraboloid: z = ax^2 + by^2 + c.
    """
    return a * x**2 + b * y**2 + c


def sinusoidal(x: torch.Tensor, y: torch.Tensor, a: float = 1.0, freq: float = 1.0, phase: float = 0.0) -> torch.Tensor:
    """
    Computes a 2D sinusoidal wave: z = a * sin(freq * (x + y) + phase).
    """
    return a * torch.sin(freq * (x + y) + phase)


def exponential(x: torch.Tensor, y: torch.Tensor, a: float = 1.0, b: float = 1.0, offset: float = 0.0) -> torch.Tensor:
    """
    Computes a 2D Gaussian-like exponential function: z = a * exp(-b * (x^2 + y^2)) + offset.
    """
    return a * torch.exp(-b * (x**2 + y**2)) + offset


def ripple(x: torch.Tensor, y: torch.Tensor, a: float = 1.0, freq: float = 3.0) -> torch.Tensor:
    """
    Computes a ripple/sinc-like function: z = a * sin(freq * r) / r, where r = sqrt(x^2 + y^2).
    Adds a small epsilon to avoid division by zero at the origin.
    """
    r = torch.sqrt(x**2 + y**2)
    return a * torch.sin(freq * r) / (r + 1e-5)


# --- N-Dimensional Functions ---
# These functions take a single tensor X of shape (n_samples, n_features).

def nd_paraboloid(X: torch.Tensor, a: float = 1.0, c: float = 0.0) -> torch.Tensor:
    """
    Computes an N-dimensional isotropic paraboloid: f(X) = a * sum(x_i^2) + c.
    
    Args:
        X (torch.Tensor): Input tensor of shape (n_samples, n_dimensions).
        a (float): Scaling factor.
        c (float): Vertical offset.
    """
    return a * torch.sum(X**2, dim=1) + c


def nd_exponential(X: torch.Tensor, a: float = 1.0, b: float = 1.0, offset: float = 0.0) -> torch.Tensor:
    """
    Computes an N-dimensional exponential (Gaussian-like) function: 
    f(X) = a * exp(-b * sum(x_i^2)) + offset.
    
    Args:
        X (torch.Tensor): Input tensor of shape (n_samples, n_dimensions).
        a (float): Amplitude.
        b (float): Decay rate (width control).
        offset (float): Vertical offset.
    """
    return a * torch.exp(-b * torch.sum(X**2, dim=-1)) + offset
