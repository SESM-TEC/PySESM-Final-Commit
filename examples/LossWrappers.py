"""
Loss Function Wrappers.

Custom loss implementations (e.g., Jensen-Shannon, Cross-Entropy) used in
examples and experiments to train surrogate models.

Copyright (c) 2023-2025, Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
"""
import torch

class KLDivLossWrapper(torch.nn.Module):
    def __init__(self, reduction='mean'):
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
