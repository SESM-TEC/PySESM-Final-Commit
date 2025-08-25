import torch


def validate_sesm_partial_fit(sesm, X: torch.Tensor, y: torch.Tensor):
    if sesm.n_features != X.shape[1]:
        raise ValueError(
            "[SESM] Mismatch between the number of features in SESM and the features in X. "
            f"SESM features: {sesm.n_features}, Features in X: {X.shape[1]}"
        )

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "[SESM] Mismatch between the number of samples in X and the number of targets in Y. "
            f"Observations in X: {X.shape[0]}, Targets on Y {y.shape[0]}"
        )
