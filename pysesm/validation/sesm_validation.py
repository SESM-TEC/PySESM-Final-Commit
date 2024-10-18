import logging

import torch


def validate_sesm_partial_fit(sesm, X: torch.Tensor, y: torch.Tensor):
    if sesm.n_samples != X.shape[0]:
        logging.warning(
            "[SESM] Mismatch between the number of samples in SESM and the observations in X. "
            "SESM samples: {}, Observations in X: {}".format(sesm.n_samples, X.shape[0])
        )

    if sesm.n_features != X.shape[1]:
        raise ValueError(
            "[SESM] Mismatch between the number of features in SESM and the features in X. "
            "SESM features: {}, Features in X: {}".format(sesm.n_features, X.shape[1])
        )

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "[SESM] Mismatch between the number of samples in X and the number of targets in Y. "
            "Observations in X: {}, Targets on Y {}".format(X.shape[0], y.shape[0])
        )
