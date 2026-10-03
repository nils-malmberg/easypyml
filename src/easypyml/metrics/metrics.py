"""Evaluation metrics for fitted estimators."""

import numpy as np
from numpy.typing import ArrayLike


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute the coefficient of determination R².

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    float
        The R² score. 1.0 is a perfect prediction, 0.0 is as good as always
        predicting ``mean(y_true)``, and negative values are worse than that.

    Raises
    ------
    ValueError
        If the inputs are not 1D, differ in length, have fewer than two
        samples, or contain NaN or infinity.

    Notes
    -----
    ``R² = 1 - SS_res / SS_tot`` where

    - ``SS_res = Σ(y_true - y_pred)²`` is the error of the model;
    - ``SS_tot = Σ(y_true - mean(y_true))²`` is the error of a baseline that
      always predicts the mean of the *true* values.

    R² is therefore the fraction of the baseline's error that the model
    removes. It has no lower bound: a model worse than the mean is negative.

    If ``y_true`` is constant, ``SS_tot`` is 0 and the ratio is undefined.
    Like scikit-learn, this returns 1.0 for a perfect prediction and 0.0
    otherwise. Constancy is detected by comparing values, not by testing
    ``SS_tot == 0``: rounding in the mean makes ``SS_tot`` tiny but non-zero
    for e.g. ``[0.1, 0.1, 0.1]``, where scikit-learn returns about -1.7e31.

    Examples
    --------
    >>> r2_score([1, 2, 3], [1, 2, 3])
    1.0
    >>> r2_score([1, 2, 3], [2, 2, 2])
    0.0
    >>> r2_score([1, 2, 3], [3, 2, 1])
    -3.0
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)  # used to convert list, set...
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    if y_true_arr.ndim != 1 or y_pred_arr.ndim != 1:
        raise ValueError(
            f"y_true and y_pred must be 1D, got shapes {y_true_arr.shape} and {y_pred_arr.shape}."
        )
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"y_true and y_pred must have the same length, "
            f"got {y_true_arr.size} and {y_pred_arr.size}."
        )
    if y_true_arr.size < 2:
        raise ValueError("R² is not defined for fewer than two samples.")
    if not (np.isfinite(y_true_arr).all() and np.isfinite(y_pred_arr).all()):
        raise ValueError("y_true and y_pred must not contain NaN or infinity.")

    ss_res = float(np.sum((y_true_arr - y_pred_arr) ** 2))
    if np.all(y_true_arr == y_true_arr[0]):
        return 1.0 if ss_res == 0.0 else 0.0

    ss_tot = float(np.sum((y_true_arr - y_true_arr.mean()) ** 2))
    return 1.0 - ss_res / ss_tot
