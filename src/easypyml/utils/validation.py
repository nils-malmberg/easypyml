"""Input validation shared by all estimators and metrics."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from easypyml.exceptions import NotFittedError


def check_array(X: ArrayLike, n_features: int | None = None) -> NDArray[np.float64]:
    """Validate a feature matrix and convert it to a float64 array.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data (list, NumPy array, DataFrame...).
    n_features : int or None, default=None
        Expected number of columns. ``None`` skips this check (used by
        ``fit``, where it is not known yet); ``predict`` passes the
        estimator's ``n_features_in_``.

    Returns
    -------
    X_arr : ndarray of shape (n_samples, n_features), dtype float64
        A new array if a conversion was needed, otherwise ``X`` itself:
        callers must not modify it in place.

    Raises
    ------
    ValueError
        If ``X`` is not 2D, has no samples, has the wrong number of
        columns, or contains NaN or infinity.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_features), got shape {X_arr.shape}.")
    if X_arr.shape[0] == 0:
        raise ValueError("X must contain at least one sample.")
    if n_features is not None and X_arr.shape[1] != n_features:
        raise ValueError(f"X has {X_arr.shape[1]} features, but {n_features} were expected.")
    if not np.isfinite(X_arr).all():
        raise ValueError("X must not contain NaN or infinity.")
    return X_arr


def check_X_y(X: ArrayLike, y: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate training data and convert it to float64 arrays.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,)
        Target values.

    Returns
    -------
    X_arr : ndarray of shape (n_samples, n_features), dtype float64
        Validated ``X`` (see ``check_array``).
    y_arr : ndarray of shape (n_samples,), dtype float64
        Validated ``y``.

    Raises
    ------
    ValueError
        If ``X`` is invalid (see ``check_array``), ``y`` is not 1D,
        ``X`` and ``y`` differ in length, or ``y`` contains NaN or infinity.
    """
    X_arr = check_array(X)
    y_arr = np.asarray(y, dtype=np.float64)
    if y_arr.ndim != 1:
        raise ValueError(f"y must be 1D (n_samples,), got shape {y_arr.shape}.")
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(f"X has {X_arr.shape[0]} samples but y has {y_arr.shape[0]}.")
    if not np.isfinite(y_arr).all():
        raise ValueError("y must not contain NaN or infinity.")
    return X_arr, y_arr


def check_is_fitted(estimator: object) -> None:
    """Raise ``NotFittedError`` if ``estimator`` has not been fitted.

    By convention, attributes learned during ``fit`` end with an underscore
    (``coef_``, ``n_features_in_``...) and do not exist before. An estimator
    is therefore considered fitted if it has at least one such attribute.

    Parameters
    ----------
    estimator : object
        The estimator to check.

    Raises
    ------
    NotFittedError
        If no attribute ending with ``_`` is set on ``estimator``.
    """
    fitted = [name for name in vars(estimator) if name.endswith("_") and not name.startswith("__")]
    if not fitted:
        raise NotFittedError(
            f"This {type(estimator).__name__} instance is not fitted yet. Call 'fit' first."
        )
