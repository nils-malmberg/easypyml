"""Ordinary least squares linear regression."""

from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray


class LinearRegression:
    """Ordinary least squares (OLS) linear regression.

    Fits ``y ≈ X @ coef_ + intercept_`` by minimising the sum of squared
    residuals ``||y - X @ w - b||²``.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to learn the intercept ``b``. If False, the model is forced
        through the origin and ``intercept_`` is 0.0.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Learned weights ``w``.
    intercept_ : float
        Learned bias ``b``.
    rank_ : int
        Rank of the (centred) design matrix. ``rank_ < n_features_in_`` means
        some features are redundant: infinitely many solutions exist (see
        Notes, point 4).
    n_features_in_ : int
        Number of features seen during ``fit``.

    Notes
    -----
    How the solution is computed, and why.

    1. **Closed form.** Without an intercept, minimising ``||y - Xw||²``
       gives the normal equation ``XᵀX w = Xᵀy``, i.e.
       ``w = (XᵀX)⁻¹ Xᵀ y``.

    2. **The intercept cannot be added afterwards.** Solving for ``w``
       without ``b`` and then setting ``b = mean(y - Xw)`` is wrong: the
       first step forces the line through the origin, so the slope is
       distorted to compensate (on ``y = 3x + 10`` it finds ``w ≈ 6.33``
       and ``b ≈ 1.67`` instead of 3 and 10). ``w`` and ``b`` must be
       estimated together.

    3. **Two ways to estimate them together.**

       a. Prepend a column of ones to ``X``: ``b`` becomes one more weight,
          learned with the others.
       b. Centre ``X`` and ``y`` (subtract their means). The OLS hyperplane
          always passes through the point ``(mean(X), mean(y))``, so on
          centred data it passes through the origin and the formula of
          point 1 is exact. The intercept is then recovered as
          ``b = mean(y) - mean(X) @ w``.

       Both give the same predictions. This class uses (b), like
       scikit-learn, because it keeps ``b`` out of the weight vector, which
       matters in point 4.

    4. **Infinitely many solutions.** If a column is a linear combination of
       the others (perfect multicollinearity, e.g. ``x2 = x1 + 1``), ``XᵀX``
       is singular: ``np.linalg.inv`` raises ``LinAlgError``, and infinitely
       many ``w`` give exactly the same predictions. Among them,
       ``np.linalg.lstsq`` returns the one with the smallest norm ``||w||``.
       With option (a), ``b`` is part of that norm and gets shrunk too; with
       option (b) it is not. That is why, on ``X = [[1, 2], [3, 4], [5, 6]]``
       and ``y = [1, 2, 3]``, option (a) gives ``b = 1/6, w = [1/6, 1/3]``
       while this class (and scikit-learn) gives ``b = 0.25, w = [0.25, 0.25]``.
       Both fit the data perfectly.

    5. **Never invert XᵀX.** Forming ``XᵀX`` squares the condition number of
       ``X`` (``cond(XᵀX) = cond(X)²``), which amplifies rounding errors.
       ``np.linalg.lstsq`` works on ``X`` directly through an SVD: it is more
       accurate and handles the rank-deficient case of point 4.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1.0], [2.0], [3.0], [4.0]])
    >>> y = 3 * X[:, 0] + 10
    >>> model = LinearRegression().fit(X, y)
    >>> model.coef_.round(6), round(model.intercept_, 6)
    (array([3.]), 10.0)
    """

    coef_: NDArray[np.float64]
    intercept_: float
    rank_: int
    n_features_in_: int

    def __init__(self, fit_intercept: bool = True) -> None:
        self.fit_intercept = fit_intercept

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Fit the model by least squares.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : LinearRegression
            The fitted estimator.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D (n_samples, n_features), got shape {X_arr.shape}.")
        if y_arr.ndim != 1:
            raise ValueError(f"y must be 1D (n_samples,), got shape {y_arr.shape}.")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(f"X has {X_arr.shape[0]} samples but y has {y_arr.shape[0]}.")
        if not (np.isfinite(X_arr).all() and np.isfinite(y_arr).all()):
            raise ValueError("X and y must not contain NaN or infinity.")

        if self.fit_intercept:
            X_mean = X_arr.mean(axis=0)
            y_mean = float(y_arr.mean())
        else:
            X_mean = np.zeros(X_arr.shape[1])
            y_mean = 0.0

        # Notes 3b and 5: solve on centred data with an SVD-based solver.
        # The subtractions create new arrays, so the caller's X and y are untouched.
        coef, _, rank, _ = np.linalg.lstsq(X_arr - X_mean, y_arr - y_mean, rcond=None)

        self.coef_ = coef
        self.intercept_ = float(y_mean - X_mean @ coef)
        self.rank_ = int(
            rank
        )  # number of independent columns, if rank<shape[1] => infinity of solutions
        self.n_features_in_ = X_arr.shape[1]  # to avoid .predict() on wrong shapes
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict targets for ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        if not hasattr(self, "coef_"):
            raise AttributeError(
                "This LinearRegression instance is not fitted yet. Call 'fit' first."
            )
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2 or X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must have shape (n_samples, {self.n_features_in_}), got {X_arr.shape}."
            )
        return X_arr @ self.coef_ + self.intercept_
