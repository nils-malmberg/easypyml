"""Base classes shared by all EasyPyML estimators.

Every estimator is a (non-frozen) dataclass:

- hyperparameters are ordinary dataclass fields, so ``__init__`` and
  ``__repr__`` are generated and ``get_params`` can list them generically;
- attributes learned by ``fit`` end with ``_`` and are declared with
  ``field(init=False, repr=False, compare=False)``: they are not constructor
  arguments, do not clutter the repr, are not compared by ``==``, and do not
  exist before ``fit`` (which is what ``check_is_fitted`` relies on).

The dataclass cannot be frozen, because ``fit`` must store the learned
attributes on ``self`` and return ``self`` (the scikit-learn contract).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from easypyml.metrics import r2_score


@dataclass
class BaseEstimator(ABC):
    """Common interface of every estimator: hyperparameters and ``fit``.

    Subclasses must be decorated with ``@dataclass`` and declare their
    hyperparameters as fields.
    """

    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Learn from ``X`` and ``y``, store the learned attributes, return ``self``."""

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the hyperparameters, i.e. the constructor arguments.

        Parameters
        ----------
        deep : bool, default=True
            Accepted for scikit-learn compatibility (``clone`` calls
            ``get_params(deep=False)``). It only matters for estimators that
            contain other estimators, which EasyPyML does not have yet.

        Returns
        -------
        dict
            Mapping from hyperparameter name to its current value.
        """
        return {f.name: getattr(self, f.name) for f in fields(self) if f.init}

    def set_params(self, **params: Any) -> Self:
        """Set hyperparameters and return ``self``.

        Parameters
        ----------
        **params
            Hyperparameter names and their new values.

        Returns
        -------
        self
            The estimator, so that calls can be chained.

        Raises
        ------
        ValueError
            If a name is not a hyperparameter of this estimator.
        """
        valid = self.get_params()
        for name, value in params.items():
            if name not in valid:
                raise ValueError(
                    f"Invalid parameter {name!r} for {type(self).__name__}. "
                    f"Valid parameters are: {sorted(valid)}."
                )
            setattr(self, name, value)
        return self


@dataclass
class BaseRegressor(BaseEstimator):
    """Base class for regressors: adds ``predict`` and an R² ``score``."""

    @abstractmethod
    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict continuous targets for ``X``."""

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the R² of ``predict(X)`` with respect to ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True targets.

        Returns
        -------
        float
            R² score, see ``easypyml.metrics.r2_score``.
        """
        return r2_score(y, self.predict(X))
