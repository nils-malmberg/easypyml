import numpy as np
import pytest
from numpy.typing import ArrayLike

from easypyml.exceptions import NotFittedError
from easypyml.utils import check_array, check_is_fitted, check_X_y


def test_check_X_y_converts_lists_to_float64() -> None:
    X_arr, y_arr = check_X_y([[1, 2], [3, 4]], [5, 6])
    assert X_arr.dtype == np.float64
    assert y_arr.dtype == np.float64
    np.testing.assert_array_equal(X_arr, [[1.0, 2.0], [3.0, 4.0]])


@pytest.mark.parametrize(
    ("X", "y", "match"),
    [
        ([1.0, 2.0], [1.0, 2.0], "X must be 2D"),
        ([[1.0], [2.0]], [[1.0], [2.0]], "y must be 1D"),
        ([[1.0], [2.0]], [1.0, 2.0, 3.0], "2 samples but y has 3"),
        ([[1.0], [np.nan]], [1.0, 2.0], "X must not contain NaN"),
        ([[1.0], [2.0]], [1.0, np.inf], "y must not contain NaN"),
    ],
)
def test_check_X_y_invalid_input_raises(X: ArrayLike, y: ArrayLike, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        check_X_y(X, y)


def test_check_array_without_n_features_accepts_any_width() -> None:
    assert check_array([[1.0, 2.0, 3.0]]).shape == (1, 3)


@pytest.mark.parametrize(
    ("X", "n_features", "match"),
    [
        ([[1.0, 2.0]], 3, "2 features, but 3 were expected"),
        (np.empty((0, 2)), None, "at least one sample"),
        ([[np.nan, 1.0]], 2, "NaN"),
    ],
)
def test_check_array_invalid_input_raises(X: ArrayLike, n_features: int | None, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        check_array(X, n_features)


class _Estimator:
    """Minimal stand-in: fitted state is just an attribute ending with ``_``."""

    def __init__(self, fitted: bool) -> None:
        self.alpha = 1.0  # a hyperparameter: no trailing underscore
        if fitted:
            self.coef_ = np.zeros(2)


def test_check_is_fitted_raises_before_fit() -> None:
    with pytest.raises(NotFittedError, match="_Estimator instance is not fitted"):
        check_is_fitted(_Estimator(fitted=False))


def test_check_is_fitted_passes_after_fit() -> None:
    check_is_fitted(_Estimator(fitted=True))


def test_not_fitted_error_is_value_and_attribute_error() -> None:
    assert issubclass(NotFittedError, ValueError)
    assert issubclass(NotFittedError, AttributeError)
