import numpy as np
import pytest
from numpy.typing import ArrayLike
from sklearn.metrics import r2_score as sk_r2_score

from easypyml.metrics import r2_score


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_r2_matches_sklearn(seed: int) -> None:
    rng = np.random.default_rng(seed)
    y_true = rng.normal(size=50)
    y_pred = y_true + rng.normal(scale=0.5, size=50)
    np.testing.assert_allclose(r2_score(y_true, y_pred), sk_r2_score(y_true, y_pred))


def test_r2_perfect_prediction() -> None:
    assert r2_score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_r2_mean_prediction_is_zero() -> None:
    assert r2_score([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]) == pytest.approx(0.0)


def test_r2_can_be_negative() -> None:
    assert r2_score([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-3.0)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        ([5.0, 5.0, 5.0], [5.0, 5.0, 5.0], 1.0),
        ([5.0, 5.0, 5.0], [5.0, 5.0, 6.0], 0.0),
        ([0.1, 0.1, 0.1], [0.1, 0.1, 0.2], 0.0),  # sklearn: about -1.7e31
    ],
)
def test_r2_constant_y_true(y_true: list[float], y_pred: list[float], expected: float) -> None:
    assert r2_score(y_true, y_pred) == expected


def test_r2_accepts_lists_and_ints() -> None:
    assert r2_score([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0


def test_r2_returns_python_float() -> None:
    assert type(r2_score([1, 2, 3], [1, 2, 4])) is float


@pytest.mark.parametrize(
    ("y_true", "y_pred", "match"),
    [
        ([1.0, 2.0, 3.0], [1.0, 2.0], "same length"),
        ([1.0, 2.0, 3.0], [[1.0], [2.0], [3.0]], "must be 1D"),
        ([1.0], [1.0], "fewer than two samples"),
        ([1.0, np.nan], [1.0, 2.0], "NaN"),
    ],
)
def test_r2_invalid_input_raises(y_true: ArrayLike, y_pred: ArrayLike, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        r2_score(y_true, y_pred)
