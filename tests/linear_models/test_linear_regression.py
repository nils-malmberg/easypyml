import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression as SkLinearRegression

from easypyml.linear_models import LinearRegression


@pytest.fixture  # run when pytest read func attributes (X, y = noisy_data() doesn't work)
def noisy_data() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(0)  # seed fixe → test reproductible
    X = rng.normal(size=(100, 3))
    y = X @ np.array([1.5, -2.0, 0.5]) + 4.0 + rng.normal(scale=0.1, size=100)
    return X, y


@pytest.mark.parametrize("fit_intercept", [True, False])  # create 2 distincts tests
def test_matches_sklearn(
    noisy_data: tuple[NDArray[np.float64], NDArray[np.float64]], fit_intercept: bool
) -> None:
    X, y = noisy_data
    ours = LinearRegression(fit_intercept=fit_intercept).fit(X, y)
    ref = SkLinearRegression(fit_intercept=fit_intercept).fit(X, y)
    np.testing.assert_allclose(ours.coef_, ref.coef_, rtol=1e-10)
    np.testing.assert_allclose(ours.intercept_, ref.intercept_, rtol=1e-10, atol=1e-12)


def test_predict_before_fit_raises() -> None:
    with pytest.raises(AttributeError, match="not fitted"):
        LinearRegression().predict([[1.0]])
