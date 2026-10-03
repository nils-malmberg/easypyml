import numpy as np
import pytest
from sklearn.base import clone

from easypyml.base import BaseRegressor
from easypyml.linear_models import LinearRegression
from easypyml.metrics import r2_score


def test_base_regressor_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError, match="abstract"):
        BaseRegressor()  # type: ignore[abstract]


def test_score_is_r2_of_predict() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2))
    y = X @ np.array([1.0, -1.0]) + rng.normal(scale=0.3, size=30)
    model = LinearRegression().fit(X, y)
    assert model.score(X, y) == r2_score(y, model.predict(X))


def test_get_params_lists_only_hyperparameters() -> None:
    model = LinearRegression(fit_intercept=False).fit([[1.0], [2.0]], [1.0, 2.0])
    assert model.get_params() == {"fit_intercept": False}


def test_set_params_updates_and_returns_self() -> None:
    model = LinearRegression()
    assert model.set_params(fit_intercept=False) is model
    assert model.fit_intercept is False


def test_set_params_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Invalid parameter 'alpha'"):
        LinearRegression().set_params(alpha=1.0)


def test_repr_shows_hyperparameters_only() -> None:
    model = LinearRegression().fit([[1.0], [2.0]], [1.0, 2.0])
    assert repr(model) == "LinearRegression(fit_intercept=True)"


def test_equality_ignores_learned_attributes() -> None:
    a = LinearRegression().fit([[1.0], [2.0]], [1.0, 2.0])
    b = LinearRegression().fit([[1.0], [2.0]], [5.0, 0.0])
    assert a == b  # same hyperparameters, different coef_


def test_learned_attributes_do_not_exist_before_fit() -> None:
    assert not hasattr(LinearRegression(), "coef_")


def test_sklearn_clone_rebuilds_an_unfitted_copy() -> None:
    model = LinearRegression(fit_intercept=False).fit([[1.0], [2.0]], [1.0, 2.0])
    copy = clone(model)
    assert copy is not model
    assert copy.get_params() == model.get_params()
    assert not hasattr(copy, "coef_")
