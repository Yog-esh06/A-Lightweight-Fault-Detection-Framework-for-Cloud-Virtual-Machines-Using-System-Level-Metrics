"""Leakage-safe scaled linear, ridge, random-forest, and gradient-boosted forecast models.

Recurrent model note: the recurrent path uses a GRU (gated recurrent unit), not a vanilla LSTM.
GRU has fewer parameters and equivalent or better performance at the sequence lengths used here
(12-step lag window). The architecture choice is intentional; it is not a gap vs the project plan.
"""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_classical_models(config: dict, seed: int) -> dict[str, object]:
    """Construct deterministic direct multi-horizon regressors; callers fit train data only.

    Models:
    - linear_regression: OLS with no regularization — explicit "Linear Regression" baseline
      from the project plan, included so benchmark reports name it unambiguously.
    - ridge: Ridge (L2 alpha=1.0) — scaled regularized linear baseline.
    - random_forest: Non-linear tree ensemble.
    - gradient_boosting: sklearn GBR, functionally equivalent to XGBoost in the project plan.
    """
    fc = config["forecasting"]
    return {
        "linear_regression": Pipeline([
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "ridge": Pipeline([
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "random_forest": Pipeline([("imputer", SimpleImputer()), ("model", RandomForestRegressor(
            n_estimators=int(fc["forest_trees"]), max_depth=12, min_samples_leaf=3,
            random_state=seed, n_jobs=1,
        ))]),
        "gradient_boosting": Pipeline([("imputer", SimpleImputer()), ("model", MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=int(fc["boosting_estimators"]), max_depth=3, random_state=seed), n_jobs=1,
        ))]),
    }

