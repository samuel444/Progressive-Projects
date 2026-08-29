import numpy as np


def benchmark_metrics(df):
    """
    Calculate S&P 500 benchmark performance metrics.

    Required columns:
        Close
        Return

    Returns:
        total_return
        average_drawdown
        max_drawdown
        sharpe_ratio
    """

    close = df["Close"].dropna()
    returns = df["Return"].dropna()

    ########################################
    # Total Return
    ########################################

    total_return = (
        close.iloc[-1] / close.iloc[0]
    ) - 1


    ########################################
    # Drawdown
    ########################################

    running_max = close.cummax()

    drawdown = (
        close / running_max
    ) - 1

    average_drawdown = drawdown.mean()

    max_drawdown = drawdown.min()


    ########################################
    # Annualised Sharpe Ratio
    ########################################

    if returns.std() == 0:
        sharpe_ratio = 0.0

    else:
        sharpe_ratio = (
            returns.mean()
            / returns.std()
        ) * np.sqrt(252)


    ########################################
    # Results
    ########################################

    return {
        "Return": total_return,
        "Average Drawdown": average_drawdown,
        "Max Drawdown": max_drawdown,
        "Sharpe Ratio": sharpe_ratio,
    }

"""
Multi-target portfolio backtest helpers.

The public entry point is::

    run_multi_target_portfolio_backtest(...)

It is designed for the cached dataframe structure used by the Equity Selector
pipeline:

    Date, Ticker, Close, Return, <features...>, <targets...>, Split

where Split contains TRAIN and BACKTEST rows.

Selected-model metadata is expected to contain:

    Target, Model, Parameters, Target Type, Horizon, Quality Score

Target-specific feature lists can either be supplied separately or stored in a
Features column in selected_models_df.
"""


import ast
import json
import logging
import re
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.utils.class_weight import compute_sample_weight

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:  # pragma: no cover - only used if requested by metadata
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:  # pragma: no cover - only used if requested by metadata
    XGBClassifier = None
    XGBRegressor = None


logger = logging.getLogger(__name__)


########################################
# Defaults
########################################

DEFAULT_TYPE_VALUES = {
    "ALPHA": 1.00,
    "RELATIVE_ALPHA": 1.15,
    "CROSS_SECTION_ALPHA": 0.80,
    "DIRECTION": 0.50,
    "ALPHA_BINARY": 0.70,
    "BARRIER_ALPHA": 0.80,
    "VOLATILITY": -0.70,
    "VOLATILITY_EVENT": -0.80,
    "DOWNSIDE": -1.00,
    "TAIL_RISK": -1.20,
    "TAIL_EVENT": -1.30,
    "UPSIDE_RISK": 0.20,
    "UPSIDE_EVENT": 0.80,
    "RECOVERY": 0.60,
    "REVERSAL": -0.50,
    "REGIME": 0.30,
    "CORRELATION": -0.50,
    "COVARIANCE": -0.60,
    "LIQUIDITY": 0.40,
    "EXECUTION": 0.30,
    "MARKET_IMPACT": -0.50,
}


RISK_TYPES = {
    "VOLATILITY",
    "VOLATILITY_EVENT",
    "DOWNSIDE",
    "TAIL_RISK",
    "TAIL_EVENT",
    "CORRELATION",
    "COVARIANCE",
    "MARKET_IMPACT",
}


SCALE_MODELS = {
    "ols",
    "linear regression",
    "ridge",
    "lasso",
    "elastic net",
    "elasticnet",
    "huber",
    "svr",
    "knn",
    "knn regressor",
    "knn classifier",
    "mlp",
    "mlp regressor",
    "mlp classifier",
    "logistic regression",
    "logistic",
    "l1 logistic regression",
    "l1 logistic",
    "l2 logistic regression",
    "l2 logistic",
    "elastic net logistic regression",
    "elasticnet logistic regression",
    "elastic net logistic",
    "multinomial logistic regression",
    "l2 multinomial logistic regression",
    "l1 multinomial logistic regression",
    "elastic net multinomial logistic regression",
    "lda",
    "qda",
    "svm",
}


########################################
# Metadata / parameter helpers
########################################

def parse_parameters(parameters: Any) -> Dict[str, Any]:
    """Parse model parameters stored as dict / JSON / Python-literal text."""

    if parameters is None:
        return {}

    if isinstance(parameters, dict):
        parsed = parameters.copy()
    else:
        try:
            if pd.isna(parameters):
                return {}
        except (TypeError, ValueError):
            pass

        text = str(parameters).strip()
        if text == "":
            return {}

        # Preserve the behaviour of the supplied backtest code.
        text = re.sub(r"\bnull\b", "None", text, flags=re.IGNORECASE)
        text = re.sub(r"\btrue\b", "True", text, flags=re.IGNORECASE)
        text = re.sub(r"\bfalse\b", "False", text, flags=re.IGNORECASE)

        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = json.loads(str(parameters))

    if not isinstance(parsed, dict):
        raise ValueError("Model parameters must parse to a dictionary.")

    return {
        str(key).replace("model__", "", 1): value
        for key, value in parsed.items()
    }


def _parse_features(value: Any) -> Sequence[str]:
    if isinstance(value, (list, tuple, np.ndarray, pd.Index)):
        return [str(x) for x in value]

    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(text)

    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"Features must be a list, got {type(parsed).__name__}.")

    return [str(x) for x in parsed]


def _feature_map(
    selected_models_df: pd.DataFrame,
    model_features: Optional[
        Union[pd.DataFrame, Mapping[str, Sequence[str]]]
    ],
) -> Dict[str, Sequence[str]]:
    """Return Target -> selected feature names."""

    mapping: Dict[str, Sequence[str]] = {}

    if "Features" in selected_models_df.columns:
        for _, row in selected_models_df.iterrows():
            mapping[str(row["Target"])] = _parse_features(row["Features"])

    if model_features is None:
        return mapping

    if isinstance(model_features, Mapping):
        for target, features in model_features.items():
            mapping[str(target)] = _parse_features(features)
        return mapping

    if not isinstance(model_features, pd.DataFrame):
        raise TypeError(
            "model_features must be a DataFrame, mapping, or None."
        )

    required = {"Target", "Features"}
    missing = required.difference(model_features.columns)
    if missing:
        raise ValueError(
            "model_features is missing columns: "
            + ", ".join(sorted(missing))
        )

    for _, row in model_features.iterrows():
        mapping[str(row["Target"])] = _parse_features(row["Features"])

    return mapping


def infer_statistical_target_type(y: pd.Series) -> str:
    """
    Infer continuous / binary / multiclass from the training target.

    If your selected-model table later includes a Statistical Type column, that
    column takes precedence and this inference is not used.
    """

    y = pd.Series(y).replace([np.inf, -np.inf], np.nan).dropna()

    if y.empty:
        raise ValueError("Cannot infer target type from an empty target.")

    unique = pd.Series(y.unique()).dropna()
    n_unique = len(unique)

    if n_unique <= 2:
        return "binary"

    # Classification targets in the existing pipeline use small numeric class
    # sets such as -1/0/1. Avoid treating ordinary continuous targets as
    # multiclass merely because a short sample has a few distinct values.
    numeric = pd.to_numeric(unique, errors="coerce")
    if (
        n_unique <= 20
        and numeric.notna().all()
        and np.allclose(numeric, np.round(numeric))
    ):
        return "multiclass"

    return "continuous"


def _clean_binary_target(y: pd.Series, target: str) -> pd.Series:
    """Match the Future Direction handling in the supplied backtest code."""

    y = y.copy()

    if str(target).startswith("Future Direction"):
        y = pd.Series(
            np.where(y.to_numpy() > 0, 1, -1),
            index=y.index,
        )

    classes = np.sort(pd.Series(y).dropna().unique())
    if len(classes) > 2:
        raise ValueError(
            f"{target} is binary but contains classes {classes}."
        )

    return y


########################################
# Model builder - faithful to supplied code
########################################

def build_model(
    model_name: str,
    statistical_type: str,
    parameters: Any,
    num_classes: Optional[int] = None,
):
    params = parse_parameters(parameters)
    name = str(model_name).lower().strip()
    statistical_type = str(statistical_type).lower().strip()
    is_regression = statistical_type == "continuous"

    if name in {"linear regression", "ols"}:
        return LinearRegression(**params)

    if name == "ridge":
        return Ridge(**params)

    if name == "lasso":
        return Lasso(**params)

    if name in {"elastic net", "elasticnet"}:
        return ElasticNet(**params)

    if name in {"logistic regression", "logistic"}:
        params.setdefault("max_iter", 5000)
        return LogisticRegression(**params)

    if name in {"l2 logistic regression", "l2 logistic"}:
        params.setdefault("penalty", "l2")
        params.setdefault("solver", "lbfgs")
        params.setdefault("max_iter", 5000)
        return LogisticRegression(**params)

    if name in {"l1 logistic regression", "l1 logistic"}:
        params.setdefault("penalty", "l1")
        params.setdefault("solver", "liblinear")
        params.setdefault("max_iter", 5000)
        return LogisticRegression(**params)

    if name in {
        "elastic net logistic regression",
        "elasticnet logistic regression",
        "elastic net logistic",
    }:
        params.setdefault("penalty", "elasticnet")
        params.setdefault("solver", "saga")
        params.setdefault("max_iter", 5000)
        return LogisticRegression(**params)

    if name in {
        "random forest",
        "random forest regressor",
        "random forest classifier",
    }:
        if is_regression:
            return RandomForestRegressor(**params)
        return RandomForestClassifier(**params)

    if name in {
        "extra trees",
        "extra trees regressor",
        "extra trees classifier",
    }:
        if is_regression:
            return ExtraTreesRegressor(**params)
        return ExtraTreesClassifier(**params)

    if name in {
        "gradient boosting",
        "gradient boosting regressor",
        "gradient boosting classifier",
    }:
        if is_regression:
            return GradientBoostingRegressor(**params)
        return GradientBoostingClassifier(**params)

    if name in {
        "hist gradient boosting",
        "histogram gradient boosting",
        "histgradientboosting",
    }:
        if is_regression:
            return HistGradientBoostingRegressor(**params)
        return HistGradientBoostingClassifier(**params)

    if name in {"lightgbm", "lgbm"}:
        if LGBMRegressor is None or LGBMClassifier is None:
            raise ImportError(
                "lightgbm is required because a selected model uses LightGBM."
            )
        if is_regression:
            return LGBMRegressor(**params)
        return LGBMClassifier(**params)

    if is_regression:
        if name == "huber":
            params.setdefault("max_iter", 1000)
            return HuberRegressor(**params)

        if name == "xgboost":
            if XGBRegressor is None:
                raise ImportError(
                    "xgboost is required because a selected model uses XGBoost."
                )
            params.pop("objective", None)
            params.setdefault("random_state", 42)
            params.setdefault("n_jobs", -1)
            return XGBRegressor(
                **params,
                objective="reg:squarederror",
            )

        if name == "svr":
            return SVR(**params)

        if name in {"knn", "knn regressor"}:
            params.setdefault("n_jobs", -1)
            return KNeighborsRegressor(**params)

        if name in {"mlp", "mlp regressor"}:
            params.setdefault("max_iter", 1000)
            params.setdefault("random_state", 42)
            return MLPRegressor(**params)

    if name == "multinomial logistic regression":
        return LogisticRegression(
            C=np.inf,
            class_weight=params.get("class_weight"),
            solver="lbfgs",
            max_iter=5000,
            random_state=42,
        )

    if name == "l2 multinomial logistic regression":
        return LogisticRegression(
            C=params["C"],
            l1_ratio=0,
            class_weight=params.get("class_weight"),
            solver="lbfgs",
            max_iter=5000,
            random_state=42,
        )

    if name == "l1 multinomial logistic regression":
        return LogisticRegression(
            C=params["C"],
            l1_ratio=1,
            class_weight=params.get("class_weight"),
            solver="saga",
            max_iter=5000,
            random_state=42,
        )

    if name == "elastic net multinomial logistic regression":
        return LogisticRegression(
            C=params["C"],
            l1_ratio=params["l1_ratio"],
            class_weight=params.get("class_weight"),
            solver="saga",
            max_iter=5000,
            random_state=42,
        )

    if name == "lda":
        return LinearDiscriminantAnalysis(**params)

    if name == "qda":
        return QuadraticDiscriminantAnalysis(**params)

    if name == "xgboost":
        if XGBClassifier is None:
            raise ImportError(
                "xgboost is required because a selected model uses XGBoost."
            )

        params.pop("class_weight", None)
        params.pop("objective", None)
        params.pop("eval_metric", None)
        params.setdefault("random_state", 42)
        params.setdefault("n_jobs", -1)

        if statistical_type == "binary":
            return XGBClassifier(
                **params,
                objective="binary:logistic",
                eval_metric="logloss",
            )

        if num_classes is None:
            raise ValueError("num_classes is required for multiclass XGBoost.")

        return XGBClassifier(
            **params,
            objective="multi:softprob",
            num_class=int(num_classes),
            eval_metric="mlogloss",
        )

    if name == "svm":
        params.setdefault("probability", True)
        params.setdefault("random_state", 42)
        return SVC(**params)

    if name in {"knn", "knn classifier"}:
        params.setdefault("n_jobs", -1)
        return KNeighborsClassifier(**params)

    if name in {"naive bayes", "gaussian naive bayes"}:
        return GaussianNB(**params)

    if name in {"mlp", "mlp classifier"}:
        params.setdefault("max_iter", 1000)
        params.setdefault("random_state", 42)
        return MLPClassifier(**params)

    raise ValueError(
        f"Unknown model '{model_name}' for target type '{statistical_type}'."
    )


########################################
# Fit / predict helpers
########################################

def _fit_target_model(
    training_df: pd.DataFrame,
    target: str,
    portfolio_type: str,
    model_name: str,
    parameters: Any,
    features: Sequence[str],
    statistical_type: Optional[str] = None,
) -> Dict[str, Any]:
    features = list(features)

    missing_columns = [
        column
        for column in features + [target]
        if column not in training_df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"{target} is missing training columns: "
            + ", ".join(missing_columns)
        )

    model_df = (
        training_df
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=features + [target])
        .copy()
    )

    if model_df.empty:
        raise ValueError(f"No valid training rows remain for {target}.")

    X_train = model_df[features].copy()
    y_train = model_df[target].copy()

    if statistical_type is None or pd.isna(statistical_type):
        statistical_type = infer_statistical_target_type(y_train)
    statistical_type = str(statistical_type).lower().strip()

    if statistical_type == "binary":
        y_train = _clean_binary_target(y_train, target)

    if statistical_type != "continuous" and y_train.nunique() < 2:
        raise ValueError(
            f"{target} has fewer than two classes in the training data."
        )

    num_classes = (
        int(y_train.nunique())
        if statistical_type != "continuous"
        else None
    )

    model = build_model(
        model_name=model_name,
        statistical_type=statistical_type,
        parameters=parameters,
        num_classes=num_classes,
    )

    scaler = None
    model_key = str(model_name).lower().strip()
    if model_key in SCALE_MODELS:
        scaler = StandardScaler()
        X_train_model = scaler.fit_transform(X_train)
    else:
        X_train_model = X_train

    class_values = None
    label_encoder = None
    is_xgboost_classifier = (
        statistical_type != "continuous"
        and model_key == "xgboost"
    )

    if is_xgboost_classifier:
        label_encoder = LabelEncoder()
        y_fit = label_encoder.fit_transform(y_train)

        class_weight = parse_parameters(parameters).get("class_weight")
        sample_weight = None
        if class_weight is not None:
            sample_weight = compute_sample_weight(
                class_weight=class_weight,
                y=y_train,
            )

        model.fit(
            X_train_model,
            y_fit,
            sample_weight=sample_weight,
        )
        class_values = label_encoder.classes_.copy()

    else:
        model.fit(X_train_model, y_train)

        if statistical_type != "continuous":
            if not hasattr(model, "classes_"):
                raise ValueError(
                    f"{model_name} does not expose classes_ for {target}."
                )
            class_values = np.asarray(model.classes_).copy()

    # Used to orient continuous risk targets consistently. If the target is a
    # negative loss/drawdown quantity, larger raw values are actually safer, so
    # flip the prediction before ranking it as risk intensity.
    target_median = float(pd.to_numeric(y_train, errors="coerce").median())
    continuous_orientation = 1.0

    if statistical_type == "continuous" and portfolio_type in RISK_TYPES:
        name = target.lower()
        if "minimum return" in name or "min return" in name:
            continuous_orientation = -1.0
        elif target_median < 0:
            continuous_orientation = -1.0

    return {
        "model": model,
        "scaler": scaler,
        "features": features,
        "target": target,
        "portfolio_type": portfolio_type,
        "statistical_type": statistical_type,
        "model_name": model_name,
        "parameters": parameters,
        "training_rows": len(model_df),
        "class_values": class_values,
        "label_encoder": label_encoder,
        "continuous_orientation": continuous_orientation,
    }


def _predict_target_model(
    model_info: Mapping[str, Any],
    dataframe: pd.DataFrame,
) -> pd.Series:
    """Predict wherever that target's features are available; otherwise NaN."""

    features = list(model_info["features"])
    missing = [column for column in features if column not in dataframe.columns]
    if missing:
        raise ValueError(
            f"{model_info['target']} is missing backtest features: "
            + ", ".join(missing)
        )

    X = dataframe[features].copy().replace([np.inf, -np.inf], np.nan)
    valid = ~X.isna().any(axis=1)

    predictions = pd.Series(np.nan, index=dataframe.index, dtype=float)
    if not valid.any():
        return predictions

    X_valid = X.loc[valid]
    if model_info["scaler"] is not None:
        X_model = model_info["scaler"].transform(X_valid)
    else:
        X_model = X_valid

    if model_info["statistical_type"] == "continuous":
        predicted = model_info["model"].predict(X_model)
        predictions.loc[valid] = np.asarray(predicted, dtype=float)
        return predictions

    model = model_info["model"]
    if not hasattr(model, "predict_proba"):
        raise ValueError(
            f"{model_info['model_name']} does not support predict_proba "
            f"for {model_info['target']}."
        )

    probabilities = model.predict_proba(X_model)
    class_values = np.asarray(model_info["class_values"])

    try:
        numeric_classes = class_values.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Classes for {model_info['target']} must be numeric. "
            f"Got {class_values.tolist()}."
        ) from error

    if probabilities.shape[1] != len(numeric_classes):
        raise ValueError(
            f"Probability columns for {model_info['target']} do not match "
            "the stored class values."
        )

    expected_class = probabilities @ numeric_classes
    predictions.loc[valid] = expected_class
    return predictions


########################################
# Signal / horizon helpers
########################################

def _rank_to_minus_one_one(series: pd.Series) -> pd.Series:
    """Cross-sectional rank transform with exact [-1, 1] endpoints."""

    result = pd.Series(np.nan, index=series.index, dtype=float)
    valid = series.dropna()
    n = len(valid)

    if n == 0:
        return result
    if n == 1 or valid.nunique() == 1:
        result.loc[valid.index] = 0.0
        return result

    ranks = valid.rank(method="average")
    result.loc[valid.index] = 2.0 * (ranks - 1.0) / (n - 1.0) - 1.0
    return result


def _classification_signal(
    prediction: pd.Series,
    class_values: Sequence[Any],
) -> pd.Series:
    """Map expected class value to a common [-1, 1] scale."""

    classes = np.asarray(class_values, dtype=float)
    lower = float(np.min(classes))
    upper = float(np.max(classes))

    if upper == lower:
        return pd.Series(0.0, index=prediction.index)

    signal = 2.0 * (prediction - lower) / (upper - lower) - 1.0
    return signal.clip(-1.0, 1.0)


def _infer_horizon_key(target: str, horizon: Any) -> str:
    """
    Return keys such as 5m, 1h, 20d.

    The clean metadata table stores only the numeric Horizon. The target name
    retains the unit for intraday targets, so it is used to distinguish 5m from
    5d. A unitless target is treated as daily, matching the existing daily
    targets.py convention.
    """

    if pd.isna(horizon):
        return ""

    value = float(horizon)
    number = str(int(value)) if value.is_integer() else str(value)
    name = str(target).lower()

    explicit_minutes = re.search(
        rf"(?<![a-z0-9]){re.escape(number)}\s*(m|min|mins|minute|minutes)(?![a-z])",
        name,
    )
    if explicit_minutes:
        return f"{number}m"

    explicit_hours = re.search(
        rf"(?<![a-z0-9]){re.escape(number)}\s*(h|hr|hrs|hour|hours)(?![a-z])",
        name,
    )
    if explicit_hours:
        return f"{number}h"

    return f"{number}d"


def _lookup_horizon_score(
    target: str,
    portfolio_type: str,
    horizon: Any,
    row: pd.Series,
    horizon_scores: Optional[Mapping[str, Any]],
) -> float:
    """Resolve an optimized horizon score with several convenient formats."""

    if "Horizon Score" in row.index and pd.notna(row["Horizon Score"]):
        return float(row["Horizon Score"])

    if horizon_scores is None:
        return 1.0

    # Exact per-target override.
    if target in horizon_scores and np.isscalar(horizon_scores[target]):
        return float(horizon_scores[target])

    type_map = horizon_scores.get(portfolio_type)
    if type_map is None:
        type_map = horizon_scores.get(str(portfolio_type).upper())

    if type_map is None:
        return 1.0

    if np.isscalar(type_map):
        return float(type_map)

    key = _infer_horizon_key(target, horizon)
    candidates: Iterable[Any] = (
        key,
        str(horizon),
        int(horizon) if pd.notna(horizon) and float(horizon).is_integer() else horizon,
    )

    for candidate in candidates:
        if candidate in type_map:
            return float(type_map[candidate])

    return 1.0


########################################
# Portfolio optimiser
########################################

def _construct_portfolio(
    date_predictions: pd.DataFrame,
    max_weight: float,
    concentration_penalty: float,
) -> pd.DataFrame:
    portfolio = date_predictions[["Ticker", "Date", "Overall Score"]].copy()
    portfolio = portfolio.dropna(subset=["Overall Score"])

    n_stocks = len(portfolio)
    if n_stocks < 2:
        raise ValueError("At least two valid stocks are required.")

    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must be between 0 and 1.")

    if max_weight * n_stocks < 1:
        raise ValueError(
            f"max_weight={max_weight:.2%} is impossible with "
            f"only {n_stocks} valid stocks."
        )

    score = portfolio["Overall Score"].to_numpy(dtype=float)

    def objective(weights):
        return -(
            np.dot(weights, score)
            - concentration_penalty * np.sum(weights ** 2)
        )

    constraints = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1.0,
    }
    bounds = [(0.0, max_weight) for _ in range(n_stocks)]
    starting_weights = np.ones(n_stocks) / n_stocks

    result = minimize(
        objective,
        starting_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise ValueError(
            "Portfolio optimisation failed: " + str(result.message)
        )

    portfolio["Recommended Weight"] = result.x
    return portfolio.sort_values(
        "Recommended Weight",
        ascending=False,
    ).reset_index(drop=True)


########################################
# Public function
########################################

def run_multi_target_portfolio_backtest(
    dataframe: pd.DataFrame,
    selected_models_df: pd.DataFrame,
    model_features: Optional[
        Union[pd.DataFrame, Mapping[str, Sequence[str]]]
    ] = None,
    horizon_scores: Optional[Mapping[str, Any]] = None,
    type_values: Optional[Mapping[str, float]] = None,
    rebalance_every: int = 1,
    max_weight: float = 0.30,
    concentration_penalty: float = 0.10,
    trading_fee: float = 0.0,
    annualisation: int = 252,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Train every available target model, generate row-level contributions,
    construct stock weights, run the backtest, and return strategy metrics.

    Parameters
    ----------
    dataframe:
        One dataframe containing both TRAIN and BACKTEST rows. Expected base
        columns are Date, Ticker, Close, Return and Split, plus all selected
        model features and target columns.

    selected_models_df:
        Metadata with at least:

            Target
            Model
            Parameters
            Target Type
            Horizon
            Quality Score

        Target Type is the portfolio type (ALPHA, TAIL_RISK, etc.). If an
        optional "Statistical Type" column exists it is used; otherwise the
        function infers continuous/binary/multiclass from TRAIN target values.

    model_features:
        Either the separate "Selected Model Features {STOCK_TYPE}" dataframe
        with Target / Features columns, or a Target -> list-of-features mapping.
        It may be omitted if selected_models_df already contains Features.

    horizon_scores:
        Optimized horizon scores. Recommended shape::

            {
                "ALPHA": {"20d": 0.85, "60d": 1.00},
                "TAIL_EVENT": {"5m": 1.00, "1h": 0.90},
            }

        Exact target-name overrides are also accepted. If omitted, every model
        receives horizon score 1.0.

    type_values:
        Signed importance of each portfolio target type. Missing types fall
        back to DEFAULT_TYPE_VALUES.

    rebalance_every:
        Number of unique BACKTEST dates between portfolio optimizations. Weight
        decisions made on date t are applied to return t+1 to avoid lookahead.

    Returns
    -------
    dict with:
        results              -> requested strategy summary metrics
        backtest             -> daily portfolio equity / drawdown dataframe
        row_predictions      -> BACKTEST rows with Prediction / Signal /
                                Contribution for every fitted target plus
                                type scores and Recommended/Held Weight
        rebalance_weights    -> wide target weights on rebalance dates
        weights              -> daily forward-filled target weights
        fitted_models        -> fitted model/scaler metadata by target
        model_summary        -> target/model/type/horizon/quality metadata
        skipped_models       -> models that could not be used when strict=False
    """

    required_data_columns = {"Date", "Ticker", "Close", "Split"}
    missing = required_data_columns.difference(dataframe.columns)
    if missing:
        raise ValueError(
            "dataframe is missing required columns: "
            + ", ".join(sorted(missing))
        )

    required_model_columns = {
        "Target",
        "Model",
        "Parameters",
        "Target Type",
        "Horizon",
        "Quality Score",
    }
    missing = required_model_columns.difference(selected_models_df.columns)
    if missing:
        raise ValueError(
            "selected_models_df is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if rebalance_every <= 0:
        raise ValueError("rebalance_every must be greater than zero.")

    data = dataframe.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    if "Return" not in data.columns:
        data["Return"] = (
            data.groupby("Ticker", sort=False)["Close"].pct_change()
        )

    split = data["Split"].astype(str).str.upper().str.strip()
    training_df = data.loc[split == "TRAIN"].copy()
    backtest_df = data.loc[split == "BACKTEST"].copy()

    if training_df.empty:
        raise ValueError("No TRAIN rows are present in dataframe.")
    if backtest_df.empty:
        raise ValueError("No BACKTEST rows are present in dataframe.")

    feature_map = _feature_map(selected_models_df, model_features)
    type_value_map = DEFAULT_TYPE_VALUES.copy()
    if type_values is not None:
        type_value_map.update(
            {str(k).upper(): float(v) for k, v in type_values.items()}
        )

    predictions_df = backtest_df.copy()
    fitted_models: Dict[str, Dict[str, Any]] = {}
    skipped_models = []
    model_rows = []

    # One production model per target. If duplicates are supplied, use the
    # highest-quality row, which matches the selected-model build logic.
    models = selected_models_df.copy()
    models["Quality Score"] = pd.to_numeric(
        models["Quality Score"],
        errors="coerce",
    )
    models = (
        models.sort_values("Quality Score", ascending=False)
        .drop_duplicates(subset=["Target"], keep="first")
        .reset_index(drop=True)
    )

    ####################################
    # Fit every available target model
    ####################################

    for _, row in models.iterrows():
        target = str(row["Target"])
        portfolio_type = str(row["Target Type"]).upper().strip()
        model_name = str(row["Model"])
        parameters = row["Parameters"]
        horizon = row["Horizon"]
        quality = float(row["Quality Score"])
        quality = float(np.clip(quality, 0.0, 1.0))

        features = list(feature_map.get(target, []))
        if not features:
            message = f"No feature list is available for {target}."
            if strict:
                raise ValueError(message)
            logger.warning(message)
            skipped_models.append({"Target": target, "Reason": message})
            continue

        if target not in training_df.columns:
            message = f"Training target column is missing: {target}."
            if strict:
                raise ValueError(message)
            logger.warning(message)
            skipped_models.append({"Target": target, "Reason": message})
            continue

        statistical_type = None
        if "Statistical Type" in row.index and pd.notna(row["Statistical Type"]):
            statistical_type = str(row["Statistical Type"])

        try:
            model_info = _fit_target_model(
                training_df=training_df,
                target=target,
                portfolio_type=portfolio_type,
                model_name=model_name,
                parameters=parameters,
                features=features,
                statistical_type=statistical_type,
            )

            prediction = _predict_target_model(
                model_info,
                predictions_df,
            )

        except Exception as error:
            if strict:
                raise
            logger.exception("Skipping %s: %s", target, error)
            skipped_models.append(
                {"Target": target, "Reason": str(error)}
            )
            continue

        prediction_column = f"{target} Prediction"
        signal_column = f"{target} Signal"
        contribution_column = f"{target} Contribution"

        predictions_df[prediction_column] = prediction

        if model_info["statistical_type"] == "continuous":
            oriented_prediction = (
                predictions_df[prediction_column]
                * model_info["continuous_orientation"]
            )

            predictions_df[signal_column] = (
                oriented_prediction.groupby(
                    predictions_df["Date"],
                    group_keys=False,
                ).apply(_rank_to_minus_one_one)
            )

        else:
            predictions_df[signal_column] = _classification_signal(
                predictions_df[prediction_column],
                model_info["class_values"],
            )

        horizon_score = _lookup_horizon_score(
            target=target,
            portfolio_type=portfolio_type,
            horizon=horizon,
            row=row,
            horizon_scores=horizon_scores,
        )

        model_weight = float(horizon_score) * quality
        predictions_df[contribution_column] = (
            predictions_df[signal_column] * model_weight
        )

        model_info.update(
            {
                "quality_score": quality,
                "horizon": horizon,
                "horizon_score": float(horizon_score),
                "model_weight": model_weight,
                "prediction_column": prediction_column,
                "signal_column": signal_column,
                "contribution_column": contribution_column,
            }
        )
        fitted_models[target] = model_info

        model_rows.append(
            {
                "Target": target,
                "Model": model_name,
                "Parameters": parameters,
                "Target Type": portfolio_type,
                "Statistical Type": model_info["statistical_type"],
                "Horizon": horizon,
                "Horizon Score": float(horizon_score),
                "Quality Score": quality,
                "Model Weight": model_weight,
                "Training Rows": model_info["training_rows"],
            }
        )

    if not fitted_models:
        raise ValueError("No selected target model could be fitted.")

    model_summary = pd.DataFrame(model_rows)

    ####################################
    # Combine targets within each type
    #
    # Type Score = sum(Signal * H * Q) / sum(H * Q)
    #
    # The type value is applied ONCE, preventing target-count bias.
    ####################################

    types_used = sorted(
        model_summary["Target Type"].dropna().unique().tolist()
    )

    available_type_columns = []

    for portfolio_type in types_used:
        targets = model_summary.loc[
            model_summary["Target Type"] == portfolio_type,
            "Target",
        ].tolist()

        numerator = pd.Series(0.0, index=predictions_df.index)
        denominator = pd.Series(0.0, index=predictions_df.index)

        for target in targets:
            info = fitted_models[target]
            contribution = predictions_df[info["contribution_column"]]
            signal = predictions_df[info["signal_column"]]
            weight = float(info["model_weight"])

            numerator = numerator.add(
                contribution.fillna(0.0),
                fill_value=0.0,
            )
            denominator = denominator.add(
                signal.notna().astype(float) * weight,
                fill_value=0.0,
            )

        type_score_column = f"{portfolio_type} Type Score"
        type_contribution_column = f"{portfolio_type} Type Contribution"

        predictions_df[type_score_column] = np.where(
            denominator > 0,
            numerator / denominator,
            np.nan,
        )

        type_value = float(type_value_map.get(portfolio_type, 1.0))
        predictions_df[type_contribution_column] = (
            predictions_df[type_score_column] * type_value
        )

        available_type_columns.append(
            (
                type_score_column,
                type_contribution_column,
                abs(type_value),
            )
        )

    # Keep Overall Score on an approximately comparable scale even when a row
    # is missing one target family.
    overall_numerator = pd.Series(0.0, index=predictions_df.index)
    overall_denominator = pd.Series(0.0, index=predictions_df.index)

    for score_col, contribution_col, abs_type_value in available_type_columns:
        valid = predictions_df[score_col].notna()
        overall_numerator += predictions_df[contribution_col].fillna(0.0)
        overall_denominator += valid.astype(float) * abs_type_value

    predictions_df["Overall Score"] = np.where(
        overall_denominator > 0,
        overall_numerator / overall_denominator,
        np.nan,
    )

    ####################################
    # Portfolio weights
    ####################################

    backtest_dates = (
        predictions_df["Date"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    test_tickers = sorted(
        predictions_df["Ticker"].dropna().unique().tolist()
    )
    rebalance_dates = backtest_dates.iloc[::rebalance_every]

    historical_weights = []
    skipped_rebalances = 0

    for date in rebalance_dates:
        date_predictions = predictions_df.loc[
            predictions_df["Date"] == date,
            ["Ticker", "Date", "Overall Score"],
        ].dropna(subset=["Overall Score"])

        if len(date_predictions) < 2:
            skipped_rebalances += 1
            continue

        if max_weight * len(date_predictions) < 1:
            skipped_rebalances += 1
            continue

        portfolio = _construct_portfolio(
            date_predictions=date_predictions,
            max_weight=max_weight,
            concentration_penalty=concentration_penalty,
        )

        historical_weights.extend(
            portfolio[["Date", "Ticker", "Recommended Weight"]]
            .to_dict("records")
        )

    if not historical_weights:
        raise ValueError("No valid rebalance portfolio could be created.")

    weights_long = pd.DataFrame(historical_weights)
    rebalance_weights = (
        weights_long.pivot(
            index="Date",
            columns="Ticker",
            values="Recommended Weight",
        )
        .reindex(columns=test_tickers)
        .fillna(0.0)
    )

    weights_df = (
        rebalance_weights
        .reindex(backtest_dates)
        .ffill()
        .fillna(0.0)
    )

    # Today's recommendation is used from the next row/date onward, exactly as
    # in the supplied backtest's held_weights = weights_df.shift(1).
    held_weights = weights_df.shift(1).fillna(0.0)

    ####################################
    # Row-level recommended / held weights
    ####################################

    recommended_long = (
        weights_df
        .rename_axis("Ticker", axis=1)
        .reset_index()
        .melt(
            id_vars="Date",
            var_name="Ticker",
            value_name="Recommended Weight",
        )
    )
    held_long = (
        held_weights
        .rename_axis("Ticker", axis=1)
        .reset_index()
        .melt(
            id_vars="Date",
            var_name="Ticker",
            value_name="Held Weight",
        )
    )

    predictions_df = predictions_df.merge(
        recommended_long,
        on=["Date", "Ticker"],
        how="left",
    )
    predictions_df = predictions_df.merge(
        held_long,
        on=["Date", "Ticker"],
        how="left",
    )

    ####################################
    # Backtest returns
    ####################################

    returns_df = (
        predictions_df.pivot_table(
            index="Date",
            columns="Ticker",
            values="Return",
            aggfunc="first",
        )
        .reindex(index=backtest_dates, columns=test_tickers)
    )

    strategy_contributions = held_weights * returns_df.fillna(0.0)

    turnover = (
        held_weights.diff().abs().sum(axis=1).fillna(0.0)
    )
    trading_cost = turnover * float(trading_fee)
    strategy_return_series = (
        strategy_contributions.sum(axis=1) - trading_cost
    )

    active = held_weights.sum(axis=1) > 0
    if not active.any():
        raise ValueError("No portfolio became active.")

    strategy_start = active[active].index[0]

    backtest = pd.DataFrame(index=backtest_dates)
    backtest["Strategy_Return"] = strategy_return_series
    backtest = backtest.loc[strategy_start:].copy()
    backtest["Strategy"] = (1.0 + backtest["Strategy_Return"]).cumprod()

    backtest["Strategy_Peak"] = backtest["Strategy"].cummax()
    backtest["Strategy_Drawdown"] = (
        backtest["Strategy"] / backtest["Strategy_Peak"] - 1.0
    )
    backtest["Turnover"] = turnover.reindex(backtest.index).fillna(0.0)
    backtest["Trading_Cost"] = trading_cost.reindex(backtest.index).fillna(0.0)

    strategy_total_return = float(backtest["Strategy"].iloc[-1] - 1.0)
    average_drawdown = float(backtest["Strategy_Drawdown"].mean())
    maximum_drawdown = float(backtest["Strategy_Drawdown"].min())

    returns = backtest["Strategy_Return"].dropna()
    daily_std = float(returns.std())
    sharpe_ratio = (
        float(returns.mean() / daily_std * np.sqrt(annualisation))
        if daily_std > 0
        else np.nan
    )

    results = {
        "Strategy Return": strategy_total_return,
        "Average Drawdown": average_drawdown,
        "Max Drawdown": maximum_drawdown,
        "Sharpe Ratio": sharpe_ratio,
        "Rebalance Days": int(rebalance_every),
        "Max Weight": float(max_weight),
        "Concentration Penalty": float(concentration_penalty),
        "Trading Fee": float(trading_fee),
        "Fitted Targets": int(len(fitted_models)),
        "Skipped Targets": int(len(skipped_models)),
        "Skipped Rebalances": int(skipped_rebalances),
    }

    return {
        "results": results,
        "backtest": backtest,
        "row_predictions": predictions_df,
        "rebalance_weights": rebalance_weights,
        "weights": weights_df,
        "held_weights": held_weights,
        "fitted_models": fitted_models,
        "model_summary": model_summary,
        "skipped_models": pd.DataFrame(skipped_models),
    }

######################################################################
# Refactored public API: fit once -> predict once -> backtest many times
######################################################################


def _prepare_selected_models(selected_models_df):
    required = {
        "Target", "Model", "Parameters", "Target Type",
        "Horizon", "Horizon Score", "Quality Score",
    }
    missing = required.difference(selected_models_df.columns)
    if missing:
        raise ValueError(
            "selected_models_df is missing columns: "
            + ", ".join(sorted(missing))
        )

    models = selected_models_df.copy()
    models["Quality Score"] = pd.to_numeric(
        models["Quality Score"], errors="coerce"
    ).clip(0.0, 1.0)
    models["Horizon"] = pd.to_numeric(
        models["Horizon"], errors="coerce"
    )
    models["Horizon Score"] = pd.to_numeric(
        models["Horizon Score"], errors="coerce"
    ).clip(0.0, 1.0)

    models = models.dropna(
        subset=[
            "Target", "Model", "Target Type",
            "Horizon", "Horizon Score", "Quality Score",
        ]
    )

    return (
        models.sort_values("Quality Score", ascending=False)
        .drop_duplicates(subset=["Target"], keep="first")
        .reset_index(drop=True)
    )


def create_target_models(
    training_df,
    selected_models_df,
    model_features=None,
    strict=False,
):
    """
    Fit every available target model ONCE.

    Returns a dictionary containing fitted models, a model summary,
    and any skipped targets.
    """
    models_metadata = _prepare_selected_models(selected_models_df)
    feature_map = _feature_map(models_metadata, model_features)

    fitted_models = {}
    skipped_models = []
    summary_rows = []

    for _, row in models_metadata.iterrows():
        target = str(row["Target"])
        model_name = str(row["Model"])
        portfolio_type = str(row["Target Type"]).upper().strip()
        features = list(feature_map.get(target, []))

        if not features:
            message = f"No feature list is available for {target}."
            if strict:
                raise ValueError(message)
            skipped_models.append({"Target": target, "Reason": message})
            continue

        if target not in training_df.columns:
            message = f"Training target column is missing: {target}."
            if strict:
                raise ValueError(message)
            skipped_models.append({"Target": target, "Reason": message})
            continue

        statistical_type = None
        if (
            "Statistical Type" in row.index
            and pd.notna(row["Statistical Type"])
        ):
            statistical_type = str(row["Statistical Type"])

        try:
            model_info = _fit_target_model(
                training_df=training_df,
                target=target,
                portfolio_type=portfolio_type,
                model_name=model_name,
                parameters=row["Parameters"],
                features=features,
                statistical_type=statistical_type,
            )
        except Exception as error:
            if strict:
                raise
            skipped_models.append({"Target": target, "Reason": str(error)})
            continue

        model_info["horizon"] = float(row["Horizon"])
        model_info["horizon_score"] = float(row["Horizon Score"])
        model_info["quality_score"] = float(row["Quality Score"])
        fitted_models[target] = model_info

        summary_rows.append({
            "Target": target,
            "Model": model_name,
            "Parameters": row["Parameters"],
            "Target Type": portfolio_type,
            "Statistical Type": model_info["statistical_type"],
            "Horizon": float(row["Horizon"]),
            "Horizon Score": float(row["Horizon Score"]),
            "Quality Score": float(row["Quality Score"]),
            "Training Rows": int(model_info["training_rows"]),
        })

    if not fitted_models:
        raise ValueError("No target model could be fitted.")

    return {
        "models": fitted_models,
        "model_summary": pd.DataFrame(summary_rows),
        "skipped_models": pd.DataFrame(skipped_models),
    }


def create_prediction_dataframe(backtest_df, fitted_models):
    """
    Generate all BACKTEST predictions ONCE.

    The result is a long dataframe that can be cached directly in SQLite.
    It already contains Horizon, Horizon Score and Quality Score.
    """
    required = {"Date", "Ticker", "Close"}
    missing = required.difference(backtest_df.columns)
    if missing:
        raise ValueError(
            "backtest_df is missing columns: "
            + ", ".join(sorted(missing))
        )

    data = backtest_df.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    if "Return" not in data.columns:
        data["Return"] = (
            data.groupby("Ticker", sort=False)["Close"].pct_change()
        )

    parts = []

    for target, model_info in fitted_models.items():
        prediction = _predict_target_model(model_info, data)

        if model_info["statistical_type"] == "continuous":
            oriented = (
                prediction
                * float(model_info["continuous_orientation"])
            )
            signal = oriented.groupby(
                data["Date"], group_keys=False
            ).apply(_rank_to_minus_one_one)
        else:
            signal = _classification_signal(
                prediction,
                model_info["class_values"],
            )

        part = data[["Date", "Ticker", "Close", "Return"]].copy()
        part["Target"] = target
        part["Prediction"] = prediction.to_numpy()
        part["Signal"] = signal.to_numpy()
        part["Target Type"] = model_info["portfolio_type"]
        part["Statistical Type"] = model_info["statistical_type"]
        part["Horizon"] = float(model_info["horizon"])
        part["Horizon Score"] = float(model_info["horizon_score"])
        part["Quality Score"] = float(model_info["quality_score"])
        parts.append(part)

    if not parts:
        raise ValueError("No predictions were generated.")

    return pd.concat(parts, ignore_index=True)


def create_models_and_predictions(
    dataframe,
    selected_models_df,
    model_features=None,
    strict=False,
):
    """
    Convenience wrapper: split TRAIN/BACKTEST, fit once, predict once.
    """
    required = {"Split", "Date", "Ticker", "Close"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(
            "dataframe is missing columns: "
            + ", ".join(sorted(missing))
        )

    data = dataframe.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    split = data["Split"].astype(str).str.upper().str.strip()
    training_df = data.loc[split == "TRAIN"].copy()
    backtest_df = data.loc[split == "BACKTEST"].copy()

    if training_df.empty:
        raise ValueError("No TRAIN rows are present.")
    if backtest_df.empty:
        raise ValueError("No BACKTEST rows are present.")

    fitted = create_target_models(
        training_df=training_df,
        selected_models_df=selected_models_df,
        model_features=model_features,
        strict=strict,
    )

    predictions = create_prediction_dataframe(
        backtest_df=backtest_df,
        fitted_models=fitted["models"],
    )

    return {
        "models": fitted["models"],
        "model_summary": fitted["model_summary"],
        "skipped_models": fitted["skipped_models"],
        "predictions": predictions,
    }


def portfolio_returns_from_scores(
    dataframe,
    rebalance_every=1,
    max_weight = 0.30,
    concentration_penalty = 0.10,
    trading_fee = 0.00
):

    ########################################
    # Validate
    ########################################

    required_columns = {
        "Date",
        "Ticker",
        "Return",
        "Stock_Score",
    }

    missing_columns = (
        required_columns
        .difference(
            dataframe.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Missing columns: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    if rebalance_every < 1:
        raise ValueError(
            "rebalance_every must be at least 1."
        )


    ########################################
    # Clean
    ########################################

    data = dataframe[
        [
            "Date",
            "Ticker",
            "Return",
            "Stock_Score",
        ]
    ].copy()

    data[
        "Date"
    ] = pd.to_datetime(
        data[
            "Date"
        ],
        errors="coerce",
    )

    data[
        "Return"
    ] = (
        pd.to_numeric(
            data[
                "Return"
            ],
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    data[
        "Stock_Score"
    ] = (
        pd.to_numeric(
            data[
                "Stock_Score"
            ],
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(
            0.0
        )
        .clip(
            lower=0.0
        )
    )

    data = (
        data
        .dropna(
            subset=[
                "Date",
                "Ticker",
                "Return",
            ]
        )
        .sort_values(
            [
                "Date",
                "Ticker",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    ########################################
    # Weight Calculation
    ########################################

    def calculate_weights(
        daily_data,
    ):

        scores = (
            daily_data
            .set_index(
                "Ticker"
            )[
                "Stock_Score"
            ]
        )

        active = (
            scores > 0
        )

        if not active.any():
            return pd.Series(
                dtype=float
            )


        score_weights = (
            scores
            / scores.sum()
        )

        equal_weights = (
            active.astype(float)
            / active.sum()
        )


        # Blend mostly score-proportional
        # weights with 10% equal weighting.
        desired_weights = (
            (
                1.0
                - concentration_penalty
            )
            * score_weights
            +
            concentration_penalty
            * equal_weights
        )


        ####################################
        # Enforce Maximum Weight
        ####################################

        final_weights = pd.Series(
            0.0,
            index=desired_weights.index,
        )

        remaining_tickers = list(
            desired_weights.index[
                active
            ]
        )

        remaining_capital = min(
            1.0,
            len(
                remaining_tickers
            )
            * max_weight,
        )


        while (
            remaining_tickers
            and remaining_capital > 1e-12
        ):

            remaining_scores = (
                desired_weights.loc[
                    remaining_tickers
                ]
            )

            proposed = (
                remaining_scores
                / remaining_scores.sum()
                * remaining_capital
            )

            over_cap = (
                proposed > max_weight
            )

            if not over_cap.any():

                final_weights.loc[
                    remaining_tickers
                ] = proposed

                break


            capped_tickers = list(
                proposed.index[
                    over_cap
                ]
            )

            final_weights.loc[
                capped_tickers
            ] = max_weight

            remaining_capital -= (
                max_weight
                * len(
                    capped_tickers
                )
            )

            remaining_tickers = [
                ticker
                for ticker in remaining_tickers
                if ticker not in capped_tickers
            ]


        return final_weights[
            final_weights > 0
        ]


    ########################################
    # Run Through Dates
    ########################################

    dates = (
        data[
            "Date"
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    current_weights = pd.Series(
        dtype=float
    )

    previous_weights = pd.Series(
        dtype=float
    )

    return_records = []


    for date_number, date in enumerate(
        dates
    ):

        daily_data = (
            data[
                data[
                    "Date"
                ].eq(
                    date
                )
            ]
            .copy()
        )


        ####################################
        # Rebalance Only Every X Dates
        ####################################

        if (
            date_number
            % rebalance_every
            == 0
        ):

            current_weights = (
                calculate_weights(
                    daily_data
                )
            )


            all_tickers = (
                previous_weights.index
                .union(
                    current_weights.index
                )
            )

            old_weights = (
                previous_weights
                .reindex(
                    all_tickers,
                    fill_value=0.0,
                )
            )

            new_weights = (
                current_weights
                .reindex(
                    all_tickers,
                    fill_value=0.0,
                )
            )

            turnover = (
                0.5
                * (
                    new_weights
                    - old_weights
                )
                .abs()
                .sum()
            )

            previous_weights = (
                current_weights.copy()
            )

        else:

            # Continue using the portfolio
            # selected on the last rebalance.
            turnover = 0.0


        ####################################
        # Apply Held Weights
        ####################################

        daily_returns = (
            daily_data
            .set_index(
                "Ticker"
            )[
                "Return"
            ]
        )

        aligned_returns = (
            daily_returns
            .reindex(
                current_weights.index
            )
            .fillna(
                0.0
            )
        )

        gross_return = float(
            (
                current_weights
                * aligned_returns
            ).sum()
        )

        net_return = (
            gross_return
            - turnover
            * trading_fee
        )


        return_records.append(
            {
                "Date": date,
                "Return": net_return,
            }
        )


    return pd.DataFrame(
        return_records
    )

def run_portfolio_backtest_from_predictions(
    predictions_df,
    type_values=None,
    rebalance_every=1,
    max_weight=0.30,
    concentration_penalty=0.10,
    trading_fee=0.0,
    annualisation=252,
):
    """
    Run the portfolio backtest using cached predictions only.

    NO model fitting and NO prediction generation happen here.

    To test another horizon-score combination, change the
    ``Horizon Score`` column in predictions_df and call again.
    """
    required = {
        "Date", "Ticker", "Return", "Signal",
        "Horizon Score"
    }
    missing = required.difference(predictions_df.columns)
    if missing:
        raise ValueError(
            "predictions_df is missing columns: "
            + ", ".join(sorted(missing))
        )

    if rebalance_every <= 0:
        raise ValueError("rebalance_every must be greater than zero.")

    predictions = predictions_df.copy()
    predictions["Date"] = pd.to_datetime(predictions["Date"])
    predictions["Signal"] = pd.to_numeric(
        predictions["Signal"], errors="coerce"
    )
    predictions["Horizon Score"] = pd.to_numeric(
        predictions["Horizon Score"], errors="coerce"
    ).clip(0.0, 1.0)


    predictions["Contribution"] = (
        predictions["Signal"]
        * predictions["Horizon Score"]
    )

    type_value_map = DEFAULT_TYPE_VALUES.copy()
    if type_values is not None:
        type_value_map.update({
            str(key).upper().strip(): float(value)
            for key, value in type_values.items()
        })

    valid = predictions["Signal"].notna()

    predictions = (
        predictions.loc[valid]
        .groupby(["Date", "Ticker", "Portfolio Target Type"], as_index=False)
        .agg(
            Contribution_Sum=("Contribution", "sum"),
            Return=("Return", "first")
        )
    )

    BASE_TYPE_SCORES = pd.Series(
        {
            "ALPHA": 0.55,
            "RELATIVE_ALPHA": 0.55,
            "RISK_ADJUSTED_ALPHA": 0.60,
            "CROSS_SECTION_ALPHA": 0.60,
            "CROSS_SECTION_DOWNSIDE": 0.55,

            "DIRECTION": 0.55,
            "DIRECTION_MULTICLASS": 0.50,
            "ALPHA_BINARY": 0.50,
            "BARRIER_ALPHA": 0.50,

            "VOLATILITY": 0.55,
            "ABSOLUTE_MOVE": 0.50,
            "UPSIDE_VOLATILITY": 0.45,
            "DOWNSIDE_VOLATILITY": 0.55,
            "VOLATILITY_ASYMMETRY": 0.50,
            "VOLATILITY_EVENT": 0.50,

            "DOWNSIDE": 0.60,
            "TAIL_RISK": 0.60,
            "TAIL_EVENT": 0.55,
            "UPSIDE_RISK": 0.45,
            "UPSIDE_EVENT": 0.45,

            "UPSIDE_EXCURSION": 0.50,
            "DOWNSIDE_EXCURSION": 0.55,
            "TIME_TO_UPSIDE_EXCURSION": 0.45,
            "TIME_TO_DOWNSIDE_EXCURSION": 0.50,

            "RECOVERY": 0.50,
            "REVERSAL": 0.50,
            "REGIME": 0.55,
            "CORRELATION": 0.50,
            "COVARIANCE": 0.50,

            "LIQUIDITY": 0.50,
            "EXECUTION": 0.50,
            "MARKET_IMPACT": 0.50,
        },
        name="Base Type Score",
        dtype=float,
    )


    BASE_TYPE_SCORES.index.name = (
        "Portfolio Target Type"
    )


    predictions = predictions.merge(
        BASE_TYPE_SCORES,
        how="left",
        left_on="Portfolio Target Type",
        right_index=True,
        validate="many_to_one",
    )

    predictions["Type Score"] = predictions["Contribution_Sum"] * predictions["Base Type Score"]

    predictions = (
        predictions.loc[valid]
        .groupby(["Date", "Ticker"], as_index=False)
        .agg(
            Stock_Score=("Type Score", "sum"),
            Return=("Return", "first")
        )
    )

    backtest = portfolio_returns_from_scores(predictions)

    # Calculate cumulative strategy returns
    backtest["Strategy Return"] = (
        1 + backtest["Return"]
    ).cumprod()


    strategy_return = (
        backtest["Strategy Return"].iloc[-1] - 1
    )

    strategy_volatility = (
        backtest["Strategy Return"].std()
        * np.sqrt(252)
    )


    # Sharpe Ratio
    strategy_sharpe = (
        strategy_return
        / strategy_volatility
    )

    backtest["Strategy Peak"] = (
        backtest["Strategy Return"]
        .cummax()
    )

    backtest["Strategy Drawdown"] = (
        (backtest["Strategy Return"] - backtest["Strategy Peak"])
        / backtest["Strategy Peak"]
    )

    strategy_average_drawdown = backtest["Strategy Drawdown"].mean()

    strategy_max_drawdown = (
        backtest["Strategy Drawdown"].min()
    )

    return {
            "Strategy Return": strategy_return,
            "Average Drawdown": strategy_average_drawdown,
            "Max Drawdown": strategy_max_drawdown,
            "Sharpe Ratio": strategy_sharpe
    }