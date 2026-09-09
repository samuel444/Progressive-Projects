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

from equity_selector.validation import validate_chronology, purge_training_data
from .functions import target_purge_days
import numpy as np




"""Reusable target model fitting and inference; no portfolio construction."""


import ast
import json
import logging
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import pandas as pd
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

from equity_selector.parameters import parse_parameters


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
    model_features: Optional[Union[pd.DataFrame, Mapping[str, Sequence[str]]]],
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
        raise TypeError("model_features must be a DataFrame, mapping, or None.")

    required = {"Target", "Features"}
    missing = required.difference(model_features.columns)
    if missing:
        raise ValueError("model_features is missing columns: " + ", ".join(sorted(missing)))

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
    if n_unique <= 20 and numeric.notna().all() and np.allclose(numeric, np.round(numeric)):
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
        raise ValueError(f"{target} is binary but contains classes {classes}.")

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
            raise ImportError("lightgbm is required because a selected model uses LightGBM.")
        if is_regression:
            return LGBMRegressor(**params)
        return LGBMClassifier(**params)

    if is_regression:
        if name == "huber":
            params.setdefault("max_iter", 1000)
            return HuberRegressor(**params)

        if name == "xgboost":
            if XGBRegressor is None:
                raise ImportError("xgboost is required because a selected model uses XGBoost.")
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
            raise ImportError("xgboost is required because a selected model uses XGBoost.")

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

    raise ValueError(f"Unknown model '{model_name}' for target type '{statistical_type}'.")


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
        column for column in features + [target] if column not in training_df.columns
    ]
    if missing_columns:
        raise ValueError(f"{target} is missing training columns: " + ", ".join(missing_columns))

    model_df = (
        training_df.replace([np.inf, -np.inf], np.nan).dropna(subset=features + [target]).copy()
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
        raise ValueError(f"{target} has fewer than two classes in the training data.")

    num_classes = int(y_train.nunique()) if statistical_type != "continuous" else None

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
    is_xgboost_classifier = statistical_type != "continuous" and model_key == "xgboost"

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
                raise ValueError(f"{model_name} does not expose classes_ for {target}.")
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
            f"{model_info['target']} is missing backtest features: " + ", ".join(missing)
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
            f"{model_info['model_name']} does not support predict_proba for {model_info['target']}."
        )

    probabilities = model.predict_proba(X_model)
    class_values = np.asarray(model_info["class_values"])

    try:
        numeric_classes = class_values.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Classes for {model_info['target']} must be numeric. Got {class_values.tolist()}."
        ) from error

    if probabilities.shape[1] != len(numeric_classes):
        raise ValueError(
            f"Probability columns for {model_info['target']} do not match the stored class values."
        )

    expected_class = probabilities @ numeric_classes
    predictions.loc[valid] = expected_class
    return predictions


########################################
# Signal / horizon helpers
########################################










########################################
# Portfolio optimiser
########################################




########################################
# Public function
########################################




######################################################################
# Refactored public API: fit once -> predict once -> backtest many times
######################################################################


def _prepare_selected_models(selected_models_df):
    required = {
        "Target",
        "Model",
        "Parameters",
        "Target Type",
        "Horizon",
        "Horizon Score",
        "Quality Score",
    }
    missing = required.difference(selected_models_df.columns)
    if missing:
        raise ValueError("selected_models_df is missing columns: " + ", ".join(sorted(missing)))

    models = selected_models_df.copy()
    models["Quality Score"] = pd.to_numeric(models["Quality Score"], errors="coerce").clip(0.0, 1.0)
    models["Horizon"] = pd.to_numeric(models["Horizon"], errors="coerce")
    models["Horizon Score"] = pd.to_numeric(models["Horizon Score"], errors="coerce").clip(0.0, 1.0)

    models = models.dropna(
        subset=[
            "Target",
            "Model",
            "Target Type",
            "Horizon",
            "Horizon Score",
            "Quality Score",
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
    purge=True,
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
        if "Statistical Type" in row.index and pd.notna(row["Statistical Type"]):
            statistical_type = str(row["Statistical Type"])

        try:
            model_info = _fit_target_model(
                training_df=purge_training_data(training_df, target_purge_days(target))
                if purge
                else training_df,
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

        summary_rows.append(
            {
                "Target": target,
                "Model": model_name,
                "Parameters": row["Parameters"],
                "Target Type": portfolio_type,
                "Statistical Type": model_info["statistical_type"],
                "Horizon": float(row["Horizon"]),
                "Horizon Score": float(row["Horizon Score"]),
                "Quality Score": float(row["Quality Score"]),
                "Training Rows": int(model_info["training_rows"]),
            }
        )

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
        raise ValueError("backtest_df is missing columns: " + ", ".join(sorted(missing)))

    data = backtest_df.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    if "Return" not in data.columns:
        data["Return"] = data.groupby("Ticker", sort=False)["Close"].pct_change()

    parts = []

    for target, model_info in fitted_models.items():
        prediction = _predict_target_model(model_info, data)

        part = data[["Date", "Ticker", "Close", "Return"]].copy()
        part["Target"] = target
        part["Prediction"] = prediction.to_numpy()
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
    purge=True,
):
    """
    Convenience wrapper: split TRAIN/BACKTEST, fit once, predict once.
    """
    required = {"Split", "Date", "Ticker", "Close"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError("dataframe is missing columns: " + ", ".join(sorted(missing)))

    data = dataframe.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    split = data["Split"].astype(str).str.upper().str.strip()
    training_df = data.loc[split == "TRAIN"].copy()
    backtest_df = data.loc[split == "BACKTEST"].copy()

    if training_df.empty:
        raise ValueError("No TRAIN rows are present.")
    if backtest_df.empty:
        raise ValueError("No BACKTEST rows are present.")

    validate_chronology(training_df, backtest_df)
    fitted = create_target_models(
        training_df=training_df,
        selected_models_df=selected_models_df,
        model_features=model_features,
        strict=strict,
        purge=purge,
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
