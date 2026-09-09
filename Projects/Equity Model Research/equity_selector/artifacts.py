"""Frozen model exchange. Only produce_models fits; consumers only deserialize/predict.

Pickles are local, trusted research outputs, never untrusted downloads. Keys cover
training rows, feature order, metadata and purge policy, preventing silent reuse
of a fit with different provenance. BACKTEST rows never enter a fitting request.
"""

import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import numpy as np
import pandas as pd
from .config import data_root
from .validation import validate_chronology


def artifact_root():
    return Path(
        os.environ.get("EQUITY_SELECTOR_MODEL_DIR", data_root() / "confirmed_models")
    ).resolve()


def _identity(training, selected, features, purge):
    training = training.reset_index(drop=True)
    digest = hashlib.sha256(pd.util.hash_pandas_object(training, index=True).values.tobytes())
    digest.update(str(list(zip(training.columns, map(str, training.dtypes)))).encode())
    digest.update(selected.to_json(orient="split", date_format="iso").encode())
    digest.update(json.dumps(features, sort_keys=True, default=list).encode())
    digest.update(str(bool(purge)).encode())
    return digest.hexdigest()


def produce_models(training, selected, features, *, purge=True, directory=None):
    """Research API: fit exactly the requested historical training partition."""
    from main_package.model_inference import create_target_models

    directory = Path(directory or artifact_root())
    directory.mkdir(parents=True, exist_ok=True)
    key = _identity(training, selected, features, purge)
    dest = directory / (key + ".pkl")
    if dest.exists():
        return dest
    fitted = create_target_models(training, selected, features, strict=True, purge=purge)
    manifest = {
        "key": key,
        "purge": bool(purge),
        "rows": len(training),
        "train_start": str(pd.to_datetime(training.Date).min()),
        "train_end": str(pd.to_datetime(training.Date).max()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "features": features,
    }
    payload = pickle.dumps({"manifest": manifest, "fitted": fitted}, protocol=5)
    temporary = dest.with_suffix(".tmp")
    temporary.write_bytes(payload)
    temporary.replace(dest)
    manifest["sha256"] = hashlib.sha256(payload).hexdigest()
    dest.with_suffix(".json").write_text(json.dumps(manifest, indent=2, default=list) + "\n")
    return dest


def load_models_and_predictions(
    dataframe, selected_models_df, model_features=None, strict=False, purge=True
):
    """No fitting fallback. A missing fit emits a research request and fails."""
    from main_package.model_inference import create_prediction_dataframe

    data = dataframe.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    split = data.Split.astype(str).str.upper().str.strip()
    training, backtest = data.loc[split.eq("TRAIN")].copy(), data.loc[split.eq("BACKTEST")].copy()
    if training.empty or backtest.empty:
        raise ValueError("TRAIN and BACKTEST rows are required")
    validate_chronology(training, backtest)
    features = model_features or {}
    key = _identity(training, selected_models_df, features, purge)
    path = artifact_root() / (key + ".pkl")
    if not path.is_file():
        requests = Path(
            os.environ.get("EQUITY_SELECTOR_REQUEST_DIR", artifact_root().parent / "model_requests")
        )
        requests.mkdir(parents=True, exist_ok=True)
        request = requests / (key + ".pkl")
        request.write_bytes(
            pickle.dumps(
                {
                    "training": training,
                    "selected": selected_models_df,
                    "features": features,
                    "purge": purge,
                },
                protocol=5,
            )
        )
        raise FileNotFoundError(
            f"Missing confirmed fit {key}. Research request: {request}. "
            "Run Produce Confirmed Models.py in Equity Model Research, then resume this stage. No model was refit."
        )
    payload = path.read_bytes()
    manifest = json.loads(path.with_suffix(".json").read_text())
    if hashlib.sha256(payload).hexdigest() != manifest["sha256"]:
        raise ValueError("Model artifact checksum mismatch")
    bundle = pickle.loads(payload)
    if bundle["manifest"]["key"] != key:
        raise ValueError("Model artifact provenance mismatch")
    fitted = bundle["fitted"]
    return {**fitted, "predictions": create_prediction_dataframe(backtest, fitted["models"])}


def produce_requests(directory=None):
    directory = Path(
        directory
        or os.environ.get("EQUITY_SELECTOR_REQUEST_DIR", artifact_root().parent / "model_requests")
    )
    paths = sorted(directory.glob("*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No model production requests in {directory}")
    return [produce_models(**pickle.loads(p.read_bytes())) for p in paths]
