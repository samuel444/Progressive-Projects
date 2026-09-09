import numpy as np
import pandas as pd
import pytest
from main_package.model_inference import create_models_and_predictions
from equity_selector.artifacts import load_models_and_predictions, produce_requests


def test_frozen_predictions_equal_original_and_never_refit(tmp_path, monkeypatch):
    dates = pd.bdate_range("2020-01-01", periods=40)
    x = np.arange(40, dtype=float)
    data = pd.DataFrame(
        {
            "Date": dates,
            "Ticker": "A",
            "Close": 100 + x,
            "Return": 0.001,
            "x": x,
            "Future Return 1": x * 0.013 + 0.02,
            "Split": ["TRAIN"] * 30 + ["BACKTEST"] * 10,
        }
    )
    selected = pd.DataFrame(
        [
            {
                "Target": "Future Return 1",
                "Model": "Ridge",
                "Parameters": {"alpha": 1.0},
                "Target Type": "ALPHA",
                "Statistical Type": "continuous",
                "Horizon": 1,
                "Horizon Score": 1.0,
                "Quality Score": 0.8,
            }
        ]
    )
    features = {"Future Return 1": ["x"]}
    monkeypatch.setenv("EQUITY_SELECTOR_MODEL_DIR", str(tmp_path / "confirmed_models"))
    monkeypatch.setenv("EQUITY_SELECTOR_REQUEST_DIR", str(tmp_path / "requests"))
    baseline = create_models_and_predictions(
        data, selected, features, purge=False, strict=True
    )
    with pytest.raises(FileNotFoundError, match="No model was refit"):
        load_models_and_predictions(data, selected, features, purge=False)
    produce_requests()
    import main_package.model_inference as fitting

    monkeypatch.setattr(
        fitting,
        "create_target_models",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("refit")),
    )
    actual = load_models_and_predictions(data, selected, features, purge=False)
    pd.testing.assert_frame_equal(
        actual["predictions"], baseline["predictions"], check_exact=True
    )
    changed = data.copy()
    changed.loc[0, "x"] += 1
    with pytest.raises(FileNotFoundError):
        load_models_and_predictions(changed, selected, features, purge=False)
    artifact = next((tmp_path / "confirmed_models").glob("*.pkl"))
    artifact.write_bytes(artifact.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_models_and_predictions(data, selected, features, purge=False)


def test_phase_inputs_remain_frozen_when_research_changes(tmp_path, monkeypatch):
    from horizon_score.config import research_input
    research, phase = tmp_path / 'research', tmp_path / 'phase'
    research.mkdir()
    phase.mkdir()
    monkeypatch.setenv('EQUITY_SELECTOR_DATA_DIR', str(research))
    (research / 'Macro_Regimes.txt').write_text("{'new': {}}")
    assert research_input('Macro_Regimes.txt', phase) == research / 'Macro_Regimes.txt'
    (phase / 'Macro_Regimes.txt').write_text("{'frozen': {}}")
    assert research_input('Macro_Regimes.txt', phase) == phase / 'Macro_Regimes.txt'
