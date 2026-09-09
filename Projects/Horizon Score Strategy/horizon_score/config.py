"""Strategy outputs are separate from the research artifact directory."""
import os
from pathlib import Path

def data_root():
    return Path(os.environ.get("HORIZON_SCORE_DATA_DIR", Path(__file__).resolve().parents[1] / "data")).expanduser().resolve()


def research_input(name, strategy_dir=None):
    """Prefer a phase's immutable staged input; otherwise use research outputs."""
    from equity_selector.config import data_root as model_data_root
    staged = Path(strategy_dir or data_root()) / name
    return staged if staged.is_file() else model_data_root() / name
