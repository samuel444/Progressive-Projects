"""Portable paths; credentials are configured separately by service helpers."""

import os
from pathlib import Path


def data_root():
    default = Path(__file__).resolve().parents[1] / "data"
    return Path(os.environ.get("EQUITY_SELECTOR_DATA_DIR", default)).expanduser().resolve()
