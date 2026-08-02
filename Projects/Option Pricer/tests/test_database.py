
from pathlib import Path
import pandas as pd
from options_risk_engine.database import SQLiteStore


def test_write_and_read_frame(tmp_path: Path):
    store = SQLiteStore(tmp_path / "test.db")
    frame = pd.DataFrame({"x": [1, 2], "y": [3.0, 4.0]})
    store.write_frame(frame, "example", if_exists="replace")
    loaded = store.read_query("SELECT x, y FROM example ORDER BY x")
    assert loaded[["x", "y"]].equals(frame)
