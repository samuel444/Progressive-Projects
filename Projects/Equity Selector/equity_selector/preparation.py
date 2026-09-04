"""Create isolated research folders and stage immutable model inputs."""

from pathlib import Path
import shutil
import pandas as pd
from .database_audit import digest


def prepare_research(*, phase, model_dir, selection_dir, final_dir, download_fx=True):
    if phase not in {"model", "selection", "final"}:
        raise ValueError("phase must be model, selection or final")
    folders = [Path(x).resolve() for x in (model_dir, selection_dir, final_dir)]
    if len(set(folders)) != 3:
        raise ValueError("Research phase directories must be distinct")
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
    if phase == "model":
        return {"phase": phase, "directory": str(folders[0])}
    source, target = (folders[0], folders[1]) if phase == "selection" else (folders[1], folders[2])
    names = ["Final_Test_Results.db", "Selected_Features.txt", "Top_Horizon_Scores.txt"]
    for name in names:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(f"Complete the preceding phase first: {path}")
        wal = Path(str(path) + "-wal")
        if wal.exists() and wal.stat().st_size:
            raise ValueError(f"Stop database writers and checkpoint before copying {path}")
        dest = target / name
        if dest.exists() and digest(dest) != digest(path):
            raise FileExistsError(
                f"Conflicting frozen input: {dest}; use a fresh research directory"
            )
    for name in names:
        path, dest = source / name, target / name
        before = digest(path)
        if not dest.exists():
            shutil.copy2(path, dest)
        if digest(path) != before or digest(dest) != before:
            raise ValueError(
                "Input changed while staging research; stop writers and use fresh directories"
            )
    fx_path = target / "GBP_per_USD.csv"
    if download_fx and not fx_path.exists():
        import yfinance as yf

        # Each phase downloads only its own valuation dates, including its signal date.
        start, end = (
            ("2019-01-01", "2023-01-01") if phase == "selection" else ("2023-01-01", "2026-09-01")
        )
        data = yf.download(
            "GBPUSD=X", start=start, end=end, auto_adjust=False, progress=False, threads=False
        )
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            if close.shape[1] != 1:
                raise ValueError("Expected one GBPUSD close series")
            close = close.iloc[:, 0]
        if close.empty or close.isna().any() or (close <= 0).any():
            raise ValueError("No complete positive FX series was downloaded")
        dates = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
        pd.DataFrame({"Date": dates, "GBP_per_USD": 1 / close.to_numpy()}).to_csv(
            fx_path, index=False
        )
    return {"phase": phase, "directory": str(target), "fx_file": str(fx_path)}
