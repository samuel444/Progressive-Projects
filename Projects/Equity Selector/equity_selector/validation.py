"""Date-based splitting and purging; retain the panel calendar before cleaning."""

import numpy as np
import pandas as pd
from .settings import setting


def train_validation_test_split(df, test=0.2, validation=0.2):
    if not (0 < test < 1 and 0 < validation < 1 and test + validation < 1):
        raise ValueError("test and validation must be positive and sum to less than 1")
    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="raise")
    if data["Date"].isna().any():
        raise ValueError("Date must not be missing")
    if setting("RESEARCH_START") is not None:
        data = data.loc[data.Date >= pd.Timestamp(setting("RESEARCH_START"))].copy()
    research_end = setting("RESEARCH_END")
    if research_end is not None:
        data = data.loc[data.Date <= pd.Timestamp(research_end)].copy()
    train_end, validation_end = setting("MODEL_TRAIN_END"), setting("MODEL_VALIDATION_END")
    if (train_end is None) != (validation_end is None):
        raise ValueError("Set both MODEL_TRAIN_END and MODEL_VALIDATION_END")
    if train_end is not None:
        train_end, validation_end = pd.Timestamp(train_end), pd.Timestamp(validation_end)
        if not train_end < validation_end or (
            research_end is not None and not validation_end < pd.Timestamp(research_end)
        ):
            raise ValueError("Model date boundaries must increase strictly")
        parts = (
            data.loc[data.Date <= train_end],
            data.loc[(data.Date > train_end) & (data.Date <= validation_end)],
            data.loc[data.Date > validation_end],
        )
        if any(part.empty for part in parts):
            raise ValueError("Each declared model partition must contain observations")
        return tuple(
            part.sort_values(["Date", "Ticker"]).reset_index(drop=True).copy() for part in parts
        )
    dates = data["Date"].drop_duplicates().sort_values().to_numpy()
    validation_start = int(len(dates) * (1 - test - validation))
    test_start = int(len(dates) * (1 - test))
    if not 0 < validation_start < test_start < len(dates):
        raise ValueError("Not enough dates for three nonempty partitions")
    validation_date, test_date = dates[validation_start], dates[test_start]
    return tuple(
        part.sort_values(["Date", "Ticker"]).reset_index(drop=True)
        for part in (
            data.loc[data.Date < validation_date].copy(),
            data.loc[(data.Date >= validation_date) & (data.Date < test_date)].copy(),
            data.loc[data.Date >= test_date].copy(),
        )
    )


def purge_training_data(dataframe, purge_days):
    if not isinstance(purge_days, (int, np.integer)) or purge_days < 0:
        raise ValueError("purge_days must be a nonnegative integer")
    data = dataframe.copy()
    dates = pd.to_datetime(data["Date"], errors="raise")
    unique_dates = dates.drop_duplicates().sort_values()
    if purge_days == 0:
        return data
    keep = unique_dates.iloc[:-purge_days]
    return data.loc[dates.isin(keep)].copy()


def validate_chronology(training, backtest):
    if training.empty or backtest.empty:
        raise ValueError("TRAIN and BACKTEST partitions must be nonempty")
    train_dates = pd.to_datetime(training["Date"], errors="raise")
    test_dates = pd.to_datetime(backtest["Date"], errors="raise")
    if train_dates.isna().any() or test_dates.isna().any() or train_dates.max() >= test_dates.min():
        raise ValueError("TRAIN dates must precede every BACKTEST date")


def screening_training_rows(dataframe, calendar=None):
    """Copy rows before the validation boundary; accept a Date column or date index.

    An explicit calendar fixes stock eligibility to the common downloaded panel
    calendar, including missing quotes. Later quote values cannot affect decisions.
    """
    dates = pd.to_datetime(dataframe["Date"] if "Date" in dataframe else dataframe.index)
    train_end = setting("MODEL_TRAIN_END")
    if train_end is not None:
        mask = dates <= pd.Timestamp(train_end)
        if setting("RESEARCH_START") is not None:
            mask &= dates >= pd.Timestamp(setting("RESEARCH_START"))
        return dataframe.loc[mask].copy()
    reference = pd.to_datetime(calendar if calendar is not None else dates)
    reference = pd.Series(reference).drop_duplicates().sort_values()
    if setting("RESEARCH_START") is not None:
        reference = reference.loc[reference >= pd.Timestamp(setting("RESEARCH_START"))]
    if setting("RESEARCH_END") is not None:
        reference = reference.loc[reference <= pd.Timestamp(setting("RESEARCH_END"))]
    if reference.isna().any() or pd.isna(dates).any():
        raise ValueError("Screening dates must not be missing")
    boundary = int(len(reference) * (1 - 0.2 - 0.2))
    if not 0 < boundary < len(reference):
        raise ValueError("Not enough dates for screening")
    return dataframe.loc[(dates >= reference.iloc[0]) & (dates < reference.iloc[boundary])].copy()


def research_rows(data):
    """Apply declared research bounds before data-dependent classification/screening."""
    if "Date" not in data:
        return data.copy()
    dates = pd.to_datetime(data.Date, errors="raise")
    mask = dates.notna()
    if setting("RESEARCH_START") is not None:
        mask &= dates >= pd.Timestamp(setting("RESEARCH_START"))
    if setting("RESEARCH_END") is not None:
        mask &= dates <= pd.Timestamp(setting("RESEARCH_END"))
    return data.loc[mask].copy()


def required_validation_folds(data, window, requirement):
    if requirement == "all":
        _, validation, _ = train_validation_test_split(data)
        return int(np.ceil(validation.Date.nunique() / window))
    if not isinstance(requirement, int) or requirement <= 0:
        raise ValueError("MIN_VALIDATION_FOLDS must be a positive integer or 'all'")
    return requirement
