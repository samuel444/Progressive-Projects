import pandas as pd
import numpy as np

def size_stocks(df, min_length=756):
    verdict = 'keep'

    if len(df) <= min_length:
        verdict = 'drop'

    return verdict

def missingness_stocks(df, missing_threshold=0.05):
    verdict = 'keep'

    missing_portion = df.isna().mean().mean()

    if missing_portion > missing_threshold:
            verdict = 'drop'

    return verdict

def invalid_stocks(df):
    verdict = 'keep'

    columns = ["Open", "High", "Low", "Close", "Volume"]

    if (
        df[columns].isin([np.inf, -np.inf]).any().any()
        or (df[columns] < 0).any().any()
    ):
        verdict = "drop"

    return verdict

def liquidity_stocks(df, liquidity_threshold = 5000000):
    verdict = 'keep'

    dollar_volume = df["Volume"] * df["Close"]
    current_dollar_volume = np.median(dollar_volume[-20:])

    if current_dollar_volume < liquidity_threshold:
         verdict = 'drop'

    return verdict

def continuity_stocks(df, market_df):
    verdict = 'keep'

    continuity = len(df) / len(market_df)

    expected = market_df.loc[df.index.min():df.index.max()].index

    missing = ~expected.isin(df.index)

    max_gap = 0
    current_gap = 0

    for is_missing in missing:
        if is_missing:
            current_gap += 1
            max_gap = max(max_gap, current_gap)
        else:
            current_gap = 0

    if continuity <= 0.95 or max_gap >= 10:
         verdict = 'drop'



    return verdict
