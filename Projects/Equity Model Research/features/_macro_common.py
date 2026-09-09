"""Shared validation/alignment for macro features; no downloads or future fills."""
import numpy as np
import pandas as pd

METRICS = (
    'Fed_Funds','Treasury_3M','Treasury_2Y','Treasury_5Y','Treasury_10Y','Treasury_30Y',
    '10Y_2Y_Spread','10Y_3M_Spread','CPI','Core_CPI','PCE','Unemployment','Payrolls',
    'Initial_Claims','Industrial_Production','Retail_Sales','Real_GDP','High_Yield_Spread',
    'BAA_Treasury_Spread','VIX','M2','Consumer_Sentiment','Housing_Starts','USD_Index',
)
RATE_METRICS = {'Fed_Funds','Treasury_3M','Treasury_2Y','Treasury_5Y','Treasury_10Y',
                'Treasury_30Y','10Y_2Y_Spread','10Y_3M_Spread','Unemployment',
                'High_Yield_Spread','BAA_Treasury_Spread'}


def sequence(value):
    values = (value,) if isinstance(value,int) else tuple(value)
    if any(not isinstance(v,int) or v<=0 for v in values):
        raise ValueError('Windows must be positive integers')
    return values


def dates_of(df):
    if 'Date' in df:
        values = pd.to_datetime(df['Date'],errors='raise')
    elif isinstance(df.index,pd.DatetimeIndex):
        values = df.index
    else:
        raise ValueError('Supply a Date column or DatetimeIndex')
    index = pd.DatetimeIndex(values)
    if index.hasnans:
        raise ValueError('Dates cannot be missing')
    if index.tz is not None:
        index = index.tz_localize(None)
    return index.normalize()


def macro_daily(df, metrics=None):
    dates = dates_of(df)
    names = list(METRICS if metrics is None else metrics)
    unknown = set(names)-set(METRICS)
    if unknown:
        raise ValueError(f'Unknown macro metrics: {sorted(unknown)}')
    cols = [f'Macro_{m}' for m in names if f'Macro_{m}' in df]
    metadata = [c+s for c in cols for s in ('_ObservationDate','_VintageStart') if c+s in df]
    work = df[cols+metadata].copy()
    work.index = dates
    # A shared macro series must agree across all stocks for a given date.
    if len(work.columns) and (work.groupby(level=0).nunique(dropna=False)>1).any().any():
        raise ValueError('Conflicting macro values/metadata across tickers on the same date')
    work = work[~work.index.duplicated()].sort_index()
    for col in cols:
        work[col] = pd.to_numeric(work[col],errors='coerce').replace([np.inf,-np.inf],np.nan)
    return work,cols


def attach(df, values):
    aligned = pd.DataFrame({col: values[col].reindex(dates_of(df)).to_numpy() for col in values},index=df.index)
    return pd.concat([df.drop(columns=list(values.columns),errors='ignore'),aligned],axis=1)


def calendar_lag(series, days, tolerance_days=7):
    target = series.index-pd.Timedelta(days=days)
    # As-of lookup returns the actual row (including NaN), not the last non-null value.
    out = series.reindex(target,method='ffill',tolerance=pd.Timedelta(days=tolerance_days))
    return pd.Series(out.to_numpy(),index=series.index)


def ratio(a,b):
    return (a/b.where(b.abs()>1e-12)).replace([np.inf,-np.inf],np.nan)


def flag(condition, valid):
    return condition.astype(float).where(valid)


def prior_z(series, days, minimum=30):
    history = series.rolling(f'{days}D',closed='left',min_periods=minimum)
    return ratio(series-history.mean(),history.std())


def stock_groups(df, close_col=None):
    name = close_col or ('Close' if 'Close' in df else 'Price_Close')
    if name not in df:
        raise ValueError('Stock/macro features need Close or Price_Close')
    work = df.copy()
    work['_position'] = np.arange(len(df))
    work['_date'] = dates_of(df)
    work['_close'] = pd.to_numeric(work[name],errors='coerce')
    work['_ticker'] = df['Ticker'].to_numpy() if 'Ticker' in df else '__single__'
    if work.duplicated(['_ticker','_date']).any():
        raise ValueError('Duplicate ticker/date observations')
    for _,group in work.groupby('_ticker',sort=False):
        yield group.sort_values('_date')
