"""Interactions and rolling exposures: fitted coefficients use earlier bars only."""
import numpy as np
import pandas as pd
from ._macro_common import macro_daily,stock_groups,sequence,prior_z,ratio

__all__ = ['all_macro_interaction_features','all_macro_sensitivity_features']
DEFAULT_DRIVERS=('Treasury_10Y','High_Yield_Spread','VIX','USD_Index')


def all_macro_interaction_features(df, momentum_windows=(63,126),history_days=1095,
                                   minimum=90,drivers=DEFAULT_DRIVERS,close_col=None):
    daily,_=macro_daily(df,drivers)
    z={m:prior_z(daily['Macro_'+m],history_days,minimum) for m in drivers if 'Macro_'+m in daily}
    columns={}
    for group in stock_groups(df,close_col):
        # SQL panel includes weekends; use observed positive closing-price rows only.
        g=group[group._close.gt(0)&np.isfinite(group._close)].copy()
        positions=g._position.to_numpy()
        if g.empty:
            continue
        close=g._close.reset_index(drop=True)
        ret=close.pct_change(fill_method=None)
        vol=ret.rolling(21,min_periods=21).std()*np.sqrt(252)
        for driver,state in z.items():
            exposure=pd.Series(state.reindex(pd.DatetimeIndex(g._date)).to_numpy())
            for window in sequence(momentum_windows):
                momentum=close.pct_change(window,fill_method=None)
                name=f'Macro Interaction Momentum {window} x {driver} Prior Z'
                columns.setdefault(name,np.full(len(df),np.nan))[positions]=(momentum*exposure).to_numpy()
            name=f'Macro Interaction Volatility 21 x {driver} Prior Z'
            columns.setdefault(name,np.full(len(df),np.nan))[positions]=(vol*exposure).to_numpy()
        if 'Macro_VIX' in g:
            vix=pd.to_numeric(g.Macro_VIX,errors='coerce').reset_index(drop=True)/100
            name='Macro Interaction VIX Minus Stock Realized Volatility 21'
            columns.setdefault(name,np.full(len(df),np.nan))[positions]=(vix-vol).to_numpy()
    return pd.concat([df.drop(columns=list(columns),errors='ignore'),
                      pd.DataFrame(columns,index=df.index)],axis=1)


def all_macro_sensitivity_features(df,windows=(63,126,252),drivers=DEFAULT_DRIVERS,
                                   close_col=None,min_fraction=0.8):
    if not 0<min_fraction<=1:
        raise ValueError('min_fraction must be in (0,1]')
    macro_daily(df,drivers)  # validate shared macro snapshots
    columns={}
    for group in stock_groups(df,close_col):
        g=group[group._close.gt(0)&np.isfinite(group._close)].copy()
        positions=g._position.to_numpy()
        if g.empty:
            continue
        y=g._close.reset_index(drop=True).pct_change(fill_method=None)
        for driver in drivers:
            col='Macro_'+driver
            if col not in g:
                continue
            levels=pd.to_numeric(g[col],errors='coerce').reset_index(drop=True)
            # Rate/credit drivers: percentage-point differences; VIX/USD: fractional returns.
            x=levels.pct_change(fill_method=None) if driver in ('VIX','USD_Index') else levels.diff()
            x=x.replace([np.inf,-np.inf],np.nan)
            paired=x.notna()&y.notna(); xp=x.where(paired); yp=y.where(paired)
            for window in sequence(windows):
                minimum=max(3,int(np.ceil(window*min_fraction)))
                cov=yp.rolling(window,min_periods=minimum).cov(xp)
                var=xp.rolling(window,min_periods=minimum).var()
                beta=ratio(cov,var).shift(1)
                corr=yp.rolling(window,min_periods=minimum).corr(xp).shift(1)
                mean_y=yp.rolling(window,min_periods=minimum).mean().shift(1)
                mean_x=xp.rolling(window,min_periods=minimum).mean().shift(1)
                residual=y-(mean_y-beta*mean_x+beta*x)
                stem=f'Macro Sensitivity {driver} {window}'
                for suffix,values in [('Prior Beta',beta),('Prior Correlation',corr),
                                      ('Prior R Squared',corr**2),('Residual Return',residual),
                                      ('Prior Pair Count',paired.rolling(window,min_periods=1).sum().shift(1))]:
                    columns.setdefault(stem+' '+suffix,np.full(len(df),np.nan))[positions]=values.to_numpy()
    return pd.concat([df.drop(columns=list(columns),errors='ignore'),
                      pd.DataFrame(columns,index=df.index)],axis=1)
