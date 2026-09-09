"""Economic-period features recomputed from a single historical information set.

Accept raw ALFRED intervals, not just a daily latest-observation column. Windows
that clip VintageStart are validity boundaries; no release-surprise claim is made.
"""
import gzip
import heapq
import json
import sqlite3
from datetime import date
import numpy as np
import pandas as pd
from ._macro_common import dates_of,attach

__all__ = ['all_macro_vintage_features','load_alfred_intervals']
MONTHLY_GROWTH=('CPI','Core_CPI','PCE','Payrolls','Industrial_Production','Retail_Sales',
                'M2','Housing_Starts','Consumer_Sentiment','USD_Index')


def load_alfred_intervals(path):
    """Read the existing downloader cache or its old SQLite interval table. No network."""
    path=str(path)
    if path.endswith('.gz'):
        with gzip.open(path,'rt',encoding='utf-8') as handle:
            payload=json.load(handle)
        item=payload.get('alfred_intervals')
        if item is None:
            raise ValueError('Cache has no alfred_intervals; run ALFRED download first')
        # Known downloader schema; do not execute cached SQL.
        columns=['Series','Metric','Date','VintageStart','VintageEnd','Value']
        if any(len(row)!=len(columns) for row in item['rows']):
            raise ValueError('Unexpected ALFRED cache row schema')
        return pd.DataFrame(item['rows'],columns=columns)
    from pathlib import Path
    with sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True) as conn:
        return pd.read_sql_query('SELECT * FROM alfred_intervals',conn)


def _growth(a,b,power=1):
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b) or a<=0 or b<=0:
        return np.nan
    return 100*((a/b)**power-1)


def _snapshot_features(metric,state,cutoff):
    # state maps observation ordinal -> (version token, value, actual observation date)
    active={k:v for k,v in state.items() if k<=cutoff}
    if not active:
        return {}
    latest=max(active); value=active[latest][1]
    observed=pd.Timestamp(active[latest][2]); stem='Macro PIT '+metric
    out={stem+' Latest':value}
    if metric in MONTHLY_GROWTH or metric=='Unemployment':
        # These are monthly series. USD_Index is daily, so handle separately below.
        if metric=='USD_Index':
            return out
        bymonth={}
        for ordinal,(_,v,d) in sorted(active.items()):
            bymonth[pd.Timestamp(d).to_period('M')]=v
        current=observed.to_period('M')
        def at(months):
            return bymonth.get(current-months,np.nan)
        if metric=='Unemployment':
            for months in (1,3,12):
                out[f'{stem} Change {months}M PP']=value-at(months)
            recent=[at(i) for i in range(3)]
            means=[np.mean([at(i+j) for j in range(3)]) for i in range(12)]
            out[stem+' Three Month Mean']=np.mean(recent)
            out[stem+' Three Month Mean Minus Prior 12M Minimum']=np.mean(recent)-np.min(means)
        else:
            for months in (1,3,6,12):
                out[f'{stem} Growth {months}M Percent']=_growth(value,at(months))
            out[stem+' Annualized Growth 3M Percent']=_growth(value,at(3),4)
            out[stem+' Annualized Growth 6M Percent']=_growth(value,at(6),2)
            out[stem+' YoY Acceleration 3M PP']=_growth(value,at(12))-_growth(at(3),at(15))
            out[stem+' Growth 3M Versus Prior 3M PP']=_growth(value,at(3))-_growth(at(3),at(6))
    elif metric=='Real_GDP':
        byquarter={pd.Timestamp(v[2]).to_period('Q'):v[1] for _,v in sorted(active.items())}
        quarter=observed.to_period('Q')
        out[stem+' Growth 1Q Annualized Percent']=_growth(value,byquarter.get(quarter-1),4)
        out[stem+' Growth 4Q Percent']=_growth(value,byquarter.get(quarter-4))
    elif metric=='Initial_Claims':
        values=[active.get(latest-7*i,(None,np.nan,None))[1] for i in range(8)]
        mean=np.mean(values[:4]); previous=np.mean(values[4:])
        out[stem+' Four Week Mean']=mean
        out[stem+' Four Week Mean Growth Percent']=_growth(mean,previous)
    return out


def all_macro_vintage_features(df,intervals=None,release_lag_days=1,metrics=None):
    if intervals is None:
        raise ValueError('macro_vintages requires raw intervals=load_alfred_intervals(cache_path)')
    if not isinstance(release_lag_days,int) or release_lag_days<0:
        raise ValueError('release_lag_days must be a nonnegative integer')
    needed={'Metric','Date','VintageStart','VintageEnd','Value'}
    if not needed<=set(intervals):
        raise ValueError(f'Missing interval columns: {sorted(needed-set(intervals))}')
    frame=intervals.copy()
    frame['Value']=pd.to_numeric(frame.Value,errors='coerce').replace([np.inf,-np.inf],np.nan)
    keys=['Metric','Date','VintageStart']
    if (frame.groupby(keys).Value.nunique(dropna=False)>1).any():
        raise ValueError('Conflicting values for the same macro observation/vintage')
    if metrics is not None:
        requested=['Macro_'+m if not m.startswith('Macro_') else m for m in metrics]
        frame=frame[frame.Metric.isin(requested)]
    dates=pd.DatetimeIndex(sorted(set(dates_of(df))))
    outputs={}
    for label,group in frame.groupby('Metric'):
        metric=label.removeprefix('Macro_')
        events=[]
        for row in group.itertuples(index=False):
            lo=date.fromisoformat(row.VintageStart).toordinal()
            hi=date.fromisoformat(row.VintageEnd).toordinal()
            obs=date.fromisoformat(row.Date).toordinal()
            if hi<lo:
                raise ValueError('VintageEnd precedes VintageStart')
            events.append((max(lo,obs),lo,hi,obs,row.Value,row.Date))
        events.sort(key=lambda e:e[:4])
        pointer=0; state={}; expires=[]; cached={}; rows=[]
        for prediction in dates:
            cutoff=prediction.date().toordinal()-release_lag_days; changed=False
            while pointer<len(events) and events[pointer][0]<=cutoff:
                _,lo,hi,obs,value,obsdate=events[pointer]
                token=(lo,hi,pointer)
                # Newer vintage wins for a given observation; expired entries are pruned next.
                if obs not in state or token>state[obs][0]:
                    state[obs]=(token,value,obsdate); changed=True
                    heapq.heappush(expires,(hi,obs,token))
                pointer+=1
            while expires and expires[0][0]<cutoff:
                _,obs,token=heapq.heappop(expires)
                if obs in state and state[obs][0]==token:
                    del state[obs]; changed=True
            if changed:
                cached=_snapshot_features(metric,state,cutoff)
            rows.append(dict(cached))
        values=pd.DataFrame(rows,index=dates)
        for col in values:
            outputs[col]=values[col]
    result=pd.DataFrame(outputs,index=dates)
    # Differences between headline and core year-over-year inflation, same cutoff.
    for left,right,name in [('CPI','Core_CPI','Headline Minus Core Inflation PP'),
                            ('CPI','PCE','CPI Minus PCE Inflation PP')]:
        a=f'Macro PIT {left} Growth 12M Percent';b=f'Macro PIT {right} Growth 12M Percent'
        if a in result and b in result:
            result['Macro PIT '+name]=result[a]-result[b]
    return attach(df,result)
