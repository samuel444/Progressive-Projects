"""Availability and staleness: missing observations are never coded as neutral."""
import numpy as np
import pandas as pd
from ._macro_common import macro_daily, attach, flag

__all__ = ['all_macro_quality_features']


def all_macro_quality_features(df, stale_days=120, metrics=None):
    if stale_days<0:
        raise ValueError('stale_days must be nonnegative')
    daily,cols = macro_daily(df,metrics)
    out = {}
    for col in cols:
        s=daily[col]; prefix=col.replace('Macro_','Macro Quality ')
        out[prefix+' Missing']=s.isna().astype(float)
        meta=col+'_ObservationDate'
        if meta in daily:
            observed=pd.to_datetime(daily[meta],errors='coerce')
            age=(pd.Series(daily.index,index=daily.index)-observed).dt.days
            if (age<0).any():
                raise ValueError(f'{meta} lies after prediction date')
            out[prefix+' Observation Age Days']=age.where(s.notna())
            out[prefix+' Stale']=flag(age>stale_days,age.notna()&s.notna())
            out[prefix+' New Observation']=flag(observed.ne(observed.shift()),observed.notna()&observed.shift().notna())
        vintage=col+'_VintageStart'
        if vintage in daily:
            published=pd.to_datetime(daily[vintage],errors='coerce')
            age=(pd.Series(daily.index,index=daily.index)-published).dt.days
            if (age<0).any():
                raise ValueError(f'{vintage} is future information')
            out[prefix+' Version Age Days']=age.where(s.notna())
        changed=s.ne(s.shift())&s.notna()&s.shift().notna()
        # Gap breaks the observed constant-value run; no assumed release at first row.
        run=s.ne(s.shift())|s.isna()|s.shift().isna()
        start=pd.Series(daily.index,index=daily.index).where(run).ffill()
        out[prefix+' Unchanged Span Days']=(pd.Series(daily.index,index=daily.index)-start).dt.days.where(s.notna())
        out[prefix+' Changed']=flag(changed,s.notna()&s.shift().notna())
    if cols:
        out['Macro Quality Available Fraction']=daily[cols].notna().mean(axis=1)
    return attach(df,pd.DataFrame(out,index=daily.index))
