"""Transparent macro composites and stress measures; no fitted full-sample cutoffs."""
import pandas as pd
from ._macro_common import macro_daily,attach,calendar_lag,ratio,prior_z,flag

__all__ = ['all_macro_conditions_features']


def all_macro_conditions_features(df,history_days=1095,minimum=90):
    daily,_=macro_daily(df); out={}; scores={}
    directions={'VIX':1,'High_Yield_Spread':1,'Initial_Claims':1,'Unemployment':1,
                'Consumer_Sentiment':-1}
    for metric,sign in directions.items():
        col='Macro_'+metric
        if col in daily:
            scores[metric]=sign*prior_z(daily[col],history_days,minimum)
    stress=pd.DataFrame(scores,index=daily.index)
    if len(stress.columns):
        out['Macro Conditions Stress Components Available']=stress.notna().sum(axis=1).astype(float)
    # Fixed composition: omit missing component rather than silently changing the index.
    if set(directions)<=set(scores):
        out['Macro Conditions Stress Mean Z']=stress.mean(axis=1,skipna=False)
    if 'Macro_VIX' in daily:
        vix=daily.Macro_VIX
        out['Macro Conditions VIX Annual Variance']=(vix/100)**2
        out['Macro Conditions VIX Above 20']=flag(vix>20,vix.notna())
        out['Macro Conditions VIX Above 30']=flag(vix>30,vix.notna())
    if all(c in daily for c in ['Macro_CPI','Macro_Retail_Sales']):
        out['Macro Conditions Retail CPI Deflated Proxy']=100*ratio(daily.Macro_Retail_Sales,daily.Macro_CPI)
    if all(c in daily for c in ['Macro_CPI','Macro_M2']):
        out['Macro Conditions M2 CPI Deflated Proxy']=100*ratio(daily.Macro_M2,daily.Macro_CPI)
    for metric in ['Payrolls','Industrial_Production','Housing_Starts','Consumer_Sentiment','USD_Index']:
        col='Macro_'+metric
        if col in daily:
            change=daily[col]-calendar_lag(daily[col],90)
            out[f'Macro Conditions {metric} Rising 90D']=flag(change>0,change.notna())
    return attach(df,pd.DataFrame(out,index=daily.index))
