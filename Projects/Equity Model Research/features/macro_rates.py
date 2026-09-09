"""Curve shape and policy/credit conditions, in percentage-point units."""
import pandas as pd
from ._macro_common import macro_daily,attach,calendar_lag,sequence,flag,prior_z

__all__ = ['all_macro_rates_features']

PAIRS={
 'Curve 10Y 2Y':('Treasury_10Y','Treasury_2Y'),
 'Curve 10Y 3M':('Treasury_10Y','Treasury_3M'),
 'Curve 30Y 10Y':('Treasury_30Y','Treasury_10Y'),
 'Curve 5Y 2Y':('Treasury_5Y','Treasury_2Y'),
 'Curve 2Y 3M':('Treasury_2Y','Treasury_3M'),
 'Policy Gap 2Y':('Treasury_2Y','Fed_Funds'),
 'Policy Gap 10Y':('Treasury_10Y','Fed_Funds'),
 'Credit HY Minus BAA':('High_Yield_Spread','BAA_Treasury_Spread'),
}


def all_macro_rates_features(df,change_days=(30,90)):
    daily,_=macro_daily(df); out={}
    for label,(left,right) in PAIRS.items():
        a,b='Macro_'+left,'Macro_'+right
        if a not in daily or b not in daily:
            continue
        spread=daily[a]-daily[b]; prefix='Macro Rates '+label
        out[prefix]=spread
        for days in sequence(change_days):
            out[f'{prefix} Change {days}D']=spread-calendar_lag(spread,days)
        if label.startswith('Curve'):
            out[prefix+' Inverted']=flag(spread<0,spread.notna())
            valid=spread.notna(); inverted=spread<0
            resets=(~inverted)|(~valid)
            # An inversion already in progress at the first row has unknown duration.
            resetdate=pd.Series(daily.index,index=daily.index).where(resets).ffill()
            age=(pd.Series(daily.index,index=daily.index)-resetdate).dt.days
            out[prefix+' Inversion Span Days']=age.where(inverted,0).where(valid)
    terms=['Treasury_2Y','Treasury_5Y','Treasury_10Y']
    if all('Macro_'+v in daily for v in terms):
        out['Macro Rates Butterfly 2Y 5Y 10Y']=2*daily.Macro_Treasury_5Y-daily.Macro_Treasury_2Y-daily.Macro_Treasury_10Y
    if all('Macro_'+v in daily for v in ['Treasury_2Y','Treasury_10Y','Treasury_30Y']):
        out['Macro Rates Yield Level']=daily[['Macro_Treasury_2Y','Macro_Treasury_10Y','Macro_Treasury_30Y']].mean(axis=1,skipna=False)
    for metric in ['Fed_Funds','High_Yield_Spread','BAA_Treasury_Spread']:
        col='Macro_'+metric
        if col in daily:
            for days in sequence(change_days):
                delta=daily[col]-calendar_lag(daily[col],days)
                out[f'Macro Rates {metric} Rising {days}D']=flag(delta>0,delta.notna())
    return attach(df,pd.DataFrame(out,index=daily.index))
