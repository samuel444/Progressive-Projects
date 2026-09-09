"""Calendar changes in the information available then, not same-vintage growth."""
import pandas as pd
from ._macro_common import macro_daily,attach,sequence,calendar_lag,ratio,prior_z,RATE_METRICS

__all__ = ['all_macro_history_features']


def all_macro_history_features(df, change_days=(30,90,365), z_days=(365,1095),
                               smooth_days=(30,90), metrics=None, minimum=30):
    daily,cols=macro_daily(df,metrics)
    out={}
    for col in cols:
        s=daily[col]; metric=col.removeprefix('Macro_'); prefix='Macro History '+metric
        for days in sequence(change_days):
            previous=calendar_lag(s,days)
            out[f'{prefix} Change {days}D']=s-previous
            if metric not in RATE_METRICS:
                out[f'{prefix} Percent Change {days}D']=100*(ratio(s,previous)-1)
            before=calendar_lag(s,2*days)
            out[f'{prefix} Change Acceleration {days}D']=s-2*previous+before
        for days in sequence(z_days):
            out[f'{prefix} Prior Z {days}D']=prior_z(s,days,minimum)
            roll=s.rolling(f'{days}D',closed='left',min_periods=minimum)
            out[f'{prefix} Prior Range Position {days}D']=ratio(s-roll.min(),roll.max()-roll.min())
        for days in sequence(smooth_days):
            mean=s.rolling(f'{days}D',closed='left',min_periods=min(minimum,days)).mean()
            out[f'{prefix} Gap To Prior Mean {days}D']=s-mean
    return attach(df,pd.DataFrame(out,index=daily.index))
