"""Known calendar position; no fabricated release calendars or trading sessions."""
import numpy as np
import pandas as pd
from ._macro_common import dates_of

__all__ = ['all_macro_calendar_features']


def all_macro_calendar_features(df):
    dates=dates_of(df); result=df.copy()
    for name,phase,period in [('Weekday',dates.dayofweek,7),('Month',dates.month-1,12),
                              ('Day Of Year',dates.dayofyear-1,365.2425)]:
        result[f'Calendar {name} Sin']=np.sin(2*np.pi*phase/period)
        result[f'Calendar {name} Cos']=np.cos(2*np.pi*phase/period)
    result['Calendar Days To Month End']=dates.days_in_month-dates.day
    result['Calendar Month Start']=dates.is_month_start.astype(float)
    result['Calendar Month End']=dates.is_month_end.astype(float)
    result['Calendar Quarter End']=dates.is_quarter_end.astype(float)
    return result
