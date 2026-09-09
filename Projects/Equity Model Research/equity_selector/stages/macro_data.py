from equity_selector.settings import setting as get_setting, callback, configured

"""data: original sequential research stage; shared logic is in equity_selector."""

from equity_selector.config import data_root
from features import *
from targets import *
from screening import *
from main_package import *


import os
import time
import heapq
from datetime import date
import requests
import pandas as pd

MACRO_SERIES = {

    # --------------------------------------------------------
    # RATES
    # --------------------------------------------------------

    "Macro_Fed_Funds":
        "DFF",

    "Macro_Treasury_3M":
        "DGS3MO",

    "Macro_Treasury_2Y":
        "DGS2",

    "Macro_Treasury_5Y":
        "DGS5",

    "Macro_Treasury_10Y":
        "DGS10",

    "Macro_Treasury_30Y":
        "DGS30",

    # --------------------------------------------------------
    # YIELD CURVE
    # --------------------------------------------------------

    "Macro_10Y_2Y_Spread":
        "T10Y2Y",

    "Macro_10Y_3M_Spread":
        "T10Y3M",

    # --------------------------------------------------------
    # INFLATION
    # --------------------------------------------------------

    "Macro_CPI":
        "CPIAUCSL",

    "Macro_Core_CPI":
        "CPILFESL",

    "Macro_PCE":
        "PCEPI",

    # --------------------------------------------------------
    # LABOUR
    # --------------------------------------------------------

    "Macro_Unemployment":
        "UNRATE",

    "Macro_Payrolls":
        "PAYEMS",

    "Macro_Initial_Claims":
        "ICSA",

    # --------------------------------------------------------
    # ECONOMIC ACTIVITY
    # --------------------------------------------------------

    "Macro_Industrial_Production":
        "INDPRO",

    "Macro_Retail_Sales":
        "RSAFS",

    "Macro_Real_GDP":
        "GDPC1",

    # --------------------------------------------------------
    # CREDIT
    # --------------------------------------------------------

    "Macro_High_Yield_Spread":
        "BAMLH0A0HYM2",

    "Macro_BAA_Treasury_Spread":
        "BAA10Y",

    # --------------------------------------------------------
    # MARKET RISK
    # --------------------------------------------------------

    "Macro_VIX":
        "VIXCLS",

    # --------------------------------------------------------
    # MONEY / LIQUIDITY
    # --------------------------------------------------------

    "Macro_M2":
        "M2SL",

    # --------------------------------------------------------
    # SENTIMENT
    # --------------------------------------------------------

    "Macro_Consumer_Sentiment":
        "UMCSENT",

    # --------------------------------------------------------
    # HOUSING
    # --------------------------------------------------------

    "Macro_Housing_Starts":
        "HOUST",

    # --------------------------------------------------------
    # DOLLAR
    # --------------------------------------------------------

    "Macro_USD_Index":
        "DTWEXBGS",
}

def macro_asof(intervals,start,end,lag=1):
    """For each prediction day, select the latest observation known at its cutoff."""
    days = pd.date_range(start,end,freq='D')
    result = pd.DataFrame({'Date':days.strftime('%Y-%m-%d')})
    for metric,group in intervals.groupby('Metric'):
        events = []
        for row in group.itertuples(index=False):
            obs = date.fromisoformat(row.Date).toordinal()
            lo = date.fromisoformat(row.VintageStart).toordinal()
            hi = date.fromisoformat(row.VintageEnd).toordinal()
            events.append((max(obs,lo),obs,lo,hi,row.Value,row.Date,row.VintageStart))
        events.sort(key=lambda e:e[:4])
        heap = []
        pointer = 0
        values,observations,vintages = [],[],[]
        for day in days:
            cutoff = day.date().toordinal()-lag
            while pointer<len(events) and events[pointer][0]<=cutoff:
                _,obs,lo,hi,value,obsdate,vintage = events[pointer]
                heapq.heappush(heap,(-obs,-lo,-hi,pointer,hi,value,obsdate,vintage))
                pointer += 1
            while heap and heap[0][4]<cutoff:
                heapq.heappop(heap)
            if heap:
                _,_,_,_,_,value,obsdate,vintage = heap[0]
                values.append(value); observations.append(obsdate); vintages.append(vintage)
            else:
                values.append(None); observations.append(None); vintages.append(None)
        result[metric] = values
        result[metric+'_ObservationDate'] = observations
        result[metric+'_VintageStart'] = vintages
    metrics = [c for c in result if c.startswith('Macro_') and not c.endswith(('_ObservationDate','_VintageStart'))]
    if not metrics:
        return pd.DataFrame()
    return result.loc[result[metrics].notna().any(axis=1)].reset_index(drop=True)

def download_macro_intervals(api_key, start, end, series):
    """Download ALFRED vintage intervals; log and skip failed windows."""
    import logging
    import time

    import pandas as pd
    import requests

    logger = logging.getLogger(__name__)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    records = []

    def redact(message):
        text = str(message)
        return text.replace(str(api_key), "[REDACTED]") if api_key else text

    with requests.Session() as session:
        for metric, series_id in series.items():
            window = start
            successful_windows = 0
            failed_windows = 0

            while window <= end:
                stop = min(
                    window + pd.DateOffset(years=1) - pd.Timedelta(days=1),
                    end,
                )
                offset = 0
                window_records = []
                failure = None

                while True:
                    params = dict(
                        api_key=api_key,
                        file_type="json",
                        series_id=series_id,
                        realtime_start=window.strftime("%Y-%m-%d"),
                        realtime_end=stop.strftime("%Y-%m-%d"),
                        observation_start=(
                            start - pd.DateOffset(years=5)
                        ).strftime("%Y-%m-%d"),
                        observation_end=end.strftime("%Y-%m-%d"),
                        output_type=1,
                        limit=100000,
                        offset=offset,
                    )

                    payload = None

                    for attempt in range(4):
                        time.sleep(0.6 * (2 ** attempt))

                        try:
                            response = session.get(
                                "https://api.stlouisfed.org/fred/series/observations",
                                params=params,
                                timeout=120,
                            )

                            if response.status_code != 200:
                                try:
                                    detail = response.json().get(
                                        "error_message",
                                        "No explanation returned",
                                    )
                                except (ValueError, AttributeError):
                                    detail = "Non-JSON error response"

                                failure = redact(
                                    f"HTTP {response.status_code} | {detail}"
                                )

                                if (
                                    response.status_code == 429
                                    or response.status_code >= 500
                                ):
                                    continue
                                break

                            candidate = response.json()

                            if (
                                not isinstance(candidate, dict)
                                or not isinstance(
                                    candidate.get("observations"), list
                                )
                                or "count" not in candidate
                            ):
                                failure = "Invalid response: observations/count missing"
                                continue

                            payload = candidate
                            failure = None
                            break

                        except (requests.RequestException, ValueError):
                            failure = "Network failure or invalid JSON response"

                    if payload is None:
                        failure = failure or "Retries exhausted"
                        break

                    try:
                        rows = payload["observations"]
                        total = int(payload["count"])
                        if total < 0:
                            raise ValueError("Negative count")

                        page_records = [
                            dict(
                                Series=series_id,
                                Metric=metric,
                                Date=row["date"],
                                VintageStart=row["realtime_start"],
                                VintageEnd=row["realtime_end"],
                                Value=pd.to_numeric(
                                    row["value"], errors="coerce"
                                ),
                            )
                            for row in rows
                        ]
                    except (KeyError, TypeError, ValueError, OverflowError):
                        failure = "Invalid observation rows or count"
                        break

                    window_records.extend(page_records)
                    offset += len(rows)

                    if offset >= total:
                        break

                    if not rows:
                        failure = "Incomplete pagination"
                        break

                if failure is not None:
                    failed_windows += 1
                    logger.warning(
                        "ALFRED %s | %s to %s | skipped: %s",
                        series_id,
                        window.strftime("%Y-%m-%d"),
                        stop.strftime("%Y-%m-%d"),
                        failure,
                    )
                else:
                    records.extend(window_records)
                    successful_windows += 1

                window = stop + pd.Timedelta(days=1)

            logger.info(
                "ALFRED %s | %d windows downloaded, %d skipped",
                series_id,
                successful_windows,
                failed_windows,
            )

    return pd.DataFrame(
        records,
        columns=[
            "Series",
            "Metric",
            "Date",
            "VintageStart",
            "VintageEnd",
            "Value",
        ],
    )

def run():
    global \
        base_columns, \
        column, \
        columns, \
        columns_before, \
        columns_before_targets, \
        connection, \
        cross_sectional_added, \
        current_targets, \
        df, \
        dropped_by_target, \
        dropped_features, \
        feature, \
        features, \
        features_by_target, \
        file, \
        full_liquidity_spectrum_90, \
        high_liquidity_30, \
        institutional_liquidity_60, \
        keep_columns, \
        liquidity_barbell_30, \
        logger, \
        logging, \
        lower_liquidity_30, \
        market_df, \
        market_feature_count, \
        market_features, \
        medium_liquidity_30, \
        mid_large_liquidity_60, \
        mid_small_liquidity_60, \
        new_market_features, \
        np, \
        original_num_columns, \
        panel_dfs, \
        pd, \
        purge_training_data, \
        ranking_targets, \
        raw_df, \
        screen_df, \
        screened_df, \
        screened_features, \
        sector_matched_liquidity_30, \
        selected_features, \
        sqlite3, \
        stock_df, \
        stock_dfs, \
        stock_drops, \
        target, \
        target_df, \
        target_dfs, \
        target_train_df, \
        targets, \
        test_df, \
        to_drop, \
        token, \
        tokens, \
        train_df, \
        type_, \
        used_features, \
        validation_df, \
        verdict, \
        warnings, \
        wide_df, \
        write_frame, \
        yf
    from equity_selector.database import write_frame
    from equity_selector.validation import purge_training_data, screening_training_rows
    import warnings
    import logging
    import pandas as pd
    import yfinance as yf
    import numpy as np

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
    )
    logger = logging.getLogger(__name__)
    high_liquidity_30 = get_setting(
        "high_liquidity_30",
        [
            "AAPL",
            "MSFT",
            "AMZN",
            "NVDA",
            "META",
            "TSLA",
            "AMD",
            "INTC",
            "QCOM",
            "MU",
            "CSCO",
            "ORCL",
            "JPM",
            "BAC",
            "WFC",
            "C",
            "XOM",
            "CVX",
            "F",
            "GM",
            "T",
            "VZ",
            "PFE",
            "JNJ",
            "WMT",
            "DIS",
            "GE",
            "HD",
            "NFLX",
            "GOOG",
        ],
    )
    medium_liquidity_30 = get_setting(
        "medium_liquidity_30",
        [
            "ADI",
            "MCHP",
            "TXN",
            "STX",
            "WDC",
            "PNC",
            "USB",
            "BK",
            "STT",
            "COF",
            "CAT",
            "DE",
            "EMR",
            "ETN",
            "ITW",
            "LOW",
            "TGT",
            "KR",
            "BBY",
            "DRI",
            "AMGN",
            "GILD",
            "BIIB",
            "BMY",
            "CVS",
            "OXY",
            "EOG",
            "SLB",
            "HAL",
            "VLO",
        ],
    )
    lower_liquidity_30 = get_setting(
        "lower_liquidity_30",
        [
            "AIT",
            "ARCB",
            "BRC",
            "CALM",
            "CHCO",
            "CPK",
            "CNMD",
            "FFIN",
            "GATX",
            "GBCI",
            "GFF",
            "HNI",
            "HVT",
            "JJSF",
            "LANC",
            "MGEE",
            "MLAB",
            "MMSI",
            "MTRN",
            "NRIM",
            "NWN",
            "RCKY",
            "RELL",
            "SCL",
            "TNC",
            "UVSP",
            "WASH",
            "WDFC",
            "NATH",
            "RES",
        ],
    )
    sector_matched_liquidity_30 = get_setting(
        "sector_matched_liquidity_30",
        [
            "MSFT",
            "ADI",
            "RELL",
            "JPM",
            "PNC",
            "CHCO",
            "GE",
            "CAT",
            "AIT",
            "AMZN",
            "LOW",
            "HVT",
            "WMT",
            "KR",
            "JJSF",
            "JNJ",
            "AMGN",
            "CNMD",
            "XOM",
            "OXY",
            "RES",
            "NEE",
            "DUK",
            "CPK",
            "FCX",
            "NUE",
            "SCL",
            "PLD",
            "SPG",
            "UHT",
        ],
    )
    liquidity_barbell_30 = get_setting(
        "liquidity_barbell_30",
        [
            "AAPL",
            "MSFT",
            "AMZN",
            "NVDA",
            "META",
            "TSLA",
            "JPM",
            "BAC",
            "XOM",
            "CVX",
            "WMT",
            "JNJ",
            "GE",
            "HD",
            "NFLX",
            "AIT",
            "ARCB",
            "BRC",
            "CALM",
            "CHCO",
            "CPK",
            "CNMD",
            "FFIN",
            "GATX",
            "HNI",
            "JJSF",
            "MGEE",
            "MTRN",
            "TNC",
            "WDFC",
        ],
    )
    institutional_liquidity_60 = get_setting(
        "institutional_liquidity_60",
        [
            "AAPL",
            "MSFT",
            "AMZN",
            "NVDA",
            "META",
            "TSLA",
            "AMD",
            "INTC",
            "QCOM",
            "MU",
            "CSCO",
            "ORCL",
            "JPM",
            "BAC",
            "WFC",
            "C",
            "XOM",
            "CVX",
            "F",
            "GM",
            "T",
            "VZ",
            "PFE",
            "JNJ",
            "WMT",
            "DIS",
            "GE",
            "HD",
            "NFLX",
            "GOOG",
            "ADI",
            "MCHP",
            "TXN",
            "STX",
            "WDC",
            "PNC",
            "USB",
            "BK",
            "STT",
            "COF",
            "CAT",
            "DE",
            "EMR",
            "ETN",
            "ITW",
            "LOW",
            "TGT",
            "KR",
            "BBY",
            "DRI",
            "AMGN",
            "GILD",
            "BIIB",
            "BMY",
            "CVS",
            "OXY",
            "EOG",
            "SLB",
            "HAL",
            "VLO",
        ],
    )
    mid_small_liquidity_60 = get_setting(
        "mid_small_liquidity_60",
        [
            "ADI",
            "MCHP",
            "TXN",
            "STX",
            "WDC",
            "PNC",
            "USB",
            "BK",
            "STT",
            "COF",
            "CAT",
            "DE",
            "EMR",
            "ETN",
            "ITW",
            "LOW",
            "TGT",
            "KR",
            "BBY",
            "DRI",
            "AMGN",
            "GILD",
            "BIIB",
            "BMY",
            "CVS",
            "OXY",
            "EOG",
            "SLB",
            "HAL",
            "VLO",
            "AIT",
            "ARCB",
            "BRC",
            "CALM",
            "CHCO",
            "CPK",
            "CNMD",
            "FFIN",
            "GATX",
            "GBCI",
            "GFF",
            "HNI",
            "HVT",
            "JJSF",
            "LANC",
            "MGEE",
            "MLAB",
            "MMSI",
            "MTRN",
            "NRIM",
            "NWN",
            "RCKY",
            "RELL",
            "SCL",
            "TNC",
            "UVSP",
            "WASH",
            "WDFC",
            "NATH",
            "RES",
        ],
    )
    mid_large_liquidity_60 = get_setting(
        "mid_large_liquidity_60",
        [
            "AAPL",
            "MSFT",
            "AMZN",
            "NVDA",
            "META",
            "TSLA",
            "AMD",
            "INTC",
            "QCOM",
            "MU",
            "CSCO",
            "ORCL",
            "JPM",
            "BAC",
            "WFC",
            "C",
            "XOM",
            "CVX",
            "F",
            "GM",
            "T",
            "VZ",
            "PFE",
            "JNJ",
            "WMT",
            "DIS",
            "GE",
            "HD",
            "NFLX",
            "GOOG",
            "ADI",
            "MCHP",
            "TXN",
            "STX",
            "WDC",
            "PNC",
            "USB",
            "BK",
            "STT",
            "COF",
            "CAT",
            "DE",
            "EMR",
            "ETN",
            "ITW",
            "LOW",
            "TGT",
            "KR",
            "BBY",
            "DRI",
            "AMGN",
            "GILD",
            "BIIB",
            "BMY",
            "CVS",
            "OXY",
            "EOG",
            "SLB",
            "HAL",
            "VLO",
        ],
    )
    full_liquidity_spectrum_90 = get_setting(
        "full_liquidity_spectrum_90",
        [
            "AAPL",
            "MSFT",
            "AMZN",
            "NVDA",
            "META",
            "TSLA",
            "AMD",
            "INTC",
            "QCOM",
            "MU",
            "CSCO",
            "ORCL",
            "JPM",
            "BAC",
            "WFC",
            "C",
            "XOM",
            "CVX",
            "F",
            "GM",
            "T",
            "VZ",
            "PFE",
            "JNJ",
            "WMT",
            "DIS",
            "GE",
            "HD",
            "NFLX",
            "GOOG",
            "ADI",
            "MCHP",
            "TXN",
            "STX",
            "WDC",
            "PNC",
            "USB",
            "BK",
            "STT",
            "COF",
            "CAT",
            "DE",
            "EMR",
            "ETN",
            "ITW",
            "LOW",
            "TGT",
            "KR",
            "BBY",
            "DRI",
            "AMGN",
            "GILD",
            "BIIB",
            "BMY",
            "CVS",
            "OXY",
            "EOG",
            "SLB",
            "HAL",
            "VLO",
            "AIT",
            "ARCB",
            "BRC",
            "CALM",
            "CHCO",
            "CPK",
            "CNMD",
            "FFIN",
            "GATX",
            "GBCI",
            "GFF",
            "HNI",
            "HVT",
            "JJSF",
            "LANC",
            "MGEE",
            "MLAB",
            "MMSI",
            "MTRN",
            "NRIM",
            "NWN",
            "RCKY",
            "RELL",
            "SCL",
            "TNC",
            "UVSP",
            "WASH",
            "WDFC",
            "NATH",
            "RES",
        ],
    )
    universe_tokens = {
        "High Liquidity 30": high_liquidity_30,
        "Medium Liquidity 30": medium_liquidity_30,
        "Lower Liquidity 30": lower_liquidity_30,
        "Sector Spread 30": sector_matched_liquidity_30,
        "Liquidity Barbell 30": liquidity_barbell_30,
        "Institutional Liquidity 60": institutional_liquidity_60,
        "Medium Small Liquidity 60": mid_small_liquidity_60,
        "Medium Large Liquidity 60": mid_large_liquidity_60,
        "All Liquidity 90": full_liquidity_spectrum_90,
    }
    universe_name = get_setting("STOCK_TYPE", "Liquidity Barbell 30")
    tokens = get_setting("TOKENS", universe_tokens.get(universe_name))
    if not tokens:
        raise ValueError("Choose a known STOCK_TYPE or provide a nonempty TOKENS list")
    logger.info("Starting equity selector pipeline for %d stocks: %s", len(tokens), tokens)
    start = pd.Timestamp(get_setting("DOWNLOAD_START", "2013-01-01"))
    end = pd.Timestamp(get_setting("DOWNLOAD_END", pd.Timestamp.today().normalize().strftime("%Y-%m-%d")))
    if end <= start:
        raise ValueError("DOWNLOAD_END must be after DOWNLOAD_START (end is exclusive)")
    import os
    api_key = 'e2b007722624040478c40d718fe9ed64'
    if not api_key:
        raise ValueError("Set ALFRED_API_KEY in settings, or the FRED_API_KEY environment variable")
    tokens = list(dict.fromkeys(str(t).strip().upper() for t in tokens))
    lag = int(get_setting("MACRO_RELEASE_LAG_DAYS", 1))
    if lag < 1:
        raise ValueError("Use at least one day of macro release lag")
    warm_start = start - pd.DateOffset(years=3) - pd.Timedelta(days=lag)
    intervals = download_macro_intervals(api_key, warm_start, end-pd.Timedelta(days=1),
                                         get_setting("MACRO_SERIES", MACRO_SERIES))
    if intervals.empty:
        raise ValueError("No ALFRED vintage observations were returned")
    df = macro_asof(intervals, warm_start, end-pd.Timedelta(days=1), lag=lag)
    df["Date"] = pd.to_datetime(df["Date"])
    df = configured(all_macro_quality_features, df)
    df = configured(all_macro_history_features, df)
    df = configured(all_macro_rates_features, df)
    df = configured(all_macro_conditions_features, df)
    df = configured(all_macro_calendar_features, df)
    df = configured(all_macro_vintage_features, df, intervals=intervals, release_lag_days=lag)
    df = df.loc[(df.Date >= start) & (df.Date < end)].copy()

    import sqlite3

    from equity_selector.macro_regimes import macro_database, save_macro_regimes

    with sqlite3.connect(macro_database()) as connection:
        write_frame(
            df,
            get_setting("STOCK_TYPE", "Liquidity Barbell 30"),
            connection,
            if_exists="replace",
            index=False,
        )


    save_macro_regimes(table=get_setting("STOCK_TYPE", "Liquidity Barbell 30"))
