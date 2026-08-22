import warnings
warnings.filterwarnings("ignore")

from features import *

from targets import *

from screening import *

from main_package import *


import logging
import pandas as pd
import yfinance as yf
import numpy as np


########################################
# Logging
########################################

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

logger = logging.getLogger(__name__)

high_liquidity_30 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "META", "TSLA", "AMD", "INTC", "QCOM", "MU",
    "CSCO", "ORCL", "JPM", "BAC", "WFC", "C", "XOM", "CVX", "F", "GM",
    "T", "VZ", "PFE", "JNJ", "WMT", "DIS", "GE", "HD", "NFLX", "GOOG"
]

medium_liquidity_30 = [
    "ADI", "MCHP", "TXN", "STX", "WDC", "PNC", "USB", "BK", "STT", "COF",
    "CAT", "DE", "EMR", "ETN", "ITW", "LOW", "TGT", "KR", "BBY", "DRI",
    "AMGN", "GILD", "BIIB", "BMY", "CVS", "OXY", "EOG", "SLB", "HAL", "VLO"
]

lower_liquidity_30 = [
    "AIT", "ARCB", "BRC", "CALM", "CHCO", "CPK", "CNMD", "FFIN", "GATX", "GBCI",
    "GFF", "HNI", "HVT", "JJSF", "LANC", "MGEE", "MLAB", "MMSI", "MTRN", "NRIM",
    "NWN", "RCKY", "RELL", "SCL", "TNC", "UVSP", "WASH", "WDFC", "NATH", "RES"
]

sector_matched_liquidity_30 = [
    "MSFT", "ADI", "RELL",
    "JPM", "PNC", "CHCO",
    "GE", "CAT", "AIT",
    "AMZN", "LOW", "HVT",
    "WMT", "KR", "JJSF",
    "JNJ", "AMGN", "CNMD",
    "XOM", "OXY", "RES",
    "NEE", "DUK", "CPK",
    "FCX", "NUE", "SCL",
    "PLD", "SPG", "UHT"
]

liquidity_barbell_30 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "META", "TSLA", "JPM", "BAC", "XOM", "CVX",
    "WMT", "JNJ", "GE", "HD", "NFLX",
    "AIT", "ARCB", "BRC", "CALM", "CHCO", "CPK", "CNMD", "FFIN", "GATX", "HNI",
    "JJSF", "MGEE", "MTRN", "TNC", "WDFC"
]

institutional_liquidity_60 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "META", "TSLA", "AMD", "INTC", "QCOM", "MU",
    "CSCO", "ORCL", "JPM", "BAC", "WFC", "C", "XOM", "CVX", "F", "GM",
    "T", "VZ", "PFE", "JNJ", "WMT", "DIS", "GE", "HD", "NFLX", "GOOG",

    "ADI", "MCHP", "TXN", "STX", "WDC", "PNC", "USB", "BK", "STT", "COF",
    "CAT", "DE", "EMR", "ETN", "ITW", "LOW", "TGT", "KR", "BBY", "DRI",
    "AMGN", "GILD", "BIIB", "BMY", "CVS", "OXY", "EOG", "SLB", "HAL", "VLO"
]

mid_small_liquidity_60 = [
    "ADI", "MCHP", "TXN", "STX", "WDC", "PNC", "USB", "BK", "STT", "COF",
    "CAT", "DE", "EMR", "ETN", "ITW", "LOW", "TGT", "KR", "BBY", "DRI",
    "AMGN", "GILD", "BIIB", "BMY", "CVS", "OXY", "EOG", "SLB", "HAL", "VLO",

    "AIT", "ARCB", "BRC", "CALM", "CHCO", "CPK", "CNMD", "FFIN", "GATX", "GBCI",
    "GFF", "HNI", "HVT", "JJSF", "LANC", "MGEE", "MLAB", "MMSI", "MTRN", "NRIM",
    "NWN", "RCKY", "RELL", "SCL", "TNC", "UVSP", "WASH", "WDFC", "NATH", "RES"
]

mid_large_liquidity_60 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "META", "TSLA", "AMD", "INTC", "QCOM", "MU",
        "CSCO", "ORCL", "JPM", "BAC", "WFC", "C", "XOM", "CVX", "F", "GM",
        "T", "VZ", "PFE", "JNJ", "WMT", "DIS", "GE", "HD", "NFLX", "GOOG",
        "ADI", "MCHP", "TXN", "STX", "WDC", "PNC", "USB", "BK", "STT", "COF",
        "CAT", "DE", "EMR", "ETN", "ITW", "LOW", "TGT", "KR", "BBY", "DRI",
        "AMGN", "GILD", "BIIB", "BMY", "CVS", "OXY", "EOG", "SLB", "HAL", "VLO"
]

full_liquidity_spectrum_90 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "META", "TSLA", "AMD", "INTC", "QCOM", "MU",
    "CSCO", "ORCL", "JPM", "BAC", "WFC", "C", "XOM", "CVX", "F", "GM",
    "T", "VZ", "PFE", "JNJ", "WMT", "DIS", "GE", "HD", "NFLX", "GOOG",

    "ADI", "MCHP", "TXN", "STX", "WDC", "PNC", "USB", "BK", "STT", "COF",
    "CAT", "DE", "EMR", "ETN", "ITW", "LOW", "TGT", "KR", "BBY", "DRI",
    "AMGN", "GILD", "BIIB", "BMY", "CVS", "OXY", "EOG", "SLB", "HAL", "VLO",

    "AIT", "ARCB", "BRC", "CALM", "CHCO", "CPK", "CNMD", "FFIN", "GATX", "GBCI",
    "GFF", "HNI", "HVT", "JJSF", "LANC", "MGEE", "MLAB", "MMSI", "MTRN", "NRIM",
    "NWN", "RCKY", "RELL", "SCL", "TNC", "UVSP", "WASH", "WDFC", "NATH", "RES"
]




tokens = liquidity_barbell_30

logger.info("Starting equity selector pipeline for %d stocks: %s", len(tokens), tokens)

########################################
# Download Data
########################################

logger.info("Downloading stock data")

raw_df = yf.download(
    tokens,
    start="2013-01-01",
    end="2023-09-30",
    auto_adjust=True,
    progress=False,
    group_by="ticker",
    multi_level_index=True
)

logger.info("Stock data downloaded: %d rows", len(raw_df))
logger.info("Downloading market benchmark")

market_df = yf.download(
    "^GSPC",
    start="2013-01-01",
    end="2023-09-30",
    auto_adjust=True,
    progress=False,
    multi_level_index=False
)

logger.info("Market benchmark downloaded: %d rows", len(market_df))

#######################################
# Screening Stocks
#######################################

stock_drops = []

logger.info("Starting stock screening for %d stocks", len(tokens))

for token in tokens:

    df = raw_df[token].copy()

    verdict = size_stocks(df)

    if verdict == "drop":
        logger.info("%s dropped: insufficient history", token)
        stock_drops.append(token)
        continue

    verdict = missingness_stocks(df)

    if verdict == "drop":
        logger.info("%s dropped: excessive missing data", token)
        stock_drops.append(token)
        continue

    verdict = invalid_stocks(df)

    if verdict == "drop":
        logger.info("%s dropped: invalid values", token)
        stock_drops.append(token)
        continue


    verdict = continuity_stocks(df, market_df)

    if verdict == "drop":
        logger.info("%s dropped: poor data continuity", token)
        stock_drops.append(token)
        continue

    logger.info("%s passed stock screening", token)


logger.info(
    "Stock screening complete: %d passed, %d dropped",
    len(tokens) - len(stock_drops),
    len(stock_drops)
)

raw_df = raw_df.drop(
    columns=stock_drops,
    level=0,
    errors="ignore"
)

tokens = [
    token for token in tokens
    if token not in stock_drops
]

if stock_drops:
    logger.info("Dropped stocks: %s", stock_drops)




########################################
# Build Features For Each Stock
########################################

stock_dfs = {}

for token in tokens:

    logger.info("Building individual-stock features for %s", token)

    df = raw_df[token].copy()

    df["Return"] = df["Close"].pct_change()

    columns_before = len(df.columns)

    # Individual stock
    df = all_return_features(df)
    df = all_momentum_features(df)
    df = all_volatility_features(df)
    df = all_range_volatility_features(df)
    df = all_trend_features(df)
    df = all_moving_average_features(df)
    df = all_drawdown_features(df)
    df = all_distribution_features(df)
    df = all_tail_risk_features(df)
    df = all_volume_features(df)
    df = all_liquidity_features(df)
    df = all_ohlc_features(df)

    # External comparison data
    df = all_market_relative_features(
        df,
        market_df=market_df
    )

    # df = all_sector_relative_features(df)

    df = all_beta_features(
        df,
        market_df=market_df
    )

    df = all_correlation_features(
        df,
        market_df=market_df
    )

    df = all_residual_features(
        df,
        market_df=market_df
    )

    # Derived
    df = all_technical_features(df)
    df = all_regime_features(df)
    df = all_interaction_features(df)
    df = all_composite_features(df)
    df = all_experimental_features(df)

    stock_dfs[token] = df

    logger.info(
        "%s complete: %d rows, %d generated feature columns",
        token,
        len(df),
        len(df.columns) - columns_before
    )


########################################
# Combine Stocks
########################################

logger.info("Combining stocks into panel dataframe")

panel_dfs = []

for token in tokens:

    stock_df = stock_dfs[token].copy()

    stock_df["Ticker"] = token
    stock_df["Date"] = stock_df.index

    panel_dfs.append(
        stock_df.reset_index(drop=True)
    )

df = pd.concat(
    panel_dfs,
    ignore_index=True
)

logger.info("Panel created: %d rows x %d columns", *df.shape)


base_columns = {
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Return",
    "Ticker",
    "Date"
}

features = [
    column
    for column in df.columns
    if column not in base_columns
]

logger.info("Individual-stock feature count before multi-stock features: %d", len(features))


########################################
# Multi-Stock Features
########################################

if df["Ticker"].nunique() > 1:

    logger.info("Multiple stocks detected; creating cross-stock features")

    columns_before = set(df.columns)

    df = all_cross_sectional_features(
        df,
        columns=features,
        date_col="Date"
    )

    cross_sectional_added = len(set(df.columns) - columns_before)
    logger.info("Cross-sectional features added: %d", cross_sectional_added)


    wide_df = df.pivot(
        index="Date",
        columns="Ticker",
        values=[
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Return"
        ]
    )

    original_num_columns = len(wide_df.columns)

    market_features = all_breadth_features(
        wide_df.copy()
    )

    market_features = all_dispersion_features(
        market_features
    )

    # Only keep features created by the breadth / dispersion functions
    new_market_features = market_features.iloc[
        :,
        original_num_columns:
    ].copy()

    # Remove MultiIndex from generated feature columns
    new_market_features.columns = [
        column[0] if isinstance(column, tuple) else column
        for column in new_market_features.columns
    ]

    market_feature_count = len(new_market_features.columns)

    new_market_features = new_market_features.reset_index()

    df = df.merge(
        new_market_features,
        on="Date",
        how="left"
    )

    logger.info("Breadth / dispersion features added: %d", market_feature_count)

else:
    logger.info("Only one stock detected; skipping cross-stock features")


features = [
    column
    for column in df.columns
    if column not in base_columns
]

logger.info("Total feature count before screening: %d", len(features))


########################################
# Feature Screening
########################################

screen_df = df[features].copy()

dropped_features = []

# Each screening function should return:
#     screen_df, to_drop
# where to_drop is the list of features removed by that screening.

screen_df, to_drop = missingness(screen_df)
dropped_features.extend(to_drop)
logger.info(
    "Missingness screening: dropped %d features, %d remain",
    len(to_drop),
    len(screen_df.columns)
)
logger.debug("Missingness dropped: %s", to_drop)

screen_df, to_drop = invalid_vals(screen_df)
dropped_features.extend(to_drop)
logger.info(
    "Invalid-value screening: dropped %d features, %d remain",
    len(to_drop),
    len(screen_df.columns)
)
logger.debug("Invalid-value dropped: %s", to_drop)

screen_df, to_drop = zero_variance(screen_df)
dropped_features.extend(to_drop)
logger.info(
    "Zero-variance screening: dropped %d features, %d remain",
    len(to_drop),
    len(screen_df.columns)
)
logger.debug("Zero-variance dropped: %s", to_drop)

screen_df = duplicates(screen_df)

screen_df, to_drop = correlations(screen_df)
dropped_features.extend(to_drop)
logger.info(
    "Correlation screening: dropped %d features, %d remain",
    len(to_drop),
    len(screen_df.columns)
)
logger.debug("Correlation features dropped: %s", to_drop)

features = list(screen_df.columns)

# Remove duplicates from cumulative dropped list while preserving order
dropped_features = list(dict.fromkeys(dropped_features))

logger.info(
    "Feature screening complete: %d total features dropped, %d retained",
    len(dropped_features),
    len(features)
)


########################################
# Individual Stock Targets
########################################

target_dfs = []
targets = None

for token in tokens:

    logger.info("Building targets for %s", token)

    target_df = stock_dfs[token].copy()

    columns_before_targets = set(
        target_df.columns
    )

    target_df = all_return_targets(
        target_df,
        benchmark_df=market_df
    )

    target_df = all_volatility_targets(
        target_df
    )

    target_df = all_direction_targets(
        target_df
    )

    target_df = all_barrier_targets(
        target_df
    )

    target_df = all_excursion_targets(
        target_df
    )

    target_df = all_drawdown_targets(
        target_df
    )

    target_df = all_risk_adjusted_targets(
        target_df
    )

    current_targets = [
        column
        for column in target_df.columns
        if column not in columns_before_targets
    ]

    if targets is None:
        targets = current_targets

    target_df["Ticker"] = token
    target_df["Date"] = target_df.index

    target_dfs.append(
        target_df[
            ["Date", "Ticker"] + current_targets
        ].reset_index(drop=True)
    )

    logger.info("%s targets created: %d", token, len(current_targets))


target_df = pd.concat(
    target_dfs,
    ignore_index=True
)

logger.info("Individual-stock target panel created: %d rows x %d columns", *target_df.shape)


########################################
# Final DataFrame
########################################

keep_columns = (
    [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Return",
        "Ticker",
        "Date"
    ]
    + features
)

df = df[keep_columns]

df = df.merge(
    target_df,
    on=["Date", "Ticker"],
    how="left"
)


########################################
# Multi-Stock Targets
########################################

if df["Ticker"].nunique() > 1:

    logger.info("Creating cross-sectional ranking targets")

    columns_before = set(df.columns)

    df = all_ranking_targets(
        df,
        ticker_col="Ticker",
        date_col="Date",
        price_col="Close"
    )

    ranking_targets = [
        column
        for column in df.columns
        if column not in columns_before
    ]

    targets += ranking_targets

    logger.info("Cross-sectional ranking targets added: %d", len(ranking_targets))


logger.info(
    "Pipeline complete: %d rows, %d features, %d targets, %d dropped features",
    len(df),
    len(features),
    len(targets),
    len(dropped_features)
)


########################################
# Feature Screening by Target
########################################

features_by_target = {
    target: features.copy()
    for target in targets
}

dropped_by_target = {}

train_df, validation_df, test_df = train_validation_test_split(df)

for target in targets:

    type_ = target_type(train_df, target)

    logger.info(
        f"[{targets.index(target)}/{len(targets)}] Screening target: %s | type: %s | starting features: %d",
        target,
        type_,
        len(features_by_target[target])
    )

    selected_features = features_by_target[target]
    dropped_by_target[target] = {}


    ########################################
    # All Target Types
    ########################################

    selected_features = run_screen(
        "Coverage",
        feature_target_coverage,
        train_df,
        selected_features,
        target,
        dropped_by_target
    )


    ########################################
    # Continuous Targets
    ########################################

    if type_ == "continuous":


        selected_features = run_screen(
            "Quantile Spread",
            quantile_spread,
            train_df,
            selected_features,
            target,
            dropped_by_target
        )

        selected_features = run_screen(
            "Quantile Monotonicity",
            quantile_monotonicity,
            train_df,
            selected_features,
            target,
            dropped_by_target
        )

        selected_features = run_screen(
                "Pearson",
                pearson_correlation,
                train_df,
                selected_features,
                target,
                dropped_by_target
        )

        if len(tokens) >= 30:
            selected_features = run_screen(
                "IC Screening",
               ic_screen,
                train_df,   
                selected_features,
                target,
                dropped_by_target
            )
                


    ########################################
    # Binary Targets
    ########################################

    elif type_ == "binary":

        selected_features = run_screen(
            "Quantile Spread",
            quantile_spread,
            train_df,
            selected_features,
            target,
            dropped_by_target
        )

        selected_features = run_screen(
            "Quantile Monotonicity",
            quantile_monotonicity,
            train_df,
            selected_features,
            target,
            dropped_by_target
        )

        selected_features = run_screen(
                "Pearson",
                pearson_correlation,
                train_df,
                selected_features,
                target,
                dropped_by_target
        )


    ########################################
    # Multiclass Targets
    ########################################

    elif type_ == "multiclass":

        # Only the universally applicable screens above:
        # Coverage
        # Time Stability
        pass

    selected_features = run_screen(
        "Time Stability",
        time_stability,
        train_df,
        selected_features,
        target,
        dropped_by_target
    )


    features_by_target[target] = selected_features

    logger.info(
        "Finished target: %s | final features: %d",
        target,
        len(selected_features)
    )


file = open('/Users/sam/Progressive-Projects/Projects/Equity Selector/data/Selected_Features.txt', 'a')
file.write(f'\n{str(features_by_target)}')
file.close()

used_features = set().union(*features_by_target.values())

screened_features = [
    feature for feature in features
    if feature in used_features
]

columns = (
    ["Date", "Ticker", "Open", "Close", "Low", "High", "Volume"]
    + targets
    + screened_features
)

# Remove duplicates while preserving order
columns = list(dict.fromkeys(columns))

screened_df = df[columns].copy()


import sqlite3

with sqlite3.connect(
    "/Users/sam/Progressive-Projects/Projects/Equity Selector/data/Features_Targets_Data.db"
) as connection:

    screened_df.to_sql(
        "Liquidity Barbell 30",
        connection,
        if_exists="replace",
        index=False
    )