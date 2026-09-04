from equity_selector.settings import setting as get_setting, callback, configured

"""data: original sequential research stage; shared logic is in equity_selector."""

from equity_selector.config import data_root
from features import *
from targets import *
from screening import *
from main_package import *


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
    logger.info("Downloading stock data")
    raw_df = yf.download(
        tokens,
        start=get_setting("DOWNLOAD_START", "2013-01-01"),
        end=get_setting("DOWNLOAD_END", "2023-09-30"),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        multi_level_index=True,
    )
    logger.info("Stock data downloaded: %d rows", len(raw_df))
    logger.info("Downloading market benchmark")
    market_df = yf.download(
        get_setting("MARKET_TICKER", "^GSPC"),
        start=get_setting("DOWNLOAD_START", "2013-01-01"),
        end=get_setting("DOWNLOAD_END", "2023-09-30"),
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )
    logger.info("Market benchmark downloaded: %d rows", len(market_df))
    stock_drops = []
    logger.info("Starting stock screening for %d stocks", len(tokens))
    screening_market = screening_training_rows(market_df, raw_df.index)
    for token in tokens:
        df = screening_training_rows(raw_df[token], raw_df.index)
        verdict = configured(size_stocks, df)
        if verdict == "drop":
            logger.info("%s dropped: insufficient history", token)
            stock_drops.append(token)
            continue
        verdict = configured(missingness_stocks, df)
        if verdict == "drop":
            logger.info("%s dropped: excessive missing data", token)
            stock_drops.append(token)
            continue
        verdict = configured(invalid_stocks, df)
        if verdict == "drop":
            logger.info("%s dropped: invalid values", token)
            stock_drops.append(token)
            continue
        verdict = configured(continuity_stocks, df, screening_market)
        if verdict == "drop":
            logger.info("%s dropped: poor data continuity", token)
            stock_drops.append(token)
            continue
        logger.info("%s passed stock screening", token)
    logger.info(
        "Stock screening complete: %d passed, %d dropped",
        len(tokens) - len(stock_drops),
        len(stock_drops),
    )
    raw_df = raw_df.drop(columns=stock_drops, level=0, errors="ignore")
    tokens = [token for token in tokens if token not in stock_drops]
    if stock_drops:
        logger.info("Dropped stocks: %s", stock_drops)
    stock_dfs = {}
    for token in tokens:
        logger.info("Building individual-stock features for %s", token)
        df = raw_df[token].copy()
        df["Return"] = df["Close"].pct_change()
        columns_before = len(df.columns)
        df = configured(all_return_features, df)
        df = configured(all_momentum_features, df)
        df = configured(all_volatility_features, df)
        df = configured(all_range_volatility_features, df)
        df = configured(all_trend_features, df)
        df = configured(all_moving_average_features, df)
        df = configured(all_drawdown_features, df)
        df = configured(all_distribution_features, df)
        df = configured(all_tail_risk_features, df)
        df = configured(all_volume_features, df)
        df = configured(all_liquidity_features, df)
        df = configured(all_ohlc_features, df)
        df = configured(all_market_relative_features, df, market_df=market_df)
        df = configured(all_beta_features, df, market_df=market_df)
        df = configured(all_correlation_features, df, market_df=market_df)
        df = configured(all_residual_features, df, market_df=market_df)
        df = configured(all_technical_features, df)
        df = configured(all_regime_features, df)
        df = configured(all_interaction_features, df)
        df = configured(all_composite_features, df)
        df = configured(all_experimental_features, df)
        stock_dfs[token] = df
        logger.info(
            "%s complete: %d rows, %d generated feature columns",
            token,
            len(df),
            len(df.columns) - columns_before,
        )
    logger.info("Combining stocks into panel dataframe")
    panel_dfs = []
    for token in tokens:
        stock_df = stock_dfs[token].copy()
        stock_df["Ticker"] = token
        stock_df["Date"] = stock_df.index
        panel_dfs.append(stock_df.reset_index(drop=True))
    df = pd.concat(panel_dfs, ignore_index=True)
    logger.info("Panel created: %d rows x %d columns", *df.shape)
    base_columns = {"Open", "High", "Low", "Close", "Volume", "Return", "Ticker", "Date"}
    features = [column for column in df.columns if column not in base_columns]
    logger.info("Individual-stock feature count before multi-stock features: %d", len(features))
    if df["Ticker"].nunique() > 1:
        logger.info("Multiple stocks detected; creating cross-stock features")
        columns_before = set(df.columns)
        df = configured(all_cross_sectional_features, df, columns=features, date_col="Date")
        cross_sectional_added = len(set(df.columns) - columns_before)
        logger.info("Cross-sectional features added: %d", cross_sectional_added)
        wide_df = df.pivot(
            index="Date",
            columns="Ticker",
            values=["Open", "High", "Low", "Close", "Volume", "Return"],
        )
        original_num_columns = len(wide_df.columns)
        market_features = configured(all_breadth_features, wide_df.copy())
        market_features = configured(all_dispersion_features, market_features)
        new_market_features = market_features.iloc[:, original_num_columns:].copy()
        new_market_features.columns = [
            column[0] if isinstance(column, tuple) else column
            for column in new_market_features.columns
        ]
        market_feature_count = len(new_market_features.columns)
        new_market_features = new_market_features.reset_index()
        df = df.merge(new_market_features, on="Date", how="left")
        logger.info("Breadth / dispersion features added: %d", market_feature_count)
    else:
        logger.info("Only one stock detected; skipping cross-stock features")
    features = [column for column in df.columns if column not in base_columns]
    logger.info("Total feature count before screening: %d", len(features))
    screen_df = screening_training_rows(df)[features].copy()
    dropped_features = []
    screen_df, to_drop = configured(missingness, screen_df)
    dropped_features.extend(to_drop)
    logger.info(
        "Missingness screening: dropped %d features, %d remain",
        len(to_drop),
        len(screen_df.columns),
    )
    logger.debug("Missingness dropped: %s", to_drop)
    screen_df, to_drop = configured(invalid_vals, screen_df)
    dropped_features.extend(to_drop)
    logger.info(
        "Invalid-value screening: dropped %d features, %d remain",
        len(to_drop),
        len(screen_df.columns),
    )
    logger.debug("Invalid-value dropped: %s", to_drop)
    screen_df, to_drop = configured(zero_variance, screen_df)
    dropped_features.extend(to_drop)
    logger.info(
        "Zero-variance screening: dropped %d features, %d remain",
        len(to_drop),
        len(screen_df.columns),
    )
    logger.debug("Zero-variance dropped: %s", to_drop)
    screen_df = configured(duplicates, screen_df)
    screen_df, to_drop = configured(correlations, screen_df)
    dropped_features.extend(to_drop)
    logger.info(
        "Correlation screening: dropped %d features, %d remain",
        len(to_drop),
        len(screen_df.columns),
    )
    logger.debug("Correlation features dropped: %s", to_drop)
    features = list(screen_df.columns)
    dropped_features = list(dict.fromkeys(dropped_features))
    logger.info(
        "Feature screening complete: %d total features dropped, %d retained",
        len(dropped_features),
        len(features),
    )
    target_dfs = []
    targets = None
    for token in tokens:
        logger.info("Building targets for %s", token)
        target_df = stock_dfs[token].copy()
        columns_before_targets = set(target_df.columns)
        target_df = configured(all_return_targets, target_df, benchmark_df=market_df)
        target_df = configured(all_volatility_targets, target_df)
        target_df = configured(all_direction_targets, target_df)
        target_df = configured(all_barrier_targets, target_df)
        target_df = configured(all_excursion_targets, target_df)
        target_df = configured(all_drawdown_targets, target_df)
        target_df = configured(all_risk_adjusted_targets, target_df)
        current_targets = [
            column for column in target_df.columns if column not in columns_before_targets
        ]
        if targets is None:
            targets = current_targets
        target_df["Ticker"] = token
        target_df["Date"] = target_df.index
        target_dfs.append(target_df[["Date", "Ticker"] + current_targets].reset_index(drop=True))
        logger.info("%s targets created: %d", token, len(current_targets))
    target_df = pd.concat(target_dfs, ignore_index=True)
    logger.info("Individual-stock target panel created: %d rows x %d columns", *target_df.shape)
    keep_columns = ["Open", "High", "Low", "Close", "Volume", "Return", "Ticker", "Date"] + features
    df = df[keep_columns]
    df = df.merge(target_df, on=["Date", "Ticker"], how="left")
    if df["Ticker"].nunique() > 1:
        logger.info("Creating cross-sectional ranking targets")
        columns_before = set(df.columns)
        df = configured(
            all_ranking_targets, df, ticker_col="Ticker", date_col="Date", price_col="Close"
        )
        ranking_targets = [column for column in df.columns if column not in columns_before]
        targets += ranking_targets
        logger.info("Cross-sectional ranking targets added: %d", len(ranking_targets))
    logger.info(
        "Pipeline complete: %d rows, %d features, %d targets, %d dropped features",
        len(df),
        len(features),
        len(targets),
        len(dropped_features),
    )
    features_by_target = {target: features.copy() for target in targets}
    dropped_by_target = {}
    train_df, validation_df, test_df = train_validation_test_split(df)
    for target in targets:
        target_train_df = purge_training_data(train_df, target_purge_days(target))
        type_ = target_type(target_train_df, target)
        logger.info(
            f"[{targets.index(target)}/{len(targets)}] Screening target: %s | type: %s | starting features: %d",
            target,
            type_,
            len(features_by_target[target]),
        )
        selected_features = features_by_target[target]
        dropped_by_target[target] = {}
        selected_features = run_screen(
            "Coverage",
            feature_target_coverage,
            target_train_df,
            selected_features,
            target,
            dropped_by_target,
        )
        if type_ == "continuous":
            selected_features = run_screen(
                "Quantile Spread",
                quantile_spread,
                target_train_df,
                selected_features,
                target,
                dropped_by_target,
            )
            selected_features = run_screen(
                "Quantile Monotonicity",
                quantile_monotonicity,
                target_train_df,
                selected_features,
                target,
                dropped_by_target,
            )
            selected_features = run_screen(
                "Pearson",
                pearson_correlation,
                target_train_df,
                selected_features,
                target,
                dropped_by_target,
            )
            if len(tokens) >= 30:
                selected_features = run_screen(
                    "IC Screening",
                    ic_screen,
                    target_train_df,
                    selected_features,
                    target,
                    dropped_by_target,
                )
        elif type_ == "binary":
            selected_features = run_screen(
                "Quantile Spread",
                quantile_spread,
                target_train_df,
                selected_features,
                target,
                dropped_by_target,
            )
            selected_features = run_screen(
                "Quantile Monotonicity",
                quantile_monotonicity,
                target_train_df,
                selected_features,
                target,
                dropped_by_target,
            )
            selected_features = run_screen(
                "Pearson",
                pearson_correlation,
                target_train_df,
                selected_features,
                target,
                dropped_by_target,
            )
        elif type_ == "multiclass":
            pass
        selected_features = run_screen(
            "Time Stability",
            time_stability,
            target_train_df,
            selected_features,
            target,
            dropped_by_target,
        )
        features_by_target[target] = selected_features
        logger.info("Finished target: %s | final features: %d", target, len(selected_features))
    from equity_selector.feature_mapping import updated_feature_mapping_text
    from equity_selector.files import commit_with_text

    mapping_text = updated_feature_mapping_text(
        data_root() / "Selected_Features.txt",
        get_setting("STOCK_TYPE", "Liquidity Barbell 30"),
        features_by_target,
    )
    used_features = set().union(*features_by_target.values())
    screened_features = [feature for feature in features if feature in used_features]
    columns = (
        ["Date", "Ticker", "Open", "Close", "Low", "High", "Volume"] + targets + screened_features
    )
    columns = list(dict.fromkeys(columns))
    screened_df = df[columns].copy()
    import sqlite3

    with sqlite3.connect(str(data_root()) + "/Features_Targets_Data.db") as connection:
        write_frame(
            screened_df,
            get_setting("STOCK_TYPE", "Liquidity Barbell 30"),
            connection,
            if_exists="replace",
            index=False,
        )
        commit_with_text(connection, data_root() / "Selected_Features.txt", mapping_text)
