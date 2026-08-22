
"""Option-chain download, cleaning, pricing comparisons and recommendations."""

import logging
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from options_risk_engine.config import (
    MAX_MONEYNESS, MAX_SPREAD_PCT, MIN_ASK, MIN_MONEYNESS,
    MIN_OPEN_INTEREST, MIN_VOLUME,
)
from options_risk_engine.data.market_data import latest_feature_value
from options_risk_engine.domain import OptionTicker
from options_risk_engine.pricing.black_scholes import black_scholes
from options_risk_engine.pricing.greeks import add_greek_columns
from options_risk_engine.pricing.implied_volatility import add_implied_volatility_columns
from options_risk_engine.utils import safe_relative_edge

logger = logging.getLogger(__name__)
OptionChainRecord = tuple[pd.DataFrame, str]

def choose_closest_expiry(
    expiries: list[str],
    target_date: pd.Timestamp,
) -> str:
    """Choose the listed expiry closest to the requested target date."""

    expiry_list = list(expiries)

    if not expiry_list:
        raise ValueError("No option expiries were supplied")

    return min(
        expiry_list,
        key=lambda expiry: abs(
            pd.Timestamp(expiry).normalize()
            - target_date.normalize()
        ),
    )


def clean_downloaded_chain(
    chain: pd.DataFrame,
    iv_floor: float = 0.000011,
) -> pd.DataFrame:
    """Copy an option chain and replace unusable quotes and IV values."""

    cleaned = chain.copy()

    if "impliedVolatility" in cleaned.columns:
        cleaned["impliedVolatility"] = cleaned[
            "impliedVolatility"
        ].where(
            cleaned["impliedVolatility"] > iv_floor
        )

    quote_columns = [
        column
        for column in ("bid", "ask")
        if column in cleaned.columns
    ]

    if quote_columns:
        cleaned[quote_columns] = cleaned[quote_columns].replace(
            0.0,
            np.nan,
        )

    return cleaned


def download_option_chains(
    tickers: list[OptionTicker],
) -> list[OptionTicker]:
    """Download and attach the nearest target-DTE chain to each ticker."""

    logger.info(
        "Downloading option chains for %d ticker objects",
        len(tickers),
    )

    loaded = 0

    for ticker in tickers:
        logger.info(
            "Downloading option chain for %s nearest to %s",
            ticker.symbol,
            ticker.target_date.date(),
        )

        try:
            yf_ticker = yf.Ticker(ticker.symbol)
            expiries = yf_ticker.options

            if not expiries:
                logger.warning(
                    "No option expiries found for %s",
                    ticker.symbol,
                )
                continue

            expiry = choose_closest_expiry(
                expiries=expiries,
                target_date=ticker.target_date,
            )
            downloaded_chain = yf_ticker.option_chain(expiry)

            calls = clean_downloaded_chain(downloaded_chain.calls)
            puts = clean_downloaded_chain(downloaded_chain.puts)

            ticker.attach_option_chains(
                calls=calls,
                puts=puts,
                expiry=expiry,
            )
            loaded += 1

            logger.info(
                "%s chain loaded for %s: %d calls and %d puts",
                ticker.symbol,
                ticker.expiry_string,
                len(calls),
                len(puts),
            )

        except Exception:
            logger.exception(
                "Failed to download option chain for %s",
                ticker.symbol,
            )

    logger.info(
        "Option-chain download complete: %d of %d tickers loaded",
        loaded,
        len(tickers),
    )

    return tickers


def get_volatility_inputs(
    data: pd.DataFrame,
    ticker: OptionTicker,
) -> dict[str, float]:
    """Collect and store realised and forecast volatility inputs."""

    target_values = (
        data[("Target_RV", ticker.symbol)]
        .dropna()
        .tail(ticker.forecast_lookback)
    )

    if target_values.empty:
        raise ValueError(
            f"No Target_RV values available for {ticker.symbol}"
        )

    volatility_inputs = {
        "RV20": latest_feature_value(data, "RV20", ticker),
        "RV60": latest_feature_value(data, "RV60", ticker),
        "RV252": latest_feature_value(data, "RV252", ticker),
        "ForeV": float(target_values.mean()),
    }

    ticker.set_volatility_inputs(volatility_inputs)

    logger.info(
        "%s volatility inputs - RV20: %.4f, RV60: %.4f, "
        "RV252: %.4f, forecast: %.4f",
        ticker.symbol,
        ticker.volatility_inputs["RV20"],
        ticker.volatility_inputs["RV60"],
        ticker.volatility_inputs["RV252"],
        ticker.volatility_inputs["ForeV"],
    )

    return ticker.volatility_inputs


def as_aligned_series(
    values: Any,
    index: pd.Index,
    name: str,
) -> pd.Series:
    """Convert model output to a Series aligned to an option-chain index."""

    array = np.asarray(values, dtype=float)

    if array.ndim != 1 or len(array) != len(index):
        raise ValueError(
            f"{name} has shape {array.shape}; expected ({len(index)},)"
        )

    return pd.Series(array, index=index, name=name)


def calculate_black_scholes_scenarios(
    ticker: OptionTicker,
) -> dict[str, dict[str, pd.Series]]:
    """Price calls and puts under the ticker object's volatility inputs."""

    ticker.ensure_pricing_ready()

    scenario_prices: dict[str, dict[str, pd.Series]] = {
        "calls": {},
        "puts": {},
    }

    iv_call_prices = black_scholes(
        ticker=ticker,
        spot=ticker.current_price,
        strike=ticker.call_chain["strike"],
        time_to_expiry=ticker.time_to_expiry,
        risk_free_rate=ticker.risk_free_rate,
        dividend_yield=ticker.dividend_yield,
        volatility=ticker.call_chain["impliedVolatility"],
        option_type="call",
    )

    iv_put_prices = black_scholes(
        ticker=ticker,
        spot=ticker.current_price,
        strike=ticker.put_chain["strike"],
        time_to_expiry=ticker.time_to_expiry,
        risk_free_rate=ticker.risk_free_rate,
        dividend_yield=ticker.dividend_yield,
        volatility=ticker.put_chain["impliedVolatility"],
        option_type="put",
    )

    scenario_prices["calls"]["IV"] = as_aligned_series(
        iv_call_prices,
        ticker.call_chain.index,
        "BS_IV",
    )
    scenario_prices["puts"]["IV"] = as_aligned_series(
        iv_put_prices,
        ticker.put_chain.index,
        "BS_IV",
    )

    for model_name, sigma in ticker.volatility_inputs.items():
        model_call_prices = black_scholes(
            ticker=ticker,
            spot=ticker.current_price,
            strike=ticker.call_chain["strike"],
            time_to_expiry=ticker.time_to_expiry,
            risk_free_rate=ticker.risk_free_rate,
            dividend_yield=ticker.dividend_yield,
            volatility=sigma,
            option_type="call"
        )

        model_put_prices = black_scholes(
            ticker=ticker,
            spot=ticker.current_price,
            strike=ticker.put_chain["strike"],
            time_to_expiry=ticker.time_to_expiry,
            risk_free_rate=ticker.risk_free_rate,
            dividend_yield=ticker.dividend_yield,
            volatility=sigma,
            option_type="put"
        )

        scenario_prices["calls"][model_name] = as_aligned_series(
            model_call_prices,
            ticker.call_chain.index,
            f"BS_{model_name}",
        )
        scenario_prices["puts"][model_name] = as_aligned_series(
            model_put_prices,
            ticker.put_chain.index,
            f"BS_{model_name}",
        )

    return scenario_prices


def build_comparison_table(
    chain: pd.DataFrame,
    model_prices: Mapping[str, pd.Series],
    ticker: OptionTicker,
) -> pd.DataFrame:
    """Build one market-versus-model comparison table."""

    base_columns = [
        "contractSymbol",
        "strike",
        "bid",
        "ask",
        "lastPrice",
        "lastTradeDate",
        "impliedVolatility",
        "volume",
        "openInterest",
    ]

    comparison = chain.reindex(columns=base_columns).copy()
    comparison["Current Stock Price"] = ticker.current_price
    comparison["Expiry"] = ticker.expiry
    comparison["MarketMid"] = (
        comparison["bid"] + comparison["ask"]
    ) / 2

    for model_name, prices in model_prices.items():
        price_column = f"BS_{model_name}"

        if model_name == "IV":
            comparison["IV Used"] = comparison["impliedVolatility"]
        else:
            comparison[f"{model_name} Used"] = (
                ticker.volatility_inputs[model_name]
            )

        comparison[price_column] = prices.reindex(comparison.index)

        for market_column, market_name in (
            ("MarketMid", "Mid"),
            ("ask", "Ask"),
            ("bid", "Bid"),
        ):
            comparison[f"{price_column} - {market_name}"] = (
                comparison[price_column]
                - comparison[market_column]
            )
            comparison[f"{price_column} {market_name}Edge"] = (
                safe_relative_edge(
                    comparison[price_column],
                    comparison[market_column],
                )
            )

    return comparison


def check_put_call_parity(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    ticker: OptionTicker,
    tolerance: Optional[float] = None,
) -> pd.DataFrame:
    """Check European put-call parity using the ticker object's inputs."""

    tolerance = (
        ticker.parity_tolerance
        if tolerance is None
        else tolerance
    )

    call_values = (
        calls[["strike", "BS_ForeV"]]
        .dropna()
        .drop_duplicates(subset="strike")
        .rename(columns={"BS_ForeV": "Call_BS_ForeV"})
    )
    put_values = (
        puts[["strike", "BS_ForeV"]]
        .dropna()
        .drop_duplicates(subset="strike")
        .rename(columns={"BS_ForeV": "Put_BS_ForeV"})
    )

    parity = call_values.merge(
        put_values,
        on="strike",
        how="inner",
        validate="one_to_one",
    )

    if parity.empty:
        logger.warning(
            "%s has no matching call and put strikes for parity checking",
            ticker.symbol,
        )
        return parity

    parity["Theoretical C-P"] = (
        ticker.current_price
        * np.exp(
            -ticker.dividend_yield
            * ticker.time_to_expiry
        )
        - parity["strike"]
        * np.exp(
            -ticker.risk_free_rate
            * ticker.time_to_expiry
        )
    )
    parity["Observed C-P"] = (
        parity["Call_BS_ForeV"]
        - parity["Put_BS_ForeV"]
    )
    parity["Parity Error"] = (
        parity["Observed C-P"]
        - parity["Theoretical C-P"]
    )
    parity["Absolute Parity Error"] = parity["Parity Error"].abs()
    parity["Within Tolerance"] = (
        parity["Absolute Parity Error"] <= tolerance
    )

    maximum_error = float(parity["Absolute Parity Error"].max())
    failed_count = int((~parity["Within Tolerance"]).sum())

    logger.info(
        "%s put-call parity check: %d matched strikes, "
        "maximum absolute error %.10f",
        ticker.symbol,
        len(parity),
        maximum_error,
    )

    if failed_count:
        logger.warning(
            "%s has %d strikes outside the %.2e parity tolerance",
            ticker.symbol,
            failed_count,
            tolerance,
        )

    ticker.parity_table = parity
    return parity


def attach_parity_errors(
    options: pd.DataFrame,
    parity: pd.DataFrame,
) -> pd.DataFrame:
    """Attach parity-error columns to a call or put comparison table."""

    if parity.empty:
        result = options.copy()
        result["Parity Error"] = np.nan
        result["Absolute Parity Error"] = np.nan
        result["Parity Valid"] = False
        return result

    parity_columns = parity[[
        "strike",
        "Parity Error",
        "Absolute Parity Error",
        "Within Tolerance",
    ]].rename(columns={"Within Tolerance": "Parity Valid"})

    return options.merge(
        parity_columns,
        on="strike",
        how="left",
        validate="many_to_one",
    )


def add_recommendation_metrics(
    options: pd.DataFrame,
    buy_edge: float,
) -> pd.DataFrame:
    """Add spread, moneyness and eligibility-aware recommendations."""

    result = options.copy()

    valid_midpoint = result["MarketMid"] > 0
    result["SpreadPct"] = np.nan
    result.loc[valid_midpoint, "SpreadPct"] = (
        result.loc[valid_midpoint, "ask"]
        - result.loc[valid_midpoint, "bid"]
    ) / result.loc[valid_midpoint, "MarketMid"]

    result["Moneyness"] = (
        result["strike"]
        / result["Current Stock Price"]
    )

    edge = result["BS_ForeV AskEdge"]
    missing_quote = result["bid"].isna() | result["ask"].isna()
    quote_valid = result["Quote Valid"].fillna(False)
    eligible = result["Recommendation Eligible"].fillna(False)

    result["Initial Recommended Action"] = np.select(
        condlist=[
            missing_quote,
            ~quote_valid,
            ~eligible,
            edge.isna(),
            edge >= buy_edge,
            (edge > 0) & (edge < buy_edge),
            edge <= 0,
        ],
        choicelist=[
            "No Data",
            "Invalid Quote",
            "Ineligible",
            "No Data",
            "Buy",
            "Positive Edge",
            "Do Not Buy",
        ],
        default="No Data",
    )

    return result


def filter_highlighted_options(
    options: pd.DataFrame,
    buy_edge: float,
    max_spread_pct: float = MAX_SPREAD_PCT,
    min_ask: float = MIN_ASK,
    min_open_interest: int = MIN_OPEN_INTEREST,
    min_volume: int = MIN_VOLUME,
    min_moneyness: float = MIN_MONEYNESS,
    max_moneyness: float = MAX_MONEYNESS,
) -> pd.DataFrame:
    """Return eligible, liquid contracts with the required forecast edge."""

    mask = (
        options["Recommendation Eligible"].eq(True)
        & options["Initial Recommended Action"].eq("Buy")
        & options["bid"].notna()
        & options["ask"].notna()
        & (options["SpreadPct"] <= max_spread_pct)
        & (options["ask"] >= min_ask)
        & (options["openInterest"].fillna(0) >= min_open_interest)
        & (options["volume"].fillna(0) >= min_volume)
        & (options["Moneyness"] >= min_moneyness)
        & (options["Moneyness"] <= max_moneyness)
        & (options["BS_ForeV AskEdge"] >= buy_edge)
    )

    return options.loc[mask].copy()


def price_option_universe(
    tickers: list[OptionTicker],
    data: pd.DataFrame,
    raise_on_error: bool = True,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
]:
    """Price and validate all ticker objects using their stored assumptions."""

    cleaned_options: dict[str, pd.DataFrame] = {}
    highlighted_options: dict[str, pd.DataFrame] = {}
    parity_results: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}

    logger.info(
        "Starting option pricing for %d ticker objects",
        len(tickers),
    )

    for ticker in tickers:
        stage = "checking attached option chains"

        if ticker.call_chain is None or ticker.put_chain is None:
            logger.warning(
                "%s skipped because its option chains are unavailable",
                ticker.symbol,
            )
            continue

        logger.info(
            "%s beginning option-chain pricing",
            ticker.symbol,
        )

        try:
            stage = "reading current stock price"
            ticker.set_current_price(
                latest_feature_value(
                    data=data,
                    feature="Close",
                    ticker=ticker,
                )
            )

            stage = "collecting volatility inputs"
            get_volatility_inputs(data=data, ticker=ticker)

            stage = "calculating Black-Scholes scenarios"
            scenario_prices = calculate_black_scholes_scenarios(ticker)

            stage = "building comparison tables"
            calls = build_comparison_table(
                chain=ticker.call_chain,
                model_prices=scenario_prices["calls"],
                ticker=ticker,
            )
            puts = build_comparison_table(
                chain=ticker.put_chain,
                model_prices=scenario_prices["puts"],
                ticker=ticker,
            )

            # Calculate bid, midpoint and ask implied volatility
            stage = "calculating market implied volatilities"

            calls = add_implied_volatility_columns(
                options=calls,
                ticker=ticker,
                option_type="call",
            )

            puts = add_implied_volatility_columns(
                options=puts,
                ticker=ticker,
                option_type="put",
            )

            calls["BS_From_IV_Mid"] = black_scholes(
                ticker=ticker,
                spot=ticker.current_price,
                strike=calls["strike"],
                time_to_expiry=ticker.time_to_expiry,
                risk_free_rate=ticker.risk_free_rate,
                dividend_yield=ticker.dividend_yield,
                option_type="call",
                volatility=calls["IV_Mid"],
)

            puts["BS_From_IV_Mid"] = black_scholes(
                ticker=ticker,
                spot=ticker.current_price,
                strike=puts["strike"],
                time_to_expiry=ticker.time_to_expiry,
                risk_free_rate=ticker.risk_free_rate,
                dividend_yield=ticker.dividend_yield,
                option_type="put",
                volatility=puts["IV_Mid"],
            )


            puts["IV_Repricing_Error"] = puts["BS_From_IV_Mid"] - puts["MarketMid"]
            calls["IV_Repricing_Error"] = calls["BS_From_IV_Mid"] - calls["MarketMid"]

            calls["IV_Repricing_Valid"] = abs(calls["IV_Repricing_Error"]) <= 1e-4
            puts["IV_Repricing_Valid"] = abs(puts["IV_Repricing_Error"]) <= 1e-4

            stage = "calculating analytical Greeks"

            calls = add_greek_columns(
                options=calls,
                ticker=ticker,
                option_type="call",
            )

            puts = add_greek_columns(
                options=puts,
                ticker=ticker,
                option_type="put",
            )

            
            logger.info(
                "%s implied volatility calculations complete",
                ticker.symbol,
            )

            stage = "checking put-call parity"
            parity = check_put_call_parity(
                calls=calls,
                puts=puts,
                ticker=ticker,
            )
            parity_results[ticker.symbol] = parity
            calls = attach_parity_errors(calls, parity)
            puts = attach_parity_errors(puts, parity)

            stage = "adding shared validation information"

            for option_type, options in (
                ("Calls", calls),
                ("Puts", puts),
            ):
                options["Ticker"] = ticker.symbol
                options["Option_Type"] = option_type
                options["Current Stock Price"] = ticker.current_price
                options["Expiry"] = ticker.expiry
                options["Calendar DTE"] = ticker.calendar_dte
                options["Trading DTE"] = ticker.trading_dte
                options["Forecast Horizon"] = ticker.forecast_horizon
                options["Horizon Difference"] = (
                    ticker.trading_dte
                    - ticker.forecast_horizon
                )
                options["Horizon Aligned"] = (
                    options["Horizon Difference"].abs().le(3)
                )
                options["Time to Expiry"] = ticker.time_to_expiry
                options["Risk-Free Rate"] = ticker.risk_free_rate
                options["Dividend Yield"] = ticker.dividend_yield
                options["Forward Price"] = ticker.forward_price
                options["Forward Moneyness"] = (
                    options["strike"] / ticker.forward_price
                )
                options["Volatility Spread"] = (
                    options["ForeV Used"] - options["IV Used"]
                )

                options["Moneyness"] = (
                    options["strike"] / ticker.current_price
                )
                valid_midpoint = options["MarketMid"] > 0
                options["SpreadPct"] = np.nan
                options.loc[valid_midpoint, "SpreadPct"] = (
                    options.loc[valid_midpoint, "ask"]
                    - options.loc[valid_midpoint, "bid"]
                ) / options.loc[valid_midpoint, "MarketMid"]

                discounted_strike = (
                    options["strike"]
                    * np.exp(
                        -ticker.risk_free_rate
                        * ticker.time_to_expiry
                    )
                )

                options["Maximum Buy Price"] = (
                    options["BS_ForeV"]
                    / (1 + ticker.buy_edge)
                )
                options["Ask Below Maximum"] = (
                    options["ask"].notna()
                    & (
                        options["ask"]
                        <= options["Maximum Buy Price"]
                    )
                )

                if option_type == "Calls":
                    options["Intrinsic Value"] = (
                        ticker.current_price - options["strike"]
                    ).clip(lower=0)
                    options["Lower Price Bound"] = (
                        ticker.discounted_spot - discounted_strike
                    ).clip(lower=0)
                    options["Upper Price Bound"] = ticker.discounted_spot
                    options["Break-Even Price"] = (
                        options["strike"] + options["ask"]
                    )
                else:
                    options["Intrinsic Value"] = (
                        options["strike"] - ticker.current_price
                    ).clip(lower=0)
                    options["Lower Price Bound"] = (
                        discounted_strike - ticker.discounted_spot
                    ).clip(lower=0)
                    options["Upper Price Bound"] = discounted_strike
                    options["Break-Even Price"] = (
                        options["strike"] - options["ask"]
                    )

                options["Time Value"] = (
                    options["MarketMid"]
                    - options["Intrinsic Value"]
                )
                options["Distance Above Lower Bound"] = (
                    options["MarketMid"]
                    - options["Lower Price Bound"]
                )
                options["Distance Below Upper Bound"] = (
                    options["Upper Price Bound"]
                    - options["MarketMid"]
                )
                options["Pricing Bounds Valid"] = (
                    options["MarketMid"].notna()
                    & (
                        options["MarketMid"]
                        >= options["Lower Price Bound"] - 1e-6
                    )
                    & (
                        options["MarketMid"]
                        <= options["Upper Price Bound"] + 1e-6
                    )
                )

                if "lastTradeDate" in options.columns:
                    last_trade_date = (
                        pd.to_datetime(
                            options["lastTradeDate"],
                            errors="coerce",
                            utc=True,
                        )
                        .dt.tz_convert(None)
                        .dt.normalize()
                    )
                    options["Quote Age Days"] = (
                        ticker.valuation_date - last_trade_date
                    ).dt.days
                    quote_is_recent = (
                        options["Quote Age Days"]
                        .le(5)
                        .fillna(False)
                    )
                else:
                    options["Quote Age Days"] = np.nan
                    quote_is_recent = pd.Series(
                        True,
                        index=options.index,
                    )

                valid_bid_ask = (
                    options["bid"].notna()
                    & options["ask"].notna()
                    & options["bid"].ge(0)
                    & options["ask"].gt(0)
                    & options["ask"].ge(options["bid"])
                )

                options["Quote Valid"] = (
                    valid_bid_ask
                    & options["Pricing Bounds Valid"]
                )
                options["Quote Issue"] = np.select(
                    [
                        options["bid"].isna() | options["ask"].isna(),
                        options["bid"].lt(0),
                        options["ask"].le(0),
                        options["ask"].lt(options["bid"]),
                        ~options["Pricing Bounds Valid"],
                    ],
                    [
                        "Missing bid or ask",
                        "Negative bid",
                        "Non-positive ask",
                        "Ask below bid",
                        "Outside pricing bounds",
                    ],
                    default="",
                )

                # Put-call parity validates the pricing implementation; it is
                # not a liquidity or quote-quality requirement for one contract.
                options["Recommendation Eligible"] = (
                    options["Quote Valid"]
                    & options["Horizon Aligned"].eq(True)
                    & options["BS_ForeV"].notna()
                    & options["ForeV Used"].notna()
                    & options["ask"].notna()
                )

                options["Greek Volatility"] = (
                    options["IV_Mid"]
                    .fillna(options["IV Used"])
                    .fillna(options["ForeV Used"])
                )

                options["Greek Volatility Source"] = np.select(
                    [
                        options["IV_Mid"].notna(),
                        options["IV Used"].notna(),
                        options["ForeV Used"].notna(),
                    ],
                    [
                        "Calculated Mid IV",
                        "Downloaded IV",
                        "Forecast RV",
                    ],
                    default="No Volatility",
                )

            stage = "adding recommendations and filters"
            calls = add_recommendation_metrics(
                options=calls,
                buy_edge=ticker.buy_edge,
            )
            puts = add_recommendation_metrics(
                options=puts,
                buy_edge=ticker.buy_edge,
            )

            highlighted_calls = filter_highlighted_options(
                options=calls,
                buy_edge=ticker.buy_edge,
            )
            highlighted_puts = filter_highlighted_options(
                options=puts,
                buy_edge=ticker.buy_edge,
            )

            ticker.call_analysis = calls
            ticker.put_analysis = puts

            cleaned_options[ticker.table_key("Calls")] = calls
            cleaned_options[ticker.table_key("Puts")] = puts
            highlighted_options[ticker.table_key("Calls")] = highlighted_calls
            highlighted_options[ticker.table_key("Puts")] = highlighted_puts

            logger.info(
                "%s pricing complete — %d total calls, %d total puts, "
                "%d highlighted calls and %d highlighted puts",
                ticker.symbol,
                len(calls),
                len(puts),
                len(highlighted_calls),
                len(highlighted_puts),
            )

        except Exception as error:
            failures[ticker.symbol] = (
                f"{type(error).__name__}: {error}"
            )
            logger.exception(
                "%s failed during stage '%s'",
                ticker.symbol,
                stage,
            )
            if raise_on_error:
                raise

    logger.info(
        "Option pricing finished — %d full tables, %d highlighted "
        "tables, %d parity tables and %d failed tickers",
        len(cleaned_options),
        len(highlighted_options),
        len(parity_results),
        len(failures),
    )

    if not cleaned_options:
        raise RuntimeError(
            "No option tables were created. Failures: "
            f"{failures}"
        )

    return cleaned_options, highlighted_options, parity_results
