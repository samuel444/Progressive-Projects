from equity_selector.settings import configured
import logging
import pandas as pd
import yfinance as yf
import numpy as np
import warnings

logger = logging.getLogger(__name__)


def run_screen(name, function, df, selected_features, target, dropped_by_target):

    selected_features, to_drop = configured(function, df, selected_features, target)

    dropped_by_target[target][name] = to_drop

    logger.info(
        "%s | %s: dropped %d, %d remain", target, name, len(to_drop), len(selected_features)
    )

    return selected_features


def target_type(df, target):

    values = set(df[target].dropna().unique())

    ########################################
    # Expected Binary Targets
    ########################################

    if (
        target.startswith("Future Direction")
        or target.startswith("Future Return Above")
        or target.startswith("Top ")
        or target.startswith("Bottom ")
    ):
        # Sanity check
        if len(values) > 2:
            raise ValueError(f"{target} expected binary but has values: {values}")

        return "binary"

    ########################################
    # Expected Multiclass Targets
    ########################################

    if (
        target.startswith("Three Class Direction")
        or target.startswith("Barrier ")
        or target.startswith("Volatility Barrier")
    ):
        # If target actually only contains two classes,
        # treat it as binary
        if len(values) == 2:
            return "binary"

        return "multiclass"

    ########################################
    # Everything Else
    ########################################

    return "continuous"


from equity_selector.validation import train_validation_test_split


def target_purge_days(target):

    ########################################
    # Horizon Is Final Value
    ########################################

    if (
        target.startswith("Forward Return ")
        or target.startswith("Forward Log Return ")
        or target.startswith("Forward Excess Return ")
        or target.startswith("Future Volatility ")
        or target.startswith("Future Variance ")
        or target.startswith("Future Upside Volatility ")
        or target.startswith("Future Downside Volatility ")
        or target.startswith("Future Downside Upside Volatility Ratio ")
        or target.startswith("Future Mean Absolute Return ")
        or target.startswith("Future Maximum Absolute Return ")
        or target.startswith("Future Direction ")
        or target.startswith("Future Return Above ")
        or target.startswith("Three Class Direction ")
        or target.startswith("Barrier ")
        or target.startswith("Maximum Favourable Excursion ")
        or target.startswith("Maximum Adverse Excursion ")
        or target.startswith("Time To Maximum Favourable Excursion ")
        or target.startswith("Time To Maximum Adverse Excursion ")
        or target.startswith("Future Drawdown At Horizon ")
        or target.startswith("Future Maximum Drawdown ")
        or target.startswith("Future Minimum Return ")
        or target.startswith("Future Return Volatility Ratio ")
        or target.startswith("Future Sortino Ratio ")
        or target.startswith("Future Return Drawdown Ratio ")
        or target.startswith("Future Return Rank ")
        or target.startswith("Top ")
        or target.startswith("Bottom ")
    ):
        return int(target.split()[-1])

    ########################################
    # Future Return Minus Risk
    #
    # Future Return Minus Risk 20 0.5
    ########################################

    if target.startswith("Future Return Minus Risk "):
        return int(target.split()[-2])

    ########################################
    # Volatility Barrier
    #
    # Volatility Barrier 20 60 1 2
    #                    ^  ^
    #                 window horizon
    ########################################

    if target.startswith("Volatility Barrier "):
        return int(target.split()[3])

    ########################################
    # No Rolling/Future Horizon
    ########################################

    return 0
