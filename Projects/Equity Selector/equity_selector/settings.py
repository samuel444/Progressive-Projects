"""Scoped launcher settings: no persistent module-global configuration leaks."""

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy

_ACTIVE = ContextVar("equity_selector_settings", default={})
_CALLBACKS = ContextVar("equity_selector_callbacks", default={})


def setting(name, default=None):
    value = _ACTIVE.get().get(name)
    return deepcopy(default if value is None else value)


def callback(name, default):
    return _CALLBACKS.get().get(name, default)


@contextmanager
def stage_settings(settings=None, callbacks=None):
    if callbacks and any(not callable(value) for value in callbacks.values()):
        raise TypeError("Every callback override must be callable")
    settings_token = _ACTIVE.set(deepcopy(settings or {}))
    callbacks_token = _CALLBACKS.set(dict(callbacks or {}))
    try:
        yield
    finally:
        _ACTIVE.reset(settings_token)
        _CALLBACKS.reset(callbacks_token)


def choose_catalogue(name, generated, legacy):
    mode = setting("MODEL_CATALOGUE_MODE", "legacy")
    if mode not in {"legacy", "grid"}:
        raise ValueError("MODEL_CATALOGUE_MODE must be legacy or grid")
    return setting(name, generated if mode == "grid" else legacy)


def configured(function, *args, **kwargs):
    options = setting("FUNCTION_KWARGS", {}).get(function.__name__, {})
    return callback(function.__name__, function)(*args, **{**options, **kwargs})


def required_date(name):
    import pandas as pd

    value = setting(name)
    if value is None:
        raise ValueError(f"Set {name} in SETTINGS in the original launcher script")
    try:
        date = pd.Timestamp(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"Invalid {name}: use YYYY-MM-DD in launcher SETTINGS") from error
    if pd.isna(date):
        raise ValueError(f"Invalid {name}: date must not be missing")
    return date


def choose_model_row(eligible, target):
    from .parameters import parameters_to_json

    mode = setting("MODEL_SELECTION_MODE", "rank_one")
    if eligible.empty:
        raise ValueError(f"{target}: no eligible models")
    ordered = eligible.sort_values("Test Selection Rank")
    if mode == "rank_one":
        return ordered.iloc[0]
    if mode != "explicit":
        raise ValueError("MODEL_SELECTION_MODE must be rank_one or explicit")
    choice = setting("MODEL_SELECTIONS", {}).get(target)
    if isinstance(choice, int) and not isinstance(choice, bool):
        matched = ordered.loc[ordered["Test Selection Rank"].eq(choice)]
    elif isinstance(choice, dict) and {"Model", "Parameters"} <= choice.keys():
        matched = ordered.loc[
            ordered.Model.eq(choice["Model"])
            & ordered.Parameters.map(parameters_to_json).eq(
                parameters_to_json(choice["Parameters"])
            )
        ]
    else:
        raise ValueError(
            f"Set MODEL_SELECTIONS[{target!r}] to a selection rank or Model/Parameters dictionary in launcher SETTINGS"
        )
    if len(matched) != 1:
        raise ValueError(f"{target}: explicit selection must match exactly one eligible model")
    return matched.iloc[0]


def run_screen_schedule(screen, count, initial, extra, limit):
    """Finite configured screening schedule; never prompt or silently run an oversized grid."""
    import math

    if not isinstance(limit, int) or limit < 1:
        raise ValueError("MAX_EXHAUSTIVE_CONFIGURATIONS must be a positive integer")
    for iterations, threshold in [*initial, *extra]:
        if (
            not isinstance(iterations, int)
            or iterations < 1
            or not math.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            raise ValueError(
                "Random screen entries require positive iterations and threshold in [0, 1]"
            )
    result = None
    for iterations, threshold in initial:
        result = screen(iterations=iterations, threshold=threshold)
    for iterations, threshold in extra:
        if count() <= limit:
            break
        result = screen(iterations=iterations, threshold=threshold)
    if count() > limit:
        raise ValueError(
            f"{count()} horizon configurations remain, above limit {limit}; set EXTRA_RANDOM_SCREENS or MAX_EXHAUSTIVE_CONFIGURATIONS in Horizon Score Backtests.py SETTINGS"
        )
    return result
