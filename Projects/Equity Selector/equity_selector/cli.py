"""Explicit entry points for the original research stages."""

import argparse
import importlib
import logging
import os
from time import perf_counter

STAGES = {
    "data": "Build and screen market features and targets.",
    "intraday": "Convert stored features and targets to session-safe intraday data.",
    "training": "Search and validate target models with purged walk-forward folds.",
    "final_test": "Evaluate selected models on the final model-test partition.",
    "horizons": "Research horizon-score configurations using cached predictions.",
    "cache": "Build the strategy backtest database.",
    "simulations": "Simulate and filter portfolio configurations.",
    "precise": "Evaluate selected strategies with detailed diagnostics.",
}


def run_stage(stage, argv=None, *, settings=None, callbacks=None):
    from contextlib import ExitStack
    from copy import deepcopy
    from pathlib import Path
    from .settings import stage_settings
    from .settings_catalogue import STAGE_KEYS, STAGE_CALLBACKS

    settings, callbacks = dict(settings or {}), dict(callbacks or {})
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    unknown = set(settings) - set(STAGE_KEYS[stage])
    unknown_hooks = set(callbacks) - set(STAGE_CALLBACKS[stage])
    unknown_hooks |= set(settings.get("FUNCTION_KWARGS", {}) or {}) - set(STAGE_CALLBACKS[stage])
    if unknown or unknown_hooks:
        raise ValueError(
            f"Unknown settings: {sorted(unknown)}; unknown callbacks: {sorted(unknown_hooks)}"
        )
    if stage != "precise":
        parser = argparse.ArgumentParser(description=STAGES[stage])
        parser.add_argument("--data-dir", default=settings.get("DATA_DIR"))
        parser.add_argument(
            "--log-level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            default=settings.get("LOG_LEVEL", "INFO"),
        )
        args = parser.parse_args(argv)
        settings.update(DATA_DIR=args.data_dir, LOG_LEVEL=args.log_level)
    with ExitStack() as stack:
        stack.enter_context(stage_settings(settings, callbacks))
        original_dir = os.environ.get("EQUITY_SELECTOR_DATA_DIR")

        def restore_directory():
            if original_dir is None:
                os.environ.pop("EQUITY_SELECTOR_DATA_DIR", None)
            else:
                os.environ["EQUITY_SELECTOR_DATA_DIR"] = original_dir

        stack.callback(restore_directory)
        if settings.get("DATA_DIR"):
            os.environ["EQUITY_SELECTOR_DATA_DIR"] = str(settings["DATA_DIR"])
        logging.basicConfig(
            level=getattr(logging, settings.get("LOG_LEVEL", "INFO")),
            format="%(asctime)s | %(levelname)s | %(message)s",
        )
        logger = logging.getLogger(__name__)
        started = perf_counter()
        logger.info("Starting stage %s", stage)
        module = importlib.import_module(".stages." + stage, __package__)
        if stage in {"precise", "intraday"}:
            from .config import data_root

            overrides = {
                key: value
                for key, value in settings.items()
                if hasattr(module, key) and value is not None
            }
            if stage == "intraday":
                overrides.update(
                    DATA_DIR=data_root(),
                    DATABASE=data_root() / "Features_Targets_Data.db",
                    SELECTED_FEATURES_FILE=data_root() / "Selected_Features.txt",
                )
            originals = {key: getattr(module, key) for key in overrides}
            for key, value in overrides.items():
                setattr(module, key, deepcopy(value))

            def restore_globals():
                for key, value in originals.items():
                    setattr(module, key, value)

            stack.callback(restore_globals)
        result = module.main(argv) if stage == "precise" else module.run()
        logger.info("Completed stage %s | elapsed=%.2fs", stage, perf_counter() - started)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Equity Selector research pipeline")
    parser.add_argument("stage", choices=STAGES)
    args, rest = parser.parse_known_args(argv)
    return run_stage(args.stage, rest)
