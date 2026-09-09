"""Walk-forward orchestration with explicit dependencies for offline testing."""

from equity_selector.database import write_frame
import logging
import sqlite3
import numpy as np
import pandas as pd
from .parameters import configuration_key, parameters_to_json, unique_models
from .validation import train_validation_test_split, purge_training_data
from .database import quote_identifier
from .settings import setting

logger = logging.getLogger(__name__)


def walk_forward(
    models_to_do,
    df,
    features,
    target,
    purge_days,
    type,
    portfolio_type,
    previous_results,
    no_table,
    validation_window=20,
    *,
    database_path,
    fit_function,
    prune_function,
    role_function,
):
    if not isinstance(validation_window, int) or validation_window <= 0:
        raise ValueError("validation_window must be a positive integer")
    if not isinstance(purge_days, (int, np.integer)) or purge_days < 0:
        raise ValueError("purge_days must be a nonnegative integer")
    if not models_to_do:
        return previous_results.copy()
    models_to_do = unique_models(models_to_do)
    VALIDATION_DATABASE_PATH = database_path
    run_single_fold = fit_function
    prune = prune_function
    portfolio_selection_role = role_function

    logger.info(
        "%s | Starting walk-forward validation | Statistical type: %s | Portfolio type: %s | Selection role: %s | %d models | validation window: %d | purge days: %d",
        target,
        type,
        portfolio_type,
        portfolio_selection_role(portfolio_type),
        len(models_to_do),
        validation_window,
        purge_days,
    )

    original_previous_results = previous_results.copy()

    logger.info("%s | Loaded %d previous model configurations", target, len(previous_results))

    full_train_df, full_validation_df, test_df = train_validation_test_split(df, 0.2, 0.2)

    logger.info(
        "%s | Split complete | train rows: %d | validation rows: %d | test rows: %d",
        target,
        len(full_train_df),
        len(full_validation_df),
        len(test_df),
    )

    validation_dates = sorted(full_validation_df["Date"].unique())

    logger.info("%s | Validation contains %d unique dates", target, len(validation_dates))

    validation_results = []

    all_validation_results = []
    failed_configurations = set()

    try:
        with sqlite3.connect(VALIDATION_DATABASE_PATH) as conn:
            fold_results = pd.read_sql_query(
                f"SELECT * FROM {quote_identifier(target + '__folds')}", conn
            )

    except pd.errors.DatabaseError as error:
        if "no such table" not in str(error):
            raise
        fold_results = pd.DataFrame()

    fold = 1
    start = 0

    while start < len(validation_dates):
        fold_dates = validation_dates[start : start + validation_window]

        if len(fold_dates) == 0:
            break

        logger.info(
            "%s | Fold %d | Validation dates: %s -> %s | %d dates",
            target,
            fold,
            fold_dates[0],
            fold_dates[-1],
            len(fold_dates),
        )

        validation_df = full_validation_df[full_validation_df["Date"].isin(fold_dates)].copy()

        validation_start = fold_dates[0]

        train_df = pd.concat(
            [full_train_df, full_validation_df[full_validation_df["Date"] < validation_start]]
        ).copy()

        logger.info(
            "%s | Fold %d | Train rows before purge: %d | Validation rows: %d",
            target,
            fold,
            len(train_df),
            len(validation_df),
        )

        train_df = purge_training_data(train_df, purge_days)

        logger.info(
            "%s | Fold %d | Train rows after purge: %d | Removed final %d training dates",
            target,
            fold,
            len(train_df),
            purge_days,
        )

        logger.info(
            "%s | Fold %d | Running %d model configurations", target, fold, len(models_to_do)
        )

        remove_model = []

        for model_number, model in enumerate(models_to_do, start=1):
            model_fold_result = run_single_fold(
                train_df, validation_df, features, target, type, fold, model
            )

            if model_fold_result is None:
                logger.warning(
                    "%s | Fold %d | Model %d/%d | %s | Failed, Model Removed",
                    target,
                    fold,
                    model_number,
                    len(models_to_do),
                    model["name"],
                )

                remove_model.append(model)
                failed_configurations.add(configuration_key(model))

            else:
                model_fold_result = dict(model_fold_result)
                model_fold_result.update(
                    {
                        "Train Start": str(train_df.Date.min()),
                        "Train End": str(train_df.Date.max()),
                        "Validation Start": str(validation_df.Date.min()),
                        "Validation End": str(validation_df.Date.max()),
                        "Purge Periods": purge_days,
                    }
                )
                validation_results.append(model_fold_result)

                all_validation_results.append(model_fold_result)

                logger.debug(
                    "%s | Fold %d | Model %d/%d | %s | Complete",
                    target,
                    fold,
                    model_number,
                    len(models_to_do),
                    model["name"],
                )

        logger.info("%s | Fold %d | Complete", target, fold)

        if remove_model:
            models_to_do = [
                model
                for model in models_to_do
                if not any(
                    model["name"] == removing_model["name"]
                    and model["params"] == removing_model["params"]
                    for removing_model in remove_model
                )
            ]

            validation_results = [
                result
                for result in validation_results
                if any(
                    result["Model"] == model["name"] and result["Parameters"] == model["params"]
                    for model in models_to_do
                )
            ]

        if not models_to_do:
            logger.info("%s | No active models remain after fold %d", target, fold)
            break

        PRUNING_STAGES = setting(
            "PRUNING_STAGES",
            {
                3: (0.95, 20000),
                5: (0.9, 5000),
                7: (0.8, 1000),
                9: (0.65, 300),
                11: (0.5, 70),
                14: (0.5, 15),
            },
        )
        if fold in PRUNING_STAGES:
            models_to_do = prune(
                models_to_do,
                validation_results,
                fold_results,
                fold,
                target,
                type,
                portfolio_type,
                PRUNING_STAGES[fold][0],
                PRUNING_STAGES[fold][1],
            )

            validation_results = [
                result
                for result in validation_results
                if any(
                    result["Model"] == model["name"] and result["Parameters"] == model["params"]
                    for model in models_to_do
                )
            ]

            if not models_to_do:
                start = len(validation_dates)

        fold += 1
        start += validation_window

    if not all_validation_results:
        logger.info("%s | No successful validation fits", target)
        return original_previous_results
    new_results = pd.DataFrame(all_validation_results)

    new_results["Parameters"] = new_results["Parameters"].apply(parameters_to_json)

    with sqlite3.connect(VALIDATION_DATABASE_PATH) as conn:
        write_frame(new_results, f"{target}__folds", conn, if_exists="append", index=False)

    # Persist diagnostic folds above, but never promote a failed configuration.
    new_results = new_results.loc[
        [
            (row.Model, row.Parameters) not in failed_configurations
            for row in new_results.itertuples(index=False)
        ]
    ].copy()
    if new_results.empty:
        logger.info("%s | No successful model configurations remain", target)
        return original_previous_results

    metric_columns = [
        column
        for column in new_results.select_dtypes(include="number").columns
        if column not in ["Fold", "Purge Periods"]
    ]

    new_results = new_results.groupby(["Model", "Parameters"]).agg(
        {**{column: ["mean", "std"] for column in metric_columns}, "Fold": "max"}
    )

    new_results.columns = [
        (f"{column} {stat.title()}" if column != "Fold" else "Fold")
        for column, stat in new_results.columns
    ]

    new_results = new_results.reset_index()

    new_results["Target"] = target
    new_results["Target Type"] = type
    new_results["Portfolio Target Type"] = portfolio_type

    new_results["TPE Score?"] = 0

    new_results["Midpoint?"] = np.nan

    if not (original_previous_results.empty):
        if "Portfolio Target Type" not in original_previous_results.columns:
            original_previous_results["Portfolio Target Type"] = portfolio_type

        common_columns = original_previous_results.columns.intersection(new_results.columns)

        new_summary = pd.concat(
            [original_previous_results[common_columns], new_results[common_columns]],
            ignore_index=True,
        )

    else:
        new_summary = new_results

    return new_summary
