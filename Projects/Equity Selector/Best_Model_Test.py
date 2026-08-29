import sqlite3
import pandas as pd

from models import *

from sklearn.preprocessing import StandardScaler

from main_package import *

import json
import numpy as np

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

FEATURE_DATABASE_PATH = (
    "/Users/sam/Progressive-Projects/Projects/"
    "Equity Selector/data/Features_Targets_Data.db"
)

STOCK_TYPE = (
    "High Liquidity 30"
    #"Medium Liquidity 30"
    #"Lower Liquidity 30"
    #"Intraday Higher Liquidity 30"
    #"Intraday Medium Liquidity 30"
    #"Sector Spread 30"
    #"Liquidity Barbell 30"
    #"Institutional Liquidity 60"
    #"Medium Small Liquidity 60"
    #"Medium Large Liquidity 60"
    #"All Liquidity 90"
)

VALIDATION_DATABASE_PATH = (
    "/Users/sam/Progressive-Projects/Projects/"
    "Equity Selector/data/Validation_Model_Fits/"
    f"{STOCK_TYPE.replace(' ', '_')}.db"
)

SELECTED_FEATURES_PATH = (
    "/Users/sam/Progressive-Projects/Projects/"
    "Equity Selector/data/Selected_Features.txt"
)


def quote_sql_identifier(identifier):

    return (
        '"'
        + str(identifier).replace('"', '""')
        + '"'
    )


file = open(SELECTED_FEATURES_PATH, 'r')
selected_features = file.read()
file.close()

if STOCK_TYPE == "High Liquidity 30":
    stock_type_index = 0
elif STOCK_TYPE == "Medium Liquidity 30":
    stock_type_index = 1
elif STOCK_TYPE == "Lower Liquidity 30":
    stock_type_index = 2
elif STOCK_TYPE == "Sector Spread 30":
    stock_type_index = 3
elif STOCK_TYPE == "Intraday Higher Liquidity 30":
    stock_type_index = 4
elif STOCK_TYPE == "Intraday Medium Liquidity 30":
    stock_type_index = 5
elif STOCK_TYPE == "Liquidity Barbell 30":
    stock_type_index = 6
elif STOCK_TYPE == "Institutional Liquidity 60":
    stock_type_index = 7
elif STOCK_TYPE == "Medium Small Liquidity 60":
    stock_type_index = 8
elif STOCK_TYPE == "Medium Large Liquidity 60":
    stock_type_index = 9
elif STOCK_TYPE == "All Liquidity 90":
    stock_type_index = 10
else:
    raise ValueError(
        f"Unknown STOCK_TYPE: {STOCK_TYPE}"
    )

import ast

selected_features = selected_features.split('\n')
selected_features = selected_features[stock_type_index]
selected_features = ast.literal_eval(selected_features)


############################################################
# LOAD TEST-ELIGIBLE VALIDATION MODELS
############################################################

def testing_eligible_mask(series):

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    if pd.api.types.is_numeric_dtype(series):
        return (
            pd.to_numeric(
                series,
                errors="coerce"
            )
            .fillna(0)
            .eq(1)
        )

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .isin([
            "true",
            "1",
            "yes"
        ])
    )


def model_ranking_rules(target_type):

    if target_type == "continuous":
        return [
            ("Rank IC Mean", False),
            ("Rank IC Std", True),
            ("NRMSE Mean", True),
            ("RMSE Mean", True),
            ("MAE Mean", True),
            ("R2 Mean", False),
        ]

    if target_type == "binary":
        return [
            ("ROC AUC Mean", False),
            ("ROC AUC Std", True),
            ("PR AUC Mean", False),
            ("F1 Mean", False),
            ("Log Loss Mean", True),
        ]

    if target_type == "multiclass":
        return [
            ("Macro F1 Mean", False),
            ("Macro F1 Std", True),
            ("Balanced Accuracy Mean", False),
            ("Log Loss Mean", True),
        ]

    raise ValueError(
        f"Unknown target type: {target_type}"
    )


def target_type_from_name(target):

    if (
        target.startswith("Three Class Direction")
        or target.startswith("Barrier")
        or target.startswith("Volatility Barrier")
    ):
        return "multiclass"

    if (
        target.startswith("Future Direction")
        or target.startswith("Future Return Above")
        or target.startswith("Top 20 Percent Future Return")
        or target.startswith("Top 25 Percent Future Return")
    ):
        return "binary"

    return "continuous"


def rank_eligible_models(
    leaderboard,
    target_type
):

    available_rules = [
        (column, ascending)
        for column, ascending in model_ranking_rules(
            target_type
        )
        if column in leaderboard.columns
    ]

    if not available_rules:
        return leaderboard.copy()

    ranked = leaderboard.sort_values(
        by=[column for column, _ in available_rules],
        ascending=[ascending for _, ascending in available_rules],
        na_position="last",
        kind="stable"
    ).copy()

    ranked["Test Selection Rank"] = np.arange(
        1,
        len(ranked) + 1
    )

    return ranked


def load_test_eligible_models(
    selected_features,
    validation_database_path
):

    eligible_by_target = {}

    with sqlite3.connect(
        validation_database_path
    ) as validation_connection:

        table_names = pd.read_sql_query(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            """,
            validation_connection
        )["name"].tolist()

        table_names = set(
            table_names
        )

        for target in selected_features.keys():

            # Exact normal target-table match.
            if target not in table_names:
                continue

            leaderboard = pd.read_sql_query(
                f"""
                SELECT *
                FROM {quote_sql_identifier(target)}
                """,
                validation_connection
            )

            # No eligible validation results for this target.
            if leaderboard.empty:
                continue

            eligible = rank_eligible_models(
                leaderboard=leaderboard,
                target_type=target_type_from_name(
                    target
                )
            )

            eligible_by_target[target] = (
                eligible.reset_index(
                    drop=True
                )
            )

    return eligible_by_target


eligible_models_by_target = (
    load_test_eligible_models(
        selected_features=selected_features,
        validation_database_path=(
            VALIDATION_DATABASE_PATH
        )
    )
)


targets = [
    target
    for target in selected_features.keys()
    if target in eligible_models_by_target
]

logger.info(
    "Validation database: %s",
    VALIDATION_DATABASE_PATH
)

logger.info(
    "Test-eligible targets: %d / %d",
    len(targets),
    len(selected_features)
)


############################################################
# MEMORY-SAFE SOURCE DATABASE
#
# Only the columns required for the current target are read.
# The full Features_Targets_Data table is never loaded.
############################################################

data_connection = sqlite3.connect(
    FEATURE_DATABASE_PATH
)

data_connection.execute(
    "PRAGMA query_only = ON"
)

source_table_info = pd.read_sql_query(
    f"PRAGMA table_info({quote_sql_identifier(STOCK_TYPE)})",
    data_connection
)

if source_table_info.empty:
    raise ValueError(
        f"Source table {STOCK_TYPE!r} does not exist "
        f"in {FEATURE_DATABASE_PATH}"
    )

SOURCE_TABLE_COLUMNS = set(
    source_table_info["name"].tolist()
)

logger.info(
    "%s | Source table ready | %d columns | full table will not be loaded",
    STOCK_TYPE,
    len(SOURCE_TABLE_COLUMNS)
)

# Model definitions are provided by full_model_source(target_type).

def get_model_function(function_name):

    function = globals().get(function_name)

    if function is None:

        raise NameError(
            f"{function_name} has not been imported yet."
        )

    return function


def purge_training_data(train_df, purge_days):

    if purge_days <= 0:
        return train_df.copy()


    dates = np.sort(
        train_df["Date"].unique()
    )


    ########################################
    # Not Enough Training History
    ########################################

    if len(dates) <= purge_days:
        return train_df.iloc[0:0].copy()


    ########################################
    # Remove Last purge_days Trading Dates
    ########################################

    purge_start_date = dates[-purge_days]

    return train_df[
        train_df["Date"] < purge_start_date
    ].copy()


def final_test(

    train_df,
    validation_df,
    test_df,

    features,

    target,
    model_name,
    parameters
):

    ########################################
    # Target Type
    ########################################

    _type = final_target_type(
        target
    )


    logger.info(
        "%s | Final Test | Type=%s | Model=%s",
        target,
        _type,
        model_name
    )


    ########################################
    # Find Existing Model Config
    ########################################

    model_config = get_selected_model_config(
        model_name=model_name,
        target_type=_type
    )


    ########################################
    # Get Existing Fit Function
    ########################################

    fit_function = get_model_function(
        model_config["function"]
    )


    logger.info(
        "%s | Function=%s | Scaled=%s",
        target,
        model_config["function"],
        model_config["scaled"]
    )


    ########################################
    # Final Training Data
    #
    # Train + Validation
    ########################################

    final_train_df = pd.concat(
        [
            train_df,
            validation_df
        ],
        ignore_index=True
    )


    ########################################
    # Purge
    ########################################

    purge_days = target_purge_days(
        target
    )

    rows_before_purge = len(
        final_train_df
    )

    final_train_df = purge_training_data(
        final_train_df,
        purge_days
    )


    logger.info(
        "%s | Train rows=%d -> %d after purge | "
        "Test rows=%d | Purge=%d",
        target,
        rows_before_purge,
        len(final_train_df),
        len(test_df),
        purge_days
    )


    ########################################
    # X / Y
    ########################################

    x_train = final_train_df[
        features
    ]

    y_train = final_train_df[
        target
    ]

    x_test = test_df[
        features
    ]

    y_test = test_df[
        target
    ]


    ########################################
    # Binary Target Cleaning
    ########################################

    if _type == "binary":

        y_train = clean_binary_target(
            y_train,
            target
        )

        y_test = clean_binary_target(
            y_test,
            target
        )


    ########################################
    # Log Classes
    ########################################

    if _type in (
        "binary",
        "multiclass"
    ):

        train_classes = np.sort(
            y_train.dropna().unique()
        )

        test_classes = np.sort(
            y_test.dropna().unique()
        )

        logger.info(
            "%s | Train classes=%s | Test classes=%s",
            target,
            train_classes,
            test_classes
        )


    ########################################
    # Scaling
    ########################################

    if model_config["scaled"]:

        scaler = StandardScaler()

        x_train = pd.DataFrame(

            scaler.fit_transform(
                x_train
            ),

            columns=features,
            index=x_train.index
        )

        x_test = pd.DataFrame(

            scaler.transform(
                x_test
            ),

            columns=features,
            index=x_test.index
        )


    ########################################
    # Fit + Test
    ########################################

    logger.info(
        "%s | Fitting final model",
        target
    )

    result = fit_function(

        x_train,
        y_train,

        x_test,
        y_test,

        **parameters
    )


    if result is None:
        result = {}


    ########################################
    # Clean Result
    ########################################

    clean_result = clean_final_result(

        result=result,

        target=target,
        target_type=_type,

        model_name=model_name,
        parameters=parameters
    )


    logger.info(
        "%s | Complete | %s",
        target,
        {
            key: value
            for key, value in clean_result.items()
            if key not in [
                "Parameters"
            ]
        }
    )


    return clean_result


def get_selected_model_config(
    model_name,
    target_type
):

    models = get_models(
        target_type
    )

    for model_config in models:

        if model_config["name"] == model_name:
            return model_config

    raise ValueError(
        f"{model_name} not found for "
        f"target type {target_type}"
    )


def json_default(value):

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    return str(value)


def parameters_to_json(parameters):

    return json.dumps(
        parameters,
        sort_keys=True,
        default=json_default
    )

def clean_binary_target(y, target):

    y = y.copy()


    ########################################
    # Future Direction
    #
    # Positive     ->  1
    # Zero/Negative -> -1
    ########################################

    if target.startswith("Future Direction"):

        y = y.where(
            y > 0,
            -1
        )

        y = y.where(
            y <= 0,
            1
        )


    ########################################
    # Safety Check
    ########################################

    classes = np.sort(
        pd.Series(y)
        .dropna()
        .unique()
    )

    if len(classes) > 2:

        raise ValueError(
            f"{target} is defined as binary "
            f"but contains classes {classes}"
        )


    return y

def final_target_type(target):
    return target_type_from_name(
        target
    )


def get_models(target_type):
    model_source = full_model_source(
        target_type
    )

    return model_source


def clean_final_result(
    result,
    target,
    target_type,
    model_name,
    parameters
):

    ########################################
    # Base Information
    ########################################

    clean = {
        "Target": target,
        "Target Type": target_type,
        "Model": model_name,
        "Parameters": parameters_to_json(parameters)
    }


    ########################################
    # Continuous
    ########################################

    if target_type == "continuous":

        metrics = [
            "RMSE",
            "MAE",
            "R2",
            "Rank IC"
        ]


    ########################################
    # Binary
    ########################################

    elif target_type == "binary":

        metrics = [
            "ROC AUC",
            "PR AUC",
            "Log Loss",
            "F1"
        ]


    ########################################
    # Multiclass
    ########################################

    elif target_type == "multiclass":

        metrics = [
            "Macro F1",
            "Balanced Accuracy",
            "Log Loss"
        ]


    else:
        raise ValueError(
            f"Unknown target type: {target_type}"
        )


    ########################################
    # Only Add Metrics That Exist
    ########################################

    for metric in metrics:

        if metric in result:
            clean[metric] = result[metric]


    return clean

def predictability_score(row):

    if row["Target Type"] == "continuous":
        # Continuous predictions are consumed as cross-sectional ranks.
        # Do not use abs(): a negative IC ranks stocks in the wrong order.
        return max(row["Rank IC"], 0)

    elif row["Target Type"] == "binary":
        # Convert AUC so 0.5 = no predictive value
        auc_edge = max(row["ROC AUC"] - 0.5, 0) * 2

        return (
            0.50 * auc_edge +
            0.30 * row["PR AUC"] +
            0.20 * row["F1"]
        )

    elif row["Target Type"] == "multiclass":
        return row["Macro F1"]

    return 0


def target_category(target):
    target = target.lower()

    # Downside / loss-risk predictors
    downside_keywords = [
        "downside",
        "adverse",
        "minimum return",
        "drawdown",
        "time to maximum adverse"
    ]

    if any(word in target for word in downside_keywords):
        return "downside"

    # Volatility / magnitude-of-movement predictors
    volatility_keywords = [
        "volatility",
        "variance",
        "absolute return"
    ]

    if any(word in target for word in volatility_keywords):
        return "volatility"

    # Everything return/direction/opportunity related
    return "alpha"


############################################################
# TEST-ELIGIBLE MODEL SELECTION
############################################################

def parse_selected_parameters(value):

    if isinstance(value, dict):
        return value.copy()

    if value is None:
        return {}

    if isinstance(value, float) and np.isnan(value):
        return {}

    if isinstance(value, str):

        stripped = value.strip()

        if stripped == "":
            return {}

        try:
            parsed = json.loads(
                stripped
            )
        except Exception:
            parsed = ast.literal_eval(
                stripped
            )

        if not isinstance(parsed, dict):
            raise ValueError(
                "Stored model parameters must decode "
                "to a dictionary."
            )

        return parsed

    raise TypeError(
        "Unsupported stored parameter type: "
        f"{type(value).__name__}"
    )


def eligible_metric_columns(
    target_type,
    columns
):

    if target_type == "continuous":
        wanted = [
            "Rank IC Mean",
            "Rank IC Std",
            "NRMSE Mean",
            "RMSE Mean",
            "MAE Mean",
            "R2 Mean"
        ]

    elif target_type == "binary":
        wanted = [
            "ROC AUC Mean",
            "PR AUC Mean",
            "F1 Mean",
            "ROC AUC Std"
        ]

    elif target_type == "multiclass":
        wanted = [
            "Macro F1 Mean",
            "Balanced Accuracy Mean",
            "Log Loss Mean",
            "Macro F1 Std"
        ]

    else:
        wanted = []

    return [
        column
        for column in wanted
        if column in columns
    ]


def selected_model_from_row(
    row
):

    model_name = row[
        "Model"
    ]

    parameters = parse_selected_parameters(
        row[
            "Parameters"
        ]
    )

    return (
        model_name,
        parameters
    )


def display_test_eligible_models(
    target,
    eligible_models
):

    target_type = final_target_type(
        target
    )

    metric_columns = eligible_metric_columns(
        target_type=target_type,
        columns=eligible_models.columns
    )

    print("\n" + "=" * 100)

    print(
        f"{target} | "
        f"{len(eligible_models)} TEST-ELIGIBLE MODELS"
    )

    print("=" * 100)

    for option_number, (_, row) in enumerate(
        eligible_models.iterrows(),
        start=1
    ):

        parts = [
            f"[{option_number}]",
            str(row["Model"])
        ]

        rank = row.get(
            "Test Selection Rank"
        )

        if pd.notna(rank):

            parts.insert(
                1,
                f"Rank={int(rank)}"
            )

        for metric in metric_columns:

            value = row.get(
                metric
            )

            if pd.notna(value):

                parts.append(
                    f"{metric}={value:.6f}"
                )

        print(
            " | ".join(parts)
        )


def choose_test_eligible_model(
    target,
    eligible_models
):

    if eligible_models.empty:

        raise ValueError(
            f"{target} has no test-eligible models."
        )


    ########################################
    # Only One Choice
    ########################################

    if len(eligible_models) == 1:

        selected = eligible_models.iloc[0]

        model_name, parameters = (
            selected_model_from_row(
                selected
            )
        )

        print(
            f"\n{target}"
            f"\nOnly one eligible model: "
            f"{model_name}"
        )

        return (
            model_name,
            parameters
        )


    ########################################
    # Show Choices
    ########################################

    display_test_eligible_models(
        target=target,
        eligible_models=eligible_models
    )


    ########################################
    # Ask User
    ########################################

    while True:

        choice = input(
            f"\nSelect model for {target} "
            f"[1-{len(eligible_models)}]: "
        ).strip()

        try:

            choice_number = int(
                choice
            )

        except ValueError:

            print(
                "Please enter a number from "
                f"1 to {len(eligible_models)}."
            )

            continue


        if not (
            1
            <= choice_number
            <= len(eligible_models)
        ):

            print(
                "Please enter a number from "
                f"1 to {len(eligible_models)}."
            )

            continue


        selected = eligible_models.iloc[
            choice_number - 1
        ]

        return selected_model_from_row(
            selected
        )

def select_all_test_models(
    targets,
    eligible_models_by_target
):

    selections = {}


    ########################################
    # Global Selection Mode
    ########################################

    print("\n" + "=" * 100)
    print("FINAL TEST MODEL SELECTION")
    print("=" * 100)

    print(
        "\n[1] Automatically select Rank 1 "
        "for every target"
    )

    print(
        "[2] Manually choose the model "
        "for every target"
    )


    while True:

        mode = input(
            "\nSelection mode [1/2]: "
        ).strip()

        if mode in {
            "1",
            "2"
        }:
            break

        print(
            "Please enter 1 or 2."
        )


    auto_rank_one = (
        mode == "1"
    )


    ########################################
    # Collect ALL Selections
    ########################################

    for target_number, target in enumerate(
        targets,
        start=1
    ):

        eligible_models = (
            eligible_models_by_target[
                target
            ]
        )


        print(
            f"\n\nTARGET "
            f"[{target_number}/{len(targets)}]"
        )


        ####################################
        # Automatic Rank 1
        ####################################

        if auto_rank_one:

            # DataFrame has already been ordered
            # by Test Selection Rank.
            selected = (
                eligible_models
                .sort_values(
                    "Test Selection Rank"
                )
                .iloc[0]
            )

            model_name, parameters = (
                selected_model_from_row(
                    selected
                )
            )

            print(
                f"{target}"
                f"\nAUTO SELECTED: "
                f"Rank 1 | {model_name}"
            )


        ####################################
        # Manual Selection
        ####################################

        else:

            model_name, parameters = (
                choose_test_eligible_model(
                    target=target,
                    eligible_models=eligible_models
                )
            )


        ####################################
        # Store
        ####################################

        selections[
            target
        ] = {
            "Model": model_name,
            "Parameters": parameters
        }


    ########################################
    # Summary
    ########################################

    print("\n\n" + "=" * 100)
    print("ALL FINAL TEST MODELS SELECTED")
    print("=" * 100)

    for target_number, target in enumerate(
        targets,
        start=1
    ):

        selection = selections[
            target
        ]

        print(
            f"[{target_number}/{len(targets)}] "
            f"{target} | "
            f"{selection['Model']}"
        )


    print(
        "\nSelection complete. "
        "No further user input is required."
    )

    print(
        "Starting all final tests...\n"
    )


    return selections


selected_test_models = (
    select_all_test_models(
        targets=targets,
        eligible_models_by_target=(
            eligible_models_by_target
        )
    )
)

final_results = []
final_errors = []


logger.info(
    "Starting final testing | %d test-eligible targets",
    len(targets)
)


for target_number, target in enumerate(
    targets,
    start=1
):

    try:

        features = selected_features[
            target
        ]


        logger.info(
            "[%d/%d] %s | Starting | %d features",
            target_number,
            len(targets),
            target,
            len(features)
        )


        ########################################
        # No Features
        ########################################

        if len(features) == 0:

            logger.warning(
                "%s | Skipped | No selected features",
                target
            )

            continue


        ########################################
        # Retrieve Pre-Selected Model
        ########################################

        selection = selected_test_models[
            target
        ]

        model_name = selection[
            "Model"
        ]

        parameters = selection[
            "Parameters"
        ]


        logger.info(
            "%s | Pre-selected model=%s | Parameters=%s",
            target,
            model_name,
            parameters
        )

        ########################################
        # Data
        #
        # Read only Date, Ticker, this target,
        # and this target's selected features.
        ########################################

        columns = (
            [
                "Date",
                "Ticker",
                target
            ]
            + features
        )

        columns = list(
            dict.fromkeys(
                columns
            )
        )

        missing_columns = [
            column
            for column in columns
            if column not in SOURCE_TABLE_COLUMNS
        ]

        if len(missing_columns) > 0:
            raise KeyError(
                f"{target} is missing source columns: "
                f"{missing_columns}"
            )

        sql_columns = ", ".join(
            quote_sql_identifier(
                column
            )
            for column in columns
        )

        target_query = (
            f"SELECT {sql_columns} "
            f"FROM {quote_sql_identifier(STOCK_TYPE)}"
        )

        logger.info(
            "%s | Loading %d/%d source columns",
            target,
            len(columns),
            len(SOURCE_TABLE_COLUMNS)
        )

        current_df = pd.read_sql_query(
            target_query,
            data_connection
        )

        logger.info(
            "%s | Loaded %d rows x %d columns | %.2f MB",
            target,
            len(current_df),
            len(current_df.columns),
            current_df.memory_usage(
                deep=True
            ).sum() / (1024 ** 2)
        )


        rows_before_dropna = len(
            current_df
        )


        current_df = current_df.dropna(
            subset=[
                target,
                *features
            ]
        )


        logger.info(
            "%s | Valid rows=%d/%d",
            target,
            len(current_df),
            rows_before_dropna
        )


        current_df["Date"] = pd.to_datetime(
            current_df["Date"]
        )


        current_df = current_df.sort_values(
            [
                "Date",
                "Ticker"
            ]
        ).reset_index(
            drop=True
        )


        ########################################
        # Original Split
        ########################################

        train_df, validation_df, test_df = (
            train_validation_test_split(
                current_df
            )
        )


        logger.info(
            "%s | Split | Train=%d | Validation=%d | Test=%d",
            target,
            len(train_df),
            len(validation_df),
            len(test_df)
        )


        ########################################
        # Final Test
        ########################################

        result = final_test(

            train_df=train_df,
            validation_df=validation_df,
            test_df=test_df,

            features=features,

            target=target,

            model_name=model_name,
            parameters=parameters
        )


        final_results.append(
            result
        )


    except Exception as error:

        logger.exception(
            "%s | FAILED",
            target
        )

        final_errors.append(
            {
                "Target": target,
                "Error": str(error)
            }
        )


data_connection.close()

logger.info(
    "Source database connection closed"
)


final_results_df = pd.DataFrame(
    final_results
)

final_errors_df = pd.DataFrame(
    final_errors
)


FINAL_RESULTS_DATABASE = (
    "/Users/sam/Progressive-Projects/Projects/"
    "Equity Selector/data/Final_Test_Results.db"
)


with sqlite3.connect(
    FINAL_RESULTS_DATABASE
) as connection:

    final_results_df.to_sql(
        f"Final Test Results {STOCK_TYPE}",
        connection,
        if_exists="replace",
        index=False
    )


    if not final_errors_df.empty:

        final_errors_df.to_sql(
            "Errors",
            connection,
            if_exists="replace",
            index=False
        )


logger.info(
    "Final testing complete | "
    "Successful=%d | Failed=%d",
    len(final_results),
    len(final_errors)
)


conn = sqlite3.connect("/Users/sam/Progressive-Projects/Projects/"
    "Equity Selector/data/Final_Test_Results.db")

results = pd.read_sql_query(
    f"SELECT * FROM 'Final Test Results {STOCK_TYPE}'",
    conn
)

# Remove baseline models
results = results[
    ~results["Model"].str.contains("Baseline", case=False, na=False)
]

try:

    continuous = (
        results["Target Type"].eq("continuous")
        &
        # The selector consumes continuous predictions by rank.
        # R2/RMSE remain diagnostics, not admission requirements.
        (results["Rank IC"] >= 0.10)
    )

except KeyError:

    continuous = pd.Series(
        False,
        index=results.index
    )


try:

    binary = (
        results["Target Type"].eq("binary")
        &
        (results["ROC AUC"] >= 0.60)
        &
        (results["PR AUC"] >= 0.20)
    )

except KeyError:

    binary = pd.Series(
        False,
        index=results.index
    )


try:

    multiclass = (
        results["Target Type"].eq("multiclass")
        &
        (results["Macro F1"] >= 0.45)
    )

except KeyError:

    multiclass = pd.Series(
        False,
        index=results.index
    )


useful = results[
    continuous | binary | multiclass
].copy()

useful["Prediction Type"] = useful["Target"].apply(target_category)


useful["Predictability Score"] = useful.apply(
    predictability_score,
    axis=1
)


def calculate_quality_score(
    row,
    portfolio_type,
):

    statistical_type = str(
        row.get(
            "Target Type",
            "",
        )
    ).strip().lower()

    existing_predictability = _row_metric(
        row,
        [
            "Predictability Score",
        ],
    )

    ####################################
    # Continuous Targets
    ####################################

    if statistical_type == "continuous":

        rank_ic = _row_metric(
            row,
            [
                "Rank IC",
                "Mean Rank IC",
                "Rank IC Mean",
                "Spearman IC",
                "Spearman Correlation",
                "Spearman",
            ],
        )

        r2 = _row_metric(
            row,
            [
                "R2",
                "R^2",
                "R Squared",
                "R2 Score",
                "Test R2",
                "Test R^2",
            ],
        )

        q_ic = (
            _clip_quality(
                abs(rank_ic) / 0.30
            )
            if rank_ic is not None
            else None
        )

        q_r2 = (
            _clip_quality(
                max(r2, 0.0) / 0.20
            )
            if r2 is not None
            else None
        )

        quality = _weighted_available_mean(
            [
                (0.70, q_ic),
                (0.30, q_r2),
            ]
        )

        if quality is not None:
            return _clip_quality(
                quality
            )

    ####################################
    # Binary Targets
    ####################################

    if statistical_type == "binary":

        roc_auc = _row_metric(
            row,
            [
                "ROC AUC",
                "ROC-AUC",
                "AUC ROC",
                "AUC",
                "Test ROC AUC",
            ],
        )

        pr_auc = _row_metric(
            row,
            [
                "PR AUC",
                "PR-AUC",
                "Average Precision",
                "Average Precision Score",
                "Test PR AUC",
            ],
        )

        positive_rate = _row_metric(
            row,
            [
                "Positive Rate",
                "Positive Class Rate",
                "Positive Fraction",
                "Prevalence",
                "Base Rate",
                "Event Rate",
            ],
        )

        q_roc = (
            _clip_quality(
                (roc_auc - 0.50) / 0.25
            )
            if roc_auc is not None
            else None
        )

        q_pr = None

        if (
            pr_auc is not None
            and positive_rate is not None
            and 0.0 <= positive_rate < 1.0
        ):

            # A +0.30 PR-AUC improvement over the random/base-rate
            # classifier is treated as an excellent result and capped
            # at 1. This is much fairer for rare-event targets than raw
            # accuracy or raw PR-AUC.
            excellent_pr = min(
                1.0,
                positive_rate + 0.30,
            )

            denominator = max(
                excellent_pr - positive_rate,
                1e-12,
            )

            q_pr = _clip_quality(
                (
                    pr_auc - positive_rate
                )
                / denominator
            )

        event_types = {
            "TAIL_EVENT",
            "VOLATILITY_EVENT",
            "UPSIDE_EVENT",
        }

        if portfolio_type in event_types:
            weights = [
                (0.40, q_roc),
                (0.60, q_pr),
            ]
        else:
            weights = [
                (0.60, q_roc),
                (0.40, q_pr),
            ]

        quality = _weighted_available_mean(
            weights
        )

        if quality is not None:
            return _clip_quality(
                quality
            )

    ####################################
    # Multiclass Targets
    ####################################

    if statistical_type == "multiclass":

        macro_f1 = _row_metric(
            row,
            [
                "Macro F1",
                "Macro-F1",
                "F1 Macro",
                "Macro F1 Score",
                "Test Macro F1",
            ],
        )

        macro_auc = _row_metric(
            row,
            [
                "Macro ROC AUC",
                "Macro AUC",
                "OVR Macro AUC",
                "One Vs Rest Macro AUC",
                "Multiclass ROC AUC",
            ],
        )

        number_classes = _row_metric(
            row,
            [
                "Number Classes",
                "Number of Classes",
                "N Classes",
                "Num Classes",
            ],
        )

        if (
            number_classes is None
            or number_classes < 2
        ):
            number_classes = 3.0

        chance_f1 = 1.0 / number_classes
        excellent_f1 = 0.70

        q_f1 = None

        if macro_f1 is not None:

            denominator = max(
                excellent_f1 - chance_f1,
                1e-12,
            )

            q_f1 = _clip_quality(
                (
                    macro_f1 - chance_f1
                )
                / denominator
            )

        q_auc = (
            _clip_quality(
                (macro_auc - 0.50) / 0.25
            )
            if macro_auc is not None
            else None
        )

        quality = _weighted_available_mean(
            [
                (0.70, q_f1),
                (0.30, q_auc),
            ]
        )

        if quality is not None:
            return _clip_quality(
                quality
            )

    ####################################
    # Fallback
    ####################################

    # The existing Predictability Score is already the research
    # pipeline's best summary when the underlying component metrics
    # are not stored in the SQL table. Keep the pipeline operational
    # rather than discarding an otherwise valid model.
    if existing_predictability is not None:
        return _clip_quality(
            existing_predictability
        )

    return 0.0


########################################
# Horizon Extraction
########################################

def target_horizon(
    target,
):

    name = str(target).strip().lower()

    # Prefer values with explicit units. This allows names such as
    # "Sudden Drawdown 15m" or "Direction 1h" to be interpreted
    # correctly even if another number appears earlier in the name.
    explicit = re.findall(
        r"(?<![a-z0-9])"
        r"(\d+(?:\.\d+)?)\s*"
        r"(m|min|mins|minute|minutes|"
        r"h|hr|hrs|hour|hours|"
        r"d|day|days|"
        r"w|week|weeks)"
        r"(?![a-z])",
        name,
    )

    if explicit:
        value = float(
            explicit[-1][0]
        )

        return (
            int(value)
            if value.is_integer()
            else value
        )

    # Daily targets in the existing targets.py convention generally
    # finish with the horizon, e.g. "Forward Return 20". Taking the
    # last standalone number also avoids using an earlier threshold
    # such as the 2 in "Return Above 2% 20".
    numbers = re.findall(
        r"(?<![a-z0-9.])"
        r"(\d+(?:\.\d+)?)"
        r"(?![a-z0-9.%])",
        name,
    )

    if not numbers:
        # Do not fall back to arbitrary numbers because many target
        # names contain thresholds such as 2% or 5%. If no horizon
        # can be distinguished safely, store NULL and log it later.
        return np.nan

    value = float(
        numbers[-1]
    )

    return (
        int(value)
        if value.is_integer()
        else value
    )



def portfolio_target_type(
    target,
    prediction_type=None,
):

    name = str(
        target
    ).strip().lower()

    prediction_type = str(
        prediction_type or ""
    ).strip().lower()


    ########################################
    # Execution / Market Structure
    ########################################

    if (
        "market impact" in name
        or "price impact" in name
    ):
        return "MARKET_IMPACT"

    if (
        "execution" in name
        or "fill probability" in name
        or "fill rate" in name
        or "slippage" in name
    ):
        return "EXECUTION"

    if (
        "liquidity" in name
        or "bid ask spread" in name
        or "bid-ask spread" in name
        or "order book depth" in name
        or "order-book depth" in name
    ):
        return "LIQUIDITY"


    ########################################
    # Dependence / State
    ########################################

    if "covariance" in name:
        return "COVARIANCE"

    if "correlation" in name:
        return "CORRELATION"

    if "regime" in name:
        return "REGIME"


    ########################################
    # Recovery / Reversal
    ########################################

    if (
        "recovery" in name
        or "recover" in name
        or "bounce back" in name
    ):
        return "RECOVERY"

    if (
        "reversal" in name
        or "reverse" in name
        or "mean reversion" in name
        or "mean-reversion" in name
    ):
        return "REVERSAL"


    ########################################
    # Volatility Barriers / Events
    #
    # MUST come before generic volatility.
    ########################################

    if name.startswith(
        "volatility barrier"
    ):
        return "VOLATILITY_EVENT"

    if (
        "volatility event" in name
        or "volatility spike" in name
        or "volatility breakout" in name
    ):
        return "VOLATILITY_EVENT"


    ########################################
    # Tail Events
    ########################################

    if (
        "sudden drawdown" in name
        or "crash" in name
        or "tail event" in name
        or "extreme downside" in name
        or "downside event" in name
        or "negative spike" in name
    ):
        return "TAIL_EVENT"

    if (
        "upside spike" in name
        or "positive spike" in name
        or "upside event" in name
        or "positive event" in name
    ):
        return "UPSIDE_EVENT"


    ########################################
    # Excursions
    ########################################

    if name.startswith(
        "time to maximum favourable excursion"
    ):
        return "TIME_TO_UPSIDE_EXCURSION"

    if name.startswith(
        "time to maximum adverse excursion"
    ):
        return "TIME_TO_DOWNSIDE_EXCURSION"

    if name.startswith(
        "maximum favourable excursion"
    ):
        return "UPSIDE_EXCURSION"

    if name.startswith(
        "maximum adverse excursion"
    ):
        return "DOWNSIDE_EXCURSION"


    ########################################
    # Drawdown / Tail Risk
    ########################################

    if (
        name.startswith(
            "future maximum drawdown"
        )
        or "expected shortfall" in name
        or "conditional value at risk" in name
        or "conditional var" in name
        or "cvar" in name
        or "value at risk" in name
        or re.search(
            r"\bvar\b",
            name
        )
        or "tail risk" in name
    ):
        return "TAIL_RISK"


    ########################################
    # Minimum / Downside Return
    ########################################

    if (
        name.startswith(
            "future minimum return"
        )
        or "minimum return" in name
        or "min return" in name
    ):
        return "DOWNSIDE"


    ########################################
    # Volatility Asymmetry
    #
    # Check ratio before individual
    # upside/downside volatility.
    ########################################

    if (
        "downside upside volatility ratio"
        in name
    ):
        return "VOLATILITY_ASYMMETRY"


    ########################################
    # Downside Volatility
    ########################################

    if (
        "downside volatility" in name
        or "downside deviation" in name
    ):
        return "DOWNSIDE_VOLATILITY"


    ########################################
    # Upside Volatility
    ########################################

    if (
        "upside volatility" in name
        or "positive volatility" in name
    ):
        return "UPSIDE_VOLATILITY"


    ########################################
    # Absolute Movement
    #
    # These are NOT alpha.
    #
    # Future Mean Absolute Return
    # Future Maximum Absolute Return
    ########################################

    if (
        "mean absolute return" in name
        or "maximum absolute return" in name
        or "max absolute return" in name
        or "absolute return" in name
    ):
        return "ABSOLUTE_MOVE"


    ########################################
    # Variance
    ########################################

    if (
        name.startswith(
            "future variance"
        )
        or "variance" in name
    ):
        return "VOLATILITY"


    ########################################
    # Generic Volatility
    ########################################

    if "volatility" in name:
        return "VOLATILITY"


    ########################################
    # Risk-Adjusted Alpha
    #
    # Check BEFORE generic return rules.
    ########################################

    if (
        "return volatility ratio" in name
        or "sortino ratio" in name
        or "sharpe" in name
        or "calmar" in name
        or "return minus risk" in name
        or "return drawdown ratio" in name
        or "risk adjusted" in name
        or "risk-adjusted" in name
    ):
        return "RISK_ADJUSTED_ALPHA"


    ########################################
    # Cross-Sectional Downside
    ########################################

    if (
        "bottom 20 percent future return"
        in name
        or "bottom 25 percent future return"
        in name
        or "bottom quintile" in name
        or "bottom quartile" in name
    ):
        return "CROSS_SECTION_DOWNSIDE"


    ########################################
    # Cross-Sectional Alpha
    ########################################

    if (
        "top 20 percent future return"
        in name
        or "top 25 percent future return"
        in name
        or "top 10 percent future return"
        in name
        or "top quintile" in name
        or "top quartile" in name
        or "future return rank" in name
        or "return rank" in name
        or "return percentile" in name
        or "return quantile" in name
        or "cross sectional" in name
        or "cross-sectional" in name
    ):
        return "CROSS_SECTION_ALPHA"


    ########################################
    # Relative Alpha
    ########################################

    if (
        "excess return" in name
        or "relative return" in name
        or "abnormal return" in name
        or "residual return" in name
        or "benchmark return" in name
    ):
        return "RELATIVE_ALPHA"


    ########################################
    # Three-Class Direction
    #
    # Keep separate from binary direction
    # because it is a different prediction
    # problem and different evaluation metric.
    ########################################

    if name.startswith(
        "three class direction"
    ):
        return "DIRECTION_MULTICLASS"


    ########################################
    # Binary Direction
    ########################################

    if name.startswith(
        "future direction"
    ):
        return "DIRECTION"


    ########################################
    # Volatility barrier already caught.
    #
    # Remaining Barrier targets are
    # directional path / return barriers.
    ########################################

    if name.startswith(
        "barrier"
    ):
        return "BARRIER_ALPHA"


    ########################################
    # Binary Return Thresholds
    ########################################

    if name.startswith(
        "future return above"
    ):
        return "ALPHA_BINARY"


    ########################################
    # Ordinary Signed Forward Returns
    ########################################

    if (
        name.startswith(
            "forward return"
        )
        or name.startswith(
            "forward log return"
        )
    ):
        return "ALPHA"


    ########################################
    # Explicit Alpha / Momentum
    ########################################

    if (
        "alpha" in name
        or "momentum" in name
    ):
        return "ALPHA"


    ########################################
    # Fallbacks
    #
    # Do NOT silently call unknown targets
    # ALPHA. That hides classification bugs.
    ########################################

    if prediction_type == "volatility":
        return "VOLATILITY"

    if prediction_type == "downside":
        return "DOWNSIDE"

    raise ValueError(
        "Could not determine Portfolio Target Type "
        f"for target: {target!r}"
    )

########################################
# Cross-Target Model Quality
########################################

def _normalise_column_name(
    value,
):

    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).lower(),
    )


def _row_metric(
    row,
    aliases,
):

    column_lookup = {
        _normalise_column_name(column): column
        for column in row.index
    }

    for alias in aliases:

        key = _normalise_column_name(
            alias
        )

        if key not in column_lookup:
            continue

        value = pd.to_numeric(
            pd.Series(
                [
                    row[
                        column_lookup[key]
                    ]
                ]
            ),
            errors="coerce",
        ).iloc[0]

        if pd.notna(value):
            return float(value)

    return None


def _clip_quality(
    value,
):

    return float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )


def _weighted_available_mean(
    values,
):

    # values = [(weight, value), ...]
    available = [
        (weight, value)
        for weight, value in values
        if value is not None
        and np.isfinite(value)
    ]

    if not available:
        return None

    total_weight = sum(
        weight
        for weight, _ in available
    )

    return sum(
        weight * value
        for weight, value in available
    ) / total_weight


useful[
    "Portfolio Target Type"
] = useful.apply(
    lambda row: portfolio_target_type(
        row[
            "Target"
        ],
        row.get(
            "Prediction Type",
            "",
        ),
    ),
    axis=1,
)

useful[
    "Horizon"
] = useful[
    "Target"
].map(
    target_horizon
)

########################################
# Absolute Quality Score
########################################

useful[
    "Absolute Quality Score"
] = useful.apply(
    lambda row: calculate_quality_score(
        row,
        row[
            "Portfolio Target Type"
        ],
    ),
    axis=1,
)


########################################
# Relative Quality Score
#
# The number of targets determines the
# available score range:
#
# 1 target  -> 0.500
# 2 targets -> 0.250 to 0.750
# 3 targets -> 0.167 to 0.833
# 4 targets -> 0.125 to 0.875
#
# Position inside that range depends
# continuously on Absolute Quality Score.
########################################

def relative_quality_score(
    group
):

    number_targets = len(
        group
    )


    ####################################
    # Only One Target
    ####################################

    if number_targets == 1:

        return pd.Series(
            0.5,
            index=group.index,
            dtype=float
        )


    ####################################
    # Available Relative Score Range
    ####################################

    lower_bound = (
        0.5 / number_targets
    )

    upper_bound = (
        1.0 - lower_bound
    )


    ####################################
    # Absolute Quality Range
    ####################################

    absolute_quality = group[
        "Absolute Quality Score"
    ].astype(float)

    minimum_quality = (
        absolute_quality.min()
    )

    maximum_quality = (
        absolute_quality.max()
    )


    ####################################
    # All Targets Have Same Quality
    #
    # There is no evidence to rank them,
    # so all receive neutral 0.5.
    ####################################

    if np.isclose(
        maximum_quality,
        minimum_quality
    ):

        return pd.Series(
            0.5,
            index=group.index,
            dtype=float
        )


    ####################################
    # Continuous Min-Max Position
    ####################################

    relative_position = (
        (
            absolute_quality
            - minimum_quality
        )
        /
        (
            maximum_quality
            - minimum_quality
        )
    )


    ####################################
    # Scale Into Allowed Range
    ####################################

    relative_quality = (
        lower_bound
        +
        relative_position
        * (
            upper_bound
            - lower_bound
        )
    )


    return relative_quality


########################################
# Calculate Within Each Specific
# Portfolio Target Type
########################################

useful[
    "Quality Score"
] = useful.groupby(
    "Portfolio Target Type",
    group_keys=False
)["Absolute Quality Score"].transform(
    lambda scores: relative_quality_score(
        useful.loc[
            scores.index
        ]
    )
)

missing_horizons = useful[
    useful[
        "Horizon"
    ].isna()
]

if not missing_horizons.empty:
    logger.warning(
        "%d selected targets do not contain a parseable "
        "numeric horizon. Their Horizon value will be NULL: %s",
        len(
            missing_horizons
        ),
        ", ".join(
            missing_horizons[
                "Target"
            ].astype(str)
        ),
    )

useful = useful.sort_values(
    [
        "Target Type",
        "Quality Score"
    ],
    ascending=[
        True,
        False
    ]
)

with sqlite3.connect(
    FINAL_RESULTS_DATABASE
) as connection:

    useful.to_sql(
        f"{STOCK_TYPE} Passed Test Results",
        connection,
        if_exists="replace",
        index=False
    )
