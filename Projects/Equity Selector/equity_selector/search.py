from .parameters import parse_parameters, configuration_key, same_parameter
import ast
from numbers import Number
import re

import numpy as np
import pandas as pd


INTEGER_PARAMETERS = [
    "max_iter",
    "max_leaf_nodes",
    "max_depth",
    "min_samples_leaf",
    "min_samples_split",
    "n_estimators",
    "num_leaves",
    "min_child_samples",
    "n_neighbors",
]

import itertools


def create_tpe_model_recommendations(
    family,
    parameters,
    configuration_tpe_scores,
    family_search_df,
    x,
):

    model_recommendations = []

    ########################################
    # Store Previously Tested Configurations
    ########################################

    tested_configurations = {
        tuple(configuration.get(parameter) for parameter in parameters)
        for configuration in (family_search_df["Parameters"].apply(parse_parameters))
    }

    ########################################
    # Create Every Possible Configuration
    ########################################

    parameter_names = list(parameters.keys())

    parameter_value_combinations = itertools.product(
        *[parameters[parameter] for parameter in parameter_names]
    )

    for parameter_values in parameter_value_combinations:
        configuration = dict(
            zip(
                parameter_names,
                parameter_values,
            )
        )

        configuration_key = tuple(configuration[parameter] for parameter in parameter_names)

        ####################################
        # Remove Already-Tested Models
        ####################################

        if configuration_key in tested_configurations:
            continue

        ####################################
        # Sum Parameter TPE Scores
        ####################################

        tpe_score = sum(
            configuration_tpe_scores[parameter][configuration[parameter]]
            for parameter in parameter_names
        )

        ####################################
        # Store Recommendation
        ####################################

        model_recommendations.append(
            {
                "name": family,
                "params": configuration,
                "TPE Score": tpe_score,
            }
        )

    ########################################
    # Sort From Highest To Lowest TPE Score
    ########################################

    model_recommendations.sort(
        key=lambda model: model["TPE Score"],
        reverse=True,
    )

    x = min(x, len(model_recommendations))

    if x == len(model_recommendations):
        return model_recommendations
    ########################################
    # Return The Best X Unseen Models
    ########################################

    return model_recommendations[:x]


def is_missing_scalar(value):

    if not pd.api.types.is_scalar(value):
        return False

    return bool(pd.isna(value))


def recommend_models_to_fit(
    search_df,
    target_type,
    n,
    x,
    model_lookup,
):

    if n < 0 or x < 0:
        raise ValueError("Recommendation limits must be nonnegative")
    if n == 0 or x == 0 or search_df.empty:
        return [], []
    if not search_df.index.is_unique:
        raise ValueError("Search row index must be unique")
    current_x = 0

    model_recommendations = []
    recommendation_midpoint_column = []

    ranked_search = search_df.assign(
        _Rank=pd.to_numeric(search_df["Rank"], errors="raise")
    ).sort_values("_Rank", kind="stable")
    tested = {
        configuration_key({"name": row["Model"], "params": row["Parameters"]})
        for _, row in search_df.iterrows()
    }

    def is_new(model):
        key = configuration_key(model)
        return key not in tested and all(
            configuration_key(other) != key for other in model_recommendations
        )

    families_avaliable = ranked_search["Model"].dropna().unique()

    if len(families_avaliable) > n:
        families_avaliable = families_avaliable[:n]

    for family in families_avaliable:
        family_parameters = model_lookup[family]["params"]
        parameter_names = list(family_parameters.keys())

        ####################################
        # Keep Rows For Current Family
        ####################################

        family_search_df = (
            ranked_search[ranked_search["Model"].eq(family)]
            .drop(columns="_Rank")
            .copy()
            .rename_axis("_Search DF Index")
            .reset_index()
        )

        family_search_df = family_search_df.rename(columns={"index": "_Search DF Index"})

        ####################################
        # Convert Parameters To Dictionaries
        ####################################

        parameter_dictionaries = family_search_df["Parameters"].apply(parse_parameters)

        ####################################
        # Expand Parameters Into Columns
        ####################################

        expanded_parameters = pd.DataFrame(
            parameter_dictionaries.tolist(),
            index=family_search_df.index,
            dtype=object,
        )

        for parameter_name in parameter_names:
            if parameter_name not in expanded_parameters.columns:
                expanded_parameters[parameter_name] = None

        expected_parameter_columns = [
            parameter_name
            for parameter_name in parameter_names
            if parameter_name in expanded_parameters.columns
        ]

        additional_parameter_columns = [
            parameter_name
            for parameter_name in expanded_parameters.columns
            if parameter_name not in expected_parameter_columns
        ]

        expanded_parameters = expanded_parameters[
            expected_parameter_columns + additional_parameter_columns
        ]

        conflicting_columns = set(expanded_parameters.columns).intersection(
            family_search_df.columns
        )

        if conflicting_columns:
            raise ValueError(
                "Parameter names conflict with existing "
                "search dataframe columns | "
                f"family={family!r} | "
                f"columns={sorted(conflicting_columns)!r}"
            )

        family_search_df = pd.concat(
            [
                family_search_df,
                expanded_parameters,
            ],
            axis=1,
        )

        parameters = {}

        for parameter in expected_parameter_columns:
            unique_values = family_search_df[parameter].unique()

            parameters[parameter] = unique_values

        ####################################
        # Normalise TPE Counter
        ####################################

        family_tpe_scores = pd.to_numeric(
            family_search_df["TPE Score?"],
            errors="coerce",
        ).fillna(0)

        current_tpe_max = float(family_tpe_scores.max())

        ########################################
        # Compare TPE And Non-TPE Best Results
        ########################################

        if not family_tpe_scores.eq(current_tpe_max).all():
            ranked_family_search_df = family_search_df.assign(
                _Numeric_TPE_Score=(family_tpe_scores)
            ).sort_values(
                "Rank",
                ascending=True,
                kind="stable",
            )

            best_with_tpe = ranked_family_search_df[
                ranked_family_search_df["_Numeric_TPE_Score"].eq(current_tpe_max)
            ].head(1)

            best_without_tpe = ranked_family_search_df[
                ~ranked_family_search_df["_Numeric_TPE_Score"].eq(current_tpe_max)
            ].head(1)

            if not best_with_tpe.empty and not best_without_tpe.empty:
                best_with_tpe_score = float(best_with_tpe["Model Selection Score"].iloc[0])

                best_without_tpe_score = float(best_without_tpe["Model Selection Score"].iloc[0])

                model_selection_difference = best_with_tpe_score - best_without_tpe_score

                if model_selection_difference > 0.002:
                    family_search_df["TPE Score?"] = current_tpe_max

                    search_df.loc[
                        search_df["Model"].eq(family),
                        "TPE Score?",
                    ] = current_tpe_max
                else:
                    family_search_df["TPE Score?"] = 1

                    search_df.loc[
                        search_df["Model"].eq(family),
                        "TPE Score?",
                    ] = 1

                    current_tpe_max = 1

        if (
            len(family_search_df)
            < max(
                12,
                (2 * len(expected_parameter_columns)) + 5,
            )
            or current_tpe_max > 25
        ):
            current_tpe_max = 0

            family_search_df["TPE Score?"] = 0

            search_df.loc[
                search_df["Model"].eq(family),
                "TPE Score?",
            ] = 0

        elif len(family_search_df) < max(
            37,
            (2 * len(expected_parameter_columns)) + 30,
        ):
            if current_tpe_max == 0:
                current_tpe_max = 1

            family_search_df["TPE Score?"] = current_tpe_max

            search_df.loc[
                search_df["Model"].eq(family),
                "TPE Score?",
            ] = current_tpe_max

        ####################################
        # Add Two-Sided Midpoint Models
        ####################################

        def add_two_sided_midpoints(
            anchor_row,
            parameter,
        ):
            nonlocal current_x

            if current_x >= x or parameter not in expected_parameter_columns:
                return

            current_config = anchor_row[parameter]

            ########################################
            # Add Non-Numeric Alternatives
            ########################################

            if not isinstance(
                current_config,
                Number,
            ) or isinstance(
                current_config,
                bool,
            ):
                unique_values = []
                non_numeric_values = [
                    value
                    for value in parameters[parameter]
                    if (
                        not isinstance(
                            value,
                            Number,
                        )
                        or isinstance(
                            value,
                            bool,
                        )
                    )
                ]

                for value in non_numeric_values:
                    if same_parameter(value, current_config):
                        continue

                    if value not in unique_values:
                        unique_values.append(value)

                for parameter_value in unique_values:
                    if current_x >= x:
                        break

                    new_parameters = {
                        column: anchor_row[column] for column in expected_parameter_columns
                    }

                    new_parameters[parameter] = parameter_value

                    new_model = {
                        "name": family,
                        "params": new_parameters,
                    }

                    if is_new(new_model):
                        model_recommendations.append(new_model)

                        recommendation_midpoint_column.append(parameter)

                        current_x += 1

                return

            ########################################
            # Add Numeric Midpoints
            ########################################

            parameter_values = [
                value
                for value in parameters[parameter]
                if (
                    isinstance(
                        value,
                        Number,
                    )
                    and not isinstance(
                        value,
                        bool,
                    )
                    and not is_missing_scalar(value)
                )
            ]

            if not parameter_values:
                return

            current_config = float(current_config)

            if parameter in INTEGER_PARAMETERS:
                current_config = int(round(current_config))

            other_parameters = [
                other_parameter
                for other_parameter in expected_parameter_columns
                if other_parameter != parameter
            ]

            matching_mask = pd.Series(
                True,
                index=family_search_df.index,
            )

            for other_parameter in other_parameters:
                anchor_value = anchor_row[other_parameter]

                if is_missing_scalar(anchor_value):
                    matching_mask &= family_search_df[other_parameter].isna()

                else:
                    matching_mask &= family_search_df[other_parameter].apply(
                        lambda value: same_parameter(value, anchor_value)
                    )

            matching_parameter_values = [
                value
                for value in family_search_df.loc[
                    matching_mask,
                    parameter,
                ].tolist()
                if (
                    isinstance(
                        value,
                        Number,
                    )
                    and not isinstance(value, bool)
                    and not is_missing_scalar(value)
                )
            ]

            matching_parameter_values.sort()

            nearest_below = max(
                (value for value in matching_parameter_values if value < current_config),
                default=np.min(parameter_values),
            )

            nearest_above = min(
                (value for value in matching_parameter_values if value > current_config),
                default=np.max(parameter_values),
            )

            midpoint_values = list(parameter_values)

            midpoint_values.extend(
                [
                    current_config,
                    nearest_below,
                    nearest_above,
                ]
            )

            midpoint_values = sorted(set(midpoint_values))

            below_values = [
                value for value in midpoint_values if (nearest_below <= value <= current_config)
            ]

            above_values = [
                value for value in midpoint_values if (current_config <= value <= nearest_above)
            ]

            below = float(np.median(below_values))

            above = float(np.median(above_values))

            if parameter in INTEGER_PARAMETERS:
                below = int(round(below))

                above = int(round(above))

            for midpoint_value in (
                below,
                above,
            ):
                if current_x >= x or midpoint_value == current_config:
                    continue

                new_parameters = {
                    column: anchor_row[column] for column in expected_parameter_columns
                }

                new_parameters[parameter] = midpoint_value

                new_model = {
                    "name": family,
                    "params": new_parameters,
                }

                if is_new(new_model):
                    model_recommendations.append(new_model)

                    recommendation_midpoint_column.append(parameter)

                    current_x += 1

        ####################################
        # Continue Existing Midpoint Searches
        ####################################

        current_x = 0

        family_tpe_scores = pd.to_numeric(
            search_df.loc[
                search_df["Model"].eq(family),
                "TPE Score?",
            ],
            errors="coerce",
        ).fillna(0)

        if not family_tpe_scores.eq(0).all():
            search_df.loc[
                search_df["Model"].eq(family),
                "Midpoint?",
            ] = np.nan

            size = len(family_search_df)
            good_part_size = size // 4
            bad_part_size = size - good_part_size
            scalar = bad_part_size / good_part_size

            good_results = family_search_df.head(good_part_size)
            bad_results = family_search_df.tail(bad_part_size)

            configuration_tpe_scores = {}

            parameters = {}

            for parameter in expected_parameter_columns:
                configuration_tpe_scores[parameter] = {}

                unique_values = family_search_df[parameter].unique()

                parameters[parameter] = unique_values

                for value in unique_values:
                    good_count = (
                        good_results[parameter]
                        .apply(lambda parameter_value: same_parameter(parameter_value, value))
                        .sum()
                    )

                    bad_count = (
                        bad_results[parameter]
                        .apply(lambda parameter_value: same_parameter(parameter_value, value))
                        .sum()
                    )

                    if bad_count == 0:
                        tpe_score = float("inf")
                    elif good_count == 0:
                        tpe_score = float("-inf")
                    else:
                        tpe_score = np.log((good_count / bad_count) * scalar)
                    configuration_tpe_scores[parameter][value] = tpe_score

            finite = [
                score
                for scores in configuration_tpe_scores.values()
                for score in scores.values()
                if np.isfinite(score)
            ]
            upper = max(finite, default=0) + 1
            lower = min(finite, default=0) - 1
            for scores in configuration_tpe_scores.values():
                for key, score in scores.items():
                    if np.isposinf(score):
                        scores[key] = upper
                    elif np.isneginf(score):
                        scores[key] = lower

            models = create_tpe_model_recommendations(
                family=family,
                parameters=parameters,
                configuration_tpe_scores=configuration_tpe_scores,
                family_search_df=family_search_df,
                x=x - current_x,
            )

            models = [{"name": model["name"], "params": model["params"]} for model in models]

            num_of_models = len(models)
            randoms_created = 0
            while randoms_created < num_of_models // 4:
                clone = models[np.random.randint(len(models))]

                parameter = np.random.choice(expected_parameter_columns)
                non_selected_parameters = [p for p in expected_parameter_columns if p != parameter]
                random_parameters = {}
                random_parameters[parameter] = parameters[parameter][
                    np.random.randint(len(parameters[parameter]))
                ]
                for non_selected_parameter in non_selected_parameters:
                    random_parameters[non_selected_parameter] = clone["params"][
                        non_selected_parameter
                    ]

                random_model = {"name": family, "params": random_parameters}

                randoms_created += 1

                if is_new(random_model) and random_model not in models and len(models) < x:
                    models.append(random_model)

            model_recommendations.extend(models)
            recommendation_midpoint_column.extend([np.nan for _ in models])

            continue

        if "Midpoint?" not in family_search_df.columns:
            family_search_df["Midpoint?"] = np.nan

        midpoint_rows = family_search_df[family_search_df["Midpoint?"].notna()].copy()

        ####################################
        # Continue Named Midpoint Searches
        ####################################

        if not midpoint_rows.empty:
            if "Model Selection Score" not in midpoint_rows.columns:
                raise KeyError(
                    "Model Selection Score is required when Midpoint? contains non-missing values."
                )

            midpoint_rows["_Model Selection Score"] = pd.to_numeric(
                midpoint_rows["Model Selection Score"],
                errors="coerce",
            )

            midpoint_parameters = midpoint_rows["Midpoint?"].drop_duplicates().tolist()

            all_midpoint_parameters_converged = True

            for midpoint_parameter in midpoint_parameters:
                parameter_rows = (
                    midpoint_rows[midpoint_rows["Midpoint?"].eq(midpoint_parameter)]
                    .dropna(subset=["_Model Selection Score"])
                    .sort_values(
                        "_Model Selection Score",
                        ascending=False,
                        kind="stable",
                    )
                )

                if parameter_rows.empty:
                    all_midpoint_parameters_converged = False
                    continue

                best_row = parameter_rows.iloc[0]

                parameter_has_converged = False

                if len(parameter_rows) >= 2:
                    best_two_rows = parameter_rows.iloc[:2]

                    score_difference = abs(
                        best_two_rows.iloc[0]["_Model Selection Score"]
                        - best_two_rows.iloc[1]["_Model Selection Score"]
                    )

                    parameter_has_converged = score_difference <= 0.002

                if parameter_has_converged:
                    continue

                all_midpoint_parameters_converged = False

                non_best_indices = midpoint_rows.loc[
                    midpoint_rows["Midpoint?"].eq(midpoint_parameter)
                    & midpoint_rows["_Search DF Index"].ne(best_row["_Search DF Index"]),
                    "_Search DF Index",
                ].tolist()

                if non_best_indices:
                    search_df.loc[
                        non_best_indices,
                        "Midpoint?",
                    ] = np.nan

                add_two_sided_midpoints(
                    anchor_row=best_row,
                    parameter=midpoint_parameter,
                )

            if all_midpoint_parameters_converged:
                search_df.loc[
                    search_df["Model"].eq(family),
                    "TPE Score?",
                ] = 1

                continue

        ####################################
        # Fill Remaining Capacity Normally
        ####################################

        if current_x < x:
            for _, normal_anchor_row in family_search_df.iterrows():
                if current_x >= x:
                    break

                for parameter in expected_parameter_columns:
                    if current_x >= x:
                        break

                    add_two_sided_midpoints(
                        anchor_row=normal_anchor_row,
                        parameter=parameter,
                    )

    return model_recommendations, recommendation_midpoint_column


def add_recommendation_midpoints(
    new_summary,
    recommended_models,
    recommendation_midpoint_column,
):

    new_summary = new_summary.copy()

    if len(recommended_models) != len(recommendation_midpoint_column):
        raise ValueError(
            "recommended_models and recommendation_midpoint_column must have the same length."
        )

    if "Midpoint?" not in new_summary.columns:
        new_summary["Midpoint?"] = pd.Series(None, index=new_summary.index, dtype=object)

    new_summary["Midpoint?"] = new_summary["Midpoint?"].astype(object)

    parsed_summary_parameters = new_summary["Parameters"].apply(parse_parameters)

    for (
        recommended_model,
        midpoint_parameter,
    ) in zip(
        recommended_models,
        recommendation_midpoint_column,
    ):
        model_name = recommended_model["name"]
        model_parameters = recommended_model["params"]

        matching_mask = new_summary["Model"].eq(model_name) & parsed_summary_parameters.apply(
            lambda parameters: parameters == model_parameters
        )

        new_summary.loc[
            matching_mask,
            "Midpoint?",
        ] = midpoint_parameter

    ########################################
    # Increase TPE Scores
    ########################################

    recommended_family_counts = {}

    for recommended_model in recommended_models:
        model_name = recommended_model["name"]

        recommended_family_counts[model_name] = (
            recommended_family_counts.get(
                model_name,
                0,
            )
            + 1
        )

    numeric_tpe_scores = pd.to_numeric(
        new_summary["TPE Score?"],
        errors="coerce",
    ).fillna(0)

    for (
        model_name,
        number_recommended,
    ) in recommended_family_counts.items():
        recommended_parameter_dictionaries = [
            recommended_model["params"]
            for recommended_model in recommended_models
            if recommended_model["name"] == model_name
        ]

        matching_recommendation_mask = new_summary["Model"].eq(
            model_name
        ) & parsed_summary_parameters.apply(
            lambda row_parameters: any(
                row_parameters == recommended_parameters
                for recommended_parameters in recommended_parameter_dictionaries
            )
        )

        rows_to_update = matching_recommendation_mask & numeric_tpe_scores.gt(0)

        new_summary.loc[
            rows_to_update,
            "TPE Score?",
        ] = numeric_tpe_scores.loc[rows_to_update] + number_recommended

    return new_summary
