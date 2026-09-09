import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr


def pearson_correlation(df, features, target, threshold=0.01):

    selected_features = []
    to_drop = []

    for feature in features:
        temp_df = df[[feature, target]].dropna()

        if len(temp_df) < 2 or temp_df[feature].nunique() < 2 or temp_df[target].nunique() < 2:
            to_drop.append(feature)
            continue

        pearson = pearsonr(temp_df[feature], temp_df[target]).statistic

        if np.isfinite(pearson) and abs(pearson) > threshold:
            selected_features.append(feature)
        else:
            to_drop.append(feature)

    return selected_features, to_drop


def ic_screen(df, features, target, ic_threshold=0.01, ir_threshold=0.25, sign_threshold=0.55):

    daily_ics = []

    for date, group in df.groupby("Date"):
        feature_ranks = group[features].rank()
        target_rank = group[target].rank()

        ic = feature_ranks.corrwith(target_rank)

        daily_ics.append(ic)

    daily_ics = pd.DataFrame(daily_ics)

    mean_ic = daily_ics.mean()
    std_ic = daily_ics.std()

    ic_ir = mean_ic / std_ic

    positive_consistency = (daily_ics > 0).mean()
    negative_consistency = (daily_ics < 0).mean()

    sign_consistency = pd.concat([positive_consistency, negative_consistency], axis=1).max(axis=1)

    selected_features = []
    to_drop = []

    for feature in features:
        if (
            abs(mean_ic[feature]) >= ic_threshold
            and abs(ic_ir[feature]) >= ir_threshold
            and sign_consistency[feature] >= sign_threshold
        ):
            selected_features.append(feature)

        else:
            to_drop.append(feature)

    return selected_features, to_drop


def quantile_spread(df, features, target, threshold=0.05):

    selected_features = []
    to_drop = []

    target_std = df[target].std()

    for feature in features:
        temp_df = df[[feature, target]].dropna()

        if temp_df.empty or temp_df[feature].nunique() < 2:
            to_drop.append(feature)
            continue
        temp_df["Quantile"] = pd.qcut(temp_df[feature], 5, labels=False, duplicates="drop")

        means = temp_df.groupby("Quantile")[target].mean()

        if len(means) < 2 or not np.isfinite(target_std) or target_std <= 0:
            to_drop.append(feature)
            continue
        spread = abs(means.iloc[-1] - means.iloc[0])

        normalised_spread = spread / target_std

        if normalised_spread >= threshold:
            selected_features.append(feature)
        else:
            to_drop.append(feature)

    return selected_features, to_drop


def quantile_monotonicity(df, features, target, threshold=0.75):

    selected_features = []
    to_drop = []

    for feature in features:
        temp_df = df[[feature, target]].dropna()

        if temp_df.empty or temp_df[feature].nunique() < 2:
            to_drop.append(feature)
            continue
        temp_df["Quantile"] = pd.qcut(temp_df[feature], 5, labels=False, duplicates="drop")

        means = temp_df.groupby("Quantile")[target].mean()

        differences = means.diff().dropna()

        positive = (differences > 0).mean()
        negative = (differences < 0).mean()

        monotonicity = max(positive, negative)

        if monotonicity >= threshold:
            selected_features.append(feature)
        else:
            to_drop.append(feature)

    return selected_features, to_drop


def time_stability(df, features, target, threshold=0.55):

    selected_features = []
    to_drop = []

    df = df.copy()
    df["Year"] = pd.to_datetime(df["Date"]).dt.year

    for feature in features:
        correlations = []

        for year, group in df.groupby("Year"):
            temp_df = group[[feature, target]].dropna()

            if len(temp_df) < 3 or temp_df[feature].nunique() < 2 or temp_df[target].nunique() < 2:
                continue

            correlation = spearmanr(temp_df[feature], temp_df[target]).statistic

            if np.isfinite(correlation):
                correlations.append(correlation)

        if len(correlations) == 0:
            to_drop.append(feature)
            continue

        correlations = np.array(correlations)

        positive = np.mean(correlations > 0)
        negative = np.mean(correlations < 0)

        stability = max(positive, negative)

        if stability >= threshold:
            selected_features.append(feature)
        else:
            to_drop.append(feature)

    return selected_features, to_drop


def feature_target_coverage(df, features, target, threshold=0.60):

    selected_features = []
    to_drop = []

    num_of_rows = len(df)

    for feature in features:
        temp_df = df[[feature, target]].dropna()

        coverage = len(temp_df) / num_of_rows

        if coverage >= threshold:
            selected_features.append(feature)
        else:
            to_drop.append(feature)

    return selected_features, to_drop
