import numpy as np


def cross_sectional_rank(df, columns, date_col="Date", suffix="Rank"):
    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        if date_col in df.columns:
            df[f"{column} {suffix}"] = df.groupby(date_col)[column].rank(pct=True)
        else:
            df[f"{column} {suffix}"] = df.groupby(level=0)[column].rank(pct=True)

    return df


def cross_sectional_z_score(df, columns, date_col="Date", suffix="Cross Sectional Z Score"):
    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        if date_col in df.columns:
            grouped = df.groupby(date_col)[column]
        else:
            grouped = df.groupby(level=0)[column]

        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan)
        df[f"{column} {suffix}"] = (df[column] - mean) / std

    return df


def sector_neutral_rank(df, columns, sector_col="Sector", date_col="Date"):
    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        if date_col in df.columns:
            df[f"{column} Sector Neutral Rank"] = df.groupby([date_col, sector_col])[column].rank(pct=True)
        else:
            temp = df.copy()
            temp["__Date"] = temp.index.get_level_values(0)
            df[f"{column} Sector Neutral Rank"] = temp.groupby(["__Date", sector_col])[column].rank(pct=True).values

    return df


def winsorize_cross_section(df, columns, lower=0.01, upper=0.99, date_col="Date"):
    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        if date_col in df.columns:
            grouped = df.groupby(date_col)[column]
        else:
            grouped = df.groupby(level=0)[column]

        lower_values = grouped.transform(lambda values: values.quantile(lower))
        upper_values = grouped.transform(lambda values: values.quantile(upper))
        df[f"{column} Winsorized"] = df[column].clip(lower=lower_values, upper=upper_values)

    return df


def all_cross_sectional_features(df, columns, date_col="Date", sector_col=None):
    df = cross_sectional_rank(df, columns, date_col=date_col)
    df = cross_sectional_z_score(df, columns, date_col=date_col)

    if sector_col is not None:
        df = sector_neutral_rank(df, columns, sector_col=sector_col, date_col=date_col)

    return df
