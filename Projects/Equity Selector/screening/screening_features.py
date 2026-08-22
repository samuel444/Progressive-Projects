import pandas as pd
import numpy as np

def missingness(df, max_missing=0.3):

    features = df.columns

    to_drop = []
    
    for feature in features:
        first = df[feature].first_valid_index()
        last = df[feature].last_valid_index()

        valid = df[feature].loc[first:last]

        missing = (valid.isnull().sum())/(len(valid))

        if missing >= max_missing:
            to_drop.append(feature)

    df = df.drop(
        columns=to_drop,
        errors="ignore"
    )

    return df, to_drop

def invalid_vals(df):

    features = df.columns

    to_drop = []
    
    for feature in features:
        if df[feature].isin([np.inf, -np.inf]).any():
            to_drop.append(feature)

    df = df.drop(
        columns=to_drop,
        errors="ignore"
    )

    return df, to_drop

def zero_variance(df, threshold=0.05):

    features = df.columns

    to_drop = []
    
    for feature in features:
        mean = df[feature].mean()
        standard_deviation = df[feature].std()
        ratio = abs(standard_deviation/mean)
        if ratio < threshold:
            to_drop.append(feature)

    df = df.drop(
        columns=to_drop,
        errors="ignore"
    )


    return df, to_drop


def duplicates(df):

    df = df.loc[:, ~df.T.duplicated()]
    df = df.drop_duplicates()

    return df


def correlations(df):

    features = df.columns

    corr = df[features].corr().abs()

    upper = corr.where(
        np.triu(np.ones(corr.shape), k=1).astype(bool)
    )

    to_drop = [
        column for column in upper.columns
        if (upper[column] > 0.95).any()
    ]

    df = df.drop(columns=to_drop)

    return df, to_drop

def distribution_checks(df):

    features = df.columns

    to_drop = []
    
    for feature in features:
        pass

    df = df.drop(
        columns=to_drop,
        errors="ignore"
    )

    return df, to_drop

def stability(df):

    features = df.columns

    to_drop = []
    
    for feature in features:
        pass

    df = df.drop(
        columns=to_drop,
        errors="ignore"
    )

    return df, to_drop

def outlier_rate(df):

    features = df.columns

    to_drop = []
    
    for feature in features:
        pass

    df = df.drop(
        columns=to_drop,
        errors="ignore"
    )

    return df, to_drop

def dispersion(df):

    features = df.columns

    to_drop = []
    
    for feature in features:
        pass

    df = df.drop(
        columns=to_drop,
        errors="ignore"
    )

    return df, to_drop
