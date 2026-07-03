def z_score(series):

    mean = series.mean()

    standard_deviation = series.std()

    z_scores = (
        series - mean
    ) / standard_deviation

    return z_scores