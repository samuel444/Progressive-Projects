import pandas as pd
import polars as pl
import numpy as np
import time


rows = 5_000_000

np.random.seed(42)

data = {
    "Ticker": np.random.choice(
        ["AAPL", "MSFT", "NVDA", "AMZN"],
        size=rows
    ),

    "Return": np.random.normal(
        0,
        0.02,
        size=rows
    ),

    "Volume": np.random.randint(
        1_000,
        1_000_000,
        size=rows
    )
}


# PANDAS
pandas_df = pd.DataFrame(data)


start = time.perf_counter()


pandas_result = (
    pandas_df[
        pandas_df["Volume"] > 500_000
    ]
    .groupby("Ticker")["Return"]
    .mean()
)


pandas_time = (
    time.perf_counter() - start
)


print("Pandas:")
print(pandas_result)

print(
    f"Pandas time: {pandas_time:.4f} seconds"
)


# POLARS - going to do the same thing as we just did with Pandas
polars_df = pl.DataFrame(data)


start = time.perf_counter()


polars_result = (
    polars_df
    .filter(
        pl.col("Volume") > 500_000
    )
    .group_by("Ticker")
    .agg(
        pl.col("Return")
        .mean()
        .alias("Mean Return")
    )
)


polars_time = (
    time.perf_counter() - start
)


print("\nPolars:")
print(polars_result)

print(
    f"Polars time: {polars_time:.4f} seconds"
)