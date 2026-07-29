import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.stattools import acf, pacf

from statsmodels.tsa.ar_model import AutoReg


# Download Apple stock data
df = yf.download(
    "AAPL",
    start="2021-01-01",
    end="2026-01-01",
    auto_adjust=True,
    progress=False
)

df = df[["Close"]]

# Calculate daily returns
df["Return"] = df["Close"].pct_change()

df = df.dropna()

returns = df["Return"]


# Plot the returns
plt.plot(
    returns.index,
    returns
)

plt.xlabel("Date")
plt.ylabel("Daily Return")
plt.title("Apple Daily Returns")
plt.show()


# Plot the ACF
plot_acf(
    returns,
    lags=20
)

plt.title("ACF of Apple Daily Returns")
plt.show()


# Plot the PACF
plot_pacf(
    returns,
    lags=20,
    method="ywm"
)

plt.title("PACF of Apple Daily Returns")
plt.show()


# Get the actual ACF values
acf_values = acf(
    returns,
    nlags=20
)

print("\nACF Values:")

for lag, value in enumerate(acf_values):

    print(
        f"Lag {lag}: {value:.4f}"
    )


# Get the actual PACF values
pacf_values = pacf(
    returns,
    nlags=20,
    method="ywm"
)

print("\nPACF Values:")

for lag, value in enumerate(pacf_values):

    print(
        f"Lag {lag}: {value:.4f}"
    )


# Approximate significance boundary
n = len(returns)

significance_boundary = (
    1.96 / np.sqrt(n)
)

print("\nApproximate 95% Boundary:")
print(significance_boundary)


# Identify significant PACF lags
significant_lags = []

for lag in range(
    1,
    len(pacf_values)
):

    if (
        abs(pacf_values[lag])
        > significance_boundary
    ):

        significant_lags.append(lag)


print("\nSignificant PACF Lags:")
print(significant_lags)


# Compare a few AR orders
for p in range(0, 6):

    model = AutoReg(
        returns,
        lags=p,
        trend="c"
    )

    result = model.fit()

    print(
        f"AR({p}) "
        f"AIC: {result.aic:.2f}, "
        f"BIC: {result.bic:.2f}"
    )