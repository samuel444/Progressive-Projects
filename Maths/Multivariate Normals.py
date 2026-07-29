import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Take two stocks have average daily returns:
# Stock A = 0.05%
# Stock B = 0.03%

mean = np.array([
    0.0005,
    0.0003
])



# Covariance matrix

# Take:
# Stock A daily volatility = 2%
# Stock B daily volatility = 1.5%
# Correlation between the stocks = 0.7

vol_a = 0.02
vol_b = 0.015
correlation = 0.7

# Cov(X, Y) = correlation * std(X) * std(Y)
cov_ab = correlation * vol_a * vol_b

covariance_matrix = np.array([
    [vol_a ** 2, cov_ab],
    [cov_ab, vol_b ** 2]
])

print("\nCovariance Matrix:")
print(covariance_matrix)


# Generate joint stock returns
# Generate 1000 days of returns from a
# multivariate normal distribution

returns = np.random.multivariate_normal(
    mean=mean,
    cov=covariance_matrix,
    size=1000
)

df = pd.DataFrame(
    returns,
    columns=["Stock A", "Stock B"]
)

print("\nGenerated Returns:")
print(df.head())


# Plot the two stock returns
plt.scatter(
    df["Stock A"],
    df["Stock B"],
    alpha=0.5
)

plt.xlabel("Stock A Return")
plt.ylabel("Stock B Return")
plt.title("Correlated Stock Returns")
plt.show()


# Compare requested and observed means
print("\nSample Means:")
print(df.mean())


# Compare requested and observed covariance
print("\nSample Covariance Matrix:")
print(df.cov())


# Sample correlation
print("\nSample Correlation Matrix:")
print(df.corr())



# Portfolio example

# 60% Stock A, 40% Stock B

weights = np.array([
    0.6,
    0.4
])

# Expected portfolio return
portfolio_mean = weights @ mean

# Portfolio variance
portfolio_variance = (
    weights.T
    @ covariance_matrix
    @ weights
)

portfolio_volatility = np.sqrt(
    portfolio_variance
)

print("\nExpected Daily Portfolio Return:")
print(portfolio_mean)

print("\nDaily Portfolio Variance:")
print(portfolio_variance)

print("\nDaily Portfolio Volatility:")
print(portfolio_volatility)