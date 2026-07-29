import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.optimize import minimize


# Download stock data
df = yf.download(
    "AAPL",
    start="2021-01-01",
    end="2026-01-01",
    auto_adjust=True,
    progress=False
)

# Keep only closing prices
df = df[["Close"]]

# Calculate daily returns
df["Return"] = df["Close"].pct_change()

df = df.dropna()

returns = df["Return"].values.flatten()

print("First Returns:")
print(returns[:5])



# Estimate mu manually

# For normally distributed observations,
# the maximum likelihood estimate of mu
# is the sample mean.

mu_mle = np.mean(returns)

print("\nMLE and Numpy Mean:")
print(mu_mle)


# Estimate standard deviation
variance_mle = np.mean(
    (returns - mu_mle) ** 2
)

sigma_mle = np.sqrt(variance_mle)

print("\nMLE Standard Deviation:")
print(sigma_mle)


# NumPy can calculate the same result using ddof=0
sigma_numpy = np.std(
    returns,
    ddof=0
)

print("\nNumPy Standard Deviation:")
print(sigma_numpy)



# Calculate the likelihood of the entire dataset
densities = norm.pdf(
    returns,
    loc=mu_mle,
    scale=sigma_mle
)

# Likelihood is the product of every density
likelihood = np.prod(densities)

print("\nLikelihood:")
print(likelihood)


# Use the log-likelihood
# Multiplying many small densities can cause
# numerical underflow.
#
# Taking logs changes the product into a sum.

log_likelihood = np.sum(
    norm.logpdf(
        returns,
        loc=mu_mle,
        scale=sigma_mle
    )
)

print("\nLog-Likelihood:")
print(log_likelihood)


# Graph with every likelihood for each mu

mu_values = np.linspace(
    mu_mle - 0.005,
    mu_mle + 0.005,
    300
)

log_likelihood_mu_values = []

for candidate_mu in mu_values:

    candidate_log_likelihood = np.sum(
        norm.logpdf(
            returns,
            loc=candidate_mu,
            scale=sigma_mle
        )
    )

    log_likelihood_mu_values.append(
        candidate_log_likelihood
    )

log_likelihood_mu_values = np.array(
    log_likelihood_mu_values
)


# Find the candidate with the largest log-likelihood
best_index = np.argmax(
    log_likelihood_mu_values
)

best_mu_from_grid = mu_values[
    best_index
]

print("\nBest Mean From Grid Search:")
print(best_mu_from_grid)

print("\nSample Mean:")
print(mu_mle)


# Plot the log-likelihood for different means
plt.plot(
    mu_values,
    log_likelihood_mu_values
)

plt.axvline(
    mu_mle,
    linestyle="--",
    label="MLE Mean"
)

plt.xlabel("Candidate Mean")
plt.ylabel("Log-Likelihood")
plt.title("Log-Likelihood for Different Mean Values")
plt.legend()
plt.show()


# Graph with every likelihood for each sigma

sigma_values = np.linspace(
    sigma_mle * 0.25,
    sigma_mle * 2.5,
    300
)

log_likelihood_sigma_values = []

for candidate_sigma in sigma_values:

    candidate_log_likelihood = np.sum(
        norm.logpdf(
            returns,
            loc=mu_mle,
            scale=candidate_sigma
        )
    )

    log_likelihood_sigma_values.append(
        candidate_log_likelihood
    )

log_likelihood_sigma_values = np.array(
    log_likelihood_sigma_values
)


# Find the candidate with the largest log-likelihood
best_index = np.argmax(
    log_likelihood_sigma_values
)

best_sigma_from_grid = sigma_values[
    best_index
]

print("\nBest Std From Grid Search:")
print(best_sigma_from_grid)

print("\nNumpy Std:")
print(sigma_numpy)


# Plot the log-likelihood for different means
plt.plot(
    sigma_values,
    log_likelihood_sigma_values
)

plt.axvline(
    sigma_mle,
    linestyle="--",
    label="MLE Std"
)

plt.xlabel("Candidate Sigma")
plt.ylabel("Log-Likelihood")
plt.title("Log-Likelihood for Different Sigma Values")
plt.legend()
plt.show()


# Plot the fitted normal distribution
x_values = np.linspace(
    returns.min(),
    returns.max(),
    500
)

fitted_density = norm.pdf(
    x_values,
    loc=best_mu_from_grid,
    scale=best_sigma_from_grid
)

plt.hist(
    returns,
    bins=50,
    density=True,
    alpha=0.6,
    label="Observed Returns"
)

plt.plot(
    x_values,
    fitted_density,
    label="Fitted Normal Distribution"
)

plt.xlabel("Daily Return")
plt.ylabel("Density")
plt.title("Maximum Likelihood Normal Fit")
plt.legend()
plt.show()