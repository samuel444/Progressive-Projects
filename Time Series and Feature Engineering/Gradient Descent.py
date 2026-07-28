import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Download Apple data
df = yf.download(
    "AAPL",
    start="2021-01-01",
    end="2026-01-01",
    auto_adjust=True,
    progress=False
)

df = df[["Close"]]

# Create returns
df["Return"] = df["Close"].pct_change()

# Yesterday predicts today
df["Lag Return"] = df["Return"].shift(1)

df = df.dropna()

# Features and target
X = df["Lag Return"].values
y = df["Return"].values

# Chronological split
split = int(len(df) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]


# Gradient Descent
beta_0 = 0.0
beta_1 = 0.0

learning_rate = 0.1
epochs = 1000

losses = []

n = len(X_train)

previous_loss = float("inf")
tolerance = 1e-10

for epoch in range(10000):

    predictions = beta_0 + beta_1 * X_train
    errors = predictions - y_train

    loss = np.mean(errors ** 2)
    losses.append(loss)

    if abs(previous_loss - loss) < tolerance:
        print("Converged at epoch:", epoch)
        break

    previous_loss = loss

    gradient_beta_0 = (2 / len(X_train)) * np.sum(errors)
    gradient_beta_1 = (2 / len(X_train)) * np.sum(errors * X_train)

    beta_0 -= learning_rate * gradient_beta_0
    beta_1 -= learning_rate * gradient_beta_1

print("Intercept:", beta_0)
print("Coefficient:", beta_1)

# Test RMSE
predictions = beta_0 + beta_1 * X_test

rmse = np.sqrt(np.mean((predictions - y_test) ** 2))

print("Test RMSE:", rmse)

# Plot loss
plt.plot(losses)

plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Gradient Descent")

plt.show()