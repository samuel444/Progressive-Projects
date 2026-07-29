import numpy as np
import matplotlib.pyplot as plt

T = 1
n = 1000
dt = T / n

lower = -0.5
upper = 0.5

# Brownian motion
dW = np.sqrt(dt) * np.random.randn(n)
W = np.concatenate([[0], np.cumsum(dW)])

time = np.linspace(0, T, n + 1)

# Find first exit
exit_index = None

for i in range(len(W)):

    if W[i] <= lower or W[i] >= upper:
        exit_index = i
        break

# Plot
plt.plot(time, W)

plt.axhline(upper, linestyle="--")
plt.axhline(lower, linestyle="--")

if exit_index is not None:
    plt.scatter(
        time[exit_index],
        W[exit_index]
    )

    print(
        "First exit time:",
        time[exit_index]
    )

plt.xlabel("Time")
plt.ylabel("W(t)")
plt.title("First Exit Time")

plt.show()