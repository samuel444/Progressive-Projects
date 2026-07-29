import numpy as np
import matplotlib.pyplot as plt

T = 1
n = 1000

dt = T / n

# Brownian increments
dW = np.sqrt(dt) * np.random.randn(n)

# Brownian path
W = np.concatenate([[0], np.cumsum(dW)])

time = np.linspace(0, T, n + 1)

plt.plot(time, W)
plt.xlabel("Time")
plt.ylabel("W(t)")
plt.title("Brownian Motion")
plt.show()



