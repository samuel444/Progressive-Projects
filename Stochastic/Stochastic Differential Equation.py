import numpy as np
import matplotlib.pyplot as plt

T = 1
n = 252
dt = T / n

drift = 0.1
diffusion = 0.2

X = np.zeros(n + 1)

for t in range(n):

    dW = np.sqrt(dt) * np.random.randn()

    X[t + 1] = (
        X[t]
        + drift * dt
        + diffusion * dW
    )

plt.plot(X)
plt.title("Stochastic Differential Equation")
plt.show()