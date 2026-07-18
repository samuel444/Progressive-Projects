import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

results = []

for i in range(100000):
    roll_results = []
    for j in range(5):
        roll_results.append(random.randrange(1,7))
    results.append(np.mean(roll_results))

results = pd.Series(results)

plt.hist(results, bins=12)
plt.title("Average of 5 Dice Rolls")
plt.xlabel("5 Roll Average")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


