import pandas as pd

from matplotlib import pyplot as plt

df = pd.DataFrame({  
    "Day": [1, 2, 3, 4, 5, 6, 7, 8],
    "Stock_A_Return": [0.8, -0.4, 1.2, -0.7, 0.5, 1.0, -0.2, 0.6],
    "Stock_B_Return": [0.3, 0.2, -0.5, 0.9, -0.1, 0.4, 0.7, -0.3],
    "Market_Return": [0.4, -0.1, 0.7, -0.3, 0.2, 0.5, 0.1, 0.3],
    "Risk": [2, 3, 5, 6, 4, 7, 3, 5]
})

print(df)

# -- LINE PLOT --
plt.plot(df["Day"], df["Stock_A_Return"])

plt.title("Stock A Daily Returns")
plt.xlabel("Day")
plt.ylabel("Return (%)")
plt.grid(True)

plt.show()

# -- MULTIPLE LINE PLOT --
plt.plot(df["Day"], df["Stock_A_Return"], label="Stock A")
plt.plot(df["Day"], df["Stock_B_Return"], label="Stock B")
plt.plot(df["Day"], df["Market_Return"], label="Market")

plt.title("Daily Returns Comparison")
plt.xlabel("Day")
plt.ylabel("Return (%)")
plt.legend()
plt.grid(True)

plt.show()

# -- HISTOGRAM --
plt.hist(df["Stock_A_Return"], bins=5)

plt.title("Distribution of Stock A Returns")
plt.xlabel("Return (%)")
plt.ylabel("Frequency")
plt.grid(True)

plt.show()

# -- SCATTER PLOT --
plt.scatter(df["Risk"], df["Stock_A_Return"])

plt.title("Risk vs Stock A Return")
plt.xlabel("Risk")
plt.ylabel("Return (%)")
plt.grid(True)

plt.show()

