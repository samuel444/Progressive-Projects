import pandas as pd

from matplotlib import pyplot as plt

# ============================================================
# Very basic example DataFrame
# 
# Each row is one trading day.
# This is simple enough for beginners, but realistic enough for
# quant analyst and data science plotting examples.
#
# Columns:
# Day            = time/order variable
# Stock_A_Return = daily return for stock A
# Stock_B_Return = daily return for stock B
# Market_Return  = daily return for the overall market
# Risk           = made-up risk score for scatter plot practice
# ============================================================

df = pd.DataFrame({
    "Day": [1, 2, 3, 4, 5, 6, 7, 8],
    "Stock_A_Return": [0.8, -0.4, 1.2, -0.7, 0.5, 1.0, -0.2, 0.6],
    "Stock_B_Return": [0.3, 0.2, -0.5, 0.9, -0.1, 0.4, 0.7, -0.3],
    "Market_Return": [0.4, -0.1, 0.7, -0.3, 0.2, 0.5, 0.1, 0.3],
    "Risk": [2, 3, 5, 6, 4, 7, 3, 5]
})

print(df)

# ============================================================
# 1. LINE PLOT
#
# Type of data:
# - Time-based data
# - Ordered data
# - One value changing over time
#
# Used for:
# - Stock prices
# - Daily returns
# - Cumulative returns
# - Portfolio value over time
# - Model performance over training steps
# ============================================================

plt.figure(figsize=(8, 4))

plt.plot(df["Day"], df["Stock_A_Return"])

plt.title("Stock A Daily Returns")
plt.xlabel("Day")
plt.ylabel("Return (%)")
plt.grid(True)

plt.show()

# ============================================================
# 2. HISTOGRAM
#
# Type of data:
# - One numerical column
# - Distribution of values
#
# Used for:
# - Distribution of returns
# - Distribution of prediction errors
# - Distribution of trade profits/losses
# - Checking whether data is spread out or clustered
# ============================================================

plt.figure(figsize=(8, 4))

plt.hist(df["Stock_A_Return"], bins=5)

plt.title("Distribution of Stock A Returns")
plt.xlabel("Return (%)")
plt.ylabel("Frequency")
plt.grid(True)

plt.show()

# ============================================================
# 3. SCATTER PLOT
#
# Type of data:
# - Two numerical columns
# - Relationship between x and y
#
# Used for:
# - Risk vs return
# - Predicted values vs actual values
# - Correlation between two variables
# - Checking if one variable increases as another increases
# ============================================================

plt.figure(figsize=(8, 4))

plt.scatter(df["Risk"], df["Stock_A_Return"])

plt.title("Risk vs Stock A Return")
plt.xlabel("Risk")
plt.ylabel("Return (%)")
plt.grid(True)

plt.show()

# ============================================================
# 4. MULTIPLE LINE PLOT
#
# Type of data:
# - Time-based data
# - Multiple values changing over the same time period
#
# Used for:
# - Comparing two stocks
# - Comparing a stock against the market
# - Comparing different strategies
# - Comparing portfolio performance
# ============================================================

plt.figure(figsize=(8, 4))

plt.plot(df["Day"], df["Stock_A_Return"], label="Stock A")
plt.plot(df["Day"], df["Stock_B_Return"], label="Stock B")
plt.plot(df["Day"], df["Market_Return"], label="Market")

plt.title("Daily Returns Comparison")
plt.xlabel("Day")
plt.ylabel("Return (%)")
plt.legend()
plt.grid(True)

plt.show()

# ============================================================
# 5. HEATMAP
#
# Type of data:
# - A matrix/table of numerical relationships
# - Usually correlations between numerical columns
#
# Used for:
# - Correlation between stocks
# - Correlation between features
# - Finding variables that move together
# - Spotting duplicated or highly similar variables
#
# Note:
# A heatmap is usually easier with seaborn, but this is a pure
# Matplotlib version so beginners do not need another library yet.
# ============================================================

correlation_matrix = df[
    ["Stock_A_Return", "Stock_B_Return", "Market_Return", "Risk"]
].corr()

plt.figure(figsize=(6, 5))

plt.imshow(correlation_matrix)

plt.title("Correlation Heatmap")
plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=45
)
plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)

plt.colorbar(label="Correlation")

plt.show()