import sqlite3
import pandas as pd

connection = sqlite3.connect("/Users/sam/Progressive-Projects/SQL/Python SQL/trades.db")

query = """
SELECT *
FROM SharePrices
WHERE trade_time >= '2026-03-01 09:40:00'
"""

filtered_df = pd.read_sql(query, connection)

print(filtered_df)


import matplotlib.pyplot as plt

query = """
SELECT
    trade_time,
    share_price_change,
    RANK() OVER (
        ORDER BY trade_time ASC
    ) AS trade_rank,
    SUM(share_price_change) OVER (
        ORDER BY trade_time
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS rolling_sum
FROM SharePrices;
"""

df2 = pd.read_sql(query, connection)

plt.plot(df2['trade_rank'], df2['rolling_sum'])
plt.xlabel('Trade Rank')
plt.ylabel('Rolling Sum of Share Price Change')
plt.title('Rolling Sum of Share Price Change Over Time')
plt.show()

DescriptiveStats = df2.describe()
print(DescriptiveStats)

new_df = pd.DataFrame({
    "trade_time": [
        "2027-03-01 09:30:00",
        "2027-03-01 10:15:00",
        "2027-03-01 13:45:00",
        "2027-03-02 09:40:00",
        "2027-03-02 11:20:00",
        "2027-03-02 15:10:00",
        "2027-03-03 10:05:00",
        "2027-03-03 14:30:00"
    ],
    "share_price_change": [
        0.7,
        -0.4,
        1.1,
        0.2,
        -0.9,
        1.4,
        -0.3,
        0.8
    ]
})

concat_df = pd.concat([filtered_df, new_df], ignore_index=True)

concat_df.to_sql(
    "ConcatenatedSharePrices",
    connection,
    if_exists="replace",
    index=False
)