import numpy as np
import pandas as pd

# Create a Pandas Series
series = pd.Series([1, 2, 3, 4, 5])
print(series)

# Create a Pandas Series with custom index
indexed_series = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
print(indexed_series)

print(indexed_series['c']) # Accessing the value at index 'c'


# Create a Pandas DataFrame
dataFrame = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [5, 4, 3, 2, 1],
    'C': ['a', 'b', 'c', 'd', 'e']
}, index=['row1', 'row2', 'row3', 'row4', 'row5'])

print(dataFrame)

print(dataFrame.loc["row1"]) # Accessing the first row using loc

# Dataframe with temperature data
temperature_data = pd.DataFrame({
    'day1': [30, 32, 31, 29, 28],
    'day2': [31, 33, 30, 28, 27],
    'day3': [29, 31, 32, 30, 28]
}, index=['Los Angeles', 'Orlando', 'Detroit', 'Chicago', 'New York'])

print(temperature_data)

print(temperature_data[temperature_data['day1'] > 30])  # Filter rows where day1 temperature is greater than 30

print(temperature_data.describe())  # Summary statistics of the DataFrame


