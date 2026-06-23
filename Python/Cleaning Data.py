import pandas as pd
import numpy as np

# Create the DataFrame
df = pd.DataFrame({
    "Name": ["Sam", "Alice", "Bob", "Alice", "Tom"],
    "Age": [25, np.nan, "30", np.nan, 200],
    "Gender": ["Male", " female ", "MALE", " female ", "Male"],
    "Salary": [30000, 40000, 35000, 40000, 500000]
})

print("\nORIGINAL DATA")
print(df)

# MISSING VALUES
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Convert Age to numeric first
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

# Fill missing values with median age
median_age = df["Age"].median()
df["Age"] = df["Age"].fillna(median_age)

print("\nAfter Filling Missing Values:")
print(df)

# DUPLICATE ROWS
print("\n--- Duplicate Rows ---")
print(df.duplicated())

df = df.drop_duplicates()

print("\nAfter Removing Duplicates:")
print(df)

# DATA TYPES
print("\n--- Data Types ---")
print(df.dtypes)

# Ensure Age is numeric
df["Age"] = pd.to_numeric(df["Age"])

print("\nData Types After Conversion:")
print(df.dtypes)

# INCONSISTENT TEXT
print("\n--- Gender Values Before Cleaning ---")
print(df["Gender"].unique())

df["Gender"] = (
    df["Gender"]
    .str.strip()      # Remove spaces
    .str.lower()      # Convert to lowercase
)

print("\nGender Values After Cleaning:")
print(df["Gender"].unique())

# OUTLIERS
print("\n--- Summary Statistics ---")
print(df.describe())

# IQR Method
Q1 = df["Age"].quantile(0.25)
Q3 = df["Age"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[
    (df["Age"] < lower_bound) |
    (df["Age"] > upper_bound)
]

print("\nDetected Outliers:")
print(outliers)

# Remove outliers
df = df[
    (df["Age"] >= lower_bound) &
    (df["Age"] <= upper_bound)
]

print("\nAfter Removing Outliers:")
print(df)


print("\nCleaned Dataset")
print(df)