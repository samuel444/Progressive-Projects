import pandas as pd
import numpy as np

# Create the DataFrame
df = pd.DataFrame({
    "Name": ["Sam", "Alice", "Bob", "Alice", "Tom"],
    "Age": [25, np.nan, "30", np.nan, 200],
    "Gender": ["Male", " female ", "MALE", " female ", "Male"],
    "Salary": [30000, 40000, '35000', 40000, 500000]
})

print("\nORIGINAL DATA")
print(df)

# MISSING VALUES
print("\n--- Missing Values ---")
print(df.isnull())

print("\nCount of Missing Values in Each Column:")
print(df.isnull().sum())

# Convert Age to numeric first
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

# Drop the rows with missing values
print(df.dropna())

# Fill missing values with median age
median_age = df["Age"].median()
df["Age"] = df["Age"].fillna(median_age)

print("\nAfter Filling Missing Values:")
print(df)

# DUPLICATE ROWS
print("\n--- Duplicate Rows ---")
print(df.duplicated())

# Drop duplicate rows
df = df.drop_duplicates()

print("\nAfter Removing Duplicates:")
print(df)

# DATA TYPES
print("\n--- Data Types ---")
print(df.dtypes)


# Ensure Salary is numeric
df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')

# Other method: Make salary type int64
df['Salary'] = df['Salary'].astype('int64')

# Check data types again
print("\nData Types After Conversion:")
print(df.dtypes)

# OUTLIERS
print("\n--- Outliers ---")

for col in df.select_dtypes(include="number"):    
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    print("\nColumn Outliers: "+col)
    print(outliers)


# Remove the row instead
print("\nAfter Removing Outliers:")
print(df[df["Age"] <= 120])

# Outlier Removal using drop
print("\nAfter Removing Outliers using drop:")
print(df.drop(index=4))

# Correct the value if you KNOW it's wrong
df.loc[df["Age"] == 200, "Age"] = 20

print("\nAfter Correcting the Age:")
print(df)

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

print("\nCleaned Dataset:")
print(df)
