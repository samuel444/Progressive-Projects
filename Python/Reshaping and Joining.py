import pandas as pd
import numpy as np

# Create the DataFrame
df = pd.DataFrame({
    "Department": ["Sales", "Sales", "IT", "IT", "HR", "IT"],
    "Gender": ["Male", "Female", "Male", "Male", "Female", "Male"],
    "Salary": [30000, 35000, 40000, 45000, 28000,38000]
})



print("ORIGINAL DATAFRAME")
print(df)

# pivot_table()
pivot = pd.pivot_table(
    df,
    values="Salary",
    index="Department",
    columns="Gender"
)

print("\nPIVOT TABLE")
print(pivot)

pivot = pd.pivot_table(
    df,
    values="Salary",
    index="Department",
    columns="Gender",
    aggfunc="median",
    fill_value=0
)

print("\nPIVOT TABLE WITH MEDIAN AND FILL VALUE")
print(pivot)

# melt()
print("\nMELT")
print(pd.melt(df, id_vars="Department"))

print("\nMELT WITH SPECIFIC VALUE VARS")
print(pd.melt(df, id_vars="Department", value_vars=['Gender']))

print("\nMELT WITH NEW COLUMN NAMES")
print(pd.melt(df, id_vars="Department", value_vars=['Gender'], 
              var_name='New Name', 
              value_name='Score'))

# merge()
info = pd.DataFrame({
    "Department": ["Sales", "IT", "HR"],
    "Salary Variance": [5000, 10000, 2000]
})

print("\nMERGE")
print(pd.merge(df, info, on="Department"))

print("\nMERGE WITH RIGHT JOIN")
print(pd.merge(df, info, on="Department", how="right"))

info = pd.DataFrame({
    "Area": ["Sales", "IT", "HR"],
    "Salary Variance": [5000, 10000, 2000]
})

print("\nMERGE WITH DIFFERENT COLUMN NAMES")
print(pd.merge(df, info, left_on="Department", right_on="Area"))



# concat()
df2 = pd.DataFrame({
    "Department": ["Finance", "Marketing"],
    "Gender": ["Male","Female"],
    "Salary": [50000, 60000]
    })

print("\nCONCAT")
print(pd.concat([df, df2]))

print("\nCONCAT WITH IGNORE INDEX")
print(pd.concat([df, df2], ignore_index=True))


# cut()
print("\nCUT")
print(pd.cut(
    df["Salary"],
    bins=[0, 30000, 40000, 50000],
    labels=["Low", "Medium", "High"]
))


