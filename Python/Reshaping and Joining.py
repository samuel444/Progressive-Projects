import pandas as pd
import numpy as np

# Create the DataFrame
df = pd.DataFrame({
    "Letter": ["A", "A", "B", "B"],
    "Number": [1, 2, 1, 2],
    "Value": [10, 20, 30, 40],
    "Group": ["X", "Y", "X", "Y"]
})


print("ORIGINAL DATAFRAME")
print(df)

# pivot_table()
pivot = pd.pivot_table(
    df,
    values="Value",
    index="Letter",
    columns="Number"
)

print("\nPIVOT TABLE")
print(pivot)

# melt()
print("\nMELT")
print(pd.melt(df, id_vars=["Letter"]))

# stack()
print("\nSTACK")
print(pivot.stack())

# unstack()
print("\nUNSTACK")
print(df.set_index(["Letter", "Number"]).unstack())

# merge()
info = pd.DataFrame({
    "Letter": ["A", "B"],
    "Fruit": ["Apple", "Banana"]
})

print("\nMERGE")
print(pd.merge(df, info, on="Letter"))

# merge_ordered()
left = pd.DataFrame({
    "Date": [1, 3, 5],
    "A": [10, 20, 30]
})

right = pd.DataFrame({
    "Date": [2, 4, 6],
    "B": [100, 200, 300]
})

print("\nMERGE ORDERED")
print(pd.merge_ordered(left, right, on="Date"))

# concat()
df2 = pd.DataFrame({
    "Letter": ["C"],
    "Number": [1],
    "Value": [50],
    "Group": ["X"]
})

print("\nCONCAT")
print(pd.concat([df, df2]))

# compare()
old = pd.DataFrame({
    "A": [1, 2, 3]
})

new = pd.DataFrame({
    "A": [1, 5, 3]
})

print("\nCOMPARE")
print(old.compare(new))

# cut()
print("\nCUT")
print(pd.cut(
    df["Value"],
    bins=[0, 20, 40],
    labels=["Low", "High"]
))

# factorize()
print("\nFACTORIZE")
codes, uniques = pd.factorize(df["Letter"])
print("Codes:", codes)
print("Categories:", list(uniques))

# get_dummies()
print("\nGET DUMMIES")
print(pd.get_dummies(df["Group"]))

# explode()
tags = pd.DataFrame({
    "Letter": ["A", "B"],
    "Tags": [
        ["Red", "Blue"],
        ["Green", "Yellow"]
    ]
})

print("\nEXPLODE")
print(tags.explode("Tags"))

# crosstab()
print("\nCROSSTAB")
print(pd.crosstab(df["Letter"], df["Group"]))

# combine_first()
missing = pd.DataFrame({
    "A": [1, None, 3]
})

backup = pd.DataFrame({
    "A": [10, 20, 30]
})

print("\nCOMBINE FIRST")
print(missing.combine_first(backup))