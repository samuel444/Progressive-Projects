import pandas as pd

import sys

sys.path.append("/Users/sam/Progressive-Projects/Packages")

import my_packages as mp

df = pd.DataFrame({
    "Score": [62, 68, 71, 75, 80, 84, 91]
})

print(df)

# ------------------------------------------------------------
# Calculate z-scores
# ------------------------------------------------------------

df["Z-Score"] = mp.z_score(df["Score"])

print("\nWith Z-Scores:")
print(df)

