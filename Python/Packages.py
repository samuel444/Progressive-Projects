import pandas as pd

import sys

sys.path.append("/Users/sam/Progressive-Projects/Packages")

# Above only works on my computer, so if your using it, you may want to change 
# the path to your own directory where you have the my_packages.py file, or:
# project_root = Path(__file__).resolve().parents[1]
# sys.path.append(str(project_root / "Packages"))

import my_packages as mp

df = pd.DataFrame({
    "Score": [62, 68, 71, 75, 80, 84, 91]
})

print(df)

# Calculate z-scores
df["Z-Score"] = mp.z_score(df["Score"])

print("\nWith Z-Scores:")
print(df)

