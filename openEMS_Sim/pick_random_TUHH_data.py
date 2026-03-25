import pandas as pd

# Using "../" to go up one directory level from openEMS_Sim to the root, then into data/
csv_path = "../data/raw/Universal-Diff-SI-Array/parameter.csv"
df = pd.read_csv(csv_path)

# Pick 3 random rows
sample_df = df.sample(3, random_state=42) 

print("Here are our 3 Ground Truth Geometries:")
print(sample_df.to_string())