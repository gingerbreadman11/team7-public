import pandas as pd
import os

path = '/Users/alexanderbensland/Desktop/DS_Lab/Code/team7/Testing_Alex/Analysis/results_feature_dedup/sample_feature_dedup.csv'
df = pd.read_csv(path)

# Calculate correlations
correlations = df.corr(numeric_only=True)['track.popularity'].sort_values(ascending=False)

print("CORRELATIONS:")
print(correlations)

