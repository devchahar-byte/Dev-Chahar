import pandas as pd

df = pd.read_csv("data.csv")   # change filename if needed

# Select numerical features
num_df = df[["math score", "reading score", "writing score"]]

# Correlation
print("Correlation Matrix:\n")
print(num_df.corr())

# Covariance
print("\nCovariance Matrix:\n")
print(num_df.cov())
