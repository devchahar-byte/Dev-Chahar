# Experiment 1: Descriptive Statistics

import pandas as pd
import numpy as np

# Load the CSV dataset
df = pd.read_csv("data.csv")   # change filename if needed

# Display first few records
print("First 5 records of the dataset:\n")
print(df.head())

# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(exclude=[np.number]).columns

print("\nNumerical Columns:")
print(numerical_cols)

print("\nCategorical Columns:")
print(categorical_cols)

# Measures of Central Tendency
print("\n--- Measures of Central Tendency ---")
for col in numerical_cols:
    print(f"\n{col}")
    print("Mean:", df[col].mean())
    print("Median:", df[col].median())
    print("Mode:", df[col].mode()[0])

# Measures of Dispersion
print("\n--- Measures of Dispersion ---")
for col in numerical_cols:
    print(f"\n{col}")
    print("Minimum:", df[col].min())
    print("Maximum:", df[col].max())
    print("Sum:", df[col].sum())
    print("Variance:", df[col].var())
    print("Standard Deviation:", df[col].std())
