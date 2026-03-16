import pandas as pd

df = pd.read_csv("data.csv")   # change filename if needed

for col in ["math score", "reading score", "writing score"]:
    print(f"\n{col}")
    print("Q1:", df[col].quantile(0.25))
    print("Median (Q2):", df[col].quantile(0.50))
    print("Q3:", df[col].quantile(0.75))
