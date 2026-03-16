import pandas as pd

df = pd.read_csv("diabetes_data_upload.csv")

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Gender'] = df['Gender'].fillna("Unknown")

print(df)
