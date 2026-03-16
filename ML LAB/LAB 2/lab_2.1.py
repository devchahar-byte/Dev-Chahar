import pandas as pd

df = pd.read_csv("diabetes_data_upload.csv")

print(df.isnull())
print(df.notnull())

print("Missing values per column:")
print(df.isnull().sum())
