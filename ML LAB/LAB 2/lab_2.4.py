import pandas as pd

df = pd.read_csv("diabetes_data_upload.csv")

print("Before:")
print(df.dtypes)

df['Age'] = df['Age'].astype(int)

print("After:")
print(df.dtypes)
