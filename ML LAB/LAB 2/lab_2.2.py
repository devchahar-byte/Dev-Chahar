import pandas as pd

df = pd.read_csv("diabetes_data_upload.csv")

print("Before cleaning:", df.shape)

df_clean = df.dropna()

print("After cleaning:", df_clean.shape)
