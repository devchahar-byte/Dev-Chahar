import pandas as pd

df = pd.read_csv("diabetes_data_upload.csv")

df.replace({
    'Yes': 1,
    'No': 0,
    'Male': 1,
    'Female': 0,
    'Positive': 1,
    'Negative': 0 
}, inplace=True)

print(df.head())
