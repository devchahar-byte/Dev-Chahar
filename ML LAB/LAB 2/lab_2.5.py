import pandas as pd

df = pd.read_csv("diabetes_data_upload.csv")

df.rename(columns={
    'sudden weight loss': 'Sudden_Weight_Loss',
    'visual blurring': 'Visual_Blurring',
    'delayed healing': 'Delayed_Healing',
    'partial paresis': 'Partial_Paresis',
    'muscle stiffness': 'Muscle_Stiffness'
}, inplace=True)

print(df.head())
