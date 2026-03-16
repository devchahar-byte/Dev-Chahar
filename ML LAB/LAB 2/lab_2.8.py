import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("diabetes_data_upload.csv")

plt.scatter(df['Age'], df['Obesity'])
plt.xlabel("Age")
plt.ylabel("Obesity")
plt.title("Age vs Obesity")
plt.show()
