import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("diabetes_data_upload.csv")

plt.hist(df['Age'], bins=5)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age Distribution")
plt.show()
