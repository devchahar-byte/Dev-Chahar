import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("diabetes_data_upload.csv")

plt.boxplot(df['Age'])
plt.ylabel("Age")
plt.title("Age Boxplot")
plt.show()
