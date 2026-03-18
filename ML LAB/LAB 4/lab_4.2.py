import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# a) Importing data
# Note: Ensure you have downloaded data.csv from Kaggle for this.
df = pd.read_csv('data.csv') # Replace with your local path
print(df.info())

# b) Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df[['Volume', 'Weight', 'CO2']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# c) Outliers using Box Plot
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
sns.boxplot(df['Volume'], ax=axs[0]).set_title('Volume Outliers')
sns.boxplot(df['Weight'], ax=axs[1]).set_title('Weight Outliers')
sns.boxplot(df['CO2'], ax=axs[2]).set_title('CO2 Outliers')
plt.show()

# d) Relationship Visualization
sns.pairplot(df[['Volume', 'Weight', 'CO2']])
plt.show()

# e) Train Model (80:20)
X = df[['Volume', 'Weight']]
y = df['CO2']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# f) Weight, Intercept and Line Chart
print(f"Weights: {model.coef_}")
print(f"Intercept: {model.intercept_}")

y_pred = model.predict(X_test)
plt.plot(range(len(y_test)), y_test.values, label='True')
plt.plot(range(len(y_pred)), y_pred, label='Predicted')
plt.legend()
plt.title('True vs Predicted Outcomes')
plt.show()

# g) Metrics
print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")