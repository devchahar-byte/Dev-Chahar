import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# a) Importing data
url = "https://raw.githubusercontent.com/Yashappin/Machine-Learning/master/TvMarketing.csv"
df = pd.read_csv(url)
print("Data Structure:\n", df.head())
print("\nSummary Statistics:\n", df.describe())

# b) Visualising Data
plt.figure(figsize=(8, 5))
sns.scatterplot(x='TV', y='Sales', data=df)
plt.title('TV Budget vs Sales')
plt.show()

# c) Splitting Data (80:20)
X = df[['TV']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training shape: {X_train.shape}, Testing shape: {X_test.shape}")

# d) Train Model & Visualise Best Fit Line
lr = LinearRegression()
lr.fit(X_train, y_train)
print(f"Intercept (b0): {lr.intercept_}")
print(f"Coefficient (b1): {lr.coef_[0]}")

plt.scatter(X_train, y_train, color='blue', label='Actual Data')
plt.plot(X_train, lr.predict(X_train), color='red', label='Best Fit Line')
plt.legend()
plt.show()

# e) Display actual vs predicted
y_pred = lr.predict(X_test)
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nActual vs Predicted:\n", comparison.head())

# f) Computing RMSE and R2
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"\nRMSE: {rmse}")
print(f"R2 Value: {r2}")