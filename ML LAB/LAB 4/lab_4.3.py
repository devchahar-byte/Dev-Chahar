import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve, auc

# a) Load Data
df = pd.read_csv('advertising.csv') # Replace with local path
print(df.info())

# b) EDA & Preprocessing
# Dropping non-numeric/high-cardinality columns for simplicity
df = df.drop(['Ad Topic Line', 'City', 'Country', 'Timestamp'], axis=1)
# Checking for nulls (this dataset is usually clean)
df.fillna(df.mean(), inplace=True) 

# c) Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True)
plt.show()

# d) Train Logistic Regression
X = df.drop('Clicked on Ad', axis=1)
y = df['Clicked on Ad']

# Scaling is crucial for Logistic Regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# e) K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_val_score(log_model, X_scaled, y, cv=kf)
print(f"K-Fold Results: {cv_results}")
print(f"Mean Accuracy: {cv_results.mean()}")

# f) Classification Report
y_pred = log_model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# g) Confusion Matrix & Curves
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
y_probs = log_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()

# h) Visualise Predicted vs Actual
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).reset_index(drop=True)
results_df.head(20).plot(kind='bar', figsize=(10,5))
plt.title('Actual vs Predicted (First 20 Samples)')
plt.show()