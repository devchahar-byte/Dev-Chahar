import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import *

# Load data
X, y = load_breast_cancer(return_X_y=True)

# Split data
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Train model
model = GaussianNB().fit(Xtr, ytr)

# Predictions
yp = model.predict(Xte)
prob = model.predict_proba(Xte)[:,1]

# Accuracy
print("Accuracy:", accuracy_score(yte, yp))

# Confusion Matrix
cm = confusion_matrix(yte, yp)
ConfusionMatrixDisplay(cm).plot()
plt.show()

# Classification Report
print(classification_report(yte, yp))

# Precision-Recall Curve
p, r, _ = precision_recall_curve(yte, prob)
plt.plot(r, p)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()