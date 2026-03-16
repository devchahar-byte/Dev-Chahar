import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import *

# Load dataset
X, y = load_breast_cancer(return_X_y=True)

# Split dataset
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Train Decision Tree
model = DecisionTreeClassifier(random_state=42).fit(Xtr, ytr)

# Predictions
yp = model.predict(Xte)
prob = model.predict_proba(Xte)[:,1]

# Accuracy
print("Accuracy:", accuracy_score(yte, yp))

# Confusion Matrix
cm = confusion_matrix(yte, yp)
ConfusionMatrixDisplay(cm).plot(cmap="Purples")
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

# Decision Tree Visualization
plt.figure(figsize=(18,8))
plot_tree(model, filled=True)
plt.title("Decision Tree Visualization")
plt.show()