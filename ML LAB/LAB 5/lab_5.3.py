import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *
import numpy as np

# Load dataset
X, y = load_breast_cancer(return_X_y=True)

# Train test split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Train models
nb = GaussianNB().fit(Xtr, ytr)
dt = DecisionTreeClassifier(random_state=42).fit(Xtr, ytr)

# Accuracy
nb_train, nb_test = accuracy_score(ytr, nb.predict(Xtr)), accuracy_score(yte, nb.predict(Xte))
dt_train, dt_test = accuracy_score(ytr, dt.predict(Xtr)), accuracy_score(yte, dt.predict(Xte))

# Bar chart (train vs test accuracy)
labels = ["Naive Bayes","Decision Tree"]
train = [nb_train, dt_train]
test = [nb_test, dt_test]

x = np.arange(len(labels))
plt.bar(x-0.2, train, 0.4, label="Train")
plt.bar(x+0.2, test, 0.4, label="Test")
plt.xticks(x, labels)
plt.ylabel("Accuracy")
plt.title("Training vs Testing Accuracy")
plt.legend()
plt.show()

# ROC Curve
nb_prob = nb.predict_proba(Xte)[:,1]
dt_prob = dt.predict_proba(Xte)[:,1]

nb_fpr, nb_tpr, _ = roc_curve(yte, nb_prob)
dt_fpr, dt_tpr, _ = roc_curve(yte, dt_prob)

plt.plot(nb_fpr, nb_tpr, label="Naive Bayes")
plt.plot(dt_fpr, dt_tpr, label="Decision Tree")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# Confusion Matrix Heatmaps
nb_cm = confusion_matrix(yte, nb.predict(Xte))
dt_cm = confusion_matrix(yte, dt.predict(Xte))

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.heatmap(nb_cm, annot=True, fmt="d", cmap="Blues")
plt.title("Naive Bayes")

plt.subplot(1,2,2)
sns.heatmap(dt_cm, annot=True, fmt="d", cmap="Purples")
plt.title("Decision Tree")

plt.tight_layout()
plt.show()

# Reports
print("Naive Bayes Test Accuracy:", nb_test)
print("Decision Tree Test Accuracy:", dt_test)

print("\nNaive Bayes Classification Report\n")
print(classification_report(yte, nb.predict(Xte)))

print("\nDecision Tree Classification Report\n")
print(classification_report(yte, dt.predict(Xte)))