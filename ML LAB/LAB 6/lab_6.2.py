import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_curve, PrecisionRecallDisplay

# 1) Load and split [cite: 17]
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

# 2) Train a Decision tree classifier [cite: 18]
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)

# 3) Predict class labels [cite: 19]
y_pred = dtree.predict(X_test)

# 4) Evaluate [cite: 20]
print("--- Experiment 2: Decision Tree Results ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}") # [cite: 21]
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred)) # [cite: 22]
print("\nClassification Report:\n", classification_report(y_test, y_pred)) # [cite: 23]

# Precision-Recall Curve [cite: 24]
PrecisionRecallDisplay.from_estimator(dtree, X_test, y_test)
plt.title("Decision Tree Precision-Recall Curve")
plt.savefig('dtree_precision_recall.png')

# 5) Visualize the decision tree [cite: 25]
plt.figure(figsize=(15,10))
plot_tree(dtree, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.savefig('decision_tree_visualization.png')
print("Plots saved as dtree_precision_recall.png and decision_tree_visualization.png")