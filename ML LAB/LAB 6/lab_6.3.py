import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

# Re-run both for comparison
knn = KNeighborsClassifier().fit(X_train, y_train)
dtree = DecisionTreeClassifier().fit(X_train, y_train)

# Accuracy scores [cite: 27]
models = ['KNN', 'Decision Tree']
train_acc = [knn.score(X_train, y_train), dtree.score(X_train, y_train)]
test_acc = [knn.score(X_test, y_test), dtree.score(X_test, y_test)]

# Bar chart [cite: 27]
plt.figure(figsize=(8, 5))
x = range(len(models))
plt.bar(x, train_acc, width=0.4, label='Train Accuracy', align='center')
plt.bar(x, test_acc, width=0.4, label='Test Accuracy', align='edge')
plt.xticks(x, models)
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Comparison')
plt.savefig('accuracy_comparison.png')

# Side-by-Side Confusion Matrix Heatmap [cite: 28]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test, knn.predict(X_test)), annot=True, ax=ax1, fmt='d', cmap='Blues')
ax1.set_title('KNN Confusion Matrix')
sns.heatmap(confusion_matrix(y_test, dtree.predict(X_test)), annot=True, ax=ax2, fmt='d', cmap='Greens')
ax2.set_title('Decision Tree Confusion Matrix')
plt.savefig('confusion_matrix_comparison.png')
print("Comparison plots saved.")