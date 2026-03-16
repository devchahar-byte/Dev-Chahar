import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")   # change filename if needed
scores = ["math score", "reading score", "writing score"]

# Histograms
for col in scores:
    plt.hist(df[col])
    plt.title(f"Histogram of {col}")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.show()

# Boxplots
plt.boxplot([df[col] for col in scores], labels=scores)
plt.title("Boxplot of Scores")
plt.ylabel("Score")
plt.show()
