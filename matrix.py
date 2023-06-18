import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the confusion matrix values
confusion_matrix_values = np.array([[84, 52], [14, 203]])
#91.4%
# Define the labels
labels = ["False", "True"]

# Plot the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix_values, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(np.arange(len(labels)) + 0.5, labels)
plt.yticks(np.arange(len(labels)) + 0.5, labels)
plt.title("Confusion Matrix")
plt.show()
