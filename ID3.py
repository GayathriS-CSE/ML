#DECISION TREE ID3
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_dataset(X, y, name):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # ID3 uses entropy
    model = DecisionTreeClassifier(criterion='entropy')

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nDataset:", name)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("Recall:", recall_score(y_test, y_pred, average='weighted'))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# -------------------------
# Dataset 1: Breast Cancer
# -------------------------

cancer = load_breast_cancer()
evaluate_dataset(cancer.data, cancer.target, "Breast Cancer Dataset")

# -------------------------
# Dataset 2: Wine
# -------------------------

wine = load_wine()
evaluate_dataset(wine.data, wine.target, "Wine Dataset")
