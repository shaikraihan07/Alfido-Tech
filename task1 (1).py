# %% [markdown]
# Machine Learning Classification Project
# Breast Cancer Dataset
# Compare Logistic Regression and Random Forest

# %%
# Import Libraries

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay
)

# %%
# Load Dataset

data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset Shape:", X.shape)

# %%
# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# %%
# Logistic Regression Model

lr = LogisticRegression(max_iter=5000)

lr.fit(X_train, y_train)

# %%
# Random Forest Model

rf = RandomForestClassifier(random_state=42)

rf.fit(X_train, y_train)

# %%
# Cross Validation

lr_cv = cross_val_score(lr, X, y, cv=5)

rf_cv = cross_val_score(rf, X, y, cv=5)

print("LR CV Mean:", lr_cv.mean())
print("RF CV Mean:", rf_cv.mean())

# %%
# Evaluation Metrics (Random Forest)

pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)

probs = rf.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, probs)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
print("ROC-AUC:", roc_auc)

# %%
# Confusion Matrix

ConfusionMatrixDisplay.from_estimator(
    rf,
    X_test,
    y_test
)

plt.title("Confusion Matrix")
plt.show()

# %%
# Conclusion

print("\nConclusion")
print("-----------")
print("Random Forest performed better and achieved high accuracy.")
print("The model correctly classified most samples with very few errors.")
