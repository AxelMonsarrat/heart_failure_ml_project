# ============================
# IMPORTS (style cours)
# ============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ============================
# LOAD DATA
# ============================
df = pd.read_csv('heart_failure_clinical_records.csv')
print(df.info())
print(df.head())

# ============================
# FEATURES / LABEL
# ============================
X = df.drop("DEATH_EVENT", axis=1)
Y = df["DEATH_EVENT"]

# ============================
# TRAIN / TEST SPLIT
# ============================
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# ============================
# SCALING (for KNN)
# ============================
s = StandardScaler()
Xs_train = s.fit_transform(X_train)
Xs_test = s.transform(X_test)

# ============================
# MODELS (KNN, GradientBoosting, RandomForest)
# ============================
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=150, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=150, random_state=42)
}

# We'll store results
results = {}

# ============================
# TRAIN + EVALUATE
# ============================
for name, model in models.items():
    print("\n====================================")
    print(f"Training {name}")
    print("====================================")

    # Use scaled data for KNN, raw for tree-based models
    if name == "KNN":
        model.fit(Xs_train, Y_train)
        ypred = model.predict(Xs_test)
        if hasattr(model, "predict_proba"):
            yprob = model.predict_proba(Xs_test)[:,1]
        else:
            yprob = model.predict(Xs_test)  # fallback
    else:
        model.fit(X_train, Y_train)
        ypred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            yprob = model.predict_proba(X_test)[:,1]
        else:
            yprob = model.predict(X_test)  # fallback

    # Metrics
    acc = accuracy_score(Y_test, ypred)
    prec = precision_score(Y_test, ypred, zero_division=0)
    rec = recall_score(Y_test, ypred, zero_division=0)
    f1 = f1_score(Y_test, ypred, zero_division=0)

    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1-score :", f1)

    print("Confusion Matrix:")
    print(confusion_matrix(Y_test, ypred))

    # ROC
    fpr, tpr, _ = roc_curve(Y_test, yprob)
    roc_auc = auc(fpr, tpr)
    print("AUC :"