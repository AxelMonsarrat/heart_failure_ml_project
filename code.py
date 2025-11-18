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
    print("AUC :", roc_auc)

    # Save results
    results[name] = {
        "model": model,
        "ypred": ypred,
        "yprob": yprob,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc
    }

# ============================
# COMPARATIVE PLOT: BAR CHART FOR METRICS
# ============================
names = list(results.keys())
accs = [results[n]["acc"] for n in names]
precs = [results[n]["prec"] for n in names]
recs = [results[n]["rec"] for n in names]
f1s = [results[n]["f1"] for n in names]

x = np.arange(len(names))
width = 0.2

plt.figure(figsize=(10,6))
plt.bar(x - 1.5*width, accs, width, label='Accuracy')
plt.bar(x - 0.5*width, precs, width, label='Precision')
plt.bar(x + 0.5*width, recs, width, label='Recall')
plt.bar(x + 1.5*width, f1s, width, label='F1-score')
plt.xticks(x, names)
plt.ylabel("Score")
plt.title("Comparison of models - classification metrics")
plt.legend()
plt.ylim(0,1)
plt.show()

# ============================
# PLOT: ROC CURVES (all models)
# ============================
plt.figure(figsize=(8,6))
for name in names:
    plt.plot(results[name]["fpr"], results[name]["tpr"], label=f"{name} (AUC = {results[name]['auc']:.3f})")

plt.plot([0,1],[0,1],'k--')  # diagonal
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()

# ============================
# PLOT: Confusion Matrix for each model
# ============================
for name in names:
    cm = confusion_matrix(Y_test, results[name]["ypred"])
    plt.figure(figsize=(4,3))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {name}")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["No event","Death"], rotation=45)
    plt.yticks(tick_marks, ["No event","Death"])
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

# ============================
# FEATURE IMPORTANCE for tree models
# ============================
features = X.columns

for name in ["RandomForest", "GradientBoosting"]:
    model = results[name]["model"]
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)
    plt.figure(figsize=(6,5))
    plt.barh(features[sorted_idx], importances[sorted_idx])
    plt.title(f"Feature importance - {name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

# ============================
# OPTIONAL: Scatter example colored by true label
# ============================
plt.figure(figsize=(8,5))
plt.scatter(df["age"], df["ejection_fraction"], c=df["DEATH_EVENT"], cmap='coolwarm')
plt.xlabel("Age")
plt.ylabel("Ejection fraction")
plt.title("Age vs Ejection fraction (colored by death event)")
plt.show()

# ============================
# SUMMARY PRINT
# ============================
print("\n===== SUMMARY =====")
for name in names:
    print(f"{name} -> Acc: {results[name]['acc']:.3f}, Prec: {results[name]['prec']:.3f}, Rec: {results[name]['rec']:.3f}, F1: {results[name]['f1']:.3f}, AUC: {results[name]['auc']:.3f}")
