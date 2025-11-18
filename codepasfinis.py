import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
# SCALING (only for KNN)
# ============================
s = StandardScaler()
Xs_train = s.fit_transform(X_train)
Xs_test = s.transform(X_test)

# ============================
# MODELS (TO BE COMPLETED)
# ============================

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.metrics import confusion_matrix, roc_curve, auc

# models = {
#     "KNN": KNeighborsClassifier(n_neighbors=5),
#     "GradientBoosting": GradientBoostingClassifier(n_estimators=150, random_state=42),
#     "RandomForest": RandomForestClassifier(n_estimators=150, random_state=42)
# }

# results = {}

# ============================
# TRAINING LOOP (NOT FINISHED)
# ============================

# for name, model in models.items():
#     print("\n====================================")
#     print(f"Training {name}")
#     print("====================================")

#     if name == "KNN":
#         model.fit(Xs_train, Y_train)
#         ypred = model.predict(Xs_test)
#         yprob = model.predict_proba(Xs_test)[:,1]
#     else:
#         model.fit(X_train, Y_train)
#         ypred = model.predict(X_test)
#         yprob = model.predict_proba(X_test)[:,1]

#     acc = accuracy_score(Y_test, ypred)
#     prec = precision_score(Y_test, ypred, zero_division=0)
#     rec = recall_score(Y_test, ypred, zero_division=0)
#     f1 = f1_score(Y_test, ypred, zero_division=0)

#     print("Accuracy :", acc)
#     print("Precision:", prec)
#     print("Recall   :", rec)
#     print("F1-score :", f1)

#     print("Confusion Matrix:")
#     print(confusion_matrix(Y_test, ypred))

#     fpr, tpr, _ = roc_curve(Y_test, yprob)
#     roc_auc = auc(fpr, tpr)
#     print("AUC :", roc_auc)

#     results[name] = {
#         "model": model,
#         "ypred": ypred,
#         "yprob": yprob,
#         "acc": acc,
#         "prec": prec,
#         "rec": rec,
#         "f1": f1,
#         "fpr": fpr,
#         "tpr": tpr,
#         "auc": roc_auc
#     }

# ============================
# TODO: ADD COMPARISON PLOTS
# ============================

# # ROC CURVES
# plt.figure(figsize=(8,6))
# for name in results:
#     plt.plot(results[name]["fpr"], results[name]["tpr"], label=f"{name}")
# plt.legend()
# plt.title("ROC Curves (WIP)")
# plt.show()

# ============================
# TODO: ADD CONFUSION MATRICES
# ============================

# for name in results:
#     cm = confusion_matrix(Y_test, results[name]["ypred"])
#     plt.imshow(cm)
#     plt.title(f"Confusion Matrix - {name} (WIP)")
#     plt.show()

# ============================
# TODO: FEATURE IMPORTANCE
# ============================

# for name in ["RandomForest", "GradientBoosting"]:
#     importances = results[name]["model"].feature_importances_
#     plt.barh(X.columns, importances)
#     plt.title(f"Feature Importance - {name} (WIP)")
#     plt.show()

# ============================
# NOTES
# ============================

print("\n### MODEL TRAINING NOT FULLY IMPLEMENTED ###")
print("KNN, Gradient Boosting, Random Forest will be completed next session.")
print("Plots and evaluation metrics are still in progress.\n")
