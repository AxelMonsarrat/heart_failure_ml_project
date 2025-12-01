import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

df = pd.read_csv("heart_failure_clinical_records.csv")
print(df.info())

# ----------------------------------------------------
# UTILISATION DE 8 PARAMÈTRES CLINIQUES
# ----------------------------------------------------
X = df[['age','anaemia','high_blood_pressure','diabetes',
        'ejection_fraction','serum_creatinine','serum_sodium','time']]
Y = df['DEATH_EVENT']

# Train/Test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y)

# Normalisation
S = StandardScaler()
X_train_scaled = S.fit_transform(X_train)
X_test_scaled = S.transform(X_test)

# ======================================================
# PARTIE 1 : Différents KNN
# ======================================================
k_values = [1, 3, 5, 7, 9]
results_knn = {}

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, Y_train)
    pred = model.predict(X_test_scaled)

    acc = accuracy_score(Y_test, pred)
    cm = confusion_matrix(Y_test, pred)
    results_knn[k] = (acc, cm, model)

    print(f"\n===== KNN k={k} =====")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", cm)

# ======================================================
# PARTIE 2 : Random Forest
# ======================================================
rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train, Y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(Y_test, rf_pred)
rf_cm = confusion_matrix(Y_test, rf_pred)

print("\n===== RANDOM FOREST =====")
print("Accuracy:", rf_acc)
print(rf_cm)

# ======================================================
# PARTIE 3 : Gradient Boosting (XGBoost simplifié)
# ======================================================
xgb = GradientBoostingClassifier(random_state=42)
xgb.fit(X_train, Y_train)
xgb_pred = xgb.predict(X_test)
xgb_acc = accuracy_score(Y_test, xgb_pred)
xgb_cm = confusion_matrix(Y_test, xgb_pred)

print("\n===== XG BOOSTING =====")
print("Accuracy:", xgb_acc)
print(xgb_cm)

# ======================================================
# PLOT 1 — ROC CURVES de KNN vs RF vs XGB
# ======================================================
plt.figure(figsize=(7,6))

# KNN best k
best_k = max(results_knn, key=lambda k: results_knn[k][0])
knn_best_model = results_knn[best_k][2]

knn_probs = knn_best_model.predict_proba(X_test_scaled)[:,1]
fpr, tpr, _ = roc_curve(Y_test, knn_probs)
plt.plot(fpr, tpr, label=f"KNN (k={best_k})")

# Random Forest
rf_probs = rf.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(Y_test, rf_probs)
plt.plot(fpr, tpr, label="Random Forest")

# XGBoost
xgb_probs = xgb.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(Y_test, xgb_probs)
plt.plot(fpr, tpr, label="XGBoost")

plt.plot([0,1],[0,1],'--')
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# ======================================================
# PLOT 2 — Feature Importance Random Forest
# ======================================================
plt.figure(figsize=(8,5))
plt.barh(X.columns, rf.feature_importances_)
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.show()

# ======================================================
# PLOT 3 — Feature Importance Gradient Boosting
# ======================================================
plt.figure(figsize=(8,5))
plt.barh(X.columns, xgb.feature_importances_, color='orange')
plt.title("Feature Importance - Gradient Boosting")
plt.xlabel("Importance")
plt.show()

# ======================================================
# PLOT 4 — Confusion Matrix RF (graphique)
# ======================================================
plt.figure(figsize=(5,4))
plt.imshow(rf_cm, cmap='Blues')
plt.title("Random Forest - Confusion Matrix")
plt.colorbar()
plt.xticks([0,1], ["No Death", "Death"])
plt.yticks([0,1], ["No Death", "Death"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, rf_cm[i,j], ha='center', va='center')
plt.show()

print("\nFIN DU SCRIPT.")