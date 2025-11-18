import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score

# ============================
# LOAD DATASET
# ============================
df = pd.read_csv("heart_failure_clinical_records.csv")
print(df.info())
print(df.head())

# ============================
# FEATURES / TARGET
# ============================
X = df.drop("DEATH_EVENT", axis=1)
Y = df["DEATH_EVENT"]

# ============================
# TRAIN TEST SPLIT
# ============================
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# ============================
# SCALING (KNN NEEDS IT)
# ============================
s = StandardScaler()
Xs_train = s.fit_transform(X_train)
Xs_test = s.transform(X_test)

# ============================
# KNN MODEL
# ============================
model = KNeighborsClassifier(n_neighbors=5)
model.fit(Xs_train, Y_train)

ypred = model.predict(Xs_test)
yprob = model.predict_proba(Xs_test)[:, 1]   # for ROC curve

# ============================
# METRICS
# ============================
acc = accuracy_score(Y_test, ypred)
prec = precision_score(Y_test, ypred)
rec = recall_score(Y_test, ypred)
f1 = f1_score(Y_test, ypred)

print("\n===== KNN RESULTS =====")
print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1-score :", f1)

print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, ypred))

# ============================
# PLOT: CONFUSION MATRIX
# ============================
cm = confusion_matrix(Y_test, ypred)
plt.figure(figsize=(4,4))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix - KNN")
plt.colorbar()

labels = ["No Death", "Death"]
plt.xticks([0,1], labels)
plt.yticks([0,1], labels)

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j], ha='center', va='center', color='black')

plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# ============================
# PLOT: ROC CURVE
# ============================
fpr, tpr, _ = roc_curve(Y_test, yprob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"KNN (AUC = {roc_auc:.3f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - KNN")
plt.legend()
plt.show()

print("\nModel training done (KNN only). Further models will be implemented later.")
