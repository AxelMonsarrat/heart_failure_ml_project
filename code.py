# ============================
# IMPORTS (same style as class)
# ============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix


# ============================
# LOADING DATA
# ============================

df = pd.read_csv('heart_failure_clinical_records.csv')
print(df.info())


# ============================
# FEATURES / LABEL
# ============================

X = df.drop("DEATH_EVENT", axis=1)
Y = df["DEATH_EVENT"]

# ============================
# TRAIN / TEST SPLIT
# ============================

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# ============================
# SCALING
# ============================

s = StandardScaler()
Xs_train = s.fit_transform(X_train)
Xs_test = s.transform(X_test)

# ============================
# MODELS
# ============================

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "RandomForest": RandomForestClassifier(n_estimators=150, random_state=42)
}

# ============================
# TRAIN + EVALUATION
# ============================

for name, model in models.items():
    print("\n===============================")
    print(f"Training {name}")
    print("===============================")

    # scaled for LR + KNN
    if name in ["LogisticRegression", "KNN"]:
        model.fit(Xs_train, Y_train)
        ypred = model.predict(Xs_test)
    else:
        # RF works without scaling
        model.fit(X_train, Y_train)
        ypred = model.predict(X_test)

    # Metrics
    print("Accuracy :", accuracy_score(Y_test, ypred))
    print("Precision:", precision_score(Y_test, ypred))
    print("Recall   :", recall_score(Y_test, ypred))
    print("F1-score :", f1_score(Y_test, ypred))

    print("Confusion Matrix:")
    print(confusion_matrix(Y_test, ypred))


# ============================
# FEATURE IMPORTANCE (RF)
# ============================

rf = models["RandomForest"]

importances = rf.feature_importances_
features = X.columns

plt.figure(figsize=(8,5))
plt.barh(features, importances)
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.show()


# ============================
# PLOT: AGE vs CREATININE colored by DEATH
# ============================

plt.scatter(df["age"], df["serum_creatinine"], c=df["DEATH_EVENT"], cmap="coolwarm")
plt.xlabel("Age")
plt.ylabel("Serum Creatinine")
plt.title("Age vs Creatinine (colored by death event)")
plt.show()
