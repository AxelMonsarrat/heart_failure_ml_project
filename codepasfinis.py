import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("heart_failure_clinical_records.csv")
print(df.info())

features = ['age', 'ejection_fraction', 'serum_creatinine']
X = df[features]
Y = df['DEATH_EVENT']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

S = StandardScaler()
X_train_scaled = S.fit_transform(X_train)
X_test_scaled = S.transform(X_test)

mymodel = KNeighborsClassifier(n_neighbors=5)
knn = mymodel.fit(X_train_scaled, Y_train)

Y_pred = knn.predict(X_test_scaled)
print("Predicted values:", Y_pred)

acc = accuracy_score(Y_test, Y_pred)
print("Accuracy:", acc)

cm = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix (raw numbers):")
print(cm)

plt.figure(figsize=(4,4))
plt.imshow(cm, cmap='Blues')
plt.colorbar()

classes = ['No Death', 'Death']
plt.xticks([0, 1], classes)
plt.yticks([0, 1], classes)

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='black')

plt.title("Confusion Matrix - KNN")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

plt.scatter(df['age'], df['ejection_fraction'], c=df['DEATH_EVENT'], cmap='viridis')
plt.xlabel("Age")
plt.ylabel("Ejection Fraction")
plt.title("Age vs Ejection Fraction (colored by DEATH_EVENT)")
plt.show()

print("KNN model done.")
