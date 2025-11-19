import pandas as pd   
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler     
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix 

df = pd.read_csv("heart_failure_clinical_records.csv")
print(df.info())

X = df[['age', 'ejection_fraction', 'serum_creatinine']] 
Y = df['DEATH_EVENT'] 

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y) 

S = StandardScaler()
X_train_scaled = S.fit_transform(X_train)   # normaliser les données d'entraînement
X_test_scaled = S.transform(X_test)         # normaliser les données de test

mymodel = KNeighborsClassifier(n_neighbors=5)  
knn = mymodel.fit(X_train_scaled, Y_train)    

Y_pred = knn.predict(X_test_scaled)   # prédire les résultats sur les données de test
print("Predicted values:", Y_pred)  

acc = accuracy_score(Y_test, Y_pred)  # calculer l'accuracy du modèle
print("Accuracy:", acc)           

cm = confusion_matrix(Y_test, Y_pred) 
print("Confusion Matrix (raw numbers):")
print(cm)

plt.figure(figsize=(5,4))
df['DEATH_EVENT'].value_counts().plot(kind='bar')   # histogramme du nombre de décès
plt.title("Répartition des classes (DEATH_EVENT)")
plt.xlabel("Classe")
plt.ylabel("Fréquence")
plt.xticks([0,1], ['Survived', 'Death'], rotation=0)
plt.show()

plt.figure(figsize=(5,4))
df['sex'].value_counts().plot(kind='bar')   # répartition hommes / femmes
plt.title("Répartition des sexes")
plt.xlabel("Sexe (0 = femme, 1 = homme)")
plt.ylabel("Nombre")
plt.show()

plt.figure(figsize=(5,4))
plt.hist(df['age'], bins=20)   # distribution de l'âge
plt.title("Distribution de l'âge")
plt.xlabel("Âge")
plt.ylabel("Fréquence")
plt.show()

plt.figure(figsize=(5,4))
plt.hist(df['ejection_fraction'], bins=20)   # distribution de l'éjection fraction
plt.title("Distribution de l'ejection_fraction")
plt.xlabel("Ejection Fraction")
plt.ylabel("Fréquence")
plt.show()

plt.figure(figsize=(5,4))
plt.hist(df['serum_creatinine'], bins=20)   # distribution de la créatinine
plt.title("Distribution de serum_creatinine")
plt.xlabel("Serum Creatinine")
plt.ylabel("Fréquence")
plt.show()

plt.figure(figsize=(5,4))
plt.hist(df['serum_sodium'], bins=20)   # distribution du sodium
plt.title("Distribution de serum_sodium")
plt.xlabel("Serum Sodium")
plt.ylabel("Fréquence")
plt.show()

plt.figure(figsize=(5,4))
plt.hist(df['platelets'], bins=20)   # distribution des plaquettes
plt.title("Distribution de platelets")
plt.xlabel("Platelets")
plt.ylabel("Fréquence")
plt.show()

plt.figure(figsize=(5,4))
plt.hist(df['creatinine_phosphokinase'], bins=20)   # distribution du CPK
plt.title("Distribution de creatinine_phosphokinase")
plt.xlabel("Creatinine Phosphokinase")
plt.ylabel("Fréquence")
plt.show()

plt.figure(figsize=(4,4))
plt.imshow(cm, cmap='Greens')   # afficher la matrice de confusion en heatmap
plt.colorbar()

plt.xticks([0, 1], ['No Death', 'Death'])   # noms des classes en X
plt.yticks([0, 1], ['No Death', 'Death'])   # noms des classes en Y

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center')   # écrire les valeurs sur le graphique

plt.title("Matrice de Confusion - KNN")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.show()

plt.scatter(df['age'], df['ejection_fraction'], c=df['DEATH_EVENT'], cmap='viridis')   # scatter plot colorié par DEATH_EVENT
plt.xlabel("Âge")
plt.ylabel("Ejection Fraction")
plt.title("Âge vs Ejection Fraction (coloré par DEATH_EVENT)")
plt.show()

print("Modèle KNN terminé.") 