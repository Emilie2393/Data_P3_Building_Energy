#!/usr/bin/env python
# coding: utf-8

# # Analyse Exploratoire

# ### Import des modules

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import os 
import numpy as np

# ### Analyse Exploratoire

building_consumption = pd.read_csv("2016_Building_Energy_Benchmarking.csv")

# On regarde comment un batiment est défini dans ce jeu de données 
building_consumption.head()

# On regarde le nombre de valeurs manquantes par colonne ainsi que leur type 
building_consumption.info()

# #### TERMINER L'ANALYSE EXPLORATOIRE 

# A réaliser : 
# - Une analyse descriptive des données, y compris une explication du sens des colonnes gardées, des arguments derrière la suppression de lignes ou de colonnes, des statistiques descriptives et des visualisations pertinentes.

# Qelques pistes d'analyse : 

# Suppression des lignes concernant des immeubles d'habitation
to_delete = ["Multifamily LR (1-4)", "Multifamily MR (5-9)", "Multifamily HR (10+)"]
df = building_consumption[~building_consumption["BuildingType"].isin(to_delete)]
df = df.drop(columns=["City", "State", "DataYear", "Latitude", "Longitude", "Comments", "DefaultData"]).copy()

# * Identifier les colonnes avec une majorité de valeurs manquantes ou constantes en utilisant la méthode value_counts() de Pandas
for column in df.columns:
    print(f"\n--- {column} ---")
    print(df[column].value_counts(normalize=True, dropna=False) * 100)

print(f"\n---**** {df["LargestPropertyUseType"].value_counts(normalize=True, dropna=False) * 100} ---")

# * Mettre en evidence les différences entre les immeubles mono et multi-usages
df["PropertyActivityNumber"] = df[["SecondLargestPropertyUseType", "ThirdLargestPropertyUseType"]].notna().any(axis=1)
df["PropertyActivityNumber"] = df["PropertyActivityNumber"].map({True: "Multi-activity", False: "Mono-activity"})


# * Utiliser des pairplots et des boxplots pour faire ressortir les outliers ou des batiments avec des valeurs peu cohérentes d'un point de vue métier 
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
# Pairplot
pairplot_path = os.path.join(output_dir, "pairplot.png")
sns.pairplot(df[numeric_columns])
plt.savefig(pairplot_path, dpi=300, bbox_inches="tight")
plt.close()

# Boxplot
boxplot_path = os.path.join(output_dir, "boxplot.png")
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[numeric_columns])
plt.xticks(rotation=45)
plt.savefig(boxplot_path, dpi=300, bbox_inches="tight")
plt.close()
# Pour vous inspirer, ou comprendre l'esprit recherché dans une analyse exploratoire, vous pouvez consulter ce notebook en ligne : https://www.kaggle.com/code/pmarcelino/comprehensive-data-exploration-with-python. Il ne s'agit pas d'un modèle à suivre à la lettre ni d'un template d'analyses attendues pour ce projet. 

# # Modélisation 

# ### Import des modules 

#Selection
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV, 
    cross_validate,
)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error 
from sklearn.inspection import permutation_importance

#Preprocess
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

#Modèles
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# ### Feature Engineering

# A réaliser : Enrichir le jeu de données actuel avec de nouvelles features issues de celles existantes. 

# En règle générale : On utilise la méthode .apply() de Pandas pour créer une nouvelle colonne à partir d'une colonne existante. N'hésitez pas à regarder les exemples dans les chapitres de cours donnés en ressource

# In[ ]:

df["BuildingAge"] = df.apply(lambda row: 2025 - row["YearBuilt"], axis=1)

df["ElectricityShare"] = df.apply(
    lambda row: row["Electricity(kBtu)"] / row["SiteEnergyUse(kBtu)"]
    if pd.notna(row["SiteEnergyUse(kBtu)"]) and row["SiteEnergyUse(kBtu)"] != 0
    else None,
    axis=1
)

df["GasShare"] = df.apply(
    lambda row: row["NaturalGas(kBtu)"] / row["SiteEnergyUse(kBtu)"]
    if pd.notna(row["SiteEnergyUse(kBtu)"]) and row["SiteEnergyUse(kBtu)"] != 0
    else None,
    axis=1
)

df["SteamShare"] = df.apply(
    lambda row: row["SteamUse(kBtu)"] / row["SiteEnergyUse(kBtu)"]
    if pd.notna(row["SiteEnergyUse(kBtu)"]) and row["SiteEnergyUse(kBtu)"] != 0
    else None,
    axis=1
)

df.to_excel("2016_Building_Energy_V1.xlsx", index=False)

# CODE FEATURE ENGINEERING

# ### Préparation des features pour la modélisation

# A réaliser :
# * Si ce n'est pas déjà fait, supprimer toutes les colonnes peu pertinentes pour la modélisation.
# * Tracer la distribution de la cible pour vous familiariser avec l'ordre de grandeur. En cas d'outliers, mettez en place une démarche pour les supprimer.

def detect_outliers_iqr(series, k=5):
    """
    Détecte outliers par IQR en ignorant NaN et 0 pour le calcul des quantiles.
    k : multiplicateur IQR (1.5 classique, 3.0 plus strict)
    Retourne une Series (index -> valeur) des outliers (issus de la série originale).
    """
    # Nettoyage pour calcul des quantiles
    clean = series.dropna()
    clean = clean[clean != 0]
    if clean.empty:
        return pd.Series([], dtype=series.dtype)

    Q1 = clean.quantile(0.25)
    Q3 = clean.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR

    # Appliquer les bornes sur la série originale (pour conserver NaN et 0 non marqués)
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return outliers

# IQR
outliers_iqr = detect_outliers_iqr(df["SiteEUI(kBtu/sf)"], k=3)

# seuil absolu
seuil_physique = 400  # valeur réaliste maximale pour tout type de bâtiment
outliers_physique = df[df["SiteEUI(kBtu/sf)"] > seuil_physique]

# fusionner les deux
outliers_total = df.loc[outliers_iqr.index.union(outliers_physique.index)]

# Colonnes de référence à afficher
ref_cols = ["PropertyGFATotal", "NumberofBuildings", "LargestPropertyUseType"]

# Afficher les lignes des outliers
print(f"\nLignes contenant les outliers de SiteEUI (total {len(outliers_total)} lignes) :\n")

for idx in outliers_total.index:
    row = df.loc[idx]
    print(f"Ligne {idx}: SiteEUI = {row['SiteEUI(kBtu/sf)']}")
    for col in ref_cols:
        print(f"  {col}: {row[col]}")
    print("-" * 50)
    

# * Débarrassez-vous des features redondantes en utilisant une matrice de corrélation de Pearson. Pour cela, utiisez la méthode corr() de Pandas, couplé d'un graphique Heatmap de la librairie Seaborn 
# * Réalisez différents graphiques pour comprendre le lien entre vos features et la target (boxplots, scatterplots, pairplot si votre nombre de features numériques n'est pas très élevé).
# *  Séparez votre jeu de données en un Pandas DataFrame X (ensemble de feautures) et Pandas Series y (votre target).
# * Si vous avez des features catégorielles, il faut les encoder pour que votre modèle fonctionne. Les deux méthodes d'encodage à connaitre sont le OneHotEncoder et le LabelEncoder

# In[ ]:




# CODE PREPARATION DES FEATURES


# ### Comparaison de différents modèles supervisés

# A réaliser :
# * Pour chaque algorithme que vous allez tester, vous devez :
#     * Réaliser au préalable une séparation en jeu d'apprentissage et jeu de test via une validation croisée.
#     * Si les features quantitatives que vous souhaitez utiliser ont des ordres de grandeur très différents les uns des autres, et que vous utilisez un algorithme de regression qui est sensible à cette différence, alors il faut réaliser un scaling (normalisation) de la donnée au préalable.
#     * Entrainer le modèle sur le jeu de Train
#     * Prédire la cible sur la donnée de test (nous appelons cette étape, l'inférence).
#     * Calculer les métriques de performance R2, MAE et RMSE sur le jeu de train et de test.
#     * Interpréter les résultats pour juger de la fiabilité de l'algorithme.
# * Vous pouvez choisir par exemple de tester un modèle linéaire, un modèle à base d'arbres et un modèle de type SVM
# * Déterminer le modèle le plus performant parmi ceux testés.

# In[1]:


# CODE COMPARAISON DES MODELES


# ### Optimisation et interprétation du modèle

# A réaliser :
# * Reprennez le meilleur algorithme que vous avez sécurisé via l'étape précédente, et réalisez une GridSearch de petite taille sur au moins 3 hyperparamètres.
# * Si le meilleur modèle fait partie de la famille des modèles à arbres (RandomForest, GradientBoosting) alors utilisez la fonctionnalité feature importance pour identifier les features les plus impactantes sur la performance du modèle. Sinon, utilisez la méthode Permutation Importance de sklearn.

# In[ ]:


# CODE OPTIMISATION ET INTERPRETATION DU MODELE

