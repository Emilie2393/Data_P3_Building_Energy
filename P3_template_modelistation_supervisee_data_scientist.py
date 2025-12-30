#!/usr/bin/env python
# coding: utf-8

# # Analyse Exploratoire

# ### Import des modules

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import os 
import numpy as np
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

class BuildingEnergyStudy():

    def __init__(self):
        self.target = "SiteEUI(kBtu/sf)"
        self.df = pd.read_csv("2016_Building_Energy_Benchmarking.csv")
        self.df_filtered = None

    # ### Analyse Exploratoire

    def doc_analysis(self):
        # On regarde comment un batiment est défini dans ce jeu de données 
        self.df.head()

        # On regarde le nombre de valeurs manquantes par colonne ainsi que leur type 
        self.df.info()

        # #### TERMINER L'ANALYSE EXPLORATOIRE 

        # A réaliser : 
        # - Une analyse descriptive des données, y compris une explication du sens des colonnes gardées, des arguments derrière la suppression de lignes ou de colonnes, des statistiques descriptives et des visualisations pertinentes.

        # Qelques pistes d'analyse : 

        # Suppression des lignes concernant des immeubles d'habitation
        to_delete = ["Multifamily LR (1-4)", "Multifamily MR (5-9)", "Multifamily HR (10+)"]
        df = self.df[~self.df["BuildingType"].isin(to_delete)]
        self.df_filtered = df.drop(columns=["City", "State", "DataYear", "Latitude", "Longitude", "Comments", "DefaultData"]).copy()

        # * Identifier les colonnes avec une majorité de valeurs manquantes ou constantes en utilisant la méthode value_counts() de Pandas
        for column in self.df_filtered.columns:
            print(f"\n--- {column} ---")
            print(self.df_filtered[column].value_counts(normalize=True, dropna=False) * 100)

        print(f"\n---**** {self.df_filtered["LargestPropertyUseType"].value_counts(normalize=True, dropna=False) * 100} ---")

        # * Mettre en evidence les différences entre les immeubles mono et multi-usages
        self.df_filtered["PropertyActivityNumber"] = self.df_filtered[["SecondLargestPropertyUseType", "ThirdLargestPropertyUseType"]].notna().any(axis=1)
        self.df_filtered["PropertyActivityNumber"] = self.df_filtered["PropertyActivityNumber"].map({True: "Multi-activity", False: "Mono-activity"})

    def first_graph(self):

        # * Utiliser des pairplots et des boxplots pour faire ressortir les outliers ou des batiments avec des valeurs peu cohérentes d'un point de vue métier 
        output_dir = "plots"
        os.makedirs(output_dir, exist_ok=True)
        numeric_columns = self.df_filtered.select_dtypes(include=["float64", "int64"]).columns
        # Pairplot
        pairplot_path = os.path.join(output_dir, "pairplot.png")
        sns.pairplot(self.df_filtered[numeric_columns])
        plt.savefig(pairplot_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Boxplot
        boxplot_path = os.path.join(output_dir, "boxplot.png")
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df_filtered[numeric_columns])
        plt.xticks(rotation=45)
        plt.savefig(boxplot_path, dpi=300, bbox_inches="tight")
        plt.close()
    # Pour vous inspirer, ou comprendre l'esprit recherché dans une analyse exploratoire, vous pouvez consulter ce notebook en ligne : https://www.kaggle.com/code/pmarcelino/comprehensive-data-exploration-with-python. Il ne s'agit pas d'un modèle à suivre à la lettre ni d'un template d'analyses attendues pour ce projet. 

    # # Modélisation 

    

    # ### Feature Engineering

    # A réaliser : Enrichir le jeu de données actuel avec de nouvelles features issues de celles existantes. 

    # En règle générale : On utilise la méthode .apply() de Pandas pour créer une nouvelle colonne à partir d'une colonne existante. N'hésitez pas à regarder les exemples dans les chapitres de cours donnés en ressource

    # In[ ]:
    def new_features(self):

        self.df_filtered["BuildingAge"] = self.df_filtered.apply(lambda row: 2025 - row["YearBuilt"], axis=1)

        self.df_filtered["ElectricityShare"] = self.df_filtered.apply(
            lambda row: row["Electricity(kBtu)"] / row["SiteEnergyUse(kBtu)"]
            if pd.notna(row["SiteEnergyUse(kBtu)"]) and row["SiteEnergyUse(kBtu)"] != 0
            else None,
            axis=1
        )

        self.df_filtered["GasShare"] = self.df_filtered.apply(
            lambda row: row["NaturalGas(kBtu)"] / row["SiteEnergyUse(kBtu)"]
            if pd.notna(row["SiteEnergyUse(kBtu)"]) and row["SiteEnergyUse(kBtu)"] != 0
            else None,
            axis=1
        )

        self.df_filtered.to_excel("2016_Building_Energy_V1.xlsx", index=False)

    # CODE FEATURE ENGINEERING

    # ### Préparation des features pour la modélisation

    # A réaliser :
    # * Si ce n'est pas déjà fait, supprimer toutes les colonnes peu pertinentes pour la modélisation.
    # * Tracer la distribution de la cible pour vous familiariser avec l'ordre de grandeur. En cas d'outliers, mettez en place une démarche pour les supprimer.

    def target_distribution(self):

        plt.figure(figsize=(6, 4))
        sns.histplot(self.df_filtered[self.target], bins=40, kde=True)
        plt.xlabel(self.target)
        plt.title("Distribution de la cible SiteEUI")
        plt.savefig("plots/target_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()

    def detect_outliers_iqr(self, series, k=5):
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
    
    def delete_outliers(self):

        self.df_filtered = self.df_filtered[~self.df_filtered["ComplianceStatus"].isin(["Missing Data", "Not-Compliant"])]

        # IQR
        outliers_iqr = self.detect_outliers_iqr(self.df_filtered["SiteEUI(kBtu/sf)"], k=3)

        # seuil absolu
        seuil_physique = 400  # valeur réaliste maximale pour tout type de bâtiment
        outliers_physique = self.df_filtered[self.df_filtered["SiteEUI(kBtu/sf)"] > seuil_physique]

        # fusionner les deux
        outliers_total = self.df_filtered.loc[outliers_iqr.index.union(outliers_physique.index)]

        # Colonnes de référence à afficher
        ref_cols = ["PropertyGFATotal", "NumberofBuildings", "LargestPropertyUseType"]

        # Afficher les lignes des outliers
        print(f"\nLignes contenant les outliers de SiteEUI (total {len(outliers_total)} lignes) :\n")

        for idx in outliers_total.index:
            row = self.df_filtered.loc[idx]
            print(f"Ligne {idx}: SiteEUI = {row['SiteEUI(kBtu/sf)']}")
            for col in ref_cols:
                print(f"  {col}: {row[col]}")
            print("-" * 50)

        valid_uses = "Hospital|Care|Laboratory|Data"

        rows_to_drop = outliers_total[
            ~outliers_total["LargestPropertyUseType"]
            .fillna("")
            .str.contains(valid_uses, case=False)
        ].index

        # Suppression
        self.df_filtered = self.df_filtered.drop(index=rows_to_drop)

        print(f"{len(rows_to_drop)} lignes supprimées - outliers dont l'utilité contient d'autres termes que Hospital, Care, Laboratory et Data")

    # * Débarrassez-vous des features redondantes en utilisant une matrice de corrélation de Pearson. Pour cela, utiisez la méthode corr() de Pandas, couplé d'un graphique Heatmap de la librairie Seaborn 

    def pearson(self):
        df_num = self.df_filtered.select_dtypes(include='number')
        corr_pearson = df_num.corr(method='pearson')
        corr_spearman = df_num.corr(method='spearman')
        # figure size
        plt.figure(figsize=(30, 10))
        sns.heatmap(corr_pearson, annot=True, cmap="coolwarm", center=0)
        # change bottom indicator rotation for a better reading
        plt.xticks(rotation=40, ha="right")
        plt.title("Pearson")
        output_dir = "Buildings parameters correlation"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/Parameters_correlation_corr_heatmaps.png", dpi=500, bbox_inches="tight")
        self.df_filtered = self.df_filtered.drop(columns=["SiteEUIWN(kBtu/sf)", "SourceEUI(kBtu/sf)", "SourceEUIWN(kBtu/sf)", "Electricity(kBtu)", "SiteEnergyUse(kBtu)", "SiteEnergyUseWN(kBtu)", "YearBuilt"])


    # * Réalisez différents graphiques pour comprendre le lien entre vos features et la target (boxplots, scatterplots, pairplot si votre nombre de features numériques n'est pas très élevé).

    def pairplot(self):
        selected_features = [
            self.target,
            "BuildingAge",
            "ElectricityShare",
            "PropertyGFATotal",
            "NumberofFloors",
            "ENERGYSTARScore"
        ]

        sns.pairplot(
            self.df_filtered[selected_features],
            diag_kind="kde",
            corner=True
        )
        plt.savefig(f"plots/target_pairplot.png", dpi=500, bbox_inches="tight")

    # *  Séparez votre jeu de données en un Pandas DataFrame X (ensemble de feautures) et Pandas Series y (votre target).
    # * Si vous avez des features catégorielles, il faut les encoder pour que votre modèle fonctionne. Les deux méthodes d'encodage à connaitre sont le OneHotEncoder et le LabelEncoder
    
    def target_feature_encoder(self):

        X = self.df_filtered.drop(columns=[self.target])
        y = self.df_filtered[self.target]

        # divise le jeu de données entrainement/test pour éviter que le modèle détecte les réponses à l'avance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        property_use_cols = [
            "LargestPropertyUseType",
            "SecondLargestPropertyUseType",
            "ThirdLargestPropertyUseType"
        ]

        def group_rare(series, min_freq):
            """
            Remplace les catégories peu fréquentes par 'Other'
            """
            counts = series.value_counts()
            rare_categories = counts[counts < min_freq].index
            return series.replace(rare_categories, "Other")
        
        for col in property_use_cols:
            X[col] = group_rare(X[col], 5)

        # Récupérer les ensembles de catégories uniques (en ignorant les NaN)
        sets = {
            col: set(X[col].dropna().unique())
            for col in property_use_cols
        }

        # Créer un set de toutes les catégories uniques toutes colonnes confondues
        all_categories = set().union(*sets.values())

        # Nombre total de catégories uniques
        total_categories = len(all_categories)
        print(f"Nombre total de catégories uniques sur toutes les colonnes PropertyUse : {total_categories}")
        # Recouvrement pair à pair
        print("Recouvrement entre les colonnes :\n")

        for i in range(len(property_use_cols)):
            for j in range(i + 1, len(property_use_cols)):
                col1 = property_use_cols[i]
                col2 = property_use_cols[j]
                overlap = sets[col1] & sets[col2]

                print(f"{col1} ∩ {col2} : {len(overlap)} catégories communes")
        
        


        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        num_cols = X.select_dtypes(exclude=["object", "category"]).columns






    
    def exec_analysis(self):

        self.doc_analysis()
        #self.first_graph()
        self.new_features()
        #self.target_distribution()
        self.delete_outliers()
        self.pearson()
        #self.pairplot()
        self.target_feature_encoder()

results = BuildingEnergyStudy()
results.exec_analysis()








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

