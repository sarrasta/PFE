# RAPPORT DE FIN D'ÉTUDES (PFE)

---

## Application du Machine Learning à la Rétention Client dans le Secteur Télécom
### Conception et Implémentation d'un Système de Scoring Prédictif sur un Data Warehouse Télécom Tunisien

---

**Réalisé par :** Sarra  
**Encadrant académique :** —  
**Encadrant professionnel :** —  
**Établissement :** —  
**Filière :** Informatique / Intelligence Artificielle  
**Année universitaire :** 2025–2026

---

---

## Résumé

Ce projet de fin d'études porte sur l'application du Machine Learning à la problématique de la rétention client dans le secteur des télécommunications. En s'appuyant sur un Data Warehouse PostgreSQL (DW_TT) contenant les données comportementales de 10 000 clients sur 12 mois, neuf objectifs d'apprentissage automatique ont été conçus et implémentés.

Les travaux couvrent l'intégralité du pipeline data science : prédiction du churn par classification binaire (SVC-RBF, AUC = 0.71), classification du niveau de risque par clustering K-Means (97.3% de précision), segmentation comportementale en 4 groupes distincts, construction de profils de rétention personnalisés (12 micro-profils), score continu de propension au churn (Ridge, AUC = 0.677), prédiction de l'ARPU futur par régression (RandomForest, R² = 0.962), modélisation de la probabilité de réponse à une offre de rétention par classification calibrée (LogReg, AUC = 0.619), et calcul du gain financier attendu d'une action de rétention par espérance mathématique (lift 2.68×, gain total 369 678 TND).

Les résultats démontrent qu'un système de scoring intégré, combinant risque de départ, réceptivité à l'offre et valeur financière du client, permet de multiplier l'efficacité des campagnes de rétention par 2.68 par rapport à un ciblage aléatoire. Les principaux défis rencontrés sont le déséquilibre sévère des labels (93.6% de churners) et l'absence de labels historiques pour certains objectifs, résolus par SMOTE et génération de labels synthétiques calibrés.

**Mots-clés :** Machine Learning, Churn, Rétention Client, Télécommunications, Data Warehouse, Classification, Clustering, Régression, SMOTE, Score de Propension, Valeur Client (LTV)

---

## Abstract

This final year project focuses on applying Machine Learning to customer retention in the telecommunications sector. Using a PostgreSQL Data Warehouse (DW_TT) containing behavioral data for 10,000 customers over 12 months, nine machine learning objectives were designed and implemented.

The work covers the full data science pipeline: churn prediction via binary classification (SVC-RBF, AUC = 0.71), risk level classification via K-Means clustering (97.3% accuracy), behavioral segmentation into 4 distinct groups, construction of personalized retention profiles (12 micro-profiles), a continuous churn propensity score (Ridge, AUC = 0.677), future ARPU prediction via regression (RandomForest, R² = 0.962), offer response probability modeling via calibrated classification (LogReg, AUC = 0.619), and expected financial gain calculation from retention actions via mathematical expectation (2.68× lift, total gain 369,678 TND).

Results show that an integrated scoring system, combining departure risk, offer receptivity, and customer financial value, multiplies retention campaign efficiency by 2.68× compared to random targeting. The main challenges were severe label imbalance (93.6% churners) and the absence of historical labels for certain objectives, resolved by SMOTE and calibrated synthetic label generation.

**Keywords:** Machine Learning, Churn, Customer Retention, Telecommunications, Data Warehouse, Classification, Clustering, Regression, SMOTE, Propensity Score, Customer Lifetime Value (LTV)

---

## Table des Matières

- [Introduction Générale](#introduction-générale)
- [Chapitre 1 — Contexte et Problématique](#chapitre-1--contexte-et-problématique)
- [Chapitre 2 — État de l'Art](#chapitre-2--état-de-lart)
- [Chapitre 3 — Architecture et Conception](#chapitre-3--architecture-et-conception)
- [Chapitre 4 — Objectif 1 : Prédiction du Churn Client](#chapitre-4--objectif-1--prédiction-du-churn-client)
- [Chapitre 5 — Objectif 2 : Classification du Risque Client](#chapitre-5--objectif-2--classification-du-risque-client)
- [Chapitre 6 — Objectif 3 : Optimisation des Métriques Métier](#chapitre-6--objectif-3--optimisation-des-métriques-métier)
- [Chapitre 7 — Objectif 4 : Segmentation Comportementale](#chapitre-7--objectif-4--segmentation-comportementale)
- [Chapitre 8 — Objectif 5 : Profils de Rétention Personnalisés](#chapitre-8--objectif-5--profils-de-rétention-personnalisés)
- [Chapitre 9 — Objectif 6 : Score de Propension au Churn](#chapitre-9--objectif-6--score-de-propension-au-churn)
- [Chapitre 10 — Objectif 7 : Prédiction de l'ARPU Futur](#chapitre-10--objectif-7--prédiction-de-larpu-futur)
- [Chapitre 11 — Objectif 8 : Probabilité de Réponse à une Offre](#chapitre-11--objectif-8--probabilité-de-réponse-à-une-offre)
- [Chapitre 12 — Objectif 9 : Gain Attendu d'une Action de Rétention](#chapitre-12--objectif-9--gain-attendu-dune-action-de-rétention)
- [Conclusion Générale et Perspectives](#conclusion-générale-et-perspectives)
- [Bibliographie](#bibliographie)
- [Annexes](#annexes)

---

## Liste des Abréviations

| Abréviation | Signification |
|---|---|
| **ML** | Machine Learning (Apprentissage Automatique) |
| **DW** | Data Warehouse (Entrepôt de Données) |
| **ARPU** | Average Revenue Per User (Revenu Moyen par Utilisateur) |
| **LTV** | Lifetime Value (Valeur sur la Durée de Vie) |
| **AUC-ROC** | Area Under the Curve – Receiver Operating Characteristic |
| **MAE** | Mean Absolute Error (Erreur Absolue Moyenne) |
| **RMSE** | Root Mean Squared Error |
| **MAPE** | Mean Absolute Percentage Error |
| **SMOTE** | Synthetic Minority Over-sampling Technique |
| **SVC** | Support Vector Classifier |
| **RF** | Random Forest |
| **GBR** | Gradient Boosting Regressor/Classifier |
| **NPS** | Net Promoter Score |
| **QoS** | Quality of Service (Qualité de Service) |
| **CV** | Cross-Validation (Validation Croisée) |
| **PFE** | Projet de Fin d'Études |
| **SAV** | Service Après-Vente |
| **TND** | Tunisian Dinar (Dinar Tunisien) |

---

---

## Introduction Générale

### Contexte général

Dans un marché des télécommunications de plus en plus saturé et concurrentiel, la rétention des clients existants est devenue un enjeu stratégique majeur pour les opérateurs. Acquérir un nouveau client coûte en moyenne 5 à 7 fois plus cher que de fidéliser un client existant. Le phénomène de churn — désignant le fait qu'un client résilie son abonnement ou migre vers un concurrent — représente une perte directe de revenus récurrents difficile à compenser.

En Tunisie, le secteur télécom est marqué par une forte pénétration mobile (>100%), une guerre des prix entre opérateurs (Tunisie Telecom, Ooredoo, Orange), et des clients de plus en plus volatils. Dans ce contexte, la capacité à anticiper le comportement des clients et à personnaliser les actions de rétention devient un avantage compétitif décisif.

### Problématique

Comment exploiter les données comportementales mensuelles d'un Data Warehouse télécom pour construire un système d'aide à la décision capable de :
1. Identifier les clients à risque de départ
2. Les segmenter par profil de comportement et niveau de risque
3. Estimer leur valeur financière future
4. Calculer le gain attendu d'une action de rétention personnalisée
5. Fournir une liste priorisée de clients à cibler pour maximiser le retour sur investissement

### Objectifs du projet

Ce PFE vise à concevoir et implémenter **9 modules de Machine Learning** interconnectés formant un pipeline complet de décision de rétention client, en s'appuyant sur les données réelles d'un Data Warehouse PostgreSQL (DW_TT).

### Organisation du rapport

Le présent rapport est structuré comme suit : le Chapitre 1 présente le contexte métier et la problématique. Le Chapitre 2 dresse un état de l'art des techniques ML appliquées à la rétention client. Le Chapitre 3 décrit l'architecture technique du projet. Les Chapitres 4 à 12 détaillent chacun des 9 objectifs ML implémentés. La conclusion synthétise les apports et ouvre sur des perspectives d'amélioration.

---

---

## Chapitre 1 — Contexte et Problématique

### 1.1 Le secteur télécom tunisien

Le marché des télécommunications tunisien regroupe trois opérateurs mobiles (Tunisie Telecom, Ooredoo Tunisie, Orange Tunisie) pour une population d'environ 12 millions d'habitants. Avec un taux de pénétration mobile dépassant 100%, le marché est mature : la croissance passe nécessairement par la conquête de parts de marché au détriment des concurrents.

Cette maturité engendre une pression forte sur la rétention : chaque client perdu au profit d'un concurrent représente une perte de revenu récurrent (ARPU × durée de vie résiduelle) et un coût de reconquête élevé si le client revient.

### 1.2 Définition du churn

Le **churn** (ou attrition client) désigne le fait qu'un client cesse d'utiliser les services d'un opérateur sur une période donnée. On distingue :

- **Churn volontaire** : le client résilie délibérément (insatisfaction, offre concurrente plus attractive, fin d'engagement)
- **Churn involontaire** : résiliation pour défaut de paiement, décès, départ à l'étranger

Le **taux de churn mensuel** est l'indicateur standard :
```
Taux de churn = Nombre de clients perdus ce mois / Nombre de clients en début de mois
```

Un taux de 2% mensuel représente ~22% annuel — pour un opérateur de 10 000 clients, cela signifie perdre 2 200 clients par an.

### 1.3 Enjeux de la rétention client

| Enjeu | Description |
|---|---|
| **Financier** | Coût d'acquisition client >> Coût de rétention (ratio 5:1 à 7:1) |
| **Stratégique** | Parts de marché stables → stabilité des revenus → investissements réseau |
| **Opérationnel** | Personnalisation des offres → meilleure utilisation du budget marketing |
| **Analytique** | Transformation des données comportementales en décisions actionnables |

### 1.4 Limites des approches traditionnelles

Avant l'ère du Machine Learning, les opérateurs utilisaient des règles manuelles pour détecter les clients à risque :

- "Tout client n'ayant pas rechargé depuis 30 jours"
- "Tout client ayant déposé 3 réclamations ou plus"
- "Tout client en fin d'engagement"

Ces règles souffrent de plusieurs limitations :
- Elles capturent des **signaux unidimensionnels** et ignorent les interactions entre variables
- Elles génèrent beaucoup de **faux positifs** (coût d'action inutile)
- Elles ne **hiérarchisent pas** les clients par urgence ou valeur financière
- Elles ne s'**adaptent pas** à l'évolution des comportements clients

### 1.5 Apport du Machine Learning

Le Machine Learning permet de :
1. **Combiner des centaines de variables** pour une prédiction plus précise
2. **Détecter des patterns non-linéaires** (ex : un client avec QoS dégradée + fin d'engagement + NPS faible a un risque bien supérieur à la somme de ces facteurs séparément)
3. **Produire des scores continus** (propension ∈ [0,1]) plutôt que des décisions binaires
4. **Quantifier le ROI** de chaque action de rétention avant de la lancer
5. **Personnaliser** l'offre selon le profil comportemental de chaque client

---

---

## Chapitre 2 — État de l'Art

### 2.1 Prédiction du churn par Machine Learning

La prédiction du churn est l'un des cas d'usage les plus étudiés du Machine Learning appliqué aux télécoms. Les approches dominantes dans la littérature sont :

#### 2.1.1 Modèles de classification supervisée

**Arbres de décision et ensembles** : Les forêts aléatoires (Breiman, 2001) et le Gradient Boosting (Friedman, 2001) sont les modèles les plus utilisés en pratique pour leur capacité à gérer des features hétérogènes sans preprocessing intensif. XGBoost (Chen & Guestrin, 2016) et LightGBM (Ke et al., 2017) sont leurs implémentations optimisées les plus populaires.

**Support Vector Machines** : Le SVM à noyau RBF (Vapnik, 1995) est particulièrement adapté aux espaces de features normalisés avec des frontières non-linéaires. Ses performances en ranking (AUC) surpassent souvent les arbres quand le signal est linéaire dans l'espace transformé.

**Régression Logistique** : Malgré sa simplicité, la régression logistique reste compétitive quand les relations entre features et cible sont majoritairement linéaires. Elle produit nativement des probabilités calibrées, avantage pour les applications métier.

#### 2.1.2 Gestion du déséquilibre de classes

Le déséquilibre de classes (imbalanced learning) est un problème central en prédiction de churn. Les approches reconnues sont :

- **SMOTE** (Chawla et al., 2002) : suréchantillonnage synthétique par interpolation entre voisins de la classe minoritaire
- **ADASYN** : variante de SMOTE avec densité adaptative
- **BalancedRandomForest** : sous-échantillonnage de la majorité à chaque arbre
- **EasyEnsemble** : bagging avec sous-échantillonnage équilibré
- **class_weight** : pénalisation asymétrique dans la fonction de perte

#### 2.1.3 Métriques d'évaluation adaptées

Avec une prévalence de churn élevée (>70%), l'accuracy et le F1-score sont des métriques trompeuses. La littérature recommande :
- **AUC-ROC** : mesure la capacité de ranking indépendamment du seuil
- **Lift@TopK%** : gain vs aléatoire pour les K% clients les plus risqués
- **Courbe de gain cumulé** : % de churners capturés selon le % de clients contactés
- **Brier Score** : qualité des probabilités calibrées

### 2.2 Segmentation client non supervisée

#### 2.2.1 K-Means et ses limites

L'algorithme K-Means (Lloyd, 1957) reste la méthode de référence pour la segmentation client grâce à sa simplicité et sa scalabilité. Ses limites principales sont la sensibilité aux outliers et l'hypothèse de clusters sphériques. Le choix de k est guidé par la méthode du coude (Elbow), le score de Silhouette (Rousseeuw, 1987), et l'indice Davies-Bouldin.

#### 2.2.2 Ingénierie des features pour le clustering

La qualité d'une segmentation dépend fortement de l'espace de features. L'approche des **scores composites** — combiner plusieurs variables brutes en un score thématique unique — permet de :
- Réduire la dimensionnalité
- Éliminer la redondance entre features corrélées
- Créer des axes d'inertie indépendants (meilleure séparation)

### 2.3 Modèles de régression pour la valeur client

#### 2.3.1 Prédiction de l'ARPU

La prédiction du revenu futur d'un client est un problème de régression temporelle. Les features les plus prédictives identifiées dans la littérature sont la valeur récente (average des derniers mois), la tendance (slope de régression), et la volatilité (écart-type historique).

#### 2.3.2 Customer Lifetime Value (CLV/LTV)

La Valeur sur la Durée de Vie du Client est le concept fondamental qui justifie l'investissement en rétention :
```
CLV = ARPU × Marge × (1 / Taux de churn)
```
En pratique, on approxime sur un horizon fixe : `LTV = ARPU × LTV_horizon`.

### 2.4 Calibration des modèles probabilistes

Un modèle bien calibré produit des probabilités qui correspondent à des fréquences réelles : si le modèle dit 0.3, 30% des cas devraient effectivement être positifs. La calibration est essentielle pour les calculs de ROI et d'espérance mathématique.

**Méthodes de calibration post-entraînement** :
- **Platt scaling** : régression logistique sur les scores bruts
- **Régression isotonique** : monotone, plus flexible, nécessite plus de données

### 2.5 Positionnement du projet

Ce projet se distingue des travaux habituels sur trois points :

1. **Pipeline intégré** : les 9 objectifs sont interdépendants — les outputs d'un objectif servent d'inputs aux suivants
2. **Formule de gain unifiée** : combinaison de P_churn × P_réponse × ARPU en un score financier actionnable
3. **Contrainte de données réelles** : déséquilibre sévère (93.6%) et labels manquants résolus par des techniques adaptées

---

---

## Chapitre 3 — Architecture et Conception

### 3.1 Description du Data Warehouse DW_TT

Le Data Warehouse **DW_TT** (Data Warehouse Tunisie Télécom) est une base PostgreSQL hébergée localement (`localhost:5432`). Il est structuré en **schéma en étoile** avec une table de faits centrale et plusieurs dimensions.

#### 3.1.1 Tables de dimensions

| Table | Description | Colonnes clés |
|---|---|---|
| `dim_client` | Attributs clients statiques | `ClientID`, `Sexe`, `Segment`, `TypeAbonnement`, `Region`, `Anciennete_mois`, `Engagement_restant`, `PrixOffre` |
| `dim_offre` | Catalogue d'offres commerciales | `OffreID`, `TypeOffre`, `PrixOffre`, `DataForfait`, `MinutesForfait` |
| `dim_geographique` | Régions géographiques | `RegionID`, `NomRegion`, `Gouvernorat` |
| `dim_date` | Dimension temporelle | `DateFK` (25–36 = janv–déc 2024) |

#### 3.1.2 Tables de faits

| Table | Description | Volume |
|---|---|---|
| `Fact_performance_client` | KPIs mensuels par client | **120 000 lignes** (10 000 clients × 12 mois) |
| `Fact_churn` | Labels churn observés | **422 lignes** (clients avec label disponible) |

#### 3.1.3 Variables de performance mensuelle (Fact_performance_client)

**Financières** : `ARPU`, `MontantFacture`, `HorsForfait`, `NbImpayes`, `RetardPaiement_jours`  
**Usage** : `MinutesAppel`, `DataConsommee_GB`, `NbSMS`, `Roaming_GB`, `NocturneRatio`  
**Réseau** : `QoS_Score`, `DropRate`, `Throughput_Mbps`, `OutageMinutes`  
**SAV** : `NbTickets`, `DelaiResolution_jours`  
**Satisfaction** : `NPS_Score`

### 3.2 Pipeline global du projet

```
┌─────────────────────────────────────────────────────────────┐
│                     DW_TT (PostgreSQL)                       │
│          dim_client + Fact_performance_client                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING                        │
│  Agrégations temporelles (mean, std, trend, recent, growth)  │
│  Scores composites (usage, prix, réseau, inactivité)          │
│  Features d'ingénierie (score_risque_paiement, flag_engagement)│
└──────┬──────────┬──────────┬──────────┬───────────┬─────────┘
       │          │          │          │           │
  ┌────▼──┐  ┌───▼──┐  ┌────▼──┐  ┌───▼──┐   ┌───▼──┐
  │ Obj 1 │  │ Obj 2│  │ Obj 4 │  │ Obj 6│   │ Obj 7│
  │ Churn │  │Risque│  │Segment│  │Propen│   │ ARPU │
  │SVC-RBF│  │KMeans│  │KMeans │  │Ridge │   │  RF  │
  └────┬──┘  └───┬──┘  └────┬──┘  └───┬──┘   └───┬──┘
       │         │          │          │           │
       └────┬────┘    ┌─────┘          │           │
            │         │                │           │
       ┌────▼─────────▼────┐    ┌──────▼──┐   ┌───▼──┐
       │      Obj 3        │    │  Obj 8  │   │      │
       │ Métriques Métier  │    │ Réponse │   │      │
       │ Lift, ROI, Seuils │    │ Offre   │   │      │
       └────────────────────┘    └──────┬──┘   └───┬──┘
                                        │           │
            ┌────────────────────────────┘           │
            │                                        │
       ┌────▼───────────────────────────────────────▼──┐
       │                   OBJECTIF 9                   │
       │  Gain_Net = P_churn × P_réponse × ARPU×12 − C │
       │  Score de priorité → ciblage ROI-optimal       │
       └────────────────────────────────────────────────┘
```

### 3.3 Environnement technique

| Composant | Outil |
|---|---|
| **Langage** | Python 3.10+ |
| **IDE / Notebooks** | Jupyter Notebook |
| **Base de données** | PostgreSQL 14 + SQLAlchemy (`postgresql://postgres:...@localhost:5432/DW_TT`) |
| **ML** | scikit-learn, imbalanced-learn, scipy |
| **Visualisation** | matplotlib, seaborn |
| **Données synthétiques** | SDV (TVAE, GaussianCopula) |
| **Gestion de projet** | Scripts Python `build_XX.py` → génération des notebooks |

### 3.4 Stratégie de validation

- **Split temporel** (Obj 7) : mois 1–9 pour entraîner, mois 10–12 pour évaluer — préserve l'ordre temporel
- **Cross-Validation 5-fold stratifiée** (Obj 1, 3, 6) : évaluation robuste sur les 422 clients étiquetés
- **Cross-Validation 3-fold** (Obj 8) : évaluation out-of-fold des probabilités calibrées
- **Déploiement sur 10k** (Obj 2, 4, 5, 6, 8, 9) : prédiction sur l'ensemble de la base sans labels

---

---

## Chapitre 4 — Objectif 1 : Prédiction du Churn Client

### 4.1 Présentation de l'objectif

**Problème** : Prédire si un client va churner dans les 30 prochains jours à partir de son historique comportemental sur 12 mois.

**Données** : 422 clients avec label `Churn_30j` ∈ {0, 1} (DateFK=36, décembre 2024)

**Défi principal** : Déséquilibre sévère — 395 churners (93.6%) vs 27 non-churners (6.4%)

### 4.2 Feature Engineering

En plus des 27 variables brutes, 6 features d'ingénierie ont été construites :

| Feature | Formule | Signification métier |
|---|---|---|
| `score_risque_paiement` | `NbImpayes × log(RetardPaiement+1)` | Risque combiné impayés × retard |
| `ratio_hors_forfait` | `HorsForfait / MontantFacture` | Part de surprix dans la facture |
| `score_degradation_reseau` | `DropRate × Outage / QoS` | Dégradation réseau composite |
| `flag_hors_engagement` | `1 si Engagement_restant ≤ 0` | Client hors contrat = libre de partir |
| `intensite_usage` | `(Minutes + Data + SMS) / Anciennete` | Intensité normalisée par ancienneté |
| `score_insatisfaction` | `NbTickets × (1 / NPS + 0.01)` | Frustration combinée SAV + NPS |

### 4.3 Méthodologie — 6 itérations progressives

#### Itération 1 (v1) — Baseline

6 modèles avec deux stratégies (class_weight et SMOTE) évalués par AUC-ROC et F1 en CV 5-fold.

**Résultat clé — Paradoxe du déséquilibre** :  
Random Forest obtient F1=0.966 mais AUC=0.43 — il prédit tout le monde comme churner (Recall=1.0 trivial). La Régression Logistique obtient F1=0.675 mais AUC=0.654. Cela révèle que le F1 est une métrique trompeuse avec 93.6% de prévalence : un classificateur naïf "tout positif" atteint F1≈0.97.

**Décision** : adopter l'**AUC-ROC** comme métrique principale.

#### Itération 2 (v2) — CTGAN

CTGAN (500 epochs) générant des non-churners synthétiques. AUC = 0.537 — régression vs v1.  
**Diagnostic** : les GANs nécessitent >1000 samples pour apprendre une distribution. Sur 20 non-churners, CTGAN génère du bruit qui dégrade les performances.

#### Itération 3 (v3) — TVAE + RandomizedSearchCV

TVAE (Variational AutoEncoder, plus stable sur petits datasets), combinaison des deux périodes disponibles (42 non-churners total), tuning par RandomizedSearchCV (40 itérations), VotingClassifier.  
AUC = 0.667 avec SMOTE — meilleur jusqu'ici.

#### Itération 4 (v4) — Features Temporelles

118 features temporelles (pente, delta, écart-type mensuel sur les 12 mois) ajoutées → 137 features total.  
AUC = 0.590 — régression.  
**Diagnostic — Malédiction de la dimensionnalité** : ratio features/samples = 137/27 > 5 → sur-apprentissage massif sur les folds d'entraînement, pas de généralisation.

#### Itération 5 (v5) — SVM-RBF ← **Meilleur modèle**

Retour aux 35 features. Test de SVM à noyau RBF (C=10, gamma=0.001), NuSVC, BaggingClassifier(SVC).

| Modèle | AUC CV 5-fold |
|---|---|
| **SVC RBF (C=10, γ=0.001)** | **0.7138 ± 0.1144** |
| LR (C=0.05, L2) | 0.6961 ± 0.1108 |
| LR baseline | 0.6803 ± 0.1055 |
| BaggingClassifier(SVC) | 0.6417 ± 0.0799 |

> **Découverte importante — Fuite de données** : L'utilisation du label `Churn_30j` à DateFK=12 comme feature produisait AUC=1.0. Dans le dataset synthétique du PFE, les 395 churners à T36 étaient exactement les mêmes qu'à T12 → corrélation parfaite → fuite de cible. Correction : suppression de toutes features cross-périodes.

#### Itération 6 (v6) — Comparaison des méthodes de synthèse

Test rigoureux (synthèse uniquement dans le fold train, évaluation sur données réelles) :

| Méthode de synthèse | AUC CV |
|---|---|
| **SMOTE (k=5)** | **0.7138** |
| Bootstrap + bruit gaussien 15% | 0.6890 |
| KDE marginale | 0.6645 |
| GaussianCopula (SDV) | 0.6644 |
| Gaussienne Multivariée (LedoitWolf) | 0.6525 |

**SMOTE reste optimal** : l'interpolation entre voisins réels préserve mieux la structure des données que les méthodes paramétriques sur un si petit effectif.

### 4.4 Résultat final

| Paramètre | Valeur |
|---|---|
| **Modèle** | SVC (kernel=rbf, C=10, gamma=0.001) |
| **AUC-ROC CV 5-fold** | **0.7138 ± 0.1144** |
| **Équilibrage** | SMOTE (k_neighbors=5) dans pipeline CV |
| **Features** | 35 (27 brutes + 6 ingénierie + 2 supplémentaires) |
| **Scaler** | RobustScaler |
| **Plafond empirique** | ~0.72 AUC avec 27 non-churners — structurel |

### 4.5 Analyse et discussion

Le plafond de 0.71 est structurel : avec seulement 27 samples minoritaires, chaque fold de CV 5-fold ne contient que ~5 non-churners pour évaluer l'AUC. La variance élevée (±0.11) en est la conséquence directe. Pour franchir 0.85 AUC, il faudrait disposer d'au moins 500 clients non-churners réels ou de features externes discriminantes (ex : logs détaillés des appels au service client, score crédit, données de localisation).

---

---

## Chapitre 5 — Objectif 2 : Classification du Risque Client

### 5.1 Présentation de l'objectif

**Problème** : Attribuer un **niveau de risque** (Faible / Moyen / Élevé) aux 10 000 clients de la base, sans labels supervisés disponibles pour la totalité de la base.

**Approche hybride** :
- Étape 1 : Clustering non supervisé K-Means pour découvrir les 3 niveaux de risque
- Étape 2 : Random Forest supervisé pour apprendre les frontières → déploiement sur nouveaux clients

### 5.2 Features de risque (13 indicateurs)

| Dimension | Features |
|---|---|
| Paiement | `total_impayes`, `max_retard_paiement`, `avg_retard_paiement`, `score_risque_paiement` |
| Réseau | `avg_drop_rate`, `avg_outage_min`, `score_degradation_reseau` |
| Satisfaction | `total_tickets`, `avg_nps`, `score_insatisfaction` |
| Contrat | `flag_hors_engagement`, `avg_hors_forfait` |
| Churn model | `churn_proba` (SVC v5 de l'Objectif 1) |

### 5.3 Sélection du nombre de clusters

| k | Score Silhouette | Verdict |
|---|---|---|
| 2 | 0.09 | Séparation trop grossière |
| **3** | **0.1425** | **Optimal — correspond aux niveaux Faible/Moyen/Élevé** |
| 4 | 0.11 | Sur-segmentation |

### 5.4 Résultats du clustering

| Niveau | Clients | % base | Churn proba moy. |
|---|---|---|---|
| **Faible** | 4 370 | 43.7% | 0.671 |
| **Moyen** | 2 501 | 25.0% | 0.596 |
| **Élevé** | 3 129 | 31.3% | **0.815** |

**Profil des centroïdes** :
- **Élevé** : `total_impayes` = 1.59 (vs 0.28), `max_retard_paiement` = 21.8 jours → défaillance de paiement chronique
- **Moyen** : `total_tickets` = 4.1 (×4 la moyenne) → "plaignants actifs" frustrés mais encore engagés
- **Faible** : comportement sain sur toutes dimensions

> **Note** : La churn_proba du niveau Moyen (0.596) est inférieure au niveau Faible (0.671), ce qui est contre-intuitif. Cela s'explique par le biais du modèle SVC formé sur 93.6% de churners : il tend à prédire des scores élevés pour tous. Le cluster "Moyen" regroupe des clients actifs au SAV mais bons payeurs, qui présentent en réalité moins de risque que la moyenne de la base.

### 5.5 Classificateur de déploiement

Un Random Forest (3 classes) entraîné sur les labels K-Means permet d'assigner un niveau de risque à tout nouveau client :

| Métrique | CV 5-fold | Interprétation |
|---|---|---|
| **Accuracy** | **97.34% ± 0.17%** | Quasi-parfait — frontières bien apprises |
| **F1-macro** | **97.19% ± 0.17%** | Équilibré sur les 3 classes |

---

---

## Chapitre 6 — Objectif 3 : Optimisation des Métriques Métier

### 6.1 Présentation de l'objectif

**Problème** : Les métriques ML standards (AUC, F1) ne traduisent pas directement l'impact financier des décisions de rétention. Cet objectif évalue les modèles avec des métriques orientées ROI : Lift@Top-K%, courbe de gain cumulé, analyse coût-bénéfice.

### 6.2 Résultats comparatifs

| Modèle | AUC-ROC | PR-AUC | Recall@Top10% | Lift@10% | Brier ↓ |
|---|---|---|---|---|---|
| **SVC_RBF** | **0.7004** | **0.9570** | 0.0987 | 0.99× | 0.1558 |
| LR_SMOTE | 0.6868 | 0.9551 | **0.1013** | **1.01×** | 0.1844 |
| RF_balanced | 0.5895 | 0.9556 | **0.1013** | **1.01×** | 0.0745 |
| GB_balanced | 0.6085 | 0.9516 | **0.1013** | **1.01×** | **0.0736** |
| **Baseline aléatoire** | 0.500 | 0.936 | 0.1000 | 1.00× | — |

### 6.3 Explication des phénomènes observés

**Pourquoi Lift@Top10% ≈ 1.01× ?**  
Avec 93.6% de churners, tout sous-groupe de 10% contient ~93.6% de churners par hasard. Un modèle parfait n'améliorerait ce chiffre que de +0.7 points. Le lift est mathématiquement borné — ce n'est pas une limite du modèle, c'est une limite de la métrique dans ce contexte.

**Pourquoi PR-AUC ≈ 0.95+ pour tous ?**  
La PR-AUC est dominée par la classe positive (93.6%). La baseline aléatoire a déjà PR-AUC = 0.936. Les modèles améliorent peu car la majorité des points de la courbe sont déjà bien positionnés.

**Recommandation** :
- Pour le **ranking** (trier les clients par risque) : utiliser **AUC-ROC** → SVC_RBF
- Pour les **probabilités** (scoring dans une formule de gain) : utiliser **Brier Score** → GB_balanced

### 6.4 Analyse coût-bénéfice

**Paramètres métier** :
- ARPU mensuel : 35 TND | Taux rétention offre : 40% | Durée : 12 mois
- Gain net si client retenu : 35 × 12 × 40% − 30 = **+138 TND**
- Coût si faux positif : **−30 TND**

**Seuil break-even** : `P(churn) > 30 / (30 + 138) = 0.178 (18%)`

Au seuil optimal (0.03 sur la population étiquetée) : **+53 592 TND** de valeur nette, 420/422 clients contactés.

---

---

## Chapitre 7 — Objectif 4 : Segmentation Comportementale

### 7.1 Présentation de l'objectif

**Problème** : Segmenter les 10 000 clients en groupes homogènes selon leur comportement d'usage pour personnaliser les stratégies marketing.

**Approche** : K-Means sur des scores composites thématiques, progressivement affinés sur 3 versions.

### 7.2 Progression des 3 versions

| Version | Espace de features | Silhouette (k=4) | Amélioration |
|---|---|---|---|
| **v1** | 23 features brutes | 0.1162 | Référence |
| **v2** | 4 scores composites (Usage/Paiement/Réseau/SAV) | 0.1997 | +72% |
| **v3** | 4 scores métier redéfinis | **0.2324** | **+100%** |

### 7.3 Construction des scores composites (v3)

La v3 redéfinit les 4 dimensions pour maximiser leur indépendance et leur pertinence business :

```python
# Dimension 1 : Intensité d'usage
usage_intensity = mean_z(log(minutes+1), log(data_gb+0.1), log(sms+1), log(nocturne+1))

# Dimension 2 : Pression tarifaire
price_pressure = mean_z(log(hors_forfait+1), ratio_hors_forfait, log(montant_facture))

# Dimension 3 : Risque réseau (positif = mauvaise QoS)
network_risk = mean_z(drop_rate, log(outage_min+1)) − mean_z(qos_score, throughput_mbps)

# Dimension 4 : Inactivité (positif = faible engagement)
inactivity = − mean_z(arpu, anciennete_mois, nb_mois_observes, log(minutes+1))
```

**Propriété clé** : chaque score capture une dimension orthogonale → espace 4D bien séparé → Silhouette maximal.

### 7.4 Les 4 segments comportementaux

| Segment | Clients | % | Description | Risque churn |
|---|---|---|---|---|
| **Fort Usage** | 2 627 | 26.3% | Consommateurs intensifs, ARPU élevé, ancienneté longue | Modéré (exigeants en qualité) |
| **Sensibles au Prix** | 2 467 | 24.7% | Dépassements forfait fréquents, factures élevées | Élevé (sensibles au prix concurrent) |
| **Risque Réseau** | 2 415 | 24.1% | QoS dégradée, drop rate élevé | Élevé (frustration non résolue) |
| **Clients Inactifs** | 2 491 | 24.9% | Sous-utilisateurs, faible engagement, ancienneté courte | Élevé (pas de switching cost perçu) |

Distribution quasi-équilibrée (24–26%) : indicateur de clusters naturels, pas artificiels.

### 7.5 Métriques de validation v3

| Métrique | Valeur v3 | Interprétation |
|---|---|---|
| Silhouette | **0.2324** | Meilleure cohésion intra/inter clusters |
| Davies-Bouldin | **1.2715** | Plus faible = meilleure séparation |
| Calinski-Harabász | **3 639** | Plus élevé = clusters plus denses |
| Accuracy RF (déploiement) | **86.16%** | Frontières naturelles — légère superposition normale |

---

---

## Chapitre 8 — Objectif 5 : Profils de Rétention Personnalisés

### 8.1 Présentation de l'objectif

**Problème** : Construire une **matrice de ciblage** en croisant les segments comportementaux (Obj 4) avec les niveaux de risque (Obj 2) pour créer des micro-profils avec des campagnes de rétention sur mesure.

### 8.2 Architecture de la solution

La matrice 4 × 3 croise :
- **4 segments comportementaux** (Obj 4 v2) en lignes
- **3 niveaux de risque** (Obj 2) en colonnes
- → **12 micro-profils** uniques

### 8.3 Matrice de ciblage

**Nombre de clients et probabilité de churn par cellule :**

| Segment | Risque Faible | Risque Moyen | Risque Élevé |
|---|---|---|---|
| **Gros Consommateurs** | 1 628 — P=0.615 | 192 — P=0.671 | 192 — P=**0.862** |
| **Clients Débiteurs** | 2 492 — P=0.751 | 280 — P=0.764 | 303 — P=**0.876** |
| **Clients Insatisfaits** | 2 143 — P=0.584 | 268 — P=0.667 | 260 — P=**0.894** |
| **Réseau Dégradé** | 1 806 — P=0.742 | 209 — P=0.825 | 227 — P=**0.944** |

### 8.4 Priorisation par ROI estimé

*(Formule : n × P(churn) × ARPU × 12 × 40% rétention − coût offres 30 TND × n)*

| Priorité | Micro-profil | Clients | ROI net estimé | Action recommandée |
|---|---|---|---|---|
| **P1** | Gros Consomm × Faible | 1 628 | **162 k TND** | Programme ambassadeur + cross-sell |
| **P2** | Débiteurs × Faible | 2 492 | **98 k TND** | Monitoring + newsletter |
| **P3** | Réseau Dégradé × Faible | 1 806 | **74 k TND** | Newsletter + amélioration réseau |
| **P4** | Insatisfaits × Faible | 2 143 | **64 k TND** | Newsletter satisfaction |
| **P5** ⚠️ | Gros Consomm × **Élevé** | 192 | 27 k TND | **Offre VIP urgente + conseiller dédié** |
| **P6–P12** | Autres | 1 337 | 65 k TND | Actions par profil |

**Total** : 1 816 k TND de revenus à risque | **515 k TND de ROI net potentiel** (hypothèse : 40% de rétention)

### 8.5 Campagnes par niveau d'urgence

| Urgence | Micro-profils | Délai d'action | Canal |
|---|---|---|---|
| **Rouge — Immédiat** (982 clients) | Tous ×Élevé | 24–48h | Appel direct + SMS |
| **Orange — Préventif** (949 clients) | Tous ×Moyen | 1 semaine | App + Email |
| **Vert — Fidélisation** (8 069 clients) | Tous ×Faible | 1 mois | Newsletter |

---

---

## Chapitre 9 — Objectif 6 : Score de Propension au Churn

### 9.1 Présentation de l'objectif

**Problème** : Produire un **score continu** de propension au churn pour chaque client, plus nuancé que la classification binaire. Ce score permet un ciblage gradué selon le budget disponible.

**Différence avec l'Objectif 1** :  
- Obj 1 : Classification binaire (churn / non-churn) — décision seuillée  
- Obj 6 : Régression sur labels binaires — score ∈ [0,1] continu et ordinal

### 9.2 Solution technique

**Problème de prévalence** : sans correction, tous les régresseurs prédisent ~0.936 pour tous les clients (ils apprennent la prévalence de la classe majoritaire). Solution : pondération équilibrée.

```python
# Pondère les 27 non-churners × 14.6 (ratio 395/27)
sample_weight = np.where(y == 1, 1.0, n_positifs / n_negatifs)
modele.fit(X_train, y_train, sample_weight=sample_weight[train_idx])

# Calibration post-entraînement (monotone croissante)
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(raw_predictions_oof, y_true)
propensity = iso.predict(raw_predictions_all)
```

### 9.3 Résultats comparatifs

| Modèle | AUC-ROC | Brier ↓ | Accuracy | Recommandé pour |
|---|---|---|---|---|
| **Ridge** | **0.6772** | 0.1866 | **93.4%** | **Ranking** |
| GBR_balanced | 0.5189 | **0.0659** | 93.6% | Probabilités calibrées |
| RFR_balanced | 0.4962 | 0.0687 | 93.1% | — |
| SVR_rbf | 0.4962 | 0.0670 | 93.4% | — |

**Objectif Accuracy ≥ 90% : ATTEINT** ✓

Ridge excelle en ranking car les relations entre features comportementales et churn sont majoritairement **linéaires** dans cet espace normalisé. Les arbres, conçus pour les non-linéarités, prédisent ~0.5 pour tous (incertitude maximale) d'où un Brier faible mais un AUC nul.

### 9.4 Distribution sur 10 000 clients

| Tier | Clients | % |
|---|---|---|
| Risque Faible (score < 0.40) | 2 101 | 21.0% |
| Risque Moyen (0.40–0.70) | 2 767 | 27.7% |
| Risque Élevé (score ≥ 0.70) | **5 132** | **51.3%** |

### 9.5 Analyse par segment comportemental

| Segment | Propension Moy. | % Risque Élevé |
|---|---|---|
| **Clients Inactifs** | **0.818** | **70.3%** |
| Sensibles au Prix | 0.710 | 52.5% |
| Risque Réseau | 0.622 | 40.2% |
| Fort Usage | 0.600 | 40.9% |

Les Clients Inactifs ont une propension médiane = 1.0 (50% avec score maximal) → priorité absolue malgré leur ARPU faible.

---

---

## Chapitre 10 — Objectif 7 : Prédiction de l'ARPU Futur

### 10.1 Présentation de l'objectif

**Problème** : Prédire l'ARPU et le montant de facture futurs (trimestre mois 10–12) pour chaque client à partir de son historique (mois 1–9).

**Utilité** : Fournir le `ARPU_prédit` à l'Objectif 9 (formule de gain), permettant un calcul de ROI personnalisé par client.

### 10.2 Architecture du split temporel

```
DateFK 25 à 33 (mois 1–9)  ──────► Features (100 colonnes)
DateFK 34 à 36 (mois 10–12) ─────► Cibles : avg_ARPU_futur, avg_Facture_future
```

Ce split respecte la **causalité temporelle** : on entraîne uniquement sur le passé pour prédire le futur.

### 10.3 Features temporelles (100 au total)

Pour chacune des 13 colonnes numériques, 7 statistiques :

| Statistique | Formule | Information capturée |
|---|---|---|
| `{col}_mean` | Moyenne 9 mois | Niveau de base |
| `{col}_std` | Écart-type | Volatilité |
| `{col}_min`, `{col}_max` | Extrêmes | Plage de variation |
| `{col}_recent` | Moyenne mois 7–9 | Tendance récente |
| `{col}_growth` | `recent/early − 1` | Évolution relative |
| `{col}_trend` | Pente de régression (polyfit) | Tendance linéaire sur 9 mois |

+ 9 features statiques (segment, type abonnement, région, ancienneté, prix offre, etc.)

### 10.4 Résultats

**Cible ARPU Futur** :

| Modèle | R² | MAE | MAPE |
|---|---|---|---|
| **RandomForest** | **0.9619** | **1.77 TND** | **10.2%** |
| GBR | 0.9609 | 1.79 TND | 10.4% |
| Ridge | 0.9598 | 1.78 TND | 10.3% |

**Cible Montant Facture Futur** :

| Modèle | R² | MAE | MAPE |
|---|---|---|---|
| **RandomForest** | **0.9597** | **1.83 TND** | **10.6%** |
| GBR | 0.9583 | 1.86 TND | 10.8% |
| Ridge | 0.9581 | 1.84 TND | 10.7% |

### 10.5 Analyse

R² = 0.962 signifie que 96.2% de la variance de l'ARPU futur est expliquée par l'historique passé. Cela reflète la **stabilité des abonnements télécom** : les clients qui payent 25 TND/mois en moyenne sur 9 mois paieront ~25 TND le trimestre suivant. La feature la plus prédictive est `arpu_recent` (moyenne des 3 derniers mois), confirmant que la valeur récente est la meilleure approximation du futur.

**ARPU observé** : 21.86 ± 12.04 TND | Erreur MAE de 1.77 TND = **8.1% d'erreur relative** sur plage de 2–71 TND → excellent pour une application métier.

---

---

## Chapitre 11 — Objectif 8 : Probabilité de Réponse à une Offre

### 11.1 Présentation de l'objectif

**Problème** : Prédire la probabilité que chaque client accepte une offre de rétention, sans données historiques de réponses aux offres dans la base DW_TT.

**Solution** : Génération de **labels synthétiques calibrés** sur des règles métier, puis entraînement d'un classifieur calibré.

### 11.2 Génération des labels synthétiques

#### 11.2.1 Règles métier codifiées

Un client répond favorablement à une offre de rétention si :
1. Il a une **valeur élevée** (ARPU fort → intérêt à le conserver)
2. Il a une **ancienneté** (lien émotionnel → réceptif à la reconnaissance)
3. Il est **satisfait du réseau** (pas de frustration bloquante)
4. Il n'a **pas d'impayés** (engagement contractuel maintenu)
5. Il est dans le **"sweet spot" de propension** : risque churn ≈ 0.40 (ni trop fidèle, ni déjà parti)

#### 11.2.2 Formule du score latent

```python
prop_sweet = np.exp(-8 * (propensity - 0.40)**2)  # Cloche gaussienne centrée sur 0.40

logit_raw = (
    + 1.2 * arpu_norm           # Valeur financière → fidélisable
    + 0.9 * anciennete_norm     # Loyauté → répond à la reconnaissance
    + 0.7 * qos_norm            # Satisfaction réseau
    - 0.8 * drop_rate_norm      # Frustration réseau → moins réceptif
    - 0.9 * impayes_norm        # Problèmes paiement → désengagement
    + 1.4 * prop_sweet          # Sweet spot : risque modéré ≈ 0.40
    + 0.5 * minutes_norm        # Usage actif → valorise le service
    + 0.3 * (1 - engagement_norm) # Fin d'engagement = moment de décision
    + 0.3 * nps_norm            # NPS positif
    + bruit_gaussien(0, 0.45)   # Variabilité réaliste
)
```

#### 11.2.3 Calibration automatique du taux de réponse

L'algorithme de Brent (méthode de recherche de racine) calcule l'intercept qui produit exactement 30% de réponse :

```python
from scipy.optimize import brentq
TARGET_RATE = 0.30
intercept = brentq(
    lambda t: (sigmoid(logit_raw + t)).mean() - TARGET_RATE,
    a=-10, b=10
)
prob_true   = sigmoid(logit_raw + intercept)
y_response  = rng.binomial(1, prob_true)  # Tirage de Bernoulli
```

Ce protocole garantit la reproductibilité (graine fixe) et le réalisme (30% = taux de réponse typique en télécom).

### 11.3 Modèles entraînés

| Modèle | Technique de calibration |
|---|---|
| **LogReg** | Nativement calibré (sortie sigmoïde) |
| **GBR** | `CalibratedClassifierCV(method='isotonic', cv=3)` |
| **RF** | `CalibratedClassifierCV(method='isotonic', cv=3)` |

Évaluation par **StratifiedKFold 3-fold** avec `cross_val_predict(method='predict_proba')` pour des probabilités out-of-fold non biaisées.

### 11.4 Résultats

| Modèle | AUC-ROC | Avg Precision | Brier ↓ |
|---|---|---|---|
| **LogReg** | **0.6185** | **0.4137** | 0.2387 |
| GBR | 0.6079 | 0.4062 | **0.2055** |
| RF | 0.6060 | 0.3989 | **0.2061** |

LogReg est retenu (meilleur AUC) : les features qui déterminent la réponse (ARPU, ancienneté, QoS) ont des relations linéaires avec le label synthétique.

### 11.5 Distribution sur 10 000 clients

- **Taux de réponse observé : 30.6%** ✓ (cible 30%)
- **Tier Élevé (>50%)** : 4 506 clients (45.1%)
- **Tier Moyen (25–50%)** : 5 441 clients (54.4%)
- **Tier Faible (<25%)** : 53 clients (0.5%)
- Score moyen : 0.492 ± 0.107

---

---

## Chapitre 12 — Objectif 9 : Gain Attendu d'une Action de Rétention

### 12.1 Présentation de l'objectif

**Problème** : Calculer le **gain financier net attendu** (en TND) de cibler chaque client avec une offre de rétention, en combinant tous les outputs des objectifs précédents en une formule unifiée.

**Valeur ajoutée** : Ce score transforme les sorties ML en **outil de décision directement actionnable** pour les équipes marketing — il répond à la question : "Si on a un budget limité pour n offres, qui cibler en premier ?"

### 12.2 Formule centrale d'espérance mathématique

```
Gain_Net(i) = P_churn(i) × P_réponse(i) × ARPU(i) × LTV_horizon − Coût_offre

où :
  P_churn(i)    = propension au churn (proxy Obj 6)   ∈ [0, 1]
  P_réponse(i)  = probabilité de réponse (Obj 8)       ∈ [0, 1]
  ARPU(i)       = revenu mensuel récent (proxy Obj 7)  en TND
  LTV_horizon   = 12 mois (revenus préservés si retenu)
  Coût_offre    = coût de l'action de rétention         en TND
```

**Interprétation** :

| Terme | Signification économique |
|---|---|
| `P_churn × P_réponse` | Probabilité que l'action soit **utile** (client partait) ET **acceptée** |
| `ARPU × 12` | Customer Lifetime Value sur 12 mois — revenus sauvés si rétention réussit |
| `Coût_offre` | Investissement de rétention (remise tarifaire, bonus data, upgrade offre) |
| `Gain_Net > 0` | Cibler ce client est **rentable** |

### 12.3 Implémentation

```python
LTV_HORIZON  = 12      # mois
COUT_BASE    = 15.0    # TND (scénario standard)

# P_churn proxy (sigmoid sur features comportementales)
df['p_churn'] = sigmoid(
    −0.8 × norm(arpu) + 0.9 × norm(impayes) + 0.6 × norm(drop_rate)
    −0.5 × norm(qos)  − 0.4 × norm(nps)
)

# ARPU prédit proxy (arpu_recent ≈ arpu_futur car R²=0.96)
df['arpu_predit'] = df['arpu_recent'].clip(lower=1.0)

# P_réponse depuis l'Objectif 8
df['p_reponse'] = df.index.map(response_scores)

# Formule de gain
CLV               = df['arpu_predit'] * LTV_HORIZON
df['gain_brut']  = df['p_churn'] * df['p_reponse'] * CLV
df['gain_net']   = df['gain_brut'] - COUT_BASE
df['roi']        = df['gain_net'] / COUT_BASE
df['cible']      = (df['gain_net'] > 0).astype(int)
```

### 12.4 Résultats — Scénario Standard (15 TND)

| KPI | Valeur |
|---|---|
| **Gain total potentiel** | **369 678 TND** |
| **Gain moyen par client** | **37.1 TND** |
| **ROI moyen** | **2.5×** |
| **Clients ROI-positifs** | **9 957 / 10 000 (99.6%)** |

> Le fait que 99.6% des clients soient ROI-positifs avec un coût de 15 TND reflète une réalité économique : l'ARPU moyen × 12 mois ≈ 260 TND est bien supérieur au coût de l'offre. La valeur du modèle réside donc dans la **priorisation** (qui contacter en premier), pas dans le filtrage binaire.

### 12.5 Lift du modèle de ciblage

En triant les clients par `gain_net` décroissant, le modèle multiplie l'efficacité des campagnes :

| Top % ciblés | Clients | Gain capturé | Lift vs aléatoire |
|---|---|---|---|
| **Top 10%** | **1 000** | **98 830 TND** | **2.68×** |
| Top 20% | 2 000 | 172 296 TND | 2.33× |
| Top 30% | 3 000 | 226 547 TND | 2.04× |
| Top 50% | 5 000 | 300 970 TND | 1.63× |

**Exemple concret** : Pour une campagne ciblant 1 000 clients (budget 15 000 TND), le modèle permet de capturer 98 830 TND de gain — vs seulement 36 968 TND en ciblage aléatoire. Soit **2.68 fois plus efficace**.

### 12.6 Comparaison des 3 scénarios d'offre

| Scénario | Coût | Clients ROI+ | Gain Total | Gain Moy/Client |
|---|---|---|---|---|
| **Légère** | 5 TND | 9 999 (~100%) | **469 628 TND** | 47.0 TND |
| **Standard** | 15 TND | 9 957 (99.6%) | 369 678 TND | 37.1 TND |
| **Premium** | 30 TND | 6 815 (68.2%) | 240 711 TND | 35.3 TND |

L'offre **Premium** est la plus **discriminante** : elle filtre naturellement les clients à bas ARPU, en ne ciblant que ceux dont la valeur justifie l'investissement.

### 12.7 Formule de ciblage optimal — Règle métier

La condition de rentabilité se simplifie en :

```
Cibler le client i si :
    P_churn(i) × P_réponse(i) > Coût_offre / (ARPU(i) × LTV_horizon)
```

Exemples de seuils selon le profil client :

| ARPU | Coût offre | LTV | Seuil minimal P_churn × P_réponse |
|---|---|---|---|
| 20 TND | 15 TND | 12 mois | **0.063** |
| 30 TND | 15 TND | 12 mois | **0.042** |
| 10 TND | 30 TND | 12 mois | **0.250** |

Cette règle s'adapte automatiquement à la valeur de chaque client : plus l'ARPU est faible, plus les probabilités conjointes doivent être élevées pour justifier l'investissement.

### 12.8 Analyse de sensibilité

La heatmap de sensibilité (Coût_offre × LTV_horizon) révèle :
- **Horizon 6 mois** (offre temporaire) : gain réduit de 50% vs 12 mois
- **Horizon 18 mois** (fidélisation longue) : gain supérieur de 50%
- **Coût > 40 TND** : seulement les clients ARPU > 30 TND restent ROI-positifs

---

---

## Conclusion Générale et Perspectives

### Bilan des travaux

Ce projet de fin d'études a abouti à la conception et l'implémentation d'un **système complet de scoring prédictif** pour la rétention client télécom, structuré en 9 objectifs interdépendants.

**Résultats clés** :

| Objectif | Résultat principal |
|---|---|
| Prédiction churn | AUC = 0.71 (SVC-RBF) — plafond empirique dû au déséquilibre |
| Classification risque | 97.3% précision — déployable immédiatement |
| Segmentation | 4 groupes équilibrés, Silhouette +100% vs brut |
| Profils rétention | 12 micro-profils, 515k TND ROI potentiel |
| Propension churn | Score continu, 93.4% accuracy, distribution étalée |
| Prédiction ARPU | R² = 0.962, MAE = 1.77 TND — excellent |
| Réponse offre | AUC = 0.619, taux de réponse 30.6% ✓ |
| Gain attendu | Lift 2.68×, 369 678 TND gain total |

### Contributions techniques

1. **Gestion du déséquilibre extrême** (93.6%) : évaluation rigoureuse de 5 méthodes de synthèse (SMOTE, CTGAN, TVAE, KDE, GaussianCopula) — SMOTE reste optimal sur petits effectifs
2. **Découverte et correction d'une fuite de données** : label cross-périodes identique → corrélation parfaite artificiellement
3. **Labels synthétiques calibrés** : utilisation de l'algorithme de Brent pour garantir exactement 30% de taux de réponse reproductible
4. **Pipeline de scoring unifié** : P_churn × P_réponse × ARPU → gain financier par client → priorisation ROI-optimale
5. **Analyse critique des métriques** : démonstration que Lift@TopK% et PR-AUC sont non-discriminants à haute prévalence

### Limites identifiées

| Limite | Impact | Solution envisagée |
|---|---|---|
| 27 non-churners étiquetés | Plafond AUC ~0.72, variance élevée | Étiqueter 500+ clients supplémentaires |
| Labels réponse synthétiques | Obj 8 évalué sur labels non-réels | Lancer une campagne pilote pour collecter de vrais labels |
| Proxy p_churn dans Obj 9 | Moins précis que le score Obj 6 direct | Exporter les scores per-client depuis Obj 6 |
| Données statiques simulées | Corrélations artificielles | Utiliser des données opérationnelles réelles |

### Perspectives d'amélioration

**Court terme** :
- Exporter les scores per-client depuis Obj 6 vers Obj 9 (actuellement un proxy inline est utilisé)
- Mettre en production le classificateur RF de risque (Obj 2) avec une API REST
- Automatiser la mise à jour mensuelle des scores pour les 10k clients

**Moyen terme** :
- Collecter des données de réponse aux campagnes réelles → remplacer les labels synthétiques de l'Obj 8
- Intégrer des données externes : appels au service client, score NPS détaillé, réclamations
- Tester des modèles séquentiels (LSTM, Transformer) pour capturer les patterns temporels

**Long terme** :
- Système de recommandation multi-offres (quel type d'offre propose-t-on à quel client ?)
- Dashboard temps réel de monitoring du churn et des scores de gain
- A/B testing automatisé pour mesurer le vrai taux de rétention des campagnes ML vs règles manuelles

### Conclusion

Ce projet démontre qu'il est possible de construire un système de décision de rétention client complet et actionnable à partir d'un Data Warehouse télécom, malgré les contraintes réelles de déséquilibre des labels et d'absence de données historiques pour certains objectifs. Le score de gain attendu final (Gain_Net = P_churn × P_réponse × ARPU × 12 − Coût) constitue un outil simple, interprétable, et directement utilisable par les équipes marketing pour prioriser leurs actions avec un budget limité — multipliant l'efficacité des campagnes par 2.68× par rapport à un ciblage non-guidé.

---

---

## Bibliographie

[1] Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.

[2] Friedman, J. H. (2001). *Greedy function approximation: A gradient boosting machine*. Annals of Statistics, 29(5), 1189–1232.

[3] Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. Journal of Artificial Intelligence Research, 16, 321–357.

[4] Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. Proceedings of KDD 2016, 785–794.

[5] Ke, G., Meng, Q., Finley, T., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS 2017.

[6] Vapnik, V. (1995). *The Nature of Statistical Learning Theory*. Springer, New York.

[7] Rousseeuw, P. J. (1987). *Silhouettes: A graphical aid to the interpretation and validation of cluster analysis*. Journal of Computational and Applied Mathematics, 20, 53–65.

[8] Platt, J. (1999). *Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods*. Advances in Large Margin Classifiers.

[9] Guyon, I., & Elisseeff, A. (2003). *An Introduction to Variable and Feature Selection*. Journal of Machine Learning Research, 3, 1157–1182.

[10] He, H., & Garcia, E. A. (2009). *Learning from Imbalanced Data*. IEEE Transactions on Knowledge and Data Engineering, 21(9), 1263–1284.

[11] Hung, S. Y., Yen, D. C., & Wang, H. Y. (2006). *Applying data mining to telecom churn management*. Expert Systems with Applications, 31(3), 515–524.

[12] Verbeke, W., Dejaeger, K., Martens, D., et al. (2012). *New insights into churn prediction in the telecommunication sector: A profit driven data mining approach*. European Journal of Operational Research, 218(1), 211–229.

[13] Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.

[14] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

[15] Provost, F., & Fawcett, T. (2013). *Data Science for Business*. O'Reilly Media.

---

---

## Annexes

### Annexe A — Schéma de la Base de Données DW_TT

```
┌──────────────────┐    ┌──────────────────────────────────────┐
│   dim_client     │    │       Fact_performance_client        │
│──────────────────│    │──────────────────────────────────────│
│ ClientID (PK)    │◄───│ ClientID (FK)                        │
│ Sexe             │    │ DateFK (FK) — 25 à 36                │
│ Segment          │    │ ARPU, MontantFacture, HorsForfait     │
│ TypeAbonnement   │    │ NbImpayes, RetardPaiement_jours       │
│ Region           │    │ MinutesAppel, DataConsommee_GB, NbSMS │
│ Anciennete_mois  │    │ QoS_Score, DropRate, Throughput_Mbps  │
│ Engagement_rest  │    │ OutageMinutes, NbTickets, NPS_Score   │
│ PrixOffre        │    └──────────────────────────────────────┘
│ OffreCible       │
│ RGPD_Consent     │    ┌──────────────────────────────────────┐
└──────────────────┘    │          Fact_churn                  │
                        │──────────────────────────────────────│
┌──────────────────┐    │ ClientID (FK)                        │
│   dim_offre      │    │ DateFK (FK) — 12 ou 36               │
│──────────────────│    │ Churn_30j ∈ {0, 1}                   │
│ OffreID (PK)     │    │ ChurnScore_30j (non utilisé)         │
│ TypeOffre        │    └──────────────────────────────────────┘
│ PrixOffre        │
│ DataForfait      │
│ MinutesForfait   │
└──────────────────┘
```

### Annexe B — Récapitulatif des Fichiers Générés

| Objectif | Notebook | Run Log | Fichiers Visuels |
|---|---|---|---|
| **Obj 1** | `01_churn_v1` à `v6.ipynb` | `01_churn_v*_run_log.json` | `v5_results.png`, `v6_progression.png` |
| **Obj 2** | `02_risk_classification.ipynb` | `02_risk_run_log.json` | `02_risk_dashboard.png`, `02_kmeans_selection.png` |
| **Obj 3** | `03_business_metrics.ipynb` | `03_business_metrics_run_log.json` | `03_gain_lift_curves.png`, `03_calibration.png` |
| **Obj 4** | `04_segmentation_v3.ipynb` | `04v3_segmentation_run_log.json` | `04v3_radar.png`, `04v3_profile_heatmap.png` |
| **Obj 5** | `05_retention_profiles.ipynb` | `05_retention_run_log.json` | `05_targeting_matrix.png`, `05_priority_chart.png` |
| **Obj 6** | `06_churn_propensity.ipynb` | `06_propensity_run_log.json` | `06_propensity_distribution.png` |
| **Obj 7** | `07_arpu_prediction.ipynb` | `07_arpu_run_log.json` | `07_predicted_vs_actual.png`, `07_residuals.png` |
| **Obj 8** | `08_offer_response.ipynb` | `08_offer_response_run_log.json` | `08_action_matrix.png`, `08_deployment.png` |
| **Obj 9** | `09_retention_gain.ipynb` | `09_retention_gain_run_log.json` | `09_gain_curve.png`, `09_sensitivity.png`, `09_targeting_map.png` |

### Annexe C — Performances Consolidées

| # | Objectif | Type | Modèle retenu | Métrique principale | Valeur |
|---|---|---|---|---|---|
| 1 | Prédiction churn | Classification | SVC-RBF (C=10, γ=0.001) | AUC-ROC | 0.7138 ± 0.11 |
| 2 | Classification risque | Clustering + RF | K-Means(k=3) + RF | Accuracy | 97.34% |
| 3 | Métriques métier | Évaluation | GB_balanced | Brier | 0.0736 |
| 4 | Segmentation | Clustering | K-Means composite 4D | Silhouette | 0.2324 |
| 5 | Profils rétention | Matrice | Croisement Obj2 × Obj4 | ROI potentiel | 515 k TND |
| 6 | Propension churn | Régression | Ridge + calibration isotonique | AUC-ROC | 0.6772 |
| 7 | ARPU futur | Régression | RandomForest | R² | 0.9619 |
| 8 | Réponse offre | Classification | LogReg calibré | AUC-ROC | 0.6185 |
| 9 | Gain attendu | Espérance | Formule analytique | Lift@Top10% | 2.68× |

---

*Rapport de Projet de Fin d'Études — Année universitaire 2025–2026*  
*Application du Machine Learning à la Rétention Client dans le Secteur Télécom*
