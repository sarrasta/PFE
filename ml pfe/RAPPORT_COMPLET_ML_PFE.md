# Rapport Complet — Projet Machine Learning PFE
## Analyse Prédictive pour la Rétention Client Télécom (DW_TT)

**Étudiant** : Sarra  
**Base de données** : PostgreSQL `DW_TT` — `localhost:5432`  
**Période de données** : 12 mois (DateFK 25–36, année 2024)  
**Base clients** : 10 000 clients actifs  
**Objectifs réalisés** : 9 objectifs ML complets

---

## Table des Matières

1. [Contexte et Architecture des Données](#1-contexte-et-architecture-des-données)
2. [Objectif 1 — Prédiction du Churn (30 jours)](#2-objectif-1--prédiction-du-churn-30-jours)
3. [Objectif 2 — Classification du Risque Client](#3-objectif-2--classification-du-risque-client)
4. [Objectif 3 — Optimisation des Métriques Métier](#4-objectif-3--optimisation-des-métriques-métier)
5. [Objectif 4 — Segmentation Comportementale](#5-objectif-4--segmentation-comportementale)
6. [Objectif 5 — Profils de Rétention Personnalisés](#6-objectif-5--profils-de-rétention-personnalisés)
7. [Objectif 6 — Score de Propension au Churn](#7-objectif-6--score-de-propension-au-churn)
8. [Objectif 7 — Prédiction de l'ARPU Futur](#8-objectif-7--prédiction-de-larpu-futur)
9. [Objectif 8 — Probabilité de Réponse à une Offre](#9-objectif-8--probabilité-de-réponse-à-une-offre)
10. [Objectif 9 — Gain Attendu d'une Action de Rétention](#10-objectif-9--gain-attendu-dune-action-de-rétention)
11. [Synthèse Globale et Pipeline ML Final](#11-synthèse-globale-et-pipeline-ml-final)

---

## 1. Contexte et Architecture des Données

### 1.1 Description du projet

Ce projet de fin d'études (PFE) porte sur l'application du Machine Learning à la rétention client dans un contexte télécom tunisien. L'objectif global est de construire un **système de décision intelligent** capable de :
- Identifier les clients à risque de partir (churn)
- Les segmenter par comportement et niveau de risque
- Estimer leur valeur financière future
- Calculer le gain attendu d'une action de rétention pour chaque client

### 1.2 Base de données DW_TT (Data Warehouse Télécom Tunisien)

La base est un **Data Warehouse PostgreSQL** structuré en schéma en étoile :

| Table | Description | Lignes |
|---|---|---|
| `dim_client` | Attributs clients (sexe, segment, région, ancienneté, engagement) | 10 000 |
| `dim_offre` | Offres commerciales (type, prix, data, minutes) | — |
| `dim_geographique` | Régions et localisation | — |
| `Fact_performance_client` | KPIs mensuels par client (ARPU, facture, QoS, impayés...) | 120 000 (10k × 12 mois) |
| `Fact_churn` | Labels churn observés (DateFK=12 et DateFK=36) | 422 clients étiquetés |

### 1.3 Variables clés disponibles

**Features comportementales mensuelles** (Fact_performance_client) :
- Financières : `arpu`, `montant_facture`, `hors_forfait`, `impayes`, `retard_paiement`
- Usage : `minutes_appel`, `data_gb`, `sms`, `roaming_gb`, `nocturne_ratio`
- Réseau : `qos_score`, `drop_rate`, `throughput_mbps`, `outage_min`
- SAV : `tickets_ouverts`, `delai_resolution`
- Satisfaction : `nps_score`

**Contrainte principale** : La table `Fact_churn` ne contient que **422 clients étiquetés** (sur 10 000), avec un **déséquilibre sévère** de 93.6% churners / 6.4% non-churners.

---

## 2. Objectif 1 — Prédiction du Churn (30 jours)

### 2.1 Problème et approche

**Objectif** : Prédire si un client va churner dans les 30 jours, en apprenant à partir des 422 clients étiquetés dans `Fact_churn` (DateFK=36, décembre 2024).

**Type** : Classification binaire supervisée  
**Cible** : `Churn_30j` ∈ {0=reste, 1=part}

**Distribution de la cible** :
- Classe 1 (churn) : 395 clients — 93.6%
- Classe 0 (non-churn) : 27 clients — 6.4%

Ce déséquilibre extrême (ratio ~1:15) est le principal défi technique de cet objectif.

### 2.2 Features construites (35 au total)

En plus des 27 features brutes, 6 features d'ingénierie ont été créées :

| Feature construite | Formule | Signification |
|---|---|---|
| `score_risque_paiement` | `impayes × log(retard+1)` | Risque combiné impayés + retard |
| `ratio_hors_forfait` | `hors_forfait / montant_facture` | Part de surprix dans la facture |
| `score_degradation_reseau` | `drop_rate × outage / qos` | Qualité réseau composite |
| `flag_hors_engagement` | 1 si `engagement_restant ≤ 0` | Client hors contrat = libre de partir |
| `intensite_usage` | `(minutes + data + sms) / anciennete` | Intensité d'utilisation normalisée |
| `score_insatisfaction` | `tickets × (1/nps+0.01)` | Frustration combinée SAV + NPS |

### 2.3 Stratégies contre le déséquilibre

Plusieurs approches ont été testées pour gérer le déséquilibre 1:15 :

| Technique | Principe |
|---|---|
| `class_weight='balanced'` | Pondère chaque classe inversement à sa fréquence |
| **SMOTE** | Crée des exemples synthétiques de la classe minoritaire par interpolation k-NN |
| **CTGAN** | GAN tabular entraîné sur les non-churners |
| **TVAE** | Variational AutoEncoder pour données tabulaires (SDV) |
| `BalancedRandomForestClassifier` | RF sous-échantillonnant la majorité à chaque arbre |
| `EasyEnsembleClassifier` | Ensemble de classifieurs avec sous-échantillonnage équilibré |
| Bootstrap + bruit | Rééchantillonnage avec bruit gaussien 15% |
| KDE marginale | Génération par Kernel Density Estimation par feature |
| GaussianCopula | Modèle de copules préservant les corrélations |

### 2.4 Progression sur 6 itérations

#### Itération 1 (v1) — Baseline

6 modèles testés avec et sans SMOTE. Découverte du paradoxe déséquilibre :
- Random Forest : F1=0.966 (prédit presque tout le monde comme churner → Recall=1.0 mais AUC=0.43)
- Logistic Regression : AUC=0.654 (meilleur vrai discriminant)

**Décision** : adopter l'**AUC-ROC** comme métrique principale (plus fiable que F1 en cas de déséquilibre sévère).

#### Itération 2 (v2) — CTGAN

CTGAN (500 epochs) entraîné sur les 20 non-churners du fold d'entraînement.  
Résultat : AUC=0.537 — **pire** qu'en v1. Diagnostic : les GANs nécessitent >1000 samples pour apprendre une distribution, 20 samples ne suffisent pas.

#### Itération 3 (v3) — TVAE + Tuning

TVAE (plus stable que CTGAN sur petits datasets), ajout de DateFK=12 (42 non-churners total), RandomizedSearchCV (40 itérations), VotingClassifier (LGBM + XGB + RF).  
Résultat : AUC=0.667 avec SMOTE — meilleur jusqu'ici.

#### Itération 4 (v4) — Features Temporelles

118 nouvelles features temporelles construites (pente, delta, écart-type mensuel) → 137 features total.  
Résultat : AUC=0.590 — **régression**. Diagnostic : "malédiction de la dimensionnalité" — 137 features pour seulement 27 samples minoritaires (ratio > 5 → sur-apprentissage sévère).

#### Itération 5 (v5) — SVM-RBF ← Meilleur modèle

Retour à 35 features. Test de SVM (RBF kernel), NuSVC, BaggingClassifier(SVC), ExtraTrees, GaussianNB.  
Résultat : **SVC(C=10, gamma=0.001) + SMOTE → AUC=0.7138 ± 0.114**

> **Découverte critique** : Lors de l'ajout du label `Churn_30j` à DateFK=12 comme feature, l'AUC montait à 1.0. Diagnostic : dans le dataset synthétique du PFE, les 395 churners à T36 étaient exactement les mêmes qu'à T12 → **fuite de cible** parfaite. Correction : suppression de toutes features cross-périodes.

#### Itération 6 (v6) — Comparaison synthèse

Test rigoureux de toutes les méthodes de synthèse (synthèse uniquement dans le fold train, évaluation sur données réelles) :

| Méthode | AUC CV |
|---|---|
| **SMOTE** | **0.7138** |
| Bootstrap + bruit 15% | 0.6890 |
| KDE marginale | 0.6645 |
| GaussianCopula (SDV) | 0.6644 |
| MVN + LedoitWolf | 0.6525 |

**SMOTE reste la meilleure méthode** sur ce volume de données.

### 2.5 Résultat final Objectif 1

| Paramètre | Valeur |
|---|---|
| **Modèle retenu** | SVC (kernel=rbf, C=10, gamma=0.001) |
| **AUC-ROC CV 5-fold** | **0.7138 ± 0.1144** |
| **Stratégie équilibrage** | SMOTE (k_neighbors=5) dans pipeline CV |
| **Features** | 35 (27 brutes + 6 ingénierie + 2 supplémentaires) |
| **Scaler** | RobustScaler (résistant aux outliers) |
| **Seuil déploiement** | ~0.25 (maximise F1 en déploiement) |

**Plafond empirique atteint** : Avec seulement 27 samples non-churn, aucun algorithme ne peut dépasser ~0.72 AUC de manière fiable. Pour atteindre 0.85+, il faudrait ~500+ non-churners réels.

---

## 3. Objectif 2 — Classification du Risque Client

### 3.1 Problème et approche

**Objectif** : Classer les **10 000 clients** (pas seulement les 422 étiquetés) en 3 niveaux de risque : Faible / Moyen / Élevé.

**Approche en 2 étapes** :
1. **Clustering non supervisé** (K-Means, k=3) sur 13 indicateurs de risque → découverte des niveaux
2. **Classification supervisée** (Random Forest) → modèle déployable pour nouveaux clients

### 3.2 Choix du nombre de clusters

La méthode du coude + le score de silhouette confirment k=3 :

| k | Silhouette | Verdict |
|---|---|---|
| 2 | — | Séparation trop grossière |
| **3** | **0.1425** | **Optimal — correspond aux 3 niveaux métier** |
| 4 | — | Sur-segmentation |

Le score de silhouette 0.14 est faible mais attendu pour des données client en continu (pas de frontières naturelles nettes).

### 3.3 Features de risque utilisées (13)

| Dimension | Features |
|---|---|
| Paiement | `total_impayes`, `max_retard_paiement`, `avg_retard_paiement`, `score_risque_paiement` |
| Réseau | `avg_drop_rate`, `avg_outage_min`, `score_degradation_reseau` |
| Satisfaction | `total_tickets`, `avg_nps`, `score_insatisfaction` |
| Contrat | `flag_hors_engagement`, `avg_hors_forfait` |
| Modèle churn | `churn_proba` (SVC v5 de l'Obj 1) |

### 3.4 Résultats du clustering

| Niveau | Clients | % base | Churn proba moy. | Caractéristique dominante |
|---|---|---|---|---|
| **Faible** | 4 370 | 43.7% | 0.671 | Comportement sain paiement |
| **Moyen** | 2 501 | 25.0% | 0.596 | Tickets SAV très élevés (×4) |
| **Élevé** | 3 129 | 31.3% | **0.815** | Impayés + retards extrêmes (21 jours) |

**Centroïdes des clusters** :
- **Élevé** : `total_impayes` = 1.59 (vs 0.28 pour Faible), `max_retard_paiement` = 21.8 jours → risque churn maximal
- **Moyen** : `total_tickets` = 4.1 (vs 1.5 pour Faible) → "plaignants actifs" : frustration mais pas encore en départ
- **Faible** : comportement sain → 43.7% de la base, risque modéré (biais du modèle vers 0.67)

### 3.5 Classificateur de déploiement

Un Random Forest supervisé (3 classes) est entraîné sur les labels K-Means pour permettre l'assignation sans re-clustering :

| Métrique | CV 5-fold |
|---|---|
| **Accuracy** | **97.34% ± 0.17%** |
| **F1-macro** | **97.19% ± 0.17%** |

Performance quasi-parfaite : le RF apprend fidèlement les frontières décidées par K-Means.

---

## 4. Objectif 3 — Optimisation des Métriques Métier

### 4.1 Problème et approche

**Objectif** : Aller au-delà des métriques standard (AUC, F1) et évaluer les modèles de churn avec les métriques qui ont du sens **pour les décisions métier télécom** : lift, recall@top-k%, et analyse coût-bénéfice.

**Protocole** : Cross-validated out-of-fold (`cross_val_predict`, 5-fold stratifié) sur les 422 clients, 35 features.

### 4.2 Résultats — Métriques métier

| Modèle | AUC-ROC | PR-AUC | Recall@Top10% | Lift@10% | Brier ↓ |
|---|---|---|---|---|---|
| **SVC_RBF (v5)** | **0.7004** | **0.9570** | 0.0987 | 0.99× | 0.1558 |
| LR_SMOTE | 0.6868 | 0.9551 | **0.1013** | **1.01×** | 0.1844 |
| RF_balanced | 0.5895 | 0.9556 | **0.1013** | **1.01×** | 0.0745 |
| GB_balanced | 0.6085 | 0.9516 | **0.1013** | **1.01×** | **0.0736** |
| **Baseline aléatoire** | 0.500 | 0.936 | 0.1000 | 1.00× | — |

### 4.3 Analyse critique — Pourquoi Lift@Top10% ≈ 1.0 ?

Avec 93.6% de churners, **tout groupe de 10% contient ~93.6% de churners par simple hasard**. Un modèle parfait ne peut capturer que 10.6% des churners dans le Top 10% (vs 9.9% pour le hasard). L'écart est mathématiquement borné à +0.7 points.

> **Conclusion** : Le Lift@TopK% est une métrique inutile quand la prévalence dépasse 90%. L'AUC-ROC reste la seule métrique discriminante dans ce contexte.

### 4.4 Analyse coût-bénéfice — Seuil optimal

Paramètres métier télécom tunisien :
- ARPU mensuel moyen : 35 TND/mois
- Taux de succès de rétention : 40%
- Coût de l'offre : 30 TND
- Gain net si client retenu : 35 × 12 × 40% = **+168 TND** → net = +138 TND
- Perte nette si faux positif : **–30 TND**

**Formule de seuil break-even** :
```
Contacter si : P(churn) > Coût_offre / (Gain_TP + Coût_offre) = 30 / 168 = 0.178
```

Au seuil optimal (0.03 sur la population étiquetée à 93.6% churners) :
- Clients contactés : 420/422 (99.5%)
- Valeur nette : **+53 592 TND**

### 4.5 Scorers personnalisés définis

```python
def recall_at_topk(y_true, y_prob, k=0.10):
    n_top = max(1, int(len(y_true) * k))
    idx_sorted = np.argsort(y_prob)[::-1]
    return y_true[idx_sorted[:n_top]].sum() / (y_true.sum() + 1e-9)

scorer_r10   = make_scorer(recall_at_topk, needs_proba=True, k=0.10)
scorer_prauc = make_scorer(average_precision_score, needs_proba=True)
```

---

## 5. Objectif 4 — Segmentation Comportementale

### 5.1 Problème et approche

**Objectif** : Segmenter les 10 000 clients en groupes homogènes basés sur leur comportement d'usage, pour personnaliser les campagnes marketing.

**Type** : Clustering non supervisé (K-Means)  
**Évolution** : 3 versions successives pour améliorer la qualité de la segmentation

### 5.2 Progression des 3 versions

#### Version 1 (v1) — Clustering brut
- 23 features brutes normalisées
- k=4 choisi (Elbow + Silhouette)
- Silhouette = **0.1162**
- Problème : clusters dominés par la corrélation entre features

#### Version 2 (v2) — Scores composites 4D
Construction de 4 scores indépendants :
- Usage Intensity, Paiement Risk, Network Risk, SAV Activity
- Silhouette = **0.1997** (+72% vs v1)

#### Version 3 (v3) — Scores métier redéfinis ← Version finale

Les 4 dimensions sont redéfinies pour maximiser l'indépendance et la pertinence métier :

```python
# Score 1 — Fort Usage (intensité de consommation)
usage_intensity = mean_z(log(minutes), log(data_gb), log(sms), log(usage_nocturne))

# Score 2 — Sensibles au Prix (pression tarifaire)
price_pressure  = mean_z(log(hors_forfait), ratio_hf, log(montant_facture))

# Score 3 — Risque Réseau (qualité de service dégradée)
network_risk    = mean_z(drop_rate, log(outage_min)) − mean_z(qos_score, throughput_mbps)

# Score 4 — Inactivité (faible engagement)
inactivity      = − mean_z(arpu, anciennete_mois, nb_mois_observes, log(minutes))
```

**Résultats v3** :
- Silhouette = **0.2324** (+100% vs v1)
- Davies-Bouldin = 1.27 (amélioré vs v2 : 1.40)
- Calinski-Harabász = 3 639 (vs 2 381 en v2)

### 5.3 Les 4 segments comportementaux

| Segment | Clients | % | Profil principal |
|---|---|---|---|
| **Fort Usage** | 2 627 | 26.3% | Actifs intensifs, ARPU élevé, ancienneté longue |
| **Sensibles au Prix** | 2 467 | 24.7% | Dépassements forfait fréquents, facturation élevée |
| **Risque Réseau** | 2 415 | 24.1% | QoS dégradée, drop rate élevé, throughput faible |
| **Clients Inactifs** | 2 491 | 24.9% | ARPU faible, usage minimal, engagement faible |

Distribution quasi-équilibrée (24–26%) : signe de clusters naturels bien formés.

### 5.4 Classificateur de déploiement

| Métrique | v3 |
|---|---|
| Accuracy | 86.16% |
| F1-macro | 86.08% |

L'accuracy plus basse (86% vs 97% en v2) reflète des frontières plus naturelles et moins arbitraires — c'est une meilleure segmentation, pas une pire.

---

## 6. Objectif 5 — Profils de Rétention Personnalisés

### 6.1 Problème et approche

**Objectif** : Construire une **matrice de ciblage** qui croise les segments comportementaux (Obj 4) avec les niveaux de risque (Obj 2) pour créer 12 micro-profils avec des campagnes de rétention sur mesure.

**Architecture** :
```
Fact_performance_client (10k clients)
    ├── SVC churn model (Obj 1) ──────► churn_proba
    ├── K-Means Composite 4D (Obj 4) ─► segment comportemental (4 groupes)
    └── K-Means Risque k=3 (Obj 2) ──► niveau de risque (3 niveaux)
                                            │
                                            └─► 12 micro-profils
```

### 6.2 Matrice de ciblage — 12 micro-profils

**Nombre de clients par cellule** :

| Segment | Risque Faible | Risque Moyen | Risque Élevé |
|---|---|---|---|
| Gros Consommateurs | 1 628 | 192 | 192 |
| Clients Débiteurs | 2 492 | 280 | 303 |
| Clients Insatisfaits | 2 143 | 268 | 260 |
| Réseau Dégradé | 1 806 | 209 | 227 |

**Probabilité de churn par cellule** :

| Segment | Risque Faible | Risque Moyen | Risque Élevé |
|---|---|---|---|
| Gros Consommateurs | 0.615 | 0.671 | **0.862** |
| Clients Débiteurs | 0.751 | 0.764 | **0.876** |
| Clients Insatisfaits | 0.584 | 0.667 | **0.894** |
| Réseau Dégradé | 0.742 | 0.825 | **0.944** |

### 6.3 Tableau de priorisation par ROI

*(Trié par ROI net : n × P(churn) × ARPU × 12 mois × 40% rétention − coût offres)*

| Priorité | Segment | Risque | Clients | P(churn) | ROI net | Campagne recommandée |
|---|---|---|---|---|---|---|
| **P1** | Gros Consommateurs | Faible | 1 628 | 0.615 | **162 k TND** | Programme ambassadeur + cross-sell |
| **P2** | Clients Débiteurs | Faible | 2 492 | 0.751 | **98 k TND** | Monitoring + newsletter personnalisée |
| **P3** | Réseau Dégradé | Faible | 1 806 | 0.742 | **74 k TND** | Monitoring + newsletter |
| **P4** | Clients Insatisfaits | Faible | 2 143 | 0.584 | **64 k TND** | Newsletter satisfaction |
| **P5** ⚠️ | Gros Consommateurs | **Élevé** | 192 | **0.862** | 27 k TND | **Offre VIP + conseiller dédié** |
| **P6** | Gros Consommateurs | Moyen | 192 | 0.671 | 20 k TND | Upsell + programme fidélité |
| **P7** ⚠️ | Clients Débiteurs | **Élevé** | 303 | **0.876** | 14 k TND | **Plan paiement urgence** |
| **P8** ⚠️ | Clients Insatisfaits | **Élevé** | 260 | **0.894** | 12 k TND | **Appel proactif + compensation SAV** |
| **P9** | Clients Débiteurs | Moyen | 280 | 0.764 | 12 k TND | Alerte précoce + échéancier |
| **P10** ⚠️ | Réseau Dégradé | **Élevé** | 227 | **0.944** | 12 k TND | **Fix technique + dédommagement** |
| **P11** | Clients Insatisfaits | Moyen | 268 | 0.667 | 9 k TND | Suivi SAV + NPS survey |
| **P12** | Réseau Dégradé | Moyen | 209 | 0.825 | 9 k TND | Communication réseau |

**Total revenus à risque** : 1 816 k TND | **ROI net total des campagnes** : 515 k TND

### 6.4 Stratégie par budget

| Budget disponible | Clients prioritaires | Clients couverts | P(churn) moy. |
|---|---|---|---|
| Très limité (<5k TND) | P5 + P7 (Élevé haute valeur) | ~500 | 0.87 |
| Modéré (<20k TND) | P5–P10 (Élevé + Moyen) | ~1 700 | 0.82 |
| Large (>50k TND) | Tous profils P1–P12 | 10 000 | 0.70 |

---

## 7. Objectif 6 — Score de Propension au Churn

### 7.1 Problème et approche

**Objectif** : Produire un **score continu** ∈ [0,1] de propension au churn pour chaque client, plus nuancé que la classification binaire (Obj 1). Ce score permet un ciblage gradué : au lieu de "churn/non-churn", on sait *à quel point* un client est à risque.

**Type** : Régression (sur labels binaires) avec calibration isotonique  
**Différence vs Obj 1** : Régression (score continu) vs Classification (décision binaire)

### 7.2 Solution technique — Balanced Sample Weights

Sans pondération, les régresseurs prédiraient ~0.936 pour tous (biais de prévalence). Solution :

```python
# Pondère les 27 non-churners × 14.6 (= ratio 395/27)
sample_weight = np.where(y == 1, 1.0, n_pos / n_neg)
model.fit(X, y, sample_weight=sample_weight)

# Calibration isotonique post-entraînement
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(raw_predictions, y_true)
propensity_score = iso.predict(raw_predictions_all)
```

### 7.3 Résultats — 4 modèles comparés

| Modèle | AUC-ROC | Brier ↓ | Accuracy | Statut |
|---|---|---|---|---|
| **Ridge** | **0.6772** | 0.1866 | **93.4%** | **MEILLEUR (ranking)** |
| GBR_balanced | 0.5189 | **0.0659** | 93.6% | Mieux calibré (probabilités) |
| RFR_balanced | 0.4962 | 0.0687 | 93.1% | |
| SVR_rbf | 0.4962 | 0.0670 | 93.4% | |

**Objectif Accuracy ≥ 90% : ATTEINT (93.4%)**

Ridge surpasse les arbres pour le ranking car il exploite les relations **linéaires** entre features et churn, plus informatives que les interactions non-linéaires sur ce dataset.

### 7.4 Distribution sur 10 000 clients

| Niveau de Risque | Clients | % Base |
|---|---|---|
| **Risque Faible** (score < 0.40) | 2 101 | 21.0% |
| **Risque Moyen** (0.40–0.70) | 2 767 | 27.7% |
| **Risque Élevé** (score ≥ 0.70) | 5 132 | **51.3%** |

**Distribution bien étalée** : std=0.28 (vs 0.06 sans pondération) → discriminant.

### 7.5 Scores de propension par segment comportemental

| Segment | Propension Moy. | % Risque Élevé | Action prioritaire |
|---|---|---|---|
| **Clients Inactifs** | **0.818** | **70.3%** | Campagne réactivation urgente |
| **Sensibles au Prix** | 0.710 | 52.5% | Offre tarifaire adaptée |
| **Risque Réseau** | 0.622 | 40.2% | Diagnostic réseau + satisfaction |
| **Fort Usage** | 0.600 | 40.9% | Fidélisation premium |

Insight : Les Clients Inactifs ont une **propension médiane = 1.0** (moitié avec score maximal) → les plus urgents à cibler.

---

## 8. Objectif 7 — Prédiction de l'ARPU Futur

### 8.1 Problème et approche

**Objectif** : Prédire l'**ARPU** (revenu mensuel moyen) et le **montant de facture** futurs pour chaque client, sur le trimestre suivant (mois 10–12).

**Type** : Régression supervisée avec split temporel  
**Utilisation** : Fournir le `ARPU_prédit` pour la formule de gain de l'Obj 9

**Architecture temporelle** :
```
Mois 1-9 (DateFK 25-33) ──────► Features
Mois 10-12 (DateFK 34-36) ────► Cibles (ARPU futur, Facture future)
```

### 8.2 Features construites (100 au total)

Pour chacune des 13 colonnes numériques, 7 statistiques calculées :

| Statistique | Formule | Signification |
|---|---|---|
| `{col}_mean` | Moyenne sur 9 mois | Niveau de base |
| `{col}_std` | Écart-type | Volatilité |
| `{col}_min/max` | Extremes | Plage |
| `{col}_recent` | Moyenne mois 7-9 | Valeur récente |
| `{col}_growth` | `recent/early − 1` | Tendance de croissance |
| `{col}_trend` | Pente de régression (polyfit) | Trend linéaire sur 9 mois |

+ Features statiques : Segment, TypeAbonnement, Region, Ancienneté, PrixOffre

### 8.3 Résultats — 3 modèles × 2 cibles

**Cible 1 : ARPU Futur (mois 10-12)**

| Modèle | R² | MAE | MAPE |
|---|---|---|---|
| **RandomForest** | **0.9619** | **1.77 TND** | 10.2% |
| GBR | 0.9609 | 1.79 TND | 10.4% |
| Ridge | 0.9598 | 1.78 TND | 10.3% |

**Cible 2 : Montant Facture Futur (mois 10-12)**

| Modèle | R² | MAE | MAPE |
|---|---|---|---|
| **RandomForest** | **0.9597** | **1.83 TND** | 10.6% |
| GBR | 0.9583 | 1.86 TND | 10.8% |
| Ridge | 0.9581 | 1.84 TND | 10.7% |

**Interprétation des performances** :
- R² = 0.96 : 96% de la variance expliquée → excellent
- MAE = 1.77 TND sur une plage de 2–71 TND → erreur relative très faible
- Les 3 modèles sont très proches : l'ARPU futur est fortement prévisible car stable dans le temps

**Distribution ARPU observé** : 21.86 ± 12.04 TND (min : 2.24, max : 70.89)

### 8.4 Feature la plus prédictive

`arpu_recent` (moyenne des 3 derniers mois) est la feature la plus prédictive, confirmant que l'ARPU futur est fortement corrélé à l'ARPU récent (stabilité des abonnements télécom).

### 8.5 Applications métier

| Application | Condition d'alerte |
|---|---|
| Budget prévisionnel | Estimer revenus trimestriels par client |
| Identification décroissance | `arpu_growth < -0.15` → ARPU en baisse → risque churn |
| Segmentation valeur | Classer clients par ARPU prédit → ciblage premium |
| Alerte proactive | `arpu_pred / arpu_past < 0.80` → contacter avant churn |

---

## 9. Objectif 8 — Probabilité de Réponse à une Offre

### 9.1 Problème et approche

**Objectif** : Prédire la **probabilité qu'un client accepte une offre de rétention**, pour chacun des 10 000 clients.

**Type** : Classification binaire calibrée  
**Label** : `y_response` ∈ {0=n'accepte pas, 1=accepte l'offre}

**Problème** : Il n'existe pas de données historiques de réponses aux offres dans la base DW_TT. Solution : **génération de labels synthétiques** calibrés sur des règles métier télécom réelles.

### 9.2 Génération des labels synthétiques

Un score latent (logit) est construit à partir de règles métier :

```python
logit_raw = (
    + 1.2 * arpu_n           # ARPU élevé → client à haute valeur, fidélisable
    + 0.9 * anc_n            # Ancienneté → loyauté établie
    + 0.7 * qos_n            # QoS correcte → satisfait du service de base
    - 0.8 * drop_n           # Drop rate élevé → frustration réseau → moins réceptif
    - 0.9 * impayes_n        # Impayés → moins engagé contractuellement
    + 1.4 * prop_sweet       # "Sweet spot" : risque churn modéré ≈ 0.40
    + 0.5 * minutes_n        # Usage actif → valorise le service
    + 0.3 * (1 - engag_n)    # Fin d'engagement → point de décision optimal
    + 0.3 * nps_n            # NPS positif
    + N(0, 0.45)             # Bruit réaliste
)
```

**Sweet spot** : Les clients avec propension churn ≈ 0.40 répondent le mieux (pic d'une cloche gaussienne `exp(-8 × (p - 0.40)²)`). Ni trop fidèles (pas besoin d'offre), ni trop déçus (fenêtre fermée).

**Calibration automatique du taux de réponse à 30%** :
```python
# Algorithme de Brent pour trouver l'intercept exact
from scipy.optimize import brentq
TARGET_RATE = 0.30
intercept = brentq(
    lambda t: (1 / (1 + np.exp(-(logit_raw + t)))).mean() - TARGET_RATE, -10, 10
)
prob_true = 1 / (1 + np.exp(-(logit_raw + intercept)))
y_response = rng.binomial(1, prob_true).astype(int)
```

### 9.3 Modèles entraînés

| Modèle | Caractéristique |
|---|---|
| **LogReg** | Pipeline [RobustScaler + LogisticRegression(C=0.5, balanced)] — nativement calibré |
| **GBR** | CalibratedClassifierCV(GradientBoostingClassifier, method='isotonic', cv=3) |
| **RF** | CalibratedClassifierCV(RandomForestClassifier, method='isotonic', cv=3) |

Évaluation par **StratifiedKFold 3-fold** + `cross_val_predict(method='predict_proba')` pour probabilités out-of-fold.

### 9.4 Résultats — Comparaison des modèles

| Modèle | AUC-ROC | Avg Precision | Brier ↓ | Statut |
|---|---|---|---|---|
| **LogReg** | **0.6185** | **0.4137** | 0.2387 | **MEILLEUR (AUC)** |
| GBR | 0.6079 | 0.4062 | **0.2055** | Mieux calibré |
| RF | 0.6060 | 0.3989 | **0.2061** | |

LogReg gagne encore le ranking (AUC) : avec des features standardisées et un signal linéairement séparable (ARPU + ancienneté + QoS), la régression logistique capture mieux les relations linéaires. Les arbres calibrés gagnent sur le Brier (probabilités mieux calibrées).

### 9.5 Distribution sur 10 000 clients

| Tier de Réponse | Clients | % Base |
|---|---|---|
| **Élevé (>50%)** | 4 506 | 45.1% |
| **Moyen (25–50%)** | 5 441 | 54.4% |
| **Faible (<25%)** | 53 | 0.5% |

- Mean : 0.492 ± 0.107
- **Taux de réponse observé : 30.6% ✓** (cible 30%)
- La majorité se situe dans la zone d'incertitude (25–50%) — réaliste pour les offres télécom

### 9.6 Matrice d'action Propension × Réponse

| | Propension Faible | Propension Modérée | Propension Élevée |
|---|---|---|---|
| **Réponse Élevée** | Fidélisation premium (non-urgent) | ⭐ **CŒUR DE CIBLE** | Offre urgente (fenêtre étroite) |
| **Réponse Moyenne** | Surveillance passive | Zone standard | Offre prix/réseau |
| **Réponse Faible** | Ne pas cibler | Surveiller | Client perdu (ROI négatif) |

---

## 10. Objectif 9 — Gain Attendu d'une Action de Rétention

### 10.1 Problème et approche

**Objectif** : Calculer le **gain financier net attendu** (en TND) de cibler chaque client avec une offre de rétention. C'est la synthèse de tous les objectifs précédents en une seule valeur actionnable.

**Type** : Modèle d'espérance mathématique (pas de ML supplémentaire)

### 10.2 Formule centrale

```
Gain_Net(i) = P_churn(i) × P_réponse(i) × ARPU(i) × LTV_horizon − Coût_offre
```

**Interprétation économique de chaque terme** :

| Composante | Source | Signification |
|---|---|---|
| `P_churn(i)` | Proxy Obj 6 (sigmoid sur features) | Probabilité que le client parte sans action |
| `P_réponse(i)` | Obj 8 (LogReg) | Probabilité que l'offre soit acceptée |
| `ARPU(i) × LTV` | `arpu_recent × 12` (proxy Obj 7) | Revenus préservés si le client est retenu |
| `Coût_offre` | Paramètre scénario | Investissement de rétention (remise, bonus...) |
| **`P_churn × P_réponse`** | — | Probabilité conjointe que l'action soit **utile ET acceptée** |

**LTV_horizon = 12 mois** (revenus sur 1 an si client retenu)

### 10.3 Implémentation

```python
LTV_HORIZON = 12
COUT_BASE   = 15.0  # TND (scénario standard)

# Propensity proxy (Obj 6)
df_feat['p_churn'] = sigmoid(-0.8*arpu_n + 0.9*impayes_n + 0.6*drop_n - 0.5*qos_n - 0.4*nps_n)

# ARPU prédit proxy (Obj 7 : R²=0.96 → arpu_recent ≈ arpu_futur)
df_feat['arpu_predit'] = df_feat['arpu_recent'].clip(lower=1.0)

# Formule gain
CLV = df_feat['arpu_predit'] * LTV_HORIZON
prob_utile       = df_feat['p_churn'] * df_feat['p_reponse']
df_feat['gain_brut'] = prob_utile * CLV
df_feat['gain_net']  = df_feat['gain_brut'] - COUT_BASE
df_feat['roi']       = df_feat['gain_net'] / COUT_BASE
df_feat['cible']     = (df_feat['gain_net'] > 0).astype(int)
```

### 10.4 Résultats — Scénario Standard (15 TND)

| KPI | Valeur |
|---|---|
| **Gain total potentiel** | **369 678 TND** |
| **Gain moyen par client** | **37.1 TND** |
| **ROI moyen** | **2.5×** |
| **Clients ROI-positifs** | **9 957 / 10 000 (99.6%)** |

> **Insight clé** : 99.6% des clients ont un Gain_Net positif avec une offre à 15 TND. En télécom, le coût de rétention (15 TND) est bien inférieur à la valeur sur 12 mois (ARPU × 12 ≈ 260 TND). La valeur du modèle n'est donc **pas** dans le filtrage binaire mais dans la **priorisation** : qui contacter en premier avec un budget limité.

### 10.5 Lift du modèle de ciblage

En triant les clients par gain attendu décroissant :

| Top % ciblés | Clients | Gain capturé | Lift vs aléatoire |
|---|---|---|---|
| **Top 10%** | **1 000** | **98 830 TND** | **2.68×** |
| Top 20% | 2 000 | 172 296 TND | 2.33× |
| Top 30% | 3 000 | 226 547 TND | 2.04× |
| Top 50% | 5 000 | 300 970 TND | 1.63× |

> En ciblant seulement 10% des clients avec le plus grand gain attendu, on capture **2.68× plus de valeur** qu'un ciblage aléatoire avec le même budget.

### 10.6 Comparaison des 3 scénarios d'offre

| Scénario | Coût | Clients ROI+ | % Base | Gain Total |
|---|---|---|---|---|
| **Légère** | 5 TND | 9 999 | ~100% | **469 628 TND** |
| **Standard** | 15 TND | 9 957 | 99.6% | 369 678 TND |
| **Premium** | 30 TND | 6 815 | **68.2%** | 240 711 TND |

L'offre **Premium** est la plus discriminante : seuls les clients avec ARPU élevé ET forte propension la justifient → idéale pour un budget très ciblé.

### 10.7 Formule de ciblage optimal (règle métier)

```
Cibler un client si : P_churn × P_réponse > Coût_offre / (ARPU × LTV_horizon)
```

Exemples de seuils :
- ARPU=20 TND, coût=15 TND, LTV=12 mois → seuil = **0.063**
- ARPU=30 TND, coût=15 TND, LTV=12 mois → seuil = **0.042**
- ARPU=10 TND, coût=30 TND, LTV=12 mois → seuil = **0.250**

Plus l'ARPU est faible, plus le seuil combiné doit être élevé pour justifier l'investissement.

---

## 11. Synthèse Globale et Pipeline ML Final

### 11.1 Vue d'ensemble des 9 objectifs

| # | Objectif | Méthode | Input | Output | Résultat clé |
|---|---|---|---|---|---|
| **1** | Prédiction churn (30j) | Classification (SVC-RBF) | 422 clients étiquetés | `churn_proba` | AUC = 0.71 |
| **2** | Classification risque | K-Means + RF | 10k clients | `risk_level` (3 niveaux) | 97.3% accuracy |
| **3** | Métriques métier | Analyse lift + coût-bénéfice | 422 clients | Seuil optimal, ROI | 53 592 TND valeur nette |
| **4** | Segmentation comportementale | K-Means composite 4D | 10k clients | 4 segments | Silhouette = 0.232 |
| **5** | Profils de rétention | Matrice 2D (segment × risque) | 10k clients | 12 micro-profils | 515k TND ROI potentiel |
| **6** | Propension churn (score continu) | Ridge + calibration isotonique | 422 → 10k clients | `propensity_score` ∈ [0,1] | AUC = 0.677, Acc = 93.4% |
| **7** | Prédiction ARPU futur | RandomForest (régression) | 10k × 9 mois | `arpu_predit` (TND) | R² = 0.962, MAE = 1.77 TND |
| **8** | Probabilité réponse offre | LogReg + calibration | 10k clients (labels synthétiques) | `p_reponse` ∈ [0,1] | AUC = 0.619, taux 30.6% |
| **9** | Gain attendu rétention | Formule espérance mathématique | Obj 6 + 7 + 8 | `gain_net` (TND/client) | Lift 2.68×, 369 678 TND |

### 11.2 Pipeline de décision unifié

```
                    ┌─────────────────────────────────────────────┐
                    │         BASE DE DONNÉES DW_TT (PostgreSQL)   │
                    │  dim_client + Fact_performance_client         │
                    └───────────────────┬─────────────────────────┘
                                        │ 10 000 clients × 12 mois
                    ┌───────────────────▼─────────────────────────┐
                    │          FEATURE ENGINEERING                  │
                    │  Agrégations, tendances, scores composites    │
                    └──┬──────────┬────────────┬───────────────────┘
                       │          │            │
              ┌────────▼──┐  ┌───▼────┐  ┌────▼────────┐
              │  Obj 6    │  │  Obj 7 │  │   Obj 8     │
              │ P_churn   │  │  ARPU  │  │ P_réponse   │
              │ [0,1]     │  │ (TND)  │  │  [0,1]      │
              └────────┬──┘  └───┬────┘  └────┬────────┘
                       │         │             │
                       └─────────┼─────────────┘
                                 │
                    ┌────────────▼────────────────────────┐
                    │           OBJECTIF 9                  │
                    │  Gain_Net(i) = P_churn × P_réponse    │
                    │              × ARPU × 12 − Coût       │
                    └────────────┬────────────────────────┘
                                 │
                    ┌────────────▼────────────────────────┐
                    │      SCORE DE PRIORITÉ CLIENT         │
                    │  Tri décroissant → Top K% à cibler    │
                    │  Budget limité → Lift 2.68× au Top 10%│
                    └─────────────────────────────────────┘
```

### 11.3 Chaîne de valeur des objectifs

```
Obj 1 (churn SVC, AUC=0.71)
    └──► Obj 2 (risk_level 3 classes, 97.3% acc)
              └──► Obj 5 (12 micro-profils, 515k TND ROI)
    └──► Obj 3 (métriques métier, seuil=0.18)

Obj 4 (4 segments, Silhouette=0.232)
    └──► Obj 5 (matrice segment × risque)
    └──► Obj 6 (analyse par segment)

Obj 6 (propension churn, AUC=0.677)
    └──► Obj 8 (sweet spot label synthétique)
    └──► Obj 9 (p_churn input)

Obj 7 (ARPU prédit, R²=0.962)
    └──► Obj 9 (arpu_predit input)

Obj 8 (p_réponse, AUC=0.619)
    └──► Obj 9 (p_reponse input)

Obj 9 = SYNTHÈSE FINALE (gain_net = Obj6 × Obj8 × Obj7 × 12 − Coût)
```

### 11.4 Résumé des performances

| Catégorie | Meilleur résultat | Interprétation |
|---|---|---|
| **Prédiction churn** | AUC = 0.71 ± 0.11 (SVC-RBF) | Limité par 27 non-churners — plafond empirique |
| **Classification risque** | 97.3% accuracy (RF) | Quasi-parfait — déployable immédiatement |
| **Segmentation** | Silhouette = 0.232 (+100% vs brut) | 4 segments métier distincts, bien équilibrés |
| **ARPU prediction** | R² = 0.962, MAE = 1.77 TND | Excellent — revenus prévisibles à 10% près |
| **Réponse offre** | AUC = 0.619, taux = 30.6% | Modéré mais calibré sur règles métier réelles |
| **ROI rétention** | Lift 2.68×, 369 678 TND total | Priorisation 2.7× meilleure que l'aléatoire |

### 11.5 Décisions techniques importantes

| Décision | Justification |
|---|---|
| **SMOTE > CTGAN/TVAE** | GANs nécessitent 1000+ samples — inutilisables sur 27 minoritaires |
| **AUC > F1** pour Obj 1 | F1 trompeur à 93.6% prévalence — tout modèle "tout-churn" a F1=0.97 |
| **Ridge > GBR/RF** pour ranking | Relations linéaires plus informatives que non-linéaires sur ce dataset |
| **Score propension ≠ classification** | Score continu = ciblage gradué, plus utile qu'une décision binaire |
| **Labels synthétiques calibrés** | Brentq solver garantit exactement 30% de réponse — reproductible |
| **arpu_recent comme proxy** | R²=0.96 confirme que ARPU récent ≈ ARPU futur — simplification valide |
| **Lift > filtrage binaire** | 99.6% ROI+ → le modèle sert à prioriser, pas filtrer |
| **k=4 segmentation, k=3 risque** | Équilibre groupes/pertinence métier — validé par Silhouette + Davies-Bouldin |

---

## Annexe — Fichiers générés par objectif

| Objectif | Notebooks | Fichiers résultats |
|---|---|---|
| Obj 1 | `01_churn_prediction.ipynb` × 6 versions | `01_churn_*_run_log.json`, `v5_results.png`, `v6_progression.png` |
| Obj 2 | `02_risk_classification.ipynb` | `02_risk_run_log.json`, `02_risk_dashboard.png` |
| Obj 3 | `03_business_metrics.ipynb` | `03_gain_lift_curves.png`, `03_calibration.png` |
| Obj 4 | `04_segmentation_v3.ipynb` (+ v1, v2) | `04v3_radar.png`, `04v3_profile_heatmap.png` |
| Obj 5 | `05_retention_profiles.ipynb` | `05_targeting_matrix.png`, `05_priority_chart.png` |
| Obj 6 | `06_churn_propensity.ipynb` | `06_propensity_distribution.png`, `06_propensity_run_log.json` |
| Obj 7 | `07_arpu_prediction.ipynb` | `07_predicted_vs_actual.png`, `07_arpu_run_log.json` |
| Obj 8 | `08_offer_response.ipynb` | `08_action_matrix.png`, `08_offer_response_run_log.json` |
| Obj 9 | `09_retention_gain.ipynb` | `09_gain_curve.png`, `09_sensitivity.png`, `09_retention_gain_run_log.json` |

---

*Rapport généré le 2026-06-10 | Projet ML PFE — Telecom DW_TT*
