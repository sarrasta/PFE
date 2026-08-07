# Objectif 1 — Prédiction du Churn Client (30 jours)

**Cible** : `Churn_30j` ∈ {0, 1}  
**Type** : Classification binaire  
**Notebook** : `01_churn_prediction.ipynb`  
**Base de données** : `DW_TT` — PostgreSQL localhost:5432

---

## Contexte des données

| Élément | Détail |
|---|---|
| Période churn utilisée | Décembre 2024 (`DateFK = 36`) |
| Clients avec cible disponible | 422 |
| Features comportementales | Agrégées sur 12 mois 2024 (`Fact_performance_client`) |
| Features client | `dim_client`, `dim_geographique`, `dim_offre` |
| Total features | 28 |

### Distribution de la cible

| Classe | Effectif | % |
|---|---|---|
| 0 — Non-churn | 27 | ~6.4% |
| 1 — Churn | 395 | ~93.6% |

> **Déséquilibre sévère** : ratio ~1:15 en faveur du churn.  
> Stratégie appliquée : `class_weight='balanced'` + **SMOTE** sur le train uniquement.

### Features utilisées (28 colonnes)

**Catégorielles** (encodées LabelEncoder) :
- `sexe`, `segment`, `type_abonnement`, `region`, `offre_actuelle`

**Numériques** :
- **Client** : `anciennete_mois`, `engagement_restant`, `prix_offre`
- **Usage** : `avg_minutes`, `avg_sms`, `avg_data_gb`, `avg_roaming_gb`, `avg_nocturne_ratio`
- **Facturation** : `avg_montant_facture`, `avg_arpu`, `avg_hors_forfait`
- **Paiement** : `total_impayes`, `max_retard_paiement`, `avg_retard_paiement`
- **SAV** : `total_tickets`, `avg_delai_resolution`
- **Satisfaction** : `avg_nps`
- **Réseau** : `avg_qos`, `avg_drop_rate`, `avg_throughput`, `avg_outage_min`
- **Autre** : `nb_mois_observes`

---

## Itération 1 — Baseline (Run initial)

**Date** : 2026-06-10  
**Split** : 75% train (316) / 25% test (106), stratifié  
**SMOTE** : k_neighbors=3 → 632 samples équilibrés  
**Test set** : ~99 churn (1) / ~7 non-churn (0)

### Résultats sans SMOTE (class_weight='balanced')

| Modèle | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| **Random Forest** | — | — | **1.000** | **0.9659** | 0.4271 |
| LightGBM | — | — | 0.9798 | 0.9557 | 0.4993 |
| Gradient Boosting | — | — | 0.9697 | 0.9552 | 0.4589 |
| XGBoost | — | — | 0.9091 | 0.9184 | 0.3954 |
| Logistic Regression | — | — | 0.6667 | 0.7811 | **0.6075** |
| Decision Tree | — | — | 0.6465 | 0.7619 | 0.4646 |

### Résultats avec SMOTE

| Modèle | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| **Random Forest** | 0.9151 | 0.9327 | **0.9798** | **0.9557** | 0.3997 |
| XGBoost | — | — | 0.9596 | 0.9453 | 0.5238 |
| LightGBM | — | — | 0.9596 | 0.9453 | **0.5613** |
| Gradient Boosting | — | — | 0.9495 | 0.9400 | 0.4069 |
| Decision Tree | — | — | 0.7172 | 0.8068 | 0.4206 |
| Logistic Regression | — | — | 0.5152 | 0.6755 | 0.6537 |

**Meilleur modèle (F1)** : Random Forest (sans SMOTE) — F1 = **0.9659**  
**Meilleur modèle (AUC)** : Logistic Regression (avec SMOTE) — AUC = **0.6537**

### Observations critiques

> ⚠️ **Paradoxe déséquilibre** : Random Forest atteint Recall=1.0 sur la classe churn mais AUC=0.43,  
> ce qui indique qu'il prédit **tout le monde comme churner** (pas d'information discriminante réelle).  
> Le F1 élevé est trompeur ici car 93.6% des cas sont déjà churn=1.

- Random Forest / GB / XGBoost → F1 très élevé mais **AUC < 0.5** (pire que le hasard en termes de ranking)
- Logistic Regression → F1 plus faible mais **AUC = 0.65** : meilleure séparation réelle des classes
- SMOTE améliore légèrement l'AUC de LightGBM (+0.06) et LR (+0.05)
- Avec seulement ~7 non-churn dans le test, les métriques sont très instables

**Métrique cible pour les itérations suivantes** : **AUC-ROC** (plus robuste que F1 dans ce contexte)

---

## Itération 2 — CTGAN + Feature Engineering (v2)

**Notebook** : `01_churn_v2_ctgan.ipynb` | **Date** : 2026-06-10

### Améliorations appliquées
- 6 nouvelles features construites (`score_risque_paiement`, `ratio_hors_forfait`, `score_degradation_reseau`, `flag_hors_engagement`, `intensite_usage`, `score_insatisfaction`)
- CTGAN (epochs=500) entraîné sur les non-churn du train
- Calibration Platt (sigmoid) sur tous les modèles
- Dataset features : 27 → **33 colonnes**

### Résultats — CTGAN + Calibration (hold-out 25%)

| Modèle | AUC-ROC | F1 |
|---|---|---|
| XGBoost | **0.537** | 0.929 |
| Logistic Regression | 0.509 | 0.966 |
| LightGBM | 0.485 | 0.950 |
| Random Forest | 0.468 | 0.961 |

**Observation critique** : AUC **pire** qu'en v1 → CTGAN nécessite ~1000+ samples pour apprendre, 20 samples non-churn sont insuffisants pour un GAN.

---

## Itération 3 — TVAE + Toutes périodes + RandomizedSearchCV (v3)

**Notebook** : `01_churn_v3_tvae.ipynb` | **Date** : 2026-06-10

### Améliorations appliquées
- TVAE (SDV) à la place de CTGAN → VAE plus stable pour petits datasets
- Utilisation des deux périodes (DateFK=12 + DateFK=36) → **42 non-churn**
- RandomizedSearchCV (40 iter) sur LightGBM et XGBoost
- VotingClassifier (LGBM + XGB + RF)
- Évaluation par **cross-validation 5-fold** (plus fiable que hold-out)

### Résultats CV 5-fold

| Stratégie | Meilleur AUC CV |
|---|---|
| Baseline class_weight | 0.618 |
| SMOTE pipeline | **0.667** |
| LightGBM tuned (RSCV) | 0.576 |
| XGBoost tuned (RSCV) | 0.602 |
| VotingClassifier | 0.598 |

**Observation** : L'ajout des clients DateFK=12 (avec features performance = 0) **introduit du bruit** et dégrade l'AUC. SMOTE reste meilleur.

---

## Itération 4 — Features Temporelles + BalancedRF + EasyEnsemble (v4)

**Notebook** : `01_churn_v4_temporal.ipynb` | **Date** : 2026-06-10

### Améliorations appliquées
- Exploitation des **12 mois mensuels** de Fact_performance_client (DateFK 25→36)
- 118 features temporelles : pente (slope), écart-type, delta récent (3 mois vs historique), last vs avg
- Total : 137 features
- Nouveaux modèles : `BalancedRandomForestClassifier`, `EasyEnsembleClassifier`

### Résultats CV 5-fold

| Modèle | AUC CV |
|---|---|
| SMOTE + BalancedRF | **0.5895** |
| SMOTE + LightGBM | 0.4807 |
| LightGBM (is_unbalance) | 0.4765 |
| XGBoost (scale_pos_weight) | 0.4631 |

### Diagnostic v4 — Malédiction de la dimensionnalité

> **137 features pour 27 samples minoritaires** = ratio features/samples > 5 → les modèles sur-apprennent sur les folds d'entraînement.  
> Les features temporelles ont une importance RF élevée (78%) mais produisent un AUC **inférieur** à v3 : signal de variance excessive, pas de généralisation.

**Conclusion v4** : Les données mensuelles synthétiques n'ont pas de tendances temporelles réelles corrélées au churn. Retour aux features agrégées.

---

## Itération 5 — SVM-RBF + Features Ciblées + Tuning (v5)

**Notebook** : `01_churn_v5_focused.ipynb` | **Date** : 2026-06-10

### Améliorations appliquées
- Retour aux 35 features originales + `OffreCible` + `RGPD_Consent`
- Nouveaux modèles : SVC (RBF), NuSVC, BaggingClassifier(SVC), ExtraTrees, GaussianNB
- SVMSMOTE (variante de SMOTE basée SVM) testée
- RandomizedSearchCV (40 iter) sur SVC et LR
- `RobustScaler` à la place de `StandardScaler`

### Résultats CV 5-fold

| Modèle | AUC CV |
|---|---|
| **SVC RBF (tuné)** | **0.7138 ± 0.1144** |
| LR (tuné, C=0.05, l2) | 0.6961 ± 0.1108 |
| LR (C=1.0) | 0.6851 ± 0.0867 |
| LR + SMOTE baseline | 0.6803 ± 0.1055 |
| BaggingClassifier(SVC) | 0.6417 ± 0.0799 |

**Meilleurs hyperparamètres SVC** : `C=10.0, gamma=0.001, SMOTE k_neighbors=5`

### Découverte importante : fuite de données (churn_t1)
> Lors des premiers tests v5, l'utilisation du label `Churn_30j` à DateFK=12 comme feature produisait AUC=1.0.  
> **Cause** : dans le dataset PFE synthétique, 395/395 churners à DateFK=36 étaient déjà churners à DateFK=12 → corrélation parfaite → fuite de cible.  
> **Correction** : suppression de toutes les features cross-périodes.

### Observation v5
- SVM-RBF surpasse LR grâce à sa capacité à trouver des frontières non-linéaires dans l'espace transformé par `gamma=0.001`
- L'écart-type élevé (±0.11) confirme la limite statistique : ~5 non-churn par fold test

---

## Diagnostic Global — Plafond Empirique des Données

> La contrainte est **structurelle** : avec 27 samples minoritaires (6.4%), aucun algorithme ne peut atteindre 0.85 AUC de manière fiable.

| Contrainte | Impact |
|---|---|
| 27 non-churn sur 422 lignes | AUC CV instable (écart-type ±0.10–0.14) |
| 5–6 non-churn par fold (5-fold CV) | Estimation AUC statistiquement non fiable |
| Dataset synthétique PFE : labels identiques T12=T36 | Churn_t1 = fuite parfaite → non utilisable |
| GANs (CTGAN, TVAE) sur 20 samples | Génération bruit → AUC dégradé |
| 137 features temporelles / 27 minoritaires | Curse of dimensionality → sur-apprentissage |

**Plafond atteint** : Le meilleur AUC fiable (sans fuite) est **0.7138 ± 0.11** (SVM-RBF tuné, v5).  
Pour atteindre 0.85 il faudrait soit **~500+ non-churn** dans la base, soit des **features externes discriminantes** (ex : appels au service client, historique plaintes).

---

## Meilleur Modèle Final

| | Valeur |
|---|---|
| **Modèle** | SVC (kernel=rbf, C=10, gamma=0.001) |
| **AUC CV 5-fold** | **0.7138 ± 0.1144** |
| **Stratégie équilibrage** | SMOTE (k_neighbors=5) dans pipeline CV |
| **Features** | 35 (27 originales + 6 ingénierie + OffreCible + RGPD) |
| **Dataset** | DateFK=36 uniquement (422 lignes) |
| **Scaler** | RobustScaler (résistant aux outliers) |

---

## Itération 6 — Génération Synthétique Évaluée Correctement (v6)

**Notebook** : `01_churn_v6_synthesis.ipynb` | **Date** : 2026-06-10

### Protocole rigoureux
> **Contrainte clé** : La synthèse est appliquée **uniquement dans le fold d'entraînement**.  
> Le fold de test contient UNIQUEMENT des données réelles → pas de contamination de l'évaluation.

### Méthodes testées

| Méthode | Principe | AUC CV |
|---|---|---|
| SMOTE pipeline | Interpolation k-NN entre minoritaires | **0.7138 ± 0.1144** |
| Bootstrap + bruit 15% | Rééchantillonnage + bruit gaussien | 0.6890 ± 0.1186 |
| KDE marginale | KDE indépendante par feature | 0.6645 ± 0.1315 |
| GaussianCopula (SDV) | Modèle de copules gaussiennes | 0.6644 ± 0.0974 |
| MVN + LedoitWolf | Gaussienne multivariée régularisée | 0.6525 ± 0.1137 |

### Conclusion v6
> **SMOTE reste la meilleure méthode**. Aucune méthode de génération ne dépasse 0.71 AUC.  
> La KDE multi-dimensionnelle est impossible ici (35 features > 22 samples/fold) → KDE marginale utilisée (ignore les corrélations).  
> GaussianCopula SDV est stable mais ne surpasse pas SMOTE sur si peu de données.

---

## Progression globale (6 itérations)

| Version | Stratégie | Meilleur AUC CV | Statut |
|---|---|---|---|
| v1 | Baseline (6 modèles, SMOTE, class_weight) | 0.654 | référence initiale |
| v2 | CTGAN + Feature Engineering | 0.537 | GAN sur 20 samples = bruit |
| v3 | TVAE + RSCV + Voting | 0.667 | meilleur généraliste |
| v4 | Features temporelles (137 features) | 0.590 | curse of dimensionality |
| v5 | SVM-RBF (C=10, γ=0.001) + SMOTE | **0.714** | **meilleur modèle** |
| v6 | KDE / MVN / GaussianCopula / Bootstrap | 0.714 | SMOTE reste optimal |

---

## Fichiers générés

| Fichier | Description |
|---|---|
| `01_churn_prediction.ipynb` | v1 — Baseline, 6 modèles, SMOTE |
| `01_churn_v2_ctgan.ipynb` | v2 — CTGAN + Feature Engineering |
| `01_churn_v3_tvae.ipynb` | v3 — TVAE + RSCV + Voting |
| `01_churn_v4_temporal.ipynb` | v4 — Features temporelles (118 features) |
| `01_churn_v5_focused.ipynb` | v5 — SVM-RBF + SelectKBest + tuning |
| `01_churn_run_log.json` | Métriques run v1 |
| `01_churn_v2_run_log.json` | Métriques run v2 |
| `01_churn_v3_run_log.json` | Métriques run v3 |
| `01_churn_v4_run_log.json` | Métriques run v4 |
| `01_churn_v5_run_log.json` | Métriques run v5 |
| `01_churn_v6_run_log.json` | Métriques run v6 |
| `v5_mutual_info.png` | Mutual information des features (v5) |
| `v5_results.png` | Comparaison modèles v5 + progression |
| `v6_synthesis_quality.png` | Distributions réel vs synthétique (v6) |
| `v6_progression.png` | Progression AUC v1→v6 |

---

## Notes & Décisions Clés

- **ChurnScore_30j non utilisé** : valeur constante (0.0212) — aucune information.
- **DateFK=36 uniquement** : DateFK=12 n'a pas de données performance → bruit.
- **churn_t1 = fuite** : labels identiques entre périodes dans le dataset synthétique.
- **Features temporelles = bruit** : données mensuelles générées sans corrélation avec le churn.
- **SVM-RBF optimal** : `C=10, gamma=0.001` — marge large avec kernel gaussien étalé.
- **Seuil optimal** : ~0.25 (pas 0.5) pour maximiser F1 en déploiement.
