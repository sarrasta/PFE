# Objectif 2 — Classification du Risque Client (Faible / Moyen / Élevé)

**Cible** : `risk_level` ∈ {0=Faible, 1=Moyen, 2=Élevé}  
**Type** : Clustering non supervisé (K-Means) → classificateur supervisé (RF)  
**Notebook** : `02_risk_classification.ipynb`  
**Base de données** : `DW_TT` — PostgreSQL localhost:5432

---

## Contexte et Méthodologie

| Étape | Description |
|---|---|
| **1. Données** | Tous les 10 000 clients (dim_client + Fact_performance_client) |
| **2. Features risque** | 13 indicateurs (paiement, réseau, satisfaction, engagement, churn proba) |
| **3. Churn proba** | Modèle SVC v5 (C=10, γ=0.001) entraîné sur 422 clients étiquetés |
| **4. Clustering** | K-Means (k=3) sur features risque normalisées |
| **5. Classification** | Random Forest supervisé (3-class) sur labels découverts → CV accuracy |

### Features utilisées pour le clustering (13)

**Paiement** : `total_impayes`, `max_retard_paiement`, `avg_retard_paiement`, `score_risque_paiement`  
**Réseau** : `avg_drop_rate`, `avg_outage_min`, `score_degradation_reseau`  
**Satisfaction** : `total_tickets`, `avg_nps`, `score_insatisfaction`  
**Contrat** : `flag_hors_engagement`, `avg_hors_forfait`  
**Modèle churn** : `churn_proba` (SVC v5)

---

## Choix de k — Méthode du Coude + Silhouette

| k | Silhouette | Verdict |
|---|---|---|
| 2 | — | Séparation grossière |
| **3** | **0.1425** | **Choix retenu — 3 niveaux de risque métier** |
| 4 | — | Sur-segmentation |
| 5 | — | Trop granulaire |

> Silhouette = 0.14 est faible mais attendu sur des données client télécom où les risques sont continus et corrélés.

---

## Résultats du Clustering K-Means (k=3)

### Distribution des niveaux de risque

| Niveau | Clients | % base | Churn proba moy. | Caractéristique dominante |
|---|---|---|---|---|
| **Faible** | 4 370 | 43.7% | 0.671 | Retards paiement modérés |
| **Moyen** | 2 501 | 25.0% | 0.596 | Tickets élevés (plaintes fréquentes) |
| **Élevé** | 3 129 | 31.3% | **0.815** | Impayes + retards extrêmes |

### Profil des centroïdes par niveau

| Feature | Faible | Moyen | Élevé |
|---|---|---|---|
| `total_impayes` | 0.28 | 0.39 | **1.59** |
| `max_retard_paiement` | 2.1 j | 3.6 j | **21.8 j** |
| `total_tickets` | 1.5 | **4.1** | 2.3 |
| `avg_nps` | 35.7 | 33.1 | 35.1 |
| `avg_drop_rate` | 3.64% | 3.59% | 3.64% |
| `churn_proba` | 0.671 | 0.596 | **0.815** |

### Lecture des clusters

> **Élevé** : Clients avec de forts impayés ET longs retards de paiement → churn probable élevé (0.815)  
> **Moyen** : Clients très actifs au SAV (4× plus de tickets) mais paiement correct → risque de friction  
> **Faible** : Comportement sain sur paiement et usage — risque modéré malgré churn_proba = 0.671 (biais du modèle formé sur 93% churners)

---

## Classificateur Supervisé — Random Forest (3-class)

| Métrique | CV 5-fold | Std |
|---|---|---|
| **Accuracy** | **97.34%** | ±0.17% |
| **F1-macro** | **97.19%** | ±0.17% |

> Performance quasi-parfaite : le RF apprend très fidèlement les frontières décidées par K-Means.  
> Ce classificateur permet d'assigner un niveau de risque à TOUT nouveau client sans relancer K-Means.

---

## Diagnostic et Nuances

### Biais du churn model sur population globale
Le SVC entraîné sur les 422 clients de Fact_churn (93.6% churners) tend à prédire des probabilités élevées même pour des clients sains. La moyenne globale de churn_proba est ~0.69 sur les 10k clients (vs 0.5 attendu pour une base équilibrée). Ce biais est intégré dans le score de risque mais doit être pris en compte en déploiement.

### Cluster "Moyen" contre-intuitif
Le niveau Moyen présente `churn_proba` = 0.596 (plus faible que Faible = 0.671). Ces clients ont beaucoup de tickets (4× la moyenne) mais peu d'impayés. Cela reflète des "plaignants actifs" qui s'engagent mais ne partent pas encore → risque de frustration à surveiller.

---

## Fichiers générés

| Fichier | Description |
|---|---|
| `02_risk_classification.ipynb` | Notebook complet |
| `02_risk_run_log.json` | Métriques et distribution |
| `02_kmeans_selection.png` | Courbe du coude + silhouette |
| `02_risk_dashboard.png` | PCA, radar, distributions, feature importance |
| `02_risk_by_segment.png` | Risque par région et type abonnement |

---

## Notes & Décisions Clés

- **K=3 choisi** : correspondance naturelle avec les niveaux métier faible/moyen/élevé.
- **Churn_proba comme feature** : lie l'objectif 1 (churn) à l'objectif 2 (risque).
- **Silhouette faible** : normal pour données client continues — clusters opérationnels même avec chevauchement.
- **RF 97% accuracy** : classifier de déploiement pour nouveaux clients sans re-clustering.
- **Biais churn model** : à recalibrer avec plus de données non-churn en production.
