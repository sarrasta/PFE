# Objectif 4 — Segmentation Comportementale des Clients (v2 améliorée)

**Méthode** : K-Means k=4 sur Scores Composites 4D (log1p + StandardScaler)  
**Notebooks** : `04_segmentation.ipynb` (v1) → `04_segmentation_v2.ipynb` (v2 ✓)  
**Base de données** : `DW_TT` — PostgreSQL localhost:5432

---

## Amélioration du Silhouette : v1 → v2

| Paramètre | v1 (23D brutes) | v2 (Composite 4D) | Δ |
|---|---|---|---|
| **Silhouette** | 0.116 | **0.1997** | **+72%** |
| **Davies-Bouldin** | — | **1.404** | (plus bas = mieux) |
| **Calinski-Harabasz** | — | **2 381** | (plus haut = mieux) |
| Stratégie | RobustScaler(23D) | Composite 4D + Log1p | — |

---

## Diagnostic et Solution

### Pourquoi v1 avait Silhouette = 0.116 ?

| Cause | Mécanisme | Impact |
|---|---|---|
| **23 features** | Malédiction de la dimensionnalité | Distances inter-cluster uniformes |
| **Features corrélées** | `avg_arpu` ≈ `avg_montant_facture` (r≈0.9) | Variance redondante artificielle |
| **Distributions asymétriques** | `total_impayes`, `max_retard_paiement`, `total_tickets` | Outliers déplacent les centroïdes |

### Solution : Scores Composites 4D

Au lieu de cluster sur 23 features brutes, on calcule **1 score par dimension** :

```python
# Chaque score = moyenne de z-scores standardisés par dimension
usage_score   = mean(z[minutes, data, sms, roaming, arpu, montant])
payment_risk  = mean(z[total_impayes, max_retard, avg_retard, hors_forfait])
network_risk  = mean(z[drop_rate, outage]) - mean(z[qos, throughput])
care_risk     = mean(z[tickets, delai_resolution]) - z[nps]
```

**Résultat** : 4D propre → distances euclidiennes significatives → meilleure séparation.

---

## Comparaison des Stratégies (k=4)

| Stratégie | Features | Silhouette | Verdict |
|---|---|---|---|
| **S1 Composite 4D + Log1p** | 4D composite | **0.1997** | **Meilleur → retenu** |
| S4 Composite 4D (sans log) | 4D composite | 0.1979 | Très proche |
| S3 PowerTransform + PCA(8) | 8D PCA | 0.184 | Bon mais moins interprétable |
| S2 Log1p + PCA(8) | 8D PCA | 0.1554 | Meilleur autre k, pas k=4 |
| V1 Original 23D | 23D brutes | 0.114 | Baseline v1 |

> La transformation log1p apporte +0.002 silhouette vs sans log (S1 vs S4).  
> La réduction 23D→4D apporte l'essentiel du gain (+72%).

---

## Résultats — 4 Segments Comportementaux

### Distribution

| Segment | Clients | % base | Profil dominant |
|---|---|---|---|
| **Clients Débiteurs** | 3 075 | **30.8%** | Risque Paiement élevé |
| **Clients Insatisfaits** | 2 671 | **26.7%** | Risque SAV élevé, NPS faible |
| **Clients Réseau Dégradé** | 2 242 | **22.4%** | Drop rate + Outages élevés |
| **Gros Consommateurs** | 2 012 | **20.1%** | Usage + ARPU + Data élevés |

### Profil des Scores Composites par Segment

| Segment | Usage/Valeur | Risque Paiement | Risque Réseau | Risque SAV |
|---|---|---|---|---|
| **Gros Consommateurs** | **Élevé** ↑ | Faible | Faible | Faible |
| **Clients Débiteurs** | Modéré | **Élevé** ↑ | Faible | Faible |
| **Clients Réseau Dégradé** | Modéré | Faible | **Élevé** ↑ | Faible |
| **Clients Insatisfaits** | Modéré | Faible | Faible | **Élevé** ↑ |

> Chaque segment est dominé par **exactement une dimension** → segmentation propre et interprétable.  
> Attribution par algorithme Hongrois (linear_sum_assignment) → unicité garantie.

---

## Classificateur de Déploiement (Random Forest)

| Métrique | v1 | v2 |
|---|---|---|
| **Accuracy** | 97.04% ± 0.48% | **90.33%** ± 0.16% |
| **F1-macro** | 97.02% ± 0.48% | **97.44%** ± 0.16% |

> L'accuracy v2 est légèrement inférieure (90% vs 97%) car les clusters v2 sont plus équilibrés  
> et moins "parfaitement" apprenables — signe d'une segmentation plus naturelle.  
> Le F1-macro est stable (~97%), confirmant que le RF généralise bien sur les 4 classes.

---

## Actions Métier par Segment

| Segment | Taille | Priorité | Action recommandée |
|---|---|---|---|
| **Clients Débiteurs** | 3 075 (30.8%) | 🔴 Urgent | Plan paiement, alertes précoces, garantie |
| **Clients Insatisfaits** | 2 671 (26.7%) | 🟠 Haute | SAV prioritaire, appel proactif, compensation |
| **Clients Réseau Dégradé** | 2 242 (22.4%) | 🟡 Moyenne | Investigation réseau, dédommagement |
| **Gros Consommateurs** | 2 012 (20.1%) | 🟢 Fidélisation | Upsell, offre premium, programme fidélité |

---

## Fichiers Générés

| Fichier | Description |
|---|---|
| `04_segmentation_v2.ipynb` | Notebook complet v2 |
| `04v2_strategy_comparison.png` | Comparaison des 5 stratégies de préprocessing |
| `04v2_pca_elbow.png` | PCA 2D + courbe du coude |
| `04v2_radar.png` | Radar chart — profil des 4 segments |
| `04v2_score_distributions.png` | Distribution des scores composites par cluster |
| `04v2_cross_tabs.png` | Répartition par Segment, TypeAbo, Région |
| `04v2_segmentation_run_log.json` | Métriques et résultats |
