# Objectif 7 — Prédiction ARPU & Montant Facture Futurs (Régression)

**Méthode** : Split temporel — mois 1-9 (DateFK 25-33) → prédire mois 10-12 (DateFK 34-36)  
**Notebook** : `07_arpu_prediction.ipynb`  
**Clients** : 10 000 (toute la base — pas de labels requis)

---

## Architecture de la Solution

```
Fact_performance_client (120k lignes, 10k × 12 mois)
         │
         ├── DateFK 25-33 (9 mois) ─────► Features temporelles
         │     ├── mean, std, min, max par colonne
         │     ├── recent_mean (3 derniers mois)
         │     ├── growth_rate (récent vs début)
         │     └── trend linéaire (pente de régression)
         │
         └── DateFK 34-36 (3 mois) ─────► Cibles
               ├── avg_arpu_futur
               └── avg_facture_futur

Features statiques : Segment, TypeAbonnement, Region, Ancienneté, PrixOffre
Feature matrix : 10 000 clients × 100 features
```

---

## Résultats — Comparaison des Modèles (CV 3-Fold)

### ARPU Futur (trimestre mois 10-12)

| Modèle | R² | MAE | RMSE | MAPE | Statut |
|---|---|---|---|---|---|
| **RandomForest** | **0.9619** | **1.77 TND** | **2.27 TND** | 10.2% | **MEILLEUR** |
| GBR | 0.9609 | 1.79 TND | — | 10.4% | |
| Ridge | 0.9598 | 1.78 TND | 2.33 TND | 10.3% | |

### Montant Facture Futur (trimestre mois 10-12)

| Modèle | R² | MAE | RMSE | MAPE | Statut |
|---|---|---|---|---|---|
| **RandomForest** | **0.9597** | **1.83 TND** | **2.34 TND** | 10.6% | **MEILLEUR** |
| GBR | 0.9583 | 1.86 TND | — | 10.8% | |
| Ridge | 0.9581 | 1.84 TND | 2.39 TND | 10.7% | |

> Les 3 modèles sont très proches — R² ≈ 0.96 pour tous.  
> RandomForest retenu par légère marge (MAE=1.77 TND ARPU, 1.83 TND Facture).

---

## Interprétation des Métriques

| Métrique | ARPU | Facture | Interprétation |
|---|---|---|---|
| **R²** | **0.9619** | **0.9597** | 96% de la variance expliquée — excellent |
| **MAE** | **1.77 TND** | **1.83 TND** | Erreur absolue médiane ≈ 2 TND |
| **MAPE** | **10.2%** | **10.6%** | Erreur relative de ~10% |

**Distribution ARPU observé** : 21.86 ± 12.04 TND (min: 2.24, max: 70.89)  
**Erreur MAE = 1.77 TND** sur plage de 68 TND → erreur relative globale très faible.

---

## Données Utilisées

| Paramètre | Valeur |
|---|---|
| Clients | 10 000 (base complète) |
| Mois features | DateFK 25-33 (9 mois) |
| Mois cibles | DateFK 34-36 (3 mois futurs) |
| Features | 100 (13 cols × 4-7 stats + statiques) |
| ARPU moyen observé | 21.86 ± 12.04 TND |
| Facture moyenne observée | 21.87 ± 12.04 TND |

---

## Features Temporelles

Pour chaque colonne numérique (13 colonnes : arpu, facture, minutes, data, sms,
hors_forfait, impayes, retard, qos, drop_rate, throughput, outage, nps) :

| Stat | Description | Formule |
|---|---|---|
| `{col}_mean` | Moyenne sur 9 mois | mean(mois 1-9) |
| `{col}_std` | Volatilité | std(mois 1-9) |
| `{col}_min/max` | Extremes | — |
| `{col}_recent` | Valeur récente | mean(mois 7-9) |
| `{col}_growth` | Tendance croissance | recent/early − 1 |
| `{col}_trend` | Pente linéaire | polyfit slope sur 9 mois |

> **Feature la plus prédictive** : `arpu_mean` (moyenne ARPU passée) et `arpu_recent`  
> → L'ARPU futur est fortement corrélé à l'ARPU passé (stabilité des abonnements)

---

## Analyse par Segment

| Segment | Clients | ARPU prédit | ARPU réel | Facture prédite | Facture réelle |
|---|---|---|---|---|---|
| **Postpayé** | 2 447 | 21.73 TND | 21.77 TND | 21.73 TND | 21.80 TND |
| **Prépayé** | 7 553 | 21.94 TND | 21.91 TND | 21.94 TND | 21.92 TND |

> L'ARPU est quasi-identique entre Postpayé et Prépayé → distribution homogène.  
> Les prédictions sont très proches des valeurs réelles dans les deux segments.

---

## Utilisation Métier

| Application | Description |
|---|---|
| **Budget prévisionnel** | Estimer les revenus du trimestre suivant par client |
| **Identification décroissance** | Clients avec `arpu_growth < -0.15` → ARPU en baisse → risque churn |
| **Segmentation valeur** | Classer les clients par ARPU prédit → ciblage campagnes premium |
| **Alertes proactives** | Si ARPU prédit < ARPU actuel × 0.80 → contacter avant churn |
| **Croisement avec Obj 6** | Clients avec propension_churn élevée + ARPU_prédit élevé = priorité max |

### Formule d'alerte ARPU
```python
# Client à surveiller si ARPU prédit chute de >20%
alert_flag = (arpu_pred / arpu_past_mean) < 0.80
```

---

## Fichiers Générés

| Fichier | Description |
|---|---|
| `07_arpu_prediction.ipynb` | Notebook complet |
| `07_model_comparison.png` | R² / MAE / MAPE — 3 modèles × 2 cibles |
| `07_predicted_vs_actual.png` | Scatter prédit vs réel (hexbin density) |
| `07_residuals.png` | Distribution des résidus |
| `07_arpu_by_segment.png` | ARPU & Facture réels vs prédits par segment |
| `07_feature_importance.png` | Top 20 features (RF) pour ARPU & Facture |
| `07_summary_table.png` | Tableau comparatif |
| `07_arpu_run_log.json` | Métriques exportées |
