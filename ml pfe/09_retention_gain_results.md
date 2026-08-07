# Objectif 9 — Gain Attendu d'une Action de Rétention

**Méthode** : Modèle d'espérance mathématique — synthèse Obj 6 + Obj 7 + Obj 8  
**Notebook** : `09_retention_gain.ipynb`  
**Clients** : 10 000 — score de gain en TND par client

---

## Formule Centrale

```
Gain_Net(i) = P_churn(i) × P_réponse(i) × ARPU(i) × LTV_horizon - Coût_offre

où :
  P_churn(i)   = propension au churn (proxy Obj 6)
  P_réponse(i) = probabilité de réponse à l'offre (Obj 8)
  ARPU(i)      = revenu mensuel récent (proxy Obj 7)
  LTV_horizon  = 12 mois (revenus préservés si client retenu)
  Coût_offre   = coût de l'action de rétention (scénario)
```

### Interprétation économique

| Composante | Signification |
|---|---|
| `P_churn × P_réponse` | Probabilité que l'action soit **utile ET acceptée** |
| `ARPU × LTV` | Revenus sauvés si le client est retenu (Customer Lifetime Value) |
| `Coût_offre` | Investissement de rétention (remise, bonus data, upgrade) |
| **Gain_Net > 0** | Cibler ce client est **rentable** |

---

## Résultats — Scénario Standard (15 TND)

| KPI | Valeur |
|---|---|
| **Clients ROI-positifs** | **9 957 / 10 000 (99.6%)** |
| **Gain total potentiel** | **369 678 TND** |
| **Gain moyen par client** | **37.1 TND** |
| **ROI moyen** | **2.5×** (gain / coût = 37.1 / 15) |

> **Insight clé** : 99.6% des clients ont un Gain_Net positif avec une offre à 15 TND.
> En télécom, le coût d'une action de rétention (15 TND) est bien inférieur à la valeur
> sur 12 mois (ARPU × 12 ≈ 260 TND). La valeur du modèle n'est donc **pas** dans le
> filtrage binaire, mais dans la **priorisation** (qui contacter en premier avec un budget
> limité).

---

## Lift du Modèle de Ciblage

Valeur ajoutée vs ciblage aléatoire (même budget, meilleurs résultats) :

| Top % ciblés | Clients | Gain capturé | Lift |
|---|---|---|---|
| **Top 10%** | **1 000** | **98 830 TND** | **2.68×** |
| **Top 20%** | 2 000 | 172 296 TND | 2.33× |
| **Top 30%** | 3 000 | 226 547 TND | 2.04× |
| **Top 50%** | 5 000 | 300 970 TND | 1.63× |

> **En ciblant seulement 10% des clients** (ceux avec le plus grand gain attendu),
> on capture **2.68×** plus de valeur que si on ciblait 10% aléatoirement.
> Avec un budget d'offres limité, le modèle multiplie par ~2.5× l'efficacité des campagnes.

---

## Comparaison des 3 Scénarios d'Offre

| Scénario | Coût | Clients ROI+ | % Base | Gain Total | Gain Moy/Client |
|---|---|---|---|---|---|
| **Légère** | 5 TND | 9 999 | ~100% | 469 628 TND | 47.0 TND |
| **Standard** | 15 TND | 9 957 | 99.6% | 369 678 TND | 37.1 TND |
| **Premium** | 30 TND | 6 815 | 68.2% | 240 711 TND | 35.3 TND |

> L'offre **Légère** maximise le gain total (budget faible → quasi toute la base est ciblable).
> L'offre **Premium** a une sélectivité naturelle : seulement 68.2% des clients justifient
> l'investissement de 30 TND. C'est le scénario le plus discriminant.

---

## Sensibilité Coût × LTV

La matrice de sensibilité montre comment le gain varie selon l'horizon temporel et le coût :

- **Horizon 6 mois** (offre temporaire) : gain réduit de moitié vs 12 mois
- **Horizon 18 mois** (client fidélisé longtemps) : gain 50% supérieur
- **Coût > 40 TND** : seulement les clients ARPU > 30 TND restent ROI+

---

## Formule de Ciblage Optimal (Règle Métier)

Cibler un client si :

```python
P_churn × P_réponse > Coût_offre / (ARPU × LTV_horizon)
```

Exemples :
- ARPU = 20 TND, coût = 15 TND, LTV = 12 mois → seuil = **0.063**
- ARPU = 30 TND, coût = 15 TND, LTV = 12 mois → seuil = **0.042**
- ARPU = 10 TND, coût = 30 TND, LTV = 12 mois → seuil = **0.250**

> Plus l'ARPU est faible, plus le seuil `P_churn × P_réponse` doit être élevé pour
> justifier un investissement de rétention. Ce seuil s'adapte automatiquement à la valeur
> de chaque client.

---

## Synthèse des 4 Objectifs (Obj 6 + 7 + 8 + 9)

| Objectif | Output | Rôle dans Obj 9 |
|---|---|---|
| **Obj 6** — Propension churn | P_churn ∈ [0,1] | Prob. que l'action soit nécessaire |
| **Obj 7** — ARPU prédit | ARPU (TND) | Valeur sauvée si rétention réussie |
| **Obj 8** — Réponse offre | P_réponse ∈ [0,1] | Prob. que l'action soit acceptée |
| **Obj 9** — Gain attendu | Gain_Net (TND) | Priorisation ROI des campagnes |

```
Score_Priorité(i) = P_churn × P_réponse × ARPU × 12 - Coût_offre
                  = (Obj 6) × (Obj 8) × (Obj 7) × 12 - Coût
```

Ce score unique permet de **trier tous les clients** par ordre de priorité de rétention,
en tenant compte simultanément du risque, de la réceptivité et de la valeur financière.

---

## Fichiers Générés

| Fichier | Description |
|---|---|
| `09_retention_gain.ipynb` | Notebook complet |
| `09_gain_components.png` | Distribution des 3 composantes + gain brut/net |
| `09_gain_curve.png` | Courbe de gain cumulé + courbe de lift |
| `09_scenarios.png` | Comparaison 3 scénarios de coût |
| `09_sensitivity.png` | Heatmap sensibilité coût × LTV horizon |
| `09_gain_by_segment.png` | Gain moyen/total par segment client |
| `09_targeting_map.png` | Carte propension × réponse colorée par gain |
| `09_summary_table.png` | Tableau comparatif scénarios |
| `09_retention_gain_run_log.json` | Métriques + scores gain 10k clients |
