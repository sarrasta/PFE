# Objectif 5 — Profils Homogènes pour Campagnes de Rétention Personnalisées

**Méthode** : Matrice de ciblage 2D — Segment comportemental × Niveau de risque churn  
**Notebook** : `05_retention_profiles.ipynb`  
**Clients** : 10 000 — **12 micro-profils** identifiés

---

## Architecture de la Solution

```
Fact_performance_client (10k clients)
         │
         ├── SVC churn model (entraîné sur 422 étiquetés) ──► churn_proba
         │
         ├── K-Means Composite 4D (Obj 4 v2)  ──► segment comportemental (4 valeurs)
         │
         └── K-Means Risque k=3               ──► niveau de risque (3 valeurs)
                    │
                    └─► CROISEMENT → 12 micro-profils → campagne + ROI
```

**Métriques de clustering** :
- Segmentation comportementale 4D : Silhouette = **0.1997**
- Clustering risque k=3 : Silhouette = **0.5918** (bien défini)

---

## Matrice de Ciblage — 12 Micro-Profils

### Nombre de clients par cellule

| Segment | Risque Faible | Risque Moyen | Risque Élevé |
|---|---|---|---|
| **Gros Consommateurs** | 1 628 | 192 | 192 |
| **Clients Débiteurs** | 2 492 | 280 | 303 |
| **Clients Insatisfaits** | 2 143 | 268 | 260 |
| **Clients Réseau Dégradé** | 1 806 | 209 | 227 |

### Probabilité de churn par cellule

| Segment | Risque Faible | Risque Moyen | Risque Élevé |
|---|---|---|---|
| Gros Consommateurs | 0.615 | 0.671 | **0.862** |
| Clients Débiteurs | 0.751 | 0.764 | **0.876** |
| Clients Insatisfaits | 0.584 | 0.667 | **0.894** |
| Clients Réseau Dégradé | 0.742 | 0.825 | **0.944** |

---

## Tableau de Priorisation Complet

*(Trié par ROI net estimé : n × P(churn) × ARPU × 12 mois × 40% rétention − coût offres)*

| Priorité | Segment | Risque | Clients | P(churn) | ARPU | ROI net | Campagne |
|---|---|---|---|---|---|---|---|
| **P1** | Gros Consommateurs | Faible | 1 628 | 0.615 | — | **162 k TND** | Programme ambassadeur + cross-sell |
| **P2** | Clients Débiteurs | Faible | 2 492 | 0.751 | — | **98 k TND** | Monitoring + newsletter personnalisée |
| **P3** | Clients Réseau Dégradé | Faible | 1 806 | 0.742 | — | **74 k TND** | Monitoring + newsletter personnalisée |
| **P4** | Clients Insatisfaits | Faible | 2 143 | 0.584 | — | **64 k TND** | Monitoring + newsletter personnalisée |
| **P5** ⚠️ | Gros Consommateurs | **Élevé** | 192 | **0.862** | — | 27 k TND | **Offre VIP + conseiller dédié** |
| **P6** | Gros Consommateurs | Moyen | 192 | 0.671 | — | 20 k TND | Upsell + programme fidélité |
| **P7** ⚠️ | Clients Débiteurs | **Élevé** | 303 | **0.876** | — | 14 k TND | **Plan paiement urgence + offre rétention** |
| **P8** ⚠️ | Clients Insatisfaits | **Élevé** | 260 | **0.894** | — | 12 k TND | **Appel proactif + compensation SAV** |
| **P9** | Clients Débiteurs | Moyen | 280 | 0.764 | — | 12 k TND | Alerte précoce + échéancier flexible |
| **P10** ⚠️ | Clients Réseau Dégradé | **Élevé** | 227 | **0.944** | — | 12 k TND | **Fix technique + offre dédommagement** |
| **P11** | Clients Insatisfaits | Moyen | 268 | 0.667 | — | 9 k TND | Suivi SAV proactif + NPS survey |
| **P12** | Clients Réseau Dégradé | Moyen | 209 | 0.825 | — | 9 k TND | Amélioration réseau + communication |

> ⚠️ = action urgente — clients quasi-certains de churner

**Totaux** :
- Revenus annuels à risque : **1 816 k TND**
- ROI net total des campagnes : **515 k TND** (hypothèse : taux rétention 40%, coût offre 30 TND)

---

## Interprétation Métier

### Lecture de la matrice
Les clients **Risque Faible** génèrent le plus grand ROI absolu (P1–P4) car ils sont très nombreux (7 750 clients / 77.5%). Cependant, leur P(churn) de 0.58–0.75 reflète le biais du modèle churn formé sur 93.6% de churners.

Les clients **Risque Élevé** (P5–P10) sont peu nombreux (982 clients, ~10%) mais ont les P(churn) les plus critiques (0.86–0.94) → action immédiate requise.

### Stratégie de budget

| Budget disponible | Cibler en priorité | Clients couverts | P(churn) moy. |
|---|---|---|---|
| **Très limité** (<5k TND) | P5 + P7 (Élevé haute valeur) | ~500 clients | 0.87 |
| **Modéré** (<20k TND) | P5–P10 (tous Élevé + Moyen) | ~1 700 clients | 0.82 |
| **Large** (>50k TND) | Tous profils P1–P12 | 10 000 clients | 0.70 |

---

## Campagnes Détaillées par Micro-Profil

### 🔴 Risque Élevé — Actions immédiates (982 clients)

| Micro-profil | Action | Canal | Délai |
|---|---|---|---|
| **Gros Consomm × Élevé** (n=192) | Offre VIP, remise 15%, conseiller dédié | Appel direct | 48h |
| **Débiteurs × Élevé** (n=303) | Plan paiement 3 mois, suspension gracieuse | SMS + Agence | 24h |
| **Insatisfaits × Élevé** (n=260) | Appel proactif SAV, résolution prioritaire, bon cadeau | Appel direct | 24h |
| **Réseau Dégradé × Élevé** (n=227) | Diagnostic réseau immédiat, crédit facture | Technicien + SMS | 48h |

### 🟠 Risque Moyen — Prévention (949 clients)

| Micro-profil | Action | Canal | Délai |
|---|---|---|---|
| **Gros Consomm × Moyen** (n=192) | Upsell forfait + points fidélité | App/Email | 1 semaine |
| **Débiteurs × Moyen** (n=280) | Alerte paiement + flexibilité, offre échelon | SMS automatique | 3 jours |
| **Insatisfaits × Moyen** (n=268) | Suivi NPS mensuel, amélioration SAV | Email + Enquête | 1 semaine |
| **Réseau Dégradé × Moyen** (n=209) | Communication amélioration réseau + offre compensatoire | Email | 1 semaine |

### 🟢 Risque Faible — Fidélisation (8 069 clients)

| Micro-profil | Action | Objectif |
|---|---|---|
| **Gros Consomm × Faible** (n=1 628) | Programme ambassadeur, cross-sell assurance/roaming | Monétiser la fidélité |
| **Débiteurs × Faible** (n=2 492) | Newsletter pédagogique, rappel préventif | Maintenir le comportement |
| **Insatisfaits × Faible** (n=2 143) | Newsletter satisfaction, offres exclusives | Améliorer NPS |
| **Réseau Dégradé × Faible** (n=1 806) | Communication réseau, transparence SLA | Maintenir la confiance |

---

## Fichiers Générés

| Fichier | Description |
|---|---|
| `05_retention_profiles.ipynb` | Notebook complet |
| `05_targeting_matrix.png` | Heatmap 3D (Nb clients / P(churn) / Revenus à risque) |
| `05_priority_chart.png` | Matrice bulle + Top 6 ROI |
| `05_top4_radar.png` | Radar des 4 micro-profils prioritaires |
| `05_campaign_table.png` | Tableau complet campagnes |
| `05_retention_run_log.json` | Métriques et profils exportés |

---

## Lien avec les Objectifs Précédents

| Objectif | Contribution |
|---|---|
| **Obj 1** (Churn SVC) | churn_proba sur 10k → P(churn) par micro-profil |
| **Obj 2** (Risque k=3) | risk_level → axe vertical de la matrice |
| **Obj 4 v2** (Segmentation 4D) | segment comportemental → axe horizontal |
| **Obj 3** (Métriques métier) | ROI = n × P(churn) × ARPU × retention_rate |
