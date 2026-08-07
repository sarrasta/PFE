# Objectif 8 — Probabilité de Réponse à une Offre de Rétention

**Méthode** : Classification binaire calibrée — labels synthétiques (~30% taux de réponse)  
**Notebook** : `08_offer_response.ipynb`  
**Clients** : 10 000 — score de réponse ∈ [0,1]

---

## Principe : Probabilité de Réponse à une Offre

Un client "répond favorablement" à une offre de rétention si :
1. **ARPU élevé** → il a de la valeur à conserver
2. **Ancienneté** → relation établie, fidélité potentielle
3. **QoS correcte** → satisfait du service de base
4. **Faibles impayés** → payeur fiable, lien contractuel fort
5. **Propension churn modérée** → à risque mais pas encore décidé à partir (sweet spot)
6. **Usage actif** → il valorise le service

### Génération des labels synthétiques réalistes

```python
# Score latent (règles métier)
logit = (
    + 1.2 * arpu_norm           # Haute valeur → fidélisable
    + 0.9 * anciennete_norm     # Loyauté → répond à la reconnaissance
    + 0.7 * qos_norm            # Satisfaction réseau
    - 0.8 * drop_rate_norm      # Frustration réseau → moins réceptif
    - 0.9 * impayes_norm        # Problèmes paiement → moins engagé
    + 1.4 * prop_sweet          # Sweet spot : risque modéré ≈ 0.40
    + 0.5 * minutes_norm        # Usage actif
    + intercept_calibré          # → taux de réponse 30%
    + N(0, 0.45)                # Bruit réaliste
)
```

> **Sweet spot** : Les clients avec propension churn ≈ 0.40 (risque modéré) répondent
> le mieux aux offres — ni trop fidèles (n'ont pas besoin), ni trop déçus (déjà partis).

---

## Résultats — Comparaison des 3 Modèles (CV 3-Fold)

| Modèle | AUC-ROC | Avg Precision | Brier ↓ | Seuil opt. | Statut |
|---|---|---|---|---|---|
| **LogReg** | **0.6185** | **0.4137** | 0.2387 | 0.389 | **MEILLEUR** |
| GBR | 0.6079 | 0.4062 | **0.2055** | 0.253 | Mieux calibré |
| RF | 0.6060 | 0.3989 | **0.2061** | 0.230 | |

> **Pourquoi LogReg gagne le ranking (AUC) ?**
> Même pattern que l'Objectif 6 (Ridge vs arbres) : avec des features standardisées
> et un signal linéairement séparable (ARPU + ancienneté + QoS), la régression logistique
> capture mieux les relations linéaires. Les arbres calibrés gagnent sur le Brier score
> (meilleures probabilités calibrées) mais perdent en discriminabilité.

---

## Déploiement — Distribution sur 10k Clients

| Tier de Réponse | Clients | % Base |
|---|---|---|
| **Faible (<25%)** | 53 | 0.5% |
| **Moyen (25–50%)** | 5 441 | **54.4%** |
| **Élevé (>50%)** | 4 506 | **45.1%** |

**Distribution des scores** :
- Mean : 0.492 ± 0.107  (min : ~0.15, max : ~0.80)
- Taux de réponse observé : **30.6%** (cible 30% ✓)

> La majorité des clients se situe dans la zone incertitude (25-50%), reflétant la réalité
> télécom : les offres de rétention ont un impact limité mais mesurable (~30%).

---

## Matrice d'Action : Propension Churn × Réponse Offre

Le livrable clé de l'Objectif 8 est le croisement avec l'Objectif 6 :

| | Propension Faible | Propension Moyenne | Propension Élevée |
|---|---|---|---|
| **Réponse Élevée** | Fidélisation premium (non-urgent) | ⭐ **CŒUR DE CIBLE** (offre prioritaire) | Offre urgente (fenêtre étroite) |
| **Réponse Moyenne** | Surveillance passive | Zone standard | Offre réseau/prix (avant départ) |
| **Réponse Faible** | Ne pas cibler | Surveiller | Client perdu (ROI négatif) |

### Priorité 1 — Cœur de cible (Propension Modérée + Réponse Élevée)
- Clients avec propension churn ∈ [0.35, 0.65] **ET** response_proba > 0.50
- **ARPU prédit (Obj 7)** > 20 TND pour filtrer par valeur

```python
# Formule de ciblage optimal (croisement Obj 6 + Obj 7 + Obj 8)
cible_prioritaire = (
    (propension_churn.between(0.35, 0.65)) &   # Obj 6 : à risque modéré
    (response_proba > 0.50)                     &   # Obj 8 : réceptif à l'offre
    (arpu_predit > 20)                              # Obj 7 : valeur suffisante
)
# → ~15-20% de la base = clients ROI-positifs pour campagne de rétention
```

---

## Features les Plus Prédictives

| Rang | Feature | Interprétation |
|---|---|---|
| 1 | `arpu_mean` | ARPU moyen : proxy de valeur client |
| 2 | `anciennete_mois` | Ancienneté : fidélité potentielle |
| 3 | `qos_mean` | Satisfaction réseau |
| 4 | `impayes_mean` | Fiabilité paiement |
| 5 | `nps_mean` | Score de recommandation |
| 6 | `drop_rate_mean` | Frustration réseau |
| 7 | `minutes_mean` | Engagement vocal |
| 8 | `engagement_restant` | Fin de contrat = point de décision |

> **Feature clé** : `engagement_restant` (mois restants au contrat) — un client
> proche de la fin de son engagement est au point de décision optimal pour une offre.

---

## Lien avec les Objectifs Précédents

| Objectif | Contribution |
|---|---|
| **Obj 6** (Propension churn) | Axe X de la matrice d'action |
| **Obj 7** (ARPU prédit) | Filtre valeur client pour la formule de ciblage |
| **Obj 4 v3** (Segments comportementaux) | Analyse de réponse par profil |
| **Obj 5** (Profils de rétention) | Sélection du type d'offre par segment |

---

## Recommandations Marketing par Segment

| Segment | Score Réponse Moy. | Offre Recommandée |
|---|---|---|
| Postpayé | ~0.51 | Offre fidélité + bonus data |
| Prépayé | ~0.48 | Offre tarifaire + recharge bonifiée |

> Croiser avec les profils de l'Obj 5 pour adapter le type d'offre au segment comportemental.

---

## Fichiers Générés

| Fichier | Description |
|---|---|
| `08_offer_response.ipynb` | Notebook complet |
| `08_label_eda.png` | Distribution labels + réponse vs propension/ARPU |
| `08_model_comparison.png` | ROC + métriques + calibration |
| `08_deployment.png` | Distribution scores + pie + hexbin + segment + ARPU |
| `08_feature_importance.png` | Top 20 features (LogReg coefficients) |
| `08_action_matrix.png` | Matrice Propension × Réponse avec recommandations |
| `08_summary_table.png` | Tableau comparatif modèles |
| `08_offer_response_run_log.json` | Métriques + scores 10k clients exportés |
