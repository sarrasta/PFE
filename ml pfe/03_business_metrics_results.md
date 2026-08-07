# Objectif 3 — Optimisation des Métriques Métier

**Métriques cibles** : Recall@Top10%, PR-AUC, AUC-ROC, Lift@TopK%, Calibration  
**Notebook** : `03_business_metrics.ipynb`  
**Base de données** : `DW_TT` — PostgreSQL localhost:5432

---

## Protocole d'Évaluation

| Paramètre | Valeur |
|---|---|
| **Prédictions** | Cross-validated out-of-fold (`cross_val_predict`, 5-fold stratifié) |
| **Données** | 422 clients étiquetés (DateFK=36), 35 features (v5) |
| **Prévalence churn** | 93.6% (395 churners / 27 non-churners) |
| **Scoring optimisation** | `roc_auc`, `average_precision`, scorer custom `Recall@Top10%` |

---

## Résultats — Comparaison des Métriques Métier

| Modèle | AUC-ROC | PR-AUC | R@Top10% | Lift@10% | R@Top20% | Lift@20% | Brier ↓ |
|---|---|---|---|---|---|---|---|
| **SVC_RBF (v5 best)** | **0.7004** | **0.9570** | 0.0987 | 0.99× | 0.2051 | 1.03× | 0.1558 |
| LR_SMOTE | 0.6868 | 0.9551 | **0.1013** | **1.01×** | 0.2051 | 1.03× | 0.1844 |
| RF_balanced | 0.5895 | 0.9556 | **0.1013** | **1.01×** | **0.2076** | **1.04×** | 0.0745 |
| GB_balanced | 0.6085 | 0.9516 | **0.1013** | **1.01×** | 0.2025 | 1.01× | **0.0736** |
| **Random (baseline)** | 0.5000 | 0.9360 | 0.1000 | 1.00× | 0.2000 | 1.00× | — |

### Meilleur par métrique
| Métrique | Meilleur modèle | Valeur |
|---|---|---|
| **AUC-ROC** | SVC_RBF (v5 best) | **0.7004** |
| **PR-AUC** | SVC_RBF (v5 best) | **0.9570** |
| **Recall@Top10%** | LR / RF / GB | **0.1013** |
| **Calibration (Brier)** | GB_balanced | **0.0736** |

---

## Analyse Critique — Effet de la Prévalence

### Pourquoi Lift@Top10% ≈ 1.01× ?

Avec **93.6% de churners**, tout groupe de 10% contient ~93.6% de churners par simple hasard.  
Un modèle parfait ne peut capturer que 10.6% des churners dans les 10% supérieurs (vs 9.9% pour le hasard).  
L'écart est mathématiquement borné à **+0.7 points** de Recall.

> **Conclusion** : Lift@TopK% n'est pas une métrique discriminante quand la prévalence dépasse 90%.  
> Dans ce contexte, **AUC-ROC reste la métrique de référence** pour comparer les modèles.

### Pourquoi PR-AUC ≈ 0.95+ pour tous les modèles ?

La PR-AUC est dominée par la classe positive (churn=1 = 93.6% des données).  
La baseline aléatoire a déjà PR-AUC = prevalence = **0.936**.  
Les modèles améliorent légèrement (0.951–0.957) mais la marge est réduite.

> **Conclusion** : PR-AUC est pertinente uniquement si la classe positive est la **minorité** (typiquement <30%).  
> Ici, PR-AUC pour la classe non-churn (6.4%) serait plus informative, mais statistiquement instable (27 samples).

---

## Analyse Coût-Bénéfice — Seuil Optimal

### Paramètres métier (estimations télécom tunisien)

| Paramètre | Valeur |
|---|---|
| ARPU mensuel moyen | 35 TND/mois |
| Durée de rétention visée | 12 mois |
| Taux de succès de rétention | 40% des contacts |
| Coût de l'offre de rétention | 30 TND |
| Gain net si TP retenu | +138 TND |
| Perte nette si FP (contacté inutilement) | –30 TND |

### Calcul du seuil break-even

```
Contacter un client si : P(churn) × gain_TP + (1-P(churn)) × perte_FP > 0
=> P(churn) > 30 / (30 + 138) = 0.178 (18%)
```

> **Seuil décision recommandé : P(churn) > 0.18**  
> En dessous de ce seuil, le coût de l'offre dépasse le bénéfice attendu.

### Résultat optimal (SVC, seuil business)

| Paramètre | Valeur |
|---|---|
| Seuil optimal (maximise valeur nette) | 0.030 |
| Clients contactés | 420 / 422 (99.5%) |
| Precision | 0.938 |
| Recall | 0.997 |
| Valeur nette totale | +53 592 TND |

> **Note** : Le seuil optimal de 0.03 (=contacter presque tout le monde) confirme que le modèle  
> de churn est formé sur une population à 93.6% churners → tout client a une probabilité élevée.  
> En déploiement sur les 10k clients, la probabilité de base sera plus basse et le seuil plus discriminant.

---

## Calibration des Probabilités

| Modèle | Brier Score | Interprétation |
|---|---|---|
| **GB_balanced** | **0.0736** | Mieux calibré — probabilités fiables |
| **RF_balanced** | **0.0745** | Très proche, bon pour business |
| SVC_RBF | 0.1558 | Probabilités moins précises (SVC connu pour mauvaise calibration) |
| LR_SMOTE | 0.1844 | Oversmoothing après SMOTE |

> **Recommandation** : Pour un usage métier où les **probabilités** sont utilisées (tri des clients,  
> score de risque), préférer **GB_balanced** ou **RF_balanced** qui sont mieux calibrés.  
> Pour le **ranking pur** (tri relatif), SVC reste le meilleur (AUC-ROC=0.70).

---

## Recommandations Métier

### Stratégie de ciblage (par budget)

| Budget | Clients à contacter | Critère | Recall churners capturés |
|---|---|---|---|
| **Faible (10%)** | ~42 clients | P(churn) top 10% | ~10% — quasiment aléatoire |
| **Moyen (30%)** | ~127 clients | P(churn) > seuil | ~20% lift possible |
| **Large (80%)** | ~338 clients | P(churn) > 0.18 | >93% recall churners |
| **Maximal** | ~420 clients | P(churn) > 0.03 | ~100% recall churners |

### Métriques adaptées selon le contexte

| Contexte | Métrique recommandée | Modèle |
|---|---|---|
| Ranking pur (qui est le plus à risque) | **AUC-ROC** | SVC_RBF |
| Priorisation avec budget fixe | **Lift@TopK%** → limité par prévalence | RF/GB |
| Probabilités utilisées dans scoring | **Brier Score** | GB_balanced |
| Déploiement sur 10k clients | **Risk level** (Objectif 2) | RF 97.3% acc. |

---

## Scorers Custom Définis

```python
def recall_at_topk(y_true, y_prob, k=0.10):
    n_top = max(1, int(len(y_true) * k))
    idx_sorted = np.argsort(y_prob)[::-1]
    return y_true[idx_sorted[:n_top]].sum() / (y_true.sum() + 1e-9)

def lift_at_topk(y_true, y_prob, k=0.10):
    return recall_at_topk(y_true, y_prob, k) / k

scorer_r10   = make_scorer(recall_at_topk, needs_proba=True, k=0.10)
scorer_prauc = make_scorer(average_precision_score, needs_proba=True)
```

**Optimisation par scorer** (RandomizedSearchCV sur LR × 30 itérations) :  
Les trois scorers (AUC, PR-AUC, Recall@Top10%) convergent vers des paramètres similaires,  
confirmant que sur cette distribution, les métriques sont fortement corrélées.

---

## Fichiers Générés

| Fichier | Description |
|---|---|
| `03_business_metrics.ipynb` | Notebook complet — toutes métriques et visualisations |
| `03_gain_lift_curves.png` | Gain Chart + Lift Curve (4 modèles) |
| `03_roc_pr_curves.png` | Courbes ROC + Precision-Recall comparées |
| `03_threshold_business.png` | Analyse coût-bénéfice + Precision/Recall par seuil |
| `03_calibration.png` | Reliability Diagram + distribution des probabilités |
| `03_metrics_summary_table.png` | Tableau comparatif complet |
| `03_business_metrics_run_log.json` | Métriques numériques exportées |

---

## Conclusion

> Le modèle **SVC_RBF (v5)** reste le meilleur pour le **ranking** (AUC-ROC=0.70).  
> Pour les probabilités métier (scoring, alertes), **GB_balanced** est mieux calibré (Brier=0.074).  
> Les métriques de lift (Lift@Top10%) sont quasi-inutiles ici à cause de la **prévalence de 93.6%**.  
> La véritable valeur métier est apportée par l'**Objectif 2** (classification de risque sur 10k clients)  
> qui segmente la base entière en Faible/Moyen/Élevé avec 97.3% de précision.
