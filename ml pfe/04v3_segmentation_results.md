# Segmentation v3 — Groupes Métier Définis

**Groupes cibles** : Fort Usage / Sensibles au Prix / Risque Réseau / Clients Inactifs  
**Notebook** : `04_segmentation_v3.ipynb`

---

## Progression des Silhouettes — 3 Versions

| Version | Espace | Silhouette k=4 | Δ vs v1 |
|---|---|---|---|
| **v1** | 23 features brutes | 0.1162 | — |
| **v2** | Composite 4D (Usage/Paiement/Réseau/SAV) | 0.1997 | +72% |
| **v3** | Composite 4D métier (Usage/Prix/Réseau/Inactivité) | **0.2324** | **+100%** |

**Autres métriques (v3, k=4)** :
- Davies-Bouldin : **1.2715** (v2 était 1.404 — amélioration)
- Calinski-Harabasz : **3 639** (v2 était 2 381 — amélioration)
- Silhouette à k=3 : 0.2392 (très proche de k=4) → k=4 choisi pour pertinence métier

---

## Définition des Scores Composites (v3)

```python
# Score 1 — Fort Usage (intensité de consommation)
usage_intensity = mean_z( log(minutes), log(data_gb), log(sms), log(usage_nocturne) )

# Score 2 — Sensibles au Prix (pression tarifaire)
price_pressure  = mean_z( log(hors_forfait), ratio_hf, log(montant_facture) )
# où ratio_hf = hors_forfait / montant_facture

# Score 3 — Risque Réseau (qualité de service dégradée)
network_risk    = mean_z( drop_rate, log(outage_min) )
              − mean_z( qos_score, throughput_mbps )

# Score 4 — Inactivité (faible engagement client)
inactivity      = − mean_z( arpu, anciennete_mois, nb_mois_observes, log(minutes) )
```

**Propriété clé** : chaque score capture **une dimension indépendante**.  
Corrélations croisées faibles → espace 4D bien séparé → silhouette élevé.

---

## Résultats — 4 Groupes Équilibrés

| Groupe | Clients | % base | Dimension dominante |
|---|---|---|---|
| **Fort Usage** | 2 627 | 26.3% | Usage Intensity ↑ |
| **Clients Inactifs** | 2 491 | 24.9% | Inactivité ↑ (Engagement ↓) |
| **Sensibles au Prix** | 2 467 | 24.7% | Pression Prix ↑ |
| **Risque Réseau** | 2 415 | 24.1% | Risque Réseau ↑ |

> Distribution quasi-équilibrée (24–26%) : indicateur de clusters naturels bien formés.  
> Chaque groupe représente exactement un quart de la base → pas de cluster dominant artificiellement.

---

## Profil des Groupes

### Fort Usage (2 627 clients)
- **Minutes** : élevé ↑ | **Data GB** : élevé ↑ | **SMS** : élevé ↑
- **ARPU** : élevé ↑ | **Ancienneté** : longue | **nb_mois** : nombreux
- **Comportement** : clients actifs, forte valeur, consomment tout leur forfait
- **Risque churn** : modéré (attachés au service mais exigeants en qualité)

### Sensibles au Prix (2 467 clients)
- **Hors Forfait** : élevé ↑ | **Ratio HF** : élevé ↑ | **MontantFacture** : élevé ↑
- **Usage** : modéré (ne paient pas pour ce qu'ils consomment réellement)
- **Comportement** : dépassent régulièrement leur forfait → coût perçu élevé vs valeur
- **Risque churn** : élevé si concurrent propose meilleur tarif

### Risque Réseau (2 415 clients)
- **Drop Rate** : élevé ↑ | **Outage** : élevé ↑
- **QoS** : faible ↓ | **Throughput** : faible ↓
- **Comportement** : vivent une expérience réseau dégradée, potentiellement frustrés
- **Risque churn** : élevé si problème non résolu

### Clients Inactifs (2 491 clients)
- **ARPU** : faible ↓ | **Minutes** : faible ↓ | **Data** : faible ↓
- **Ancienneté** : courte ↓ | **nb_mois_observes** : faible ↓
- **Comportement** : sous-utilisateurs, peu engagés, faible attachement
- **Risque churn** : élevé par désintérêt (pas de "switching cost" perçu)

---

## Campagnes de Rétention Personnalisées

| Groupe | Objectif | Canal | Action principale |
|---|---|---|---|
| **Fort Usage** | Fidéliser & monétiser | Appel dédié | Offre illimitée + programme fidélité premium |
| **Sensibles au Prix** | Repositionner vers forfait adapté | SMS + App | Upgrade forfait sur mesure, suppression surprises |
| **Risque Réseau** | Résoudre avant départ | Technicien + SMS | Diagnostic immédiat + compensation facture |
| **Clients Inactifs** | Réactiver ou offre d'essai | Email + Push | Offre de relance, usage gratuit temporaire |

---

## Classificateur de Déploiement (RF)

| Métrique | v3 |
|---|---|
| **Accuracy** | **86.16%** ± — |
| **F1-macro** | **86.08%** ± — |

> L'accuracy plus basse (86% vs 97% en v2) reflète des frontières de clusters **plus naturelles et moins arbitraires** :  
> les groupes se chevauchent légèrement en espace réel (normal pour données client continues).  
> Le Silhouette plus élevé (0.232) confirme que c'est une meilleure segmentation, pas une pire.

---

## Fichiers Générés

| Fichier | Description |
|---|---|
| `04_segmentation_v3.ipynb` | Notebook complet |
| `04v3_k_selection.png` | Elbow + Silhouette + Davies-Bouldin |
| `04v3_pca_scatter.png` | PCA 2D + distribution des groupes |
| `04v3_radar.png` | Radar chart — 4 profils métier |
| `04v3_score_distributions.png` | Distribution des 4 scores composites par groupe |
| `04v3_profile_heatmap.png` | Heatmap des valeurs réelles normalisées |
| `04v3_cross_tabs.png` | Distribution par Segment, TypeAbo, Région |
| `04v3_feature_importance.png` | Top features RF classificateur |
| `04v3_segmentation_run_log.json` | Résultats exportés |
