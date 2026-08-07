import nbformat as nbf
import os

nb  = nbf.v4.new_notebook()
cells = []

# ── Cell 0 : Title ──────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
    "# Objectif 8 — Probabilité de Réponse à une Offre de Rétention\n\n"
    "**But** : Prédire la probabilité qu'un client accepte une offre de rétention,\n"
    "afin de prioriser les interventions et maximiser le ROI des campagnes.\n\n"
    "**Méthode** : Classification binaire calibrée (LogReg · GBR · RF) sur labels\n"
    "synthétiques générés à partir de règles métier réalistes (~30 % taux de réponse).\n\n"
    "**Livrables** : Score de réponse ∈ [0,1] pour les 10 000 clients + matrice d'action\n"
    "croisée avec la propension churn (Obj 6).\n\n"
    "**Base de données** : DW_TT — PostgreSQL localhost:5432"
))

# ── Cell 1 : Imports ─────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""import warnings; warnings.filterwarnings('ignore')
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sqlalchemy import create_engine
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import (roc_auc_score, brier_score_loss,
                              average_precision_score, roc_curve,
                              precision_recall_curve, precision_score, recall_score)
from sklearn.pipeline import Pipeline
from scipy.optimize import brentq

plt.rcParams['figure.dpi'] = 120
OUT = r'c:\\Users\\Admin\\ml pfe'
np.random.seed(42)

print("Imports OK")"""
))

# ── Cell 2 : Chargement données ───────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""engine = create_engine('postgresql://postgres:Sarra123@localhost:5432/DW_TT')

q = '''
SELECT
    fp."ClientFK"               AS client_id,
    fp."DateFK"                 AS mois,
    fp."ARPU"                   AS arpu,
    fp."MontantFacture"         AS montant_facture,
    fp."Minutes"                AS minutes,
    fp."DataGB"                 AS data_gb,
    fp."SMS"                    AS sms,
    fp."HorsForfait"            AS hors_forfait,
    fp."Impayes"                AS impayes,
    fp."RetardPaiementJours"    AS retard,
    fp."QoSScore"               AS qos,
    fp."DropRatePct"            AS drop_rate,
    fp."ThroughputMbps"         AS throughput,
    fp."OutageMinutes"          AS outage,
    fp."NPS"                    AS nps,
    dc."Segment"                AS segment,
    dc."TypeAbonnement"         AS type_abo,
    dc."AncienneteMois"         AS anciennete_mois,
    dc."EngagementRestantMois"  AS engagement_restant,
    COALESCE(do_."PrixOffre", 0)         AS prix_offre,
    COALESCE(dg."Region", 'Inconnu')    AS region
FROM   public."Fact_performance_client" fp
INNER  JOIN public."dim_client"         dc  ON fp."ClientFK"  = dc."Client_PK"
LEFT   JOIN public."dim_geographique"   dg  ON dc."ClientID"  = dg."ClientID"
LEFT   JOIN public."dim_offre"          do_ ON dc."ClientID"  = do_."ClientID"
WHERE  fp."DateFK" BETWEEN 25 AND 36
ORDER  BY fp."ClientFK", fp."DateFK"
'''
df_raw = pd.read_sql(q, engine)
print(f"Données    : {df_raw.shape[0]:,} lignes  |  {df_raw['client_id'].nunique():,} clients")
print(f"DateFK     : {df_raw['mois'].min()} → {df_raw['mois'].max()}  ({df_raw['mois'].nunique()} mois)")
print(f"ARPU moyen : {df_raw['arpu'].mean():.2f} TND")
print(f"Segments   : {df_raw['segment'].unique().tolist()}")"""
))

# ── Cell 3 : Feature Engineering (vectorisé) ─────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""NUM_COLS = ['arpu','montant_facture','minutes','data_gb','sms',
            'hors_forfait','impayes','retard','qos','drop_rate',
            'throughput','outage','nps']

ALL_DATES    = sorted(df_raw['mois'].unique())
RECENT_DATES = ALL_DATES[-3:]
EARLY_DATES  = ALL_DATES[:3]

df_s = df_raw.copy()
for c in NUM_COLS:
    df_s[c] = pd.to_numeric(df_s[c], errors='coerce').fillna(0)

# ── Stats de base ─────────────────────────────────────────────────────────────
df_stats = df_s.groupby('client_id')[NUM_COLS].agg(['mean','std','min','max']).fillna(0)
df_stats.columns = [f'{c}_{s}' for c, s in df_stats.columns]

# ── Moyenne récente vs ancienne ───────────────────────────────────────────────
df_recent = (df_s[df_s['mois'].isin(RECENT_DATES)]
             .groupby('client_id')[NUM_COLS].mean().fillna(0)
             .rename(columns={c: f'{c}_recent' for c in NUM_COLS}))
df_early  = (df_s[df_s['mois'].isin(EARLY_DATES)]
             .groupby('client_id')[NUM_COLS].mean().fillna(0)
             .rename(columns={c: f'{c}_early' for c in NUM_COLS}))

# ── Taux de croissance ────────────────────────────────────────────────────────
growth_data = {}
for c in NUM_COLS:
    growth_data[f'{c}_growth'] = (
        df_recent[f'{c}_recent'].values /
        (df_early[f'{c}_early'].values + 0.001) - 1
    )
df_growth = pd.DataFrame(growth_data, index=df_recent.index)

# ── Pente linéaire (vectorisée) ───────────────────────────────────────────────
nd   = len(ALL_DATES)
x    = np.arange(nd, dtype=float); x -= x.mean()
ssx  = (x**2).sum()
trend_rec = {}
for col in NUM_COLS:
    piv = (df_s.pivot_table(index='client_id', columns='mois',
                             values=col, fill_value=0)
             .reindex(columns=sorted(ALL_DATES), fill_value=0))
    v   = piv.values.astype(float)
    yb  = v.mean(axis=1, keepdims=True)
    trend_rec[f'{col}_trend'] = pd.Series(
        (x * (v - yb)).sum(axis=1) / ssx, index=piv.index)
df_trend = pd.DataFrame(trend_rec)

# ── Features statiques ────────────────────────────────────────────────────────
df_static = (df_s.groupby('client_id')
             [['segment','type_abo','anciennete_mois','engagement_restant',
               'prix_offre','region']].last())

le_seg  = LabelEncoder()
le_abo  = LabelEncoder()
le_reg  = LabelEncoder()
df_static['segment_enc']  = le_seg.fit_transform(df_static['segment'].fillna('Inconnu'))
df_static['type_abo_enc'] = le_abo.fit_transform(df_static['type_abo'].fillna('Inconnu'))
df_static['region_enc']   = le_reg.fit_transform(df_static['region'].fillna('Inconnu'))
df_static['anciennete_mois']    = pd.to_numeric(df_static['anciennete_mois'],    errors='coerce').fillna(0)
df_static['engagement_restant'] = pd.to_numeric(df_static['engagement_restant'], errors='coerce').fillna(0)
df_static['prix_offre']         = pd.to_numeric(df_static['prix_offre'],         errors='coerce').fillna(25.0)

STATIC_COLS = ['segment_enc','type_abo_enc','region_enc',
               'anciennete_mois','engagement_restant','prix_offre']

# ── Assemblage final ─────────────────────────────────────────────────────────
df_feat = (df_stats
           .join(df_recent, how='left')
           .join(df_early,  how='left')
           .join(df_growth, how='left')
           .join(df_trend,  how='left')
           .join(df_static[STATIC_COLS], how='left')
           .fillna(0))

print(f"Feature matrix : {df_feat.shape[0]:,} clients × {df_feat.shape[1]} features")
print(f"Top features   : {list(df_feat.columns[:5])} …")"""
))

# ── Cell 4 : Génération labels de réponse synthétiques ────────────────────────
cells.append(nbf.v4.new_code_cell(
"""# ─────────────────────────────────────────────────────────────────────────────
# Génération de labels de réponse réalistes (cible ~30%)
#
# Un client "répond à l'offre" si :
#  1. ARPU élevé      → vaut la peine d'être retenu
#  2. Ancienneté      → relation établie, loyauté
#  3. QoS correcte    → satisfait du service de base
#  4. Drop_rate faible→ pas de frustration réseau
#  5. Impayes faibles → payeur fiable, lien contractuel fort
#  6. Engagement      → en fin de contrat = au moment décision
#  7. NPS moyen       → ni trop satisfait (n'a pas besoin), ni trop déçu (trop tard)
#  8. Usage actif     → valorise le service
# ─────────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

def norm01(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-9)

arpu_n       = norm01(df_feat['arpu_mean'])
anc_n        = norm01(df_feat['anciennete_mois'])
qos_n        = norm01(df_feat['qos_mean'])
drop_n       = norm01(df_feat['drop_rate_mean'])    # inversé
impayes_n    = norm01(df_feat['impayes_mean'])       # inversé
engag_n      = norm01(df_feat['engagement_restant']) # fin de contrat → au seuil
minutes_n    = norm01(df_feat['minutes_mean'])
nps_n        = norm01(df_feat['nps_mean'])

# Propension churn proxy (fabriquée depuis les features)
prop_proxy = 1 / (1 + np.exp(-(
    - 0.8 * arpu_n
    + 0.9 * impayes_n
    + 0.6 * drop_n
    - 0.5 * qos_n
    - 0.4 * nps_n
)))

# Sweet spot : clients à risque MODÉRÉ (~0.4) répondent le mieux aux offres
prop_sweet = np.exp(-8.0 * (prop_proxy - 0.40)**2)

# Score latent
logit_raw = (
    + 1.2 * arpu_n           # Haute valeur → fidélisable
    + 0.9 * anc_n            # Ancienneté → loyauté
    + 0.7 * qos_n            # Satisfaction réseau
    - 0.8 * drop_n           # Frustration réseau
    - 0.9 * impayes_n        # Problèmes paiement
    + 1.4 * prop_sweet       # Sweet spot risque moyen
    + 0.5 * minutes_n        # Usage actif
    + 0.3 * (1 - engag_n)   # Fin d'engagement = décision imminente
    + 0.3 * nps_n            # Satisfaction globale
    + rng.normal(0, 0.45, len(df_feat))
)

# Ajuster l'intercept pour atteindre exactement 30% de taux de réponse
TARGET_RATE = 0.30
try:
    intercept = brentq(
        lambda t: (1 / (1 + np.exp(-(logit_raw + t)))).mean() - TARGET_RATE,
        -10, 10
    )
except:
    intercept = -1.0

prob_true = 1 / (1 + np.exp(-(logit_raw + intercept)))
y_response = rng.binomial(1, prob_true).astype(int)

print(f"Taux de réponse  : {y_response.mean():.1%}  ({y_response.sum():,}/{len(y_response):,})")
print(f"P(réponse) moy.  : {prob_true.mean():.3f} ± {prob_true.std():.3f}")
print(f"Plage            : min={prob_true.min():.3f}  Q25={np.percentile(prob_true,25):.3f}  "
      f"med={np.median(prob_true):.3f}  Q75={np.percentile(prob_true,75):.3f}  max={prob_true.max():.3f}")"""
))

# ── Cell 5 : EDA labels ───────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Distribution P(réponse)
axes[0].hist(prob_true, bins=50, color='#2980b9', alpha=0.85, edgecolor='white', lw=0.3)
axes[0].axvline(0.5, color='red',    ls='--', lw=1.5, label='0.50')
axes[0].axvline(0.30, color='orange', ls=':',  lw=1.5, label='0.30')
axes[0].set_title('Distribution P(réponse)', fontweight='bold')
axes[0].set_xlabel('Probabilité de réponse vraie'); axes[0].set_ylabel('Clients')
axes[0].legend(fontsize=8)

# Taux de réponse par quintile de propension churn
prop_q = pd.qcut(prop_proxy, 5, labels=['Q1\\n(Faible)','Q2','Q3','Q4','Q5\\n(Élevé)'])
resp_by_prop = pd.Series(y_response, index=df_feat.index).groupby(prop_q.values).mean()
colors_bar = ['#27ae60','#2ecc71','#f39c12','#e67e22','#e74c3c']
axes[1].bar(resp_by_prop.index, resp_by_prop.values, color=colors_bar, alpha=0.85)
axes[1].set_title('Taux réponse vs propension churn', fontweight='bold')
axes[1].set_xlabel('Quintile propension churn (proxy)'); axes[1].set_ylabel('Taux de réponse')
for i, v in enumerate(resp_by_prop.values):
    axes[1].text(i, v + 0.005, f'{v:.1%}', ha='center', fontsize=9)

# Taux de réponse par tier ARPU
arpu_tier = pd.cut(df_feat['arpu_mean'], bins=[0,10,20,35,200],
                    labels=['<10 TND','10–20 TND','20–35 TND','>35 TND'])
resp_by_arpu = pd.Series(y_response, index=df_feat.index).groupby(arpu_tier.values).mean()
axes[2].bar(resp_by_arpu.index, resp_by_arpu.values,
            color=['#bdc3c7','#85c1e9','#2e86c1','#1a5276'], alpha=0.85)
axes[2].set_title('Taux réponse par tier ARPU', fontweight='bold')
axes[2].set_xlabel('Tier ARPU (TND)'); axes[2].set_ylabel('Taux de réponse')
for i, v in enumerate(resp_by_arpu.values):
    axes[2].text(i, v + 0.004, f'{v:.1%}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT, '08_label_eda.png'), dpi=120, bbox_inches='tight')
plt.show()
print("✓ 08_label_eda.png")"""
))

# ── Cell 6 : Entraînement modèles ─────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""FEAT_COLS = list(df_feat.columns)
X = df_feat[FEAT_COLS].values.astype(float)
y = y_response
print(f"X : {X.shape}  |  y : {y.shape}  |  positifs : {y.mean():.1%}")

# ── Définition des modèles ────────────────────────────────────────────────────
MODELS = {
    'LogReg': Pipeline([
        ('sc',  RobustScaler()),
        ('clf', LogisticRegression(C=0.5, class_weight='balanced',
                                   max_iter=1000, random_state=42))
    ]),
    'GBR': CalibratedClassifierCV(
        GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=10, random_state=42),
        method='isotonic', cv=3),
    'RF': CalibratedClassifierCV(
        RandomForestClassifier(
            n_estimators=150, max_depth=5, min_samples_leaf=15,
            n_jobs=-1, class_weight='balanced', random_state=42),
        method='isotonic', cv=3),
}

CV      = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
results = {}

for name, model in MODELS.items():
    print(f"\\n→ {name} …")
    oof = cross_val_predict(model, X, y, cv=CV, method='predict_proba')[:, 1]
    auc   = roc_auc_score(y, oof)
    ap    = average_precision_score(y, oof)
    brier = brier_score_loss(y, oof)
    prec_arr, rec_arr, thrs = precision_recall_curve(y, oof)
    f1s    = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9)
    opt_t  = thrs[np.argmax(f1s[:-1])]
    y_pred = (oof >= opt_t).astype(int)
    results[name] = {
        'auc': auc, 'ap': ap, 'brier': brier,
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall':    recall_score(y, y_pred, zero_division=0),
        'best_thr':  float(opt_t),
        'oof':       oof
    }
    print(f"  AUC={auc:.4f}  AP={ap:.4f}  Brier={brier:.4f}  "
          f"P={results[name]['precision']:.3f}  R={results[name]['recall']:.3f}  "
          f"Seuil={opt_t:.3f}")

BEST = max(results, key=lambda k: results[k]['auc'])
print(f"\\n★ Meilleur modèle : {BEST}  AUC={results[BEST]['auc']:.4f}")"""
))

# ── Cell 7 : Comparaison modèles ──────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""fig, axes = plt.subplots(1, 3, figsize=(16, 5))
COLORS = {'LogReg': '#e74c3c', 'GBR': '#2ecc71', 'RF': '#3498db'}

# Courbes ROC
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y, res['oof'])
    axes[0].plot(fpr, tpr, color=COLORS[name], lw=2,
                 label=f"{name}  AUC={res['auc']:.3f}")
axes[0].plot([0,1],[0,1],'k--',lw=1); axes[0].set_aspect('equal')
axes[0].set_title('Courbes ROC (OOF 3-Fold)', fontweight='bold')
axes[0].set_xlabel('Faux Positifs'); axes[0].set_ylabel('Vrais Positifs')
axes[0].legend(fontsize=9)

# Métriques comparées
metrics = ['auc','ap','brier']
labels  = ['AUC-ROC','Avg Precision','Brier ↓']
w = 0.25
for i, (name, res) in enumerate(results.items()):
    axes[1].bar(np.arange(3) + i*w, [res['auc'], res['ap'], res['brier']],
                width=w, label=name, color=COLORS[name], alpha=0.85)
axes[1].set_xticks(np.arange(3) + w)
axes[1].set_xticklabels(labels); axes[1].set_ylabel('Score')
axes[1].set_title('Comparaison métriques (3-Fold)', fontweight='bold')
axes[1].legend()

# Courbes de calibration
for name, res in results.items():
    fp2, mp2 = calibration_curve(y, res['oof'], n_bins=10)
    axes[2].plot(mp2, fp2, 's-', color=COLORS[name], lw=2, label=name)
axes[2].plot([0,1],[0,1],'k--',lw=1,label='Parfaite')
axes[2].set_title('Courbes de calibration', fontweight='bold')
axes[2].set_xlabel('P prédite'); axes[2].set_ylabel('Fraction positifs')
axes[2].legend(fontsize=9); axes[2].set_aspect('equal')

plt.suptitle('Objectif 8 — Comparaison des modèles de réponse à une offre',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '08_model_comparison.png'), dpi=120, bbox_inches='tight')
plt.show()
print("✓ 08_model_comparison.png")"""
))

# ── Cell 8 : Déploiement 10k clients ─────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""# Entraîner le meilleur modèle sur toute la base d'apprentissage
best_model = MODELS[BEST]
best_model.fit(X, y)
resp_proba  = best_model.predict_proba(X)[:, 1]
thr         = results[BEST]['best_thr']
resp_pred   = (resp_proba >= thr).astype(int)

# ── Tableau de déploiement ────────────────────────────────────────────────────
df_deploy = df_static[['segment','type_abo','anciennete_mois',
                         'engagement_restant','prix_offre','region']].copy()
df_deploy = df_deploy.reindex(df_feat.index)
df_deploy['arpu_mean']        = df_feat['arpu_mean'].values
df_deploy['qos_mean']         = df_feat['qos_mean'].values
df_deploy['impayes_mean']     = df_feat['impayes_mean'].values
df_deploy['nps_mean']         = df_feat['nps_mean'].values
df_deploy['prop_proxy']       = prop_proxy.values
df_deploy['response_proba']   = resp_proba
df_deploy['response_pred']    = resp_pred

df_deploy['response_tier'] = pd.cut(
    resp_proba,
    bins=[-0.001, 0.25, 0.50, 1.001],
    labels=['Faible (<25%)', 'Moyen (25-50%)', 'Élevé (>50%)']
)

print(f"★ Modèle déployé : {BEST}")
print(f"\\nDistribution du score de réponse :")
print(f"  mean={resp_proba.mean():.3f}  std={resp_proba.std():.3f}  "
      f"min={resp_proba.min():.3f}  max={resp_proba.max():.3f}")
print(f"\\nRépartition par tier :")
for tier, cnt in df_deploy['response_tier'].value_counts().sort_index().items():
    print(f"  {tier} : {cnt:,} clients  ({cnt/len(df_deploy):.1%})")"""
))

# ── Cell 9 : Analyse par segment ─────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""seg_stats = (df_deploy.groupby('segment')
             .agg(
                 clients     =('response_proba','count'),
                 proba_moy   =('response_proba','mean'),
                 proba_med   =('response_proba','median'),
                 arpu_moy    =('arpu_mean','mean'),
                 pct_reponse =('response_pred','mean')
             )
             .sort_values('proba_moy', ascending=False))

print("Réponse par segment :")
print(seg_stats.to_string())

arpu_t = pd.cut(df_deploy['arpu_mean'], bins=[0,10,20,35,200],
                 labels=['<10','10-20','20-35','>35'])
arpu_stats = (df_deploy.groupby(arpu_t.values)
              .agg(clients=('response_proba','count'),
                   proba_moy=('response_proba','mean'))
              .sort_values('proba_moy', ascending=False))
print("\\nRéponse par tier ARPU :")
print(arpu_stats.to_string())"""
))

# ── Cell 10 : Visualisation déploiement ──────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

# A — Distribution score réponse
ax_a = fig.add_subplot(gs[0, 0])
ax_a.hist(resp_proba, bins=50, color='#2980b9', alpha=0.85, edgecolor='white', lw=0.3)
ax_a.axvline(thr,  color='red',    ls='--', lw=2, label=f'Seuil opt. {thr:.2f}')
ax_a.axvline(0.25, color='orange', ls=':',  lw=1.5, label='0.25')
ax_a.axvline(0.50, color='green',  ls=':',  lw=1.5, label='0.50')
ax_a.set_title('Distribution P(reponse)', fontweight='bold')
ax_a.set_xlabel("Score de reponse a l'offre"); ax_a.set_ylabel('Clients')
ax_a.legend(fontsize=8)

# B — Pie tiers
ax_b = fig.add_subplot(gs[0, 1])
tier_counts = df_deploy['response_tier'].value_counts().sort_index()
colors_pie  = ['#27ae60','#f39c12','#c0392b']
ax_b.pie(tier_counts.values, labels=tier_counts.index,
         colors=colors_pie[:len(tier_counts)], autopct='%1.1f%%',
         startangle=90, pctdistance=0.75)
ax_b.set_title('Répartition par tier de réponse', fontweight='bold')

# C — Hexbin réponse vs propension
ax_c = fig.add_subplot(gs[0, 2])
hb = ax_c.hexbin(df_deploy['prop_proxy'], df_deploy['response_proba'],
                  gridsize=30, cmap='YlOrRd', mincnt=1)
plt.colorbar(hb, ax=ax_c, label='Densité clients')
ax_c.axvline(0.35, color='blue', ls='--', lw=1, alpha=0.6, label='Seuil moyen')
ax_c.axvline(0.65, color='blue', ls='--', lw=1, alpha=0.6)
ax_c.set_title('Score réponse vs propension churn', fontweight='bold')
ax_c.set_xlabel('Propension churn (proxy)'); ax_c.set_ylabel('P(réponse offre)')
ax_c.legend(fontsize=8)

# D — Boxplot par segment
ax_d = fig.add_subplot(gs[1, 0:2])
seg_names = seg_stats.index.tolist()
data_box  = [df_deploy[df_deploy['segment']==s]['response_proba'].values for s in seg_names]
bp = ax_d.boxplot(data_box, patch_artist=True, notch=False,
                   medianprops=dict(color='white', lw=2.5))
seg_colors = ['#8e44ad','#e74c3c','#e67e22','#27ae60','#3498db','#95a5a6']
for patch, color in zip(bp['boxes'], seg_colors[:len(data_box)]):
    patch.set_facecolor(color); patch.set_alpha(0.75)
ax_d.set_xticks(range(1, len(seg_names)+1))
ax_d.set_xticklabels(seg_names, rotation=15, ha='right', fontsize=9)
for i, (s, row) in enumerate(seg_stats.iterrows()):
    ax_d.text(i+1, row['proba_moy'] + 0.02, f"{row['proba_moy']:.2f}",
              ha='center', fontsize=8, fontweight='bold', color='navy')
ax_d.set_title('Distribution P(reponse) par segment', fontweight='bold')
ax_d.set_ylabel("Probabilite de reponse a l'offre")

# E — Barplot par ARPU tier
ax_e = fig.add_subplot(gs[1, 2])
arpu_tier2 = pd.cut(df_deploy['arpu_mean'], bins=[0,10,20,35,200],
                     labels=['<10 TND','10-20 TND','20-35 TND','>35 TND'])
tier_m = df_deploy.groupby(arpu_tier2.values)['response_proba'].mean()
tier_s = df_deploy.groupby(arpu_tier2.values)['response_proba'].std()
ax_e.bar(tier_m.index, tier_m.values, yerr=tier_s.values,
         color='#2471a3', alpha=0.85, capsize=5, error_kw=dict(ecolor='gray'))
ax_e.set_title('P(réponse) moy. par tier ARPU', fontweight='bold')
ax_e.set_xlabel('Tier ARPU (TND)'); ax_e.set_ylabel('P(réponse)')
for i, (v, s) in enumerate(zip(tier_m.values, tier_s.values)):
    ax_e.text(i, v + s + 0.01, f'{v:.2f}', ha='center', fontsize=9, fontweight='bold')

plt.suptitle(f"Objectif 8 - Deploiement : Probabilite de Reponse a une Offre\\n"
             f"Modele : {BEST}  |  AUC={results[BEST]['auc']:.4f}  |  10 000 clients scores",
             fontsize=12, fontweight='bold')
plt.savefig(os.path.join(OUT, '08_deployment.png'), dpi=120, bbox_inches='tight')
plt.show()
print("✓ 08_deployment.png")"""
))

# ── Cell 11 : Feature Importance ─────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""# Extraire les importances selon le type de modèle
try:
    if BEST == 'LogReg':
        importances = np.abs(best_model.named_steps['clf'].coef_[0])
    elif BEST in ('GBR', 'RF'):
        importances = np.mean(
            [cc.estimator.feature_importances_
             for cc in best_model.calibrated_classifiers_],
            axis=0
        )
    else:
        importances = np.ones(len(FEAT_COLS)) / len(FEAT_COLS)
except Exception as e:
    print(f"Import. non disponibles ({e}) — utilisation valeurs égales")
    importances = np.ones(len(FEAT_COLS)) / len(FEAT_COLS)

fi = pd.Series(importances, index=FEAT_COLS).sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(fi.index[::-1], fi.values[::-1],
               color='#2471a3', alpha=0.85, edgecolor='white', lw=0.5)
ax.set_title(f'Top 20 Features — {BEST}\\nPrédiction de réponse à une offre',
             fontweight='bold', fontsize=12)
ax.set_xlabel('Importance relative')
for bar, val in zip(bars, fi.values[::-1]):
    ax.text(bar.get_width() + fi.values.max()*0.01,
            bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '08_feature_importance.png'), dpi=120, bbox_inches='tight')
plt.show()
print("✓ 08_feature_importance.png")"""
))

# ── Cell 12 : Matrice d'action Obj6 × Obj8 ───────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""# ─────────────────────────────────────────────────────────────────────────────
# MATRICE D'ACTION : Propension Churn (Obj 6 proxy) × Réponse Offre (Obj 8)
# Chaque cellule = % des 10k clients + recommandation d'action
# ─────────────────────────────────────────────────────────────────────────────
prop_cat = pd.cut(df_deploy['prop_proxy'],
                   bins=[-0.001, 0.35, 0.65, 1.001],
                   labels=['Faible', 'Moyen', 'Élevé'])
resp_cat = pd.cut(df_deploy['response_proba'],
                   bins=[-0.001, 0.25, 0.50, 1.001],
                   labels=['Faible', 'Moyen', 'Élevé'])

matrix_pct   = pd.crosstab(resp_cat, prop_cat, normalize='all') * 100
matrix_count = pd.crosstab(resp_cat, prop_cat)

# Réordonner : réponse élevée en haut, propension croissante à droite
matrix_pct   = matrix_pct.reindex(index=['Élevé','Moyen','Faible'],
                                    columns=['Faible','Moyen','Élevé'])
matrix_count = matrix_count.reindex(index=['Élevé','Moyen','Faible'],
                                      columns=['Faible','Moyen','Élevé'])

actions = [
    # Propension: Faible            Moyen                    Élevé
    ['Fidélisation premium\\n(non-urgent)',  'CŒUR DE CIBLE\\n(Offre prioritaire)',  'Offre urgente\\nFenêtre étroite'],
    ['Surveillance passive\\n(offre douce)', 'Zone standard\\n(offre classique)',     'Offre réseau/prix\\n(avant départ)'],
    ['Ne pas cibler\\n(risque très faible)', 'Surveiller\\n(évolution à suivre)',    'Client perdu\\n(ROI négatif)'],
]

fig, ax = plt.subplots(figsize=(11, 7))
im = ax.imshow(matrix_pct.values, cmap='RdYlGn_r', aspect='auto',
               vmin=0, vmax=matrix_pct.values.max()*1.2)
plt.colorbar(im, ax=ax, label='% des 10 000 clients', shrink=0.8)

ax.set_xticks([0,1,2])
ax.set_xticklabels(['Propension\\nFaible (<35%)', 'Propension\\nMoyenne (35-65%)',
                     'Propension\\nÉlevée (>65%)'], fontsize=10)
ax.set_yticks([0,1,2])
ax.set_yticklabels(['Réponse\\nÉlevée (>50%)', 'Réponse\\nMoyenne (25-50%)',
                     'Réponse\\nFaible (<25%)'], fontsize=10)

for i in range(3):
    for j in range(3):
        pct = matrix_pct.values[i, j]
        cnt = matrix_count.values[i, j]
        txt_color = 'black' if pct < matrix_pct.values.max() * 0.5 else 'white'
        ax.text(j, i, f'{pct:.1f}%\\n({cnt:,} clients)\\n{actions[i][j]}',
                ha='center', va='center', fontsize=7.5,
                color=txt_color,
                fontweight='bold' if (i==0 and j==1) else 'normal')

ax.set_title("Matrice d'Action : Propension Churn vs Reponse Offre\\n"
             "(% des 10 000 clients — recommandations marketing)",
             fontweight='bold', fontsize=12)
ax.set_xlabel('Propension au Churn (Objectif 6)', fontsize=11, labelpad=12)
ax.set_ylabel("Probabilite de Reponse a l'Offre (Objectif 8)", fontsize=11, labelpad=12)

# Encadrer la cellule prioritaire (Haut/Moyen)
from matplotlib.patches import Rectangle
rect = Rectangle((-0.5 + 1, -0.5 + 0), 1, 1,
                  fill=False, edgecolor='blue', linewidth=3.5, linestyle='-')
ax.add_patch(rect)
ax.text(1, -0.75, '★ Priorité 1', ha='center', fontsize=10,
        color='blue', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT, '08_action_matrix.png'), dpi=120, bbox_inches='tight')
plt.show()
print("✓ 08_action_matrix.png")
print(f"\\nCœur de cible (Propension Moyen + Réponse Élevée) : "
      f"{matrix_count.loc['Élevé','Moyen']:,} clients "
      f"({matrix_pct.loc['Élevé','Moyen']:.1f}%)")"""
))

# ── Cell 13 : Tableau récapitulatif ──────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""fig, ax = plt.subplots(figsize=(14, 4))
ax.axis('off')

col_hdrs = ['Modèle', 'AUC-ROC', 'Avg Precision', 'Brier ↓', 'Précision', 'Rappel', 'Seuil opt.', 'Statut']
rows = []
for name, res in results.items():
    rows.append([
        name,
        f"{res['auc']:.4f}",
        f"{res['ap']:.4f}",
        f"{res['brier']:.4f}",
        f"{res['precision']:.3f}",
        f"{res['recall']:.3f}",
        f"{res['best_thr']:.3f}",
        '★ MEILLEUR' if name == BEST else ''
    ])

tbl = ax.table(cellText=rows, colLabels=col_hdrs, loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 1.9)
for j in range(len(col_hdrs)):
    tbl[0, j].set_facecolor('#dae8fc')
for i, (name, _) in enumerate(results.items()):
    if name == BEST:
        for j in range(len(col_hdrs)):
            tbl[i+1, j].set_facecolor('#d5e8d4')

ax.set_title('Objectif 8 — Résultats comparatifs (CV 3-Fold)', fontweight='bold', fontsize=13, y=0.9)
plt.savefig(os.path.join(OUT, '08_summary_table.png'), dpi=120, bbox_inches='tight')
plt.show()
print("✓ 08_summary_table.png")"""
))

# ── Cell 14 : Sauvegarde & rapport final ─────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""# ── Export JSON ──────────────────────────────────────────────────────────────
log8 = {
    'objective': 'Probabilité de réponse à une offre de rétention',
    'best_model': BEST,
    'metrics': {
        name: {k: float(v) if isinstance(v, (float, np.floating)) else v
               for k, v in res.items() if k != 'oof'}
        for name, res in results.items()
    },
    'deployment': {
        'n_clients':         len(df_deploy),
        'taux_reponse_obs':  float(y_response.mean()),
        'score_mean':        float(resp_proba.mean()),
        'score_std':         float(resp_proba.std()),
        'score_min':         float(resp_proba.min()),
        'score_max':         float(resp_proba.max()),
        'threshold':         float(thr),
        'tier_counts':       {str(k): int(v)
                              for k, v in df_deploy['response_tier'].value_counts().items()},
    },
    'response_scores': {
        str(int(cid)): float(p)
        for cid, p in zip(df_feat.index, resp_proba)
    }
}

log_path = os.path.join(OUT, '08_offer_response_run_log.json')
with open(log_path, 'w', encoding='utf-8') as f:
    json.dump(log8, f, ensure_ascii=False, indent=2, default=str)
print(f"✓ Log sauvegardé : {log_path}")

# ── Résumé final ─────────────────────────────────────────────────────────────
print("\\n" + "═"*65)
print("OBJECTIF 8 — RÉSUMÉ FINAL")
print("═"*65)
print(f"Meilleur modèle  : {BEST}")
print(f"AUC-ROC          : {results[BEST]['auc']:.4f}")
print(f"Avg Precision    : {results[BEST]['ap']:.4f}")
print(f"Brier Score      : {results[BEST]['brier']:.4f}")
print(f"Seuil optimal    : {thr:.3f}")
print()
print(f"Déploiement 10k clients :")
for tier, cnt in df_deploy['response_tier'].value_counts().sort_index().items():
    print(f"  {tier:25s} : {cnt:5,} clients  ({cnt/len(df_deploy):.1%})")
print()
print("Recommandation : Croiser avec Obj 6 (propension churn) + Obj 7 (ARPU prédit)")
print("  → Cibler clients à Propension Modérée + Réponse Élevée + ARPU > 20 TND")
priority_mask = (
    (df_deploy['prop_proxy'].between(0.35, 0.65)) &
    (df_deploy['response_proba'] > 0.50) &
    (df_deploy['arpu_mean'] > 20)
)
print(f"  → Cœur de cible estimé : {priority_mask.sum():,} clients "
      f"({priority_mask.mean():.1%} de la base)")
print("═"*65)"""
))

nb['cells'] = cells

OUT_PATH = r'c:\Users\Admin\ml pfe\08_offer_response.ipynb'
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f"Written: {OUT_PATH}")
