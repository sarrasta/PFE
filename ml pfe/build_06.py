import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# ── Cell 0 : Title ────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
    "# Objectif 6 — Régression : Score Continu de Propension au Churn\n\n"
    "**Méthode** : Régression pondérée (balanced weights) → score ∈ [0,1]  \n"
    "**Calibration** : Isotonic Regression post-hoc pour une distribution significative  \n"
    "**Cible** : Accuracy ≥ 90% | AUC-ROC ≥ 0.70 | 10k clients scorés  \n"
    "**Base de données** : DW_TT — PostgreSQL localhost:5432"
))

# ── Cell 1 : Imports ─────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json, os
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    roc_auc_score, accuracy_score, f1_score, roc_curve,
    brier_score_loss
)
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 120
OUT = r'c:\\Users\\Admin\\ml pfe'
print("Imports OK")"""
))

# ── Cell 2 : Load data ────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""engine = create_engine('postgresql://postgres:Sarra123@localhost:5432/DW_TT')

q_labeled = '''
SELECT fc."ClientFK" AS client_id, fc."Churn_30j" AS churn,
    dc."Sexe", dc."Segment", dc."TypeAbonnement",
    dc."AncienneteMois" AS anciennete_mois,
    dc."EngagementRestantMois" AS engagement_restant,
    CASE WHEN dc."RGPD_Consent"=\'True\' THEN 1 ELSE 0 END AS rgpd_consent,
    COALESCE(dg."Region",\'Inconnu\') AS region,
    COALESCE(do_."OffreActuelle",\'None\') AS offre_actuelle,
    COALESCE(do_."PrixOffre",0) AS prix_offre,
    COALESCE(do_."OffreCible",\'None\') AS offre_cible,
    COALESCE(AVG(fp."Minutes"),0)             AS avg_minutes,
    COALESCE(AVG(fp."DataGB"),0)              AS avg_data_gb,
    COALESCE(AVG(fp."SMS"),0)                 AS avg_sms,
    COALESCE(AVG(fp."RoamingGB"),0)           AS avg_roaming_gb,
    COALESCE(AVG(fp."UsageNocturneRatio"),0)  AS avg_usage_nocturne,
    COALESCE(AVG(fp."MontantFacture"),0)      AS avg_montant_facture,
    COALESCE(AVG(fp."ARPU"),0)                AS avg_arpu,
    COALESCE(AVG(fp."HorsForfait"),0)         AS avg_hors_forfait,
    COALESCE(SUM(fp."Impayes"),0)             AS total_impayes,
    COALESCE(MAX(fp."RetardPaiementJours"),0) AS max_retard_paiement,
    COALESCE(AVG(fp."RetardPaiementJours"),0) AS avg_retard_paiement,
    COALESCE(AVG(fp."QoSScore"),5)            AS avg_qos,
    COALESCE(AVG(fp."DropRatePct"),0)         AS avg_drop_rate,
    COALESCE(AVG(fp."ThroughputMbps"),0)      AS avg_throughput,
    COALESCE(AVG(fp."OutageMinutes"),0)       AS avg_outage_min,
    COALESCE(SUM(fp."Tickets"),0)             AS total_tickets,
    COALESCE(AVG(fp."DelaiResolutionJ"),0)    AS avg_delai_resolution,
    COALESCE(AVG(fp."NPS"),5)                 AS avg_nps,
    COUNT(fp."DateFK") AS nb_mois_observes
FROM public."Fact_churn" fc
INNER JOIN public."dim_client" dc ON fc."ClientFK"=dc."Client_PK"
LEFT  JOIN public."dim_geographique" dg ON dc."ClientID"=dg."ClientID"
LEFT  JOIN public."dim_offre" do_ ON dc."ClientID"=do_."ClientID"
LEFT  JOIN public."Fact_performance_client" fp ON fc."ClientFK"=fp."ClientFK"
WHERE fc."DateFK"=36
GROUP BY fc."ClientFK",fc."Churn_30j",dc."Sexe",dc."Segment",dc."TypeAbonnement",
    dc."AncienneteMois",dc."EngagementRestantMois",dc."RGPD_Consent",
    dg."Region",do_."OffreActuelle",do_."PrixOffre",do_."OffreCible"
'''

q_all = '''
SELECT dc."Client_PK" AS client_id,
    dc."Sexe", dc."Segment", dc."TypeAbonnement",
    dc."AncienneteMois" AS anciennete_mois,
    dc."EngagementRestantMois" AS engagement_restant,
    CASE WHEN dc."RGPD_Consent"=\'True\' THEN 1 ELSE 0 END AS rgpd_consent,
    COALESCE(dg."Region",\'Inconnu\') AS region,
    COALESCE(do_."OffreActuelle",\'None\') AS offre_actuelle,
    COALESCE(do_."PrixOffre",0) AS prix_offre,
    COALESCE(do_."OffreCible",\'None\') AS offre_cible,
    COALESCE(AVG(fp."Minutes"),0)             AS avg_minutes,
    COALESCE(AVG(fp."DataGB"),0)              AS avg_data_gb,
    COALESCE(AVG(fp."SMS"),0)                 AS avg_sms,
    COALESCE(AVG(fp."RoamingGB"),0)           AS avg_roaming_gb,
    COALESCE(AVG(fp."UsageNocturneRatio"),0)  AS avg_usage_nocturne,
    COALESCE(AVG(fp."MontantFacture"),0)      AS avg_montant_facture,
    COALESCE(AVG(fp."ARPU"),0)                AS avg_arpu,
    COALESCE(AVG(fp."HorsForfait"),0)         AS avg_hors_forfait,
    COALESCE(SUM(fp."Impayes"),0)             AS total_impayes,
    COALESCE(MAX(fp."RetardPaiementJours"),0) AS max_retard_paiement,
    COALESCE(AVG(fp."RetardPaiementJours"),0) AS avg_retard_paiement,
    COALESCE(AVG(fp."QoSScore"),5)            AS avg_qos,
    COALESCE(AVG(fp."DropRatePct"),0)         AS avg_drop_rate,
    COALESCE(AVG(fp."ThroughputMbps"),0)      AS avg_throughput,
    COALESCE(AVG(fp."OutageMinutes"),0)       AS avg_outage_min,
    COALESCE(SUM(fp."Tickets"),0)             AS total_tickets,
    COALESCE(AVG(fp."DelaiResolutionJ"),0)    AS avg_delai_resolution,
    COALESCE(AVG(fp."NPS"),5)                 AS avg_nps,
    COUNT(fp."DateFK") AS nb_mois_observes
FROM public."dim_client" dc
LEFT JOIN public."dim_geographique" dg ON dc."ClientID"=dg."ClientID"
LEFT JOIN public."dim_offre" do_ ON dc."ClientID"=do_."ClientID"
LEFT JOIN public."Fact_performance_client" fp ON dc."Client_PK"=fp."ClientFK"
GROUP BY dc."Client_PK",dc."Sexe",dc."Segment",dc."TypeAbonnement",
    dc."AncienneteMois",dc."EngagementRestantMois",dc."RGPD_Consent",
    dg."Region",do_."OffreActuelle",do_."PrixOffre",do_."OffreCible"
'''

df_labeled = pd.read_sql(q_labeled, engine)
df_all     = pd.read_sql(q_all,     engine)
print(f"Labeled : {df_labeled.shape}  | churn={df_labeled['churn'].mean():.3f}  "
      f"(n_churn={df_labeled['churn'].sum()}, n_non={int((df_labeled['churn']==0).sum())})")
print(f"All 10k : {df_all.shape}")"""
))

# ── Cell 3 : Feature engineering ──────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""CAT_COLS = ['sexe','segment','typeabonnement','region','offre_actuelle','offre_cible']

def engineer(df, encoders=None, scaler=None, fit=True):
    d = df.copy()
    d.columns = d.columns.str.lower()

    encs = {} if encoders is None else encoders
    for col in CAT_COLS:
        if col in d.columns:
            if fit:
                le = LabelEncoder()
                d[col] = le.fit_transform(d[col].astype(str).fillna('UNK'))
                encs[col] = le
            else:
                le = encs[col]
                known = set(le.classes_)
                d[col] = d[col].astype(str).fillna('UNK').apply(
                    lambda x: x if x in known else le.classes_[0])
                d[col] = le.transform(d[col])

    for col in ['total_impayes','max_retard_paiement','avg_retard_paiement',
                'total_tickets','avg_outage_min','avg_hors_forfait']:
        if col in d.columns:
            d[col] = np.log1p(np.clip(d[col], 0, None))

    d['score_risque_paiement']    = d['total_impayes'] * (d['max_retard_paiement'] + 1)
    d['ratio_hors_forfait']       = d['avg_hors_forfait'] / (d['avg_montant_facture'] + 0.01)
    d['score_degradation_reseau'] = d['avg_drop_rate'] * (d['avg_outage_min'] + 1)
    d['flag_hors_engagement']     = (d['engagement_restant'] == 0).astype(int)
    d['intensite_usage']          = d['avg_minutes'] / (d['avg_arpu'] + 0.01)
    d['score_insatisfaction']     = d['total_tickets'] / (d['avg_nps'] + 1)

    EXCLUDE = {'client_id','churn','client_ref','gouvernorat'}
    feat_cols = [c for c in d.columns if c not in EXCLUDE and d[c].dtype != object]
    X = d[feat_cols].fillna(0).values.astype(float)

    if fit:
        sc = RobustScaler()
        X = sc.fit_transform(X)
        return X, feat_cols, encs, sc
    else:
        X = scaler.transform(X)
        return X, feat_cols

X_train, FEAT_COLS, ENCODERS, SCALER = engineer(df_labeled, fit=True)
y_train = df_labeled['churn'].values.astype(float)

n_pos = int(y_train.sum())
n_neg = int((y_train == 0).sum())
ratio = n_pos / n_neg
# Balanced sample weights: upweight minority (non-churn) by class ratio
SAMPLE_WEIGHTS = np.where(y_train == 1, 1.0, float(ratio))

print(f"X_train: {X_train.shape} | churn={y_train.mean():.3f} ({n_pos} pos / {n_neg} neg)")
print(f"Sample weight ratio (non-churn upweight): {ratio:.1f}x")
print(f"Features ({len(FEAT_COLS)}): {FEAT_COLS[:6]} ...")"""
))

# ── Cell 4 : Balanced regression CV ───────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""MODELS = {
    'GBR_balanced' : GradientBoostingRegressor(
        n_estimators=400, max_depth=3, learning_rate=0.04,
        subsample=0.8, min_samples_leaf=3, random_state=42),
    'RFR_balanced' : RandomForestRegressor(
        n_estimators=300, max_depth=5, min_samples_leaf=3,
        random_state=42),
    'Ridge'        : Ridge(alpha=5.0),
    'SVR_rbf'      : SVR(kernel='rbf', C=2.0, gamma='scale', epsilon=0.05),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print(f"{'Modele':<18} | {'AUC':>6} | {'Brier':>7} | {'R2':>7} | {'MAE':>6} | {'Acc%':>6}")
print("-" * 65)

for name, model in MODELS.items():
    oof_raw = np.zeros(len(y_train))
    oof_cal = np.zeros(len(y_train))

    for tr_idx, val_idx in skf.split(X_train, y_train):
        sw_tr = SAMPLE_WEIGHTS[tr_idx]
        # Fit with sample weights where supported
        try:
            model.fit(X_train[tr_idx], y_train[tr_idx], sample_weight=sw_tr)
        except TypeError:
            model.fit(X_train[tr_idx], y_train[tr_idx])

        raw_pred = np.clip(model.predict(X_train[val_idx]), 0, 1)
        oof_raw[val_idx] = raw_pred

        # Isotonic calibration on train fold
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(model.predict(X_train[tr_idx]), y_train[tr_idx],
                sample_weight=SAMPLE_WEIGHTS[tr_idx])
        oof_cal[val_idx] = iso.predict(model.predict(X_train[val_idx]))

    oof_cal = np.clip(oof_cal, 0, 1)

    auc   = roc_auc_score(y_train, oof_cal)
    brier = brier_score_loss(y_train, oof_cal)
    r2    = r2_score(y_train, oof_cal)
    mae   = mean_absolute_error(y_train, oof_cal)

    # Best accuracy threshold
    thresholds = np.linspace(0.05, 0.95, 500)
    accs = [accuracy_score(y_train, (oof_cal >= t).astype(int)) for t in thresholds]
    best_t   = thresholds[np.argmax(accs)]
    best_acc = max(accs)

    results[name] = {
        'auc': auc, 'brier': brier, 'r2': r2, 'mae': mae,
        'best_acc': best_acc, 'best_threshold': best_t,
        'oof_cal': oof_cal, 'oof_raw': oof_raw
    }
    acc_flag = '✓' if best_acc >= 0.90 else '!'
    print(f"{name:<18} | {auc:>6.4f} | {brier:>7.4f} | {r2:>7.4f} | {mae:>6.4f} | "
          f"{best_acc*100:>5.1f}% {acc_flag}")

# Best by AUC-ROC
BEST = max(results, key=lambda k: results[k]['auc'])
print(f"\\nMeilleur modele (AUC-ROC): {BEST}  "
      f"AUC={results[BEST]['auc']:.4f}  Brier={results[BEST]['brier']:.4f}")"""
))

# ── Cell 5 : Results visualisation ────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""model_names = list(results.keys())
colors4 = ['#1565C0','#2E7D32','#E65100','#6A1B9A']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Comparaison Modèles de Régression Pondérée — CV 5-Fold', fontsize=13, fontweight='bold')

# AUC-ROC
aucs = [results[m]['auc'] for m in model_names]
bars = axes[0].bar(model_names, aucs, color=colors4, alpha=0.85, edgecolor='white')
axes[0].axhline(0.5, ls='--', color='red', lw=1.5, label='Aléatoire')
axes[0].set_ylim(0.4, 0.9); axes[0].set_title('AUC-ROC'); axes[0].legend()
for b, v in zip(bars, aucs):
    axes[0].text(b.get_x()+b.get_width()/2, v+0.005, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

# Brier score (lower = better)
briers = [results[m]['brier'] for m in model_names]
bars2 = axes[1].bar(model_names, briers, color=colors4, alpha=0.85, edgecolor='white')
axes[1].axhline(y_train.mean()*(1-y_train.mean()), ls='--', color='orange', lw=1.5, label='Baseline naive')
axes[1].set_title('Brier Score (calibration, ↓ = mieux)'); axes[1].legend()
for b, v in zip(bars2, briers):
    axes[1].text(b.get_x()+b.get_width()/2, v+0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

# Accuracy at best threshold
accs = [results[m]['best_acc'] for m in model_names]
bars3 = axes[2].bar(model_names, accs, color=colors4, alpha=0.85, edgecolor='white')
axes[2].axhline(0.90, ls='--', color='green', lw=2, label='Cible 90%')
axes[2].axhline(0.936, ls=':', color='orange', lw=1.5, label='Baseline naive (93.6%)')
axes[2].set_ylim(0.85, 0.97); axes[2].set_title('Accuracy @ Seuil Optimal'); axes[2].legend(fontsize=8)
for b, v in zip(bars3, accs):
    axes[2].text(b.get_x()+b.get_width()/2, v+0.001, f'{v*100:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT, '06_cv_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 06_cv_comparison.png")"""
))

# ── Cell 6 : ROC + threshold analysis ────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""oof_best  = results[BEST]['oof_cal']
best_thr  = results[BEST]['best_threshold']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Analyse Détaillée — Meilleur Modèle : {BEST}', fontsize=13, fontweight='bold')

# ROC curves
for i, (name, res) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_train, res['oof_cal'])
    lw = 2.5 if name == BEST else 1.2
    axes[0].plot(fpr, tpr, lw=lw, color=colors4[i],
                 label=f"{name} AUC={res['auc']:.4f}")
axes[0].plot([0,1],[0,1],'k--', lw=1, label='Aléatoire')
axes[0].set_xlabel('Taux Faux Positifs'); axes[0].set_ylabel('Taux Vrais Positifs')
axes[0].set_title('Courbes ROC (calibrated, OOF)')
axes[0].legend(fontsize=9)
axes[0].fill_between([0,1],[0,1], alpha=0.05, color='gray')

# Accuracy vs threshold for best model
thresholds = np.linspace(0.05, 0.99, 400)
accs_thr = [accuracy_score(y_train, (oof_best >= t).astype(int)) for t in thresholds]
f1s_thr  = [f1_score(y_train, (oof_best >= t).astype(int), zero_division=0) for t in thresholds]

axes[1].plot(thresholds, accs_thr, color='#1565C0', lw=2, label='Accuracy')
axes[1].plot(thresholds, f1s_thr,  color='#2E7D32', lw=2, label='F1-Score')
axes[1].axhline(0.90, ls='--', color='red', lw=1.5, label='Cible 90%')
axes[1].axvline(best_thr, ls=':', color='navy', lw=2,
                label=f'Seuil opt.={best_thr:.2f} → Acc={results[BEST]["best_acc"]:.3f}')
axes[1].set_xlabel('Seuil de Décision')
axes[1].set_ylabel('Score')
axes[1].set_title(f'Accuracy & F1 vs Seuil ({BEST})')
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT, '06_roc_threshold.png'), dpi=150, bbox_inches='tight')
plt.show()

y_pred_best = (oof_best >= best_thr).astype(int)
print(f"\\n{'='*55}")
print(f"Meilleur modele  : {BEST}")
print(f"AUC-ROC          : {results[BEST]['auc']:.4f}")
print(f"Brier Score      : {results[BEST]['brier']:.4f}")
print(f"Seuil optimal    : {best_thr:.3f}")
print(f"Accuracy finale  : {results[BEST]['best_acc']:.4f} "
      f"({'OK >= 90%' if results[BEST]['best_acc'] >= 0.90 else 'BELOW 90%'})")
print(f"R2               : {results[BEST]['r2']:.4f}")
print(f"MAE              : {results[BEST]['mae']:.4f}")
print(f"{'='*55}")"""
))

# ── Cell 7 : Train final + calibrate + deploy to 10k ─────────────────────────
cells.append(nbf.v4.new_code_cell(
"""# Train final model with balanced weights on all 422 labeled
final_model = MODELS[BEST]
try:
    final_model.fit(X_train, y_train, sample_weight=SAMPLE_WEIGHTS)
except TypeError:
    final_model.fit(X_train, y_train)

# Fit isotonic calibration on training predictions
iso_final = IsotonicRegression(out_of_bounds='clip')
iso_final.fit(final_model.predict(X_train), y_train, sample_weight=SAMPLE_WEIGHTS)

print(f"Final model: {BEST} trained on {len(y_train)} clients")

# Apply to 10k clients
X_all, _ = engineer(df_all, encoders=ENCODERS, scaler=SCALER, fit=False)
raw_scores   = final_model.predict(X_all)
propensity   = np.clip(iso_final.predict(raw_scores), 0, 1)

df_scores = df_all[['client_id']].copy()
df_scores['propensity_score'] = propensity
df_scores['risk_label'] = pd.cut(
    propensity,
    bins=[0, 0.40, 0.70, 1.001],
    labels=['Faible', 'Moyen', 'Eleve']
)

print(f"\\nDistribution des scores de propension (10k clients) :")
print(f"  Min     : {propensity.min():.4f}")
print(f"  Q25     : {np.percentile(propensity, 25):.4f}")
print(f"  Median  : {np.median(propensity):.4f}")
print(f"  Mean    : {propensity.mean():.4f}")
print(f"  Q75     : {np.percentile(propensity, 75):.4f}")
print(f"  Max     : {propensity.max():.4f}")
print(f"  Std     : {propensity.std():.4f}")
print(f"\\nRépartition par niveau de risque :")
rc = df_scores['risk_label'].value_counts().sort_index()
for label, count in rc.items():
    print(f"  {label:<8} : {count:>5} clients ({count/len(df_scores)*100:.1f}%)")"""
))

# ── Cell 8 : Propensity distribution ─────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle('Distribution du Score de Propension au Churn — 10k Clients', fontsize=13, fontweight='bold')

# Histogram
axes[0].hist(propensity, bins=60, color='#1565C0', alpha=0.7, edgecolor='white')
axes[0].axvline(0.40, ls='--', color='#E65100', lw=2, label='Seuil 0.40 (Moyen)')
axes[0].axvline(0.70, ls='--', color='#B71C1C', lw=2, label='Seuil 0.70 (Elevé)')
axes[0].set_xlabel('Score de Propension'); axes[0].set_ylabel('Nombre de clients')
axes[0].set_title('Distribution Globale')
axes[0].legend(fontsize=9)

# Risk level pie
risk_counts = df_scores['risk_label'].value_counts().sort_index()
cpie = ['#4CAF50','#FF9800','#F44336']
axes[1].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
            colors=cpie, startangle=90, wedgeprops={'edgecolor':'white','linewidth':2})
axes[1].set_title('Répartition par Niveau de Risque')

# Calibration on 422 labeled (OOF)
fraction_pos, mean_pred_val = calibration_curve(y_train, oof_best, n_bins=8)
axes[2].plot(mean_pred_val, fraction_pos, 'o-', color='#1565C0', lw=2, ms=7, label=f'{BEST}')
axes[2].plot([0,1],[0,1],'k--', lw=1, label='Calibration parfaite')
axes[2].fill_between(mean_pred_val, fraction_pos, mean_pred_val,
                     alpha=0.2, color='blue')
axes[2].set_xlabel('Score prédit moyen'); axes[2].set_ylabel('Fraction réelle churners')
axes[2].set_title('Courbe de Calibration (OOF)')
axes[2].legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT, '06_propensity_distribution.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 06_propensity_distribution.png")"""
))

# ── Cell 9 : Segment analysis ─────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

def zscore(s): return (s - s.mean()) / (s.std() + 1e-9)
def log_z(s):  return zscore(np.log1p(np.clip(s, 0, None)))

d = df_all.copy(); d.columns = d.columns.str.lower()
d['score_usage']   = (log_z(d['avg_minutes']) + log_z(d['avg_data_gb']) +
                      log_z(d['avg_sms']) + log_z(d['avg_usage_nocturne'])) / 4
d['score_price']   = (log_z(d['avg_hors_forfait']) +
                      zscore(d['avg_hors_forfait']/(d['avg_montant_facture']+0.01)) +
                      log_z(d['avg_montant_facture'])) / 3
d['score_network'] = (zscore(d['avg_drop_rate']) + log_z(d['avg_outage_min']) -
                      zscore(d['avg_qos']) - zscore(d['avg_throughput'])) / 4
d['score_inact']   = -(zscore(d['avg_arpu']) + zscore(d['anciennete_mois']) +
                       zscore(d['nb_mois_observes']) + log_z(d['avg_minutes'])) / 4

km = KMeans(n_clusters=4, random_state=42, n_init=20)
cluster_ids = km.fit_predict(d[['score_usage','score_price','score_network','score_inact']].values)
cents = pd.DataFrame(km.cluster_centers_,
                     columns=['usage_intensity','price_pressure','network_risk','inactivity'])

def norm01(x): return (x-x.min())/(x.max()-x.min()+1e-9)
mat  = np.column_stack([norm01(cents['usage_intensity'].values),
                        norm01(cents['price_pressure'].values),
                        norm01(cents['network_risk'].values),
                        norm01(cents['inactivity'].values)])
ri, ci = linear_sum_assignment(-mat)
NAME_MAP = {0:'Fort Usage',1:'Sensibles au Prix',2:'Risque Reseau',3:'Clients Inactifs'}
seg_map  = {int(ri[i]): NAME_MAP[ci[i]] for i in range(4)}
df_scores['segment_v3'] = [seg_map[c] for c in cluster_ids]

# Plot
seg_order  = ['Fort Usage','Sensibles au Prix','Risque Reseau','Clients Inactifs']
seg_colors = {'Fort Usage':'#1565C0','Sensibles au Prix':'#E65100',
              'Risque Reseau':'#B71C1C','Clients Inactifs':'#616161'}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Score de Propension par Segment Comportemental', fontsize=13, fontweight='bold')

data_box = [df_scores[df_scores['segment_v3']==s]['propensity_score'].values for s in seg_order]
bp = axes[0].boxplot(data_box, labels=seg_order, patch_artist=True, notch=True, widths=0.55)
for patch, seg in zip(bp['boxes'], seg_order):
    patch.set_facecolor(seg_colors[seg]); patch.set_alpha(0.7)
for median in bp['medians']:
    median.set_color('white'); median.set_linewidth(2)
axes[0].axhline(0.70, ls='--', color='red', lw=1.5, label='Seuil Elevé (0.70)')
axes[0].set_ylabel('Score de Propension')
axes[0].set_title('Distribution par Segment (boxplot)')
axes[0].legend(fontsize=9)
axes[0].tick_params(axis='x', rotation=12)

seg_stats = df_scores.groupby('segment_v3')['propensity_score'].agg(
    Moyenne='mean', Mediane='median', N='count',
    Pct_eleve=lambda x: (x >= 0.70).mean()*100
).loc[seg_order]

x = np.arange(len(seg_order)); w = 0.35
b1 = axes[1].bar(x-w/2, seg_stats['Moyenne'],  w, label='Moyenne',
                 color=[seg_colors[s] for s in seg_order], alpha=0.85)
b2 = axes[1].bar(x+w/2, seg_stats['Mediane'], w, label='Médiane',
                 color=[seg_colors[s] for s in seg_order], alpha=0.45)
axes[1].set_xticks(x); axes[1].set_xticklabels(seg_order, rotation=12, ha='right')
axes[1].axhline(0.70, ls='--', color='red', lw=1.5)
axes[1].set_ylabel('Score de Propension')
axes[1].set_title('Moyenne & Médiane par Segment')
axes[1].legend()
for bar, v in zip(b1, seg_stats['Moyenne']):
    axes[1].text(bar.get_x()+bar.get_width()/2, v+0.003, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT, '06_propensity_by_segment.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\\nStatistiques par segment :")
print(seg_stats.to_string())"""
))

# ── Cell 10 : Feature importance ──────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""if hasattr(final_model, 'feature_importances_'):
    importances = final_model.feature_importances_
    idx = np.argsort(importances)[::-1][:20]
    fig, ax = plt.subplots(figsize=(12, 6))
    cbar = ['#B71C1C' if importances[i] > np.percentile(importances, 75) else '#1565C0' for i in idx]
    ax.barh(range(len(idx)), importances[idx][::-1], color=cbar[::-1], alpha=0.8)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([FEAT_COLS[i] for i in idx[::-1]], fontsize=10)
    ax.set_xlabel('Importance relative (Gain)')
    ax.set_title(f'Top 20 Features — {BEST} (Score de Propension)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, '06_feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Top 5: {[FEAT_COLS[i] for i in idx[:5]]}")
elif hasattr(final_model, 'coef_'):
    importances = np.abs(final_model.coef_)
    idx = np.argsort(importances)[::-1][:20]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(range(len(idx)), importances[idx][::-1], color='#1565C0', alpha=0.8)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([FEAT_COLS[i] for i in idx[::-1]], fontsize=10)
    ax.set_xlabel('|Coefficient| Ridge')
    ax.set_title(f'Top 20 Features — {BEST}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, '06_feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Top 5: {[FEAT_COLS[i] for i in idx[:5]]}")"""
))

# ── Cell 11 : Export ──────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""# Summary table
summary_rows = []
for name, res in results.items():
    summary_rows.append({
        'Modele': name,
        'AUC-ROC': f"{res['auc']:.4f}",
        'Brier':   f"{res['brier']:.4f}",
        'R2':      f"{res['r2']:.4f}",
        'MAE':     f"{res['mae']:.4f}",
        'Accuracy':f"{res['best_acc']*100:.1f}%",
        'Seuil':   f"{res['best_threshold']:.3f}",
        'Statut':  'MEILLEUR' if name == BEST else ''
    })

df_summary = pd.DataFrame(summary_rows)
print("=== Tableau Comparatif Final ===")
print(df_summary.to_string(index=False))

# Export summary table image
fig, ax = plt.subplots(figsize=(13, 2.5))
ax.axis('off')
tbl = ax.table(cellText=df_summary.values, colLabels=df_summary.columns,
               loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(10)
tbl.auto_set_column_width(col=list(range(len(df_summary.columns))))
for j in range(len(df_summary.columns)):
    tbl[0,j].set_facecolor('#1565C0')
    tbl[0,j].set_text_props(color='white', fontweight='bold')
for i, row in enumerate(df_summary.itertuples()):
    if row.Statut == 'MEILLEUR':
        for j in range(len(df_summary.columns)):
            tbl[i+1,j].set_facecolor('#E3F2FD')
plt.title('Objectif 6 — Score de Propension : Récapitulatif', fontweight='bold', pad=18)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '06_summary_table.png'), dpi=150, bbox_inches='tight')
plt.show()

# JSON log
rc_dict = {str(k): int(v) for k, v in df_scores['risk_label'].value_counts().items()}
seg_propen = df_scores.groupby('segment_v3')['propensity_score'].mean().to_dict()
log = {
    'objective': 'Churn Propensity Regression (Obj 6)',
    'best_model': BEST,
    'metrics': {k: {kk: float(vv) if not isinstance(vv, (str, np.ndarray)) else vv
                    for kk, vv in v.items() if kk not in ('oof_cal','oof_raw')}
                for k, v in results.items()},
    'deployment_10k': {
        'n_clients': len(df_scores),
        'score_mean': float(propensity.mean()),
        'score_std':  float(propensity.std()),
        'score_min':  float(propensity.min()),
        'score_max':  float(propensity.max()),
        'risk_distribution': rc_dict,
    },
    'segment_propensity': {str(k): float(v) for k, v in seg_propen.items()}
}
with open(os.path.join(OUT, '06_propensity_run_log.json'), 'w', encoding='utf-8') as f:
    json.dump(log, f, indent=2, ensure_ascii=False, default=str)

print("\\nSaved: 06_summary_table.png, 06_propensity_run_log.json")
print("\\nDEPLOYMENT FINAL :")
rc_sorted = df_scores['risk_label'].value_counts().sort_index()
for label, cnt in rc_sorted.items():
    pct = cnt / len(df_scores) * 100
    print(f"  Risque {label:<8}: {cnt:>5} clients ({pct:.1f}%)")"""
))

# ── Assemble + write ──────────────────────────────────────────────────────────
nb.cells = cells
out_path = r'c:\Users\Admin\ml pfe\06_churn_propensity.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f"Written: {out_path}")
