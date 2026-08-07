import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# ── Cell 0 : Title ─────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
    "# Objectif 7 — Régression : Prédiction ARPU & Montant Facture Futurs\n\n"
    "**Split temporel** : DateFK 25-33 (9 mois features) → DateFK 34-36 (3 mois cibles)  \n"
    "**Features** : Stats vectorisées (mean, std, tendance, récent) + profil client  \n"
    "**Modèles** : GBR · RandomForest · Ridge — CV 3-Fold sur 10 000 clients  \n"
    "**Base de données** : DW_TT — PostgreSQL localhost:5432"
))

# ── Cell 1 : Imports ────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json, os
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 120
OUT = r'c:\\Users\\Admin\\ml pfe'

TRAIN_DATES = list(range(25, 34))   # DateFK 25-33 (9 mois)
PRED_DATES  = list(range(34, 37))   # DateFK 34-36 (3 mois cibles)
print(f"Features : DateFK {TRAIN_DATES[0]}-{TRAIN_DATES[-1]}  ({len(TRAIN_DATES)} mois)")
print(f"Cibles   : DateFK {PRED_DATES[0]}-{PRED_DATES[-1]}   ({len(PRED_DATES)} mois)")"""
))

# ── Cell 2 : Load data ──────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""engine = create_engine('postgresql://postgres:Sarra123@localhost:5432/DW_TT')

q = '''
SELECT
    fp."ClientFK"            AS client_id,
    fp."DateFK"              AS mois,
    fp."ARPU"                AS arpu,
    fp."MontantFacture"      AS montant_facture,
    fp."Minutes"             AS minutes,
    fp."DataGB"              AS data_gb,
    fp."SMS"                 AS sms,
    fp."HorsForfait"         AS hors_forfait,
    fp."Impayes"             AS impayes,
    fp."RetardPaiementJours" AS retard,
    fp."QoSScore"            AS qos,
    fp."DropRatePct"         AS drop_rate,
    fp."ThroughputMbps"      AS throughput,
    fp."OutageMinutes"       AS outage,
    fp."NPS"                 AS nps,
    dc."Segment"             AS segment,
    dc."TypeAbonnement"      AS type_abo,
    dc."AncienneteMois"      AS anciennete_mois,
    dc."EngagementRestantMois" AS engagement_restant,
    COALESCE(do_."PrixOffre",0) AS prix_offre,
    COALESCE(dg."Region",\'Inconnu\') AS region
FROM public."Fact_performance_client" fp
INNER JOIN public."dim_client" dc ON fp."ClientFK"=dc."Client_PK"
LEFT  JOIN public."dim_geographique" dg ON dc."ClientID"=dg."ClientID"
LEFT  JOIN public."dim_offre" do_ ON dc."ClientID"=do_."ClientID"
ORDER BY fp."ClientFK", fp."DateFK"
'''
df_raw = pd.read_sql(q, engine)
print(f"Donnees : {df_raw.shape}  |  clients : {df_raw['client_id'].nunique()}")
print(f"ARPU moyen : {df_raw['arpu'].mean():.2f} TND   Facture moy : {df_raw['montant_facture'].mean():.2f} TND")
print(df_raw['arpu'].describe().round(2).rename('ARPU'))"""
))

# ── Cell 3 : Vectorised feature engineering ─────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""df_past   = df_raw[df_raw['mois'].isin(TRAIN_DATES)].copy()
df_future = df_raw[df_raw['mois'].isin(PRED_DATES)].copy()

NUM_COLS = ['arpu','montant_facture','minutes','data_gb','sms',
            'hors_forfait','impayes','retard','qos','drop_rate',
            'throughput','outage','nps']

df_s = df_past.sort_values(['client_id','mois'])

# ① Basic stats (fully vectorised)
df_stats = (df_s.groupby('client_id')[NUM_COLS]
            .agg(['mean','std','min','max'])
            .fillna(0))
df_stats.columns = [f'{c}_{s}' for c, s in df_stats.columns]

# ② Recent mean (last 3 months of training window)
df_recent = (df_s[df_s['mois'].isin(TRAIN_DATES[-3:])]
             .groupby('client_id')[NUM_COLS].mean().fillna(0)
             .rename(columns={c: f'{c}_recent' for c in NUM_COLS}))

# ③ Growth rate: recent vs early mean
df_early = (df_s[df_s['mois'].isin(TRAIN_DATES[:3])]
            .groupby('client_id')[NUM_COLS].mean().fillna(0))
df_growth = (df_recent[[f'{c}_recent' for c in NUM_COLS]].values /
             (df_early.values + 0.001) - 1)
df_growth = pd.DataFrame(df_growth, index=df_early.index,
                         columns=[f'{c}_growth' for c in NUM_COLS])

# ④ Linear trend (vectorised via pivot)
def trends_vectorised(df_sorted, num_cols, train_dates):
    nd = len(train_dates)
    x = np.arange(nd, dtype=float); x -= x.mean()
    ss_xx = (x**2).sum()
    records = {}
    for col in num_cols:
        pivot = (df_sorted.pivot_table(index='client_id', columns='mois',
                                       values=col, fill_value=0)
                 .reindex(columns=sorted(train_dates), fill_value=0))
        v = pivot.values.astype(float)
        y_bar = v.mean(axis=1, keepdims=True)
        records[f'{col}_trend'] = ((x * (v - y_bar)).sum(axis=1) / ss_xx)
        records[f'{col}_trend'] = pd.Series(records[f'{col}_trend'], index=pivot.index)
    return pd.DataFrame(records)

print("Computing trends...")
df_trends = trends_vectorised(df_s, NUM_COLS, TRAIN_DATES)

# ⑤ Static client features (one row per client)
static_cols = ['client_id','segment','type_abo','region',
               'anciennete_mois','engagement_restant','prix_offre']
df_static = (df_s.groupby('client_id')[static_cols[1:]]
             .first().reset_index())

# ⑥ Merge all
df_feat = (df_stats.join(df_recent).join(df_growth).join(df_trends)
           .reset_index().merge(df_static, on='client_id'))

# ⑦ Targets
df_targets = (df_future.groupby('client_id')
              .agg(arpu_futur=('arpu','mean'),
                   facture_futur=('montant_facture','mean'))
              .reset_index())

df_model = df_feat.merge(df_targets, on='client_id', how='inner')
print(f"Feature matrix : {df_model.shape}")
print(f"  ARPU futur   : {df_model['arpu_futur'].mean():.2f} ± {df_model['arpu_futur'].std():.2f} TND")
print(f"  Facture futur: {df_model['facture_futur'].mean():.2f} ± {df_model['facture_futur'].std():.2f} TND")"""
))

# ── Cell 4 : Encode + scale ─────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""CAT_COLS = ['segment','type_abo','region']
ENCODERS = {}
df_enc = df_model.copy()
for col in CAT_COLS:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col].astype(str).fillna('UNK'))
    ENCODERS[col] = le

EXCL = {'client_id','arpu_futur','facture_futur'}
FEAT_COLS = [c for c in df_enc.columns if c not in EXCL and df_enc[c].dtype != object]

X = df_enc[FEAT_COLS].fillna(0).values.astype(float)
y_arpu    = df_enc['arpu_futur'].values
y_facture = df_enc['facture_futur'].values

SCALER = RobustScaler()
X_sc = SCALER.fit_transform(X)
print(f"X : {X_sc.shape}  |  features: {len(FEAT_COLS)}")
print(f"y_arpu    : {y_arpu.mean():.2f} ± {y_arpu.std():.2f}  min={y_arpu.min():.1f}  max={y_arpu.max():.1f}")
print(f"y_facture : {y_facture.mean():.2f} ± {y_facture.std():.2f}  min={y_facture.min():.1f}  max={y_facture.max():.1f}")"""
))

# ── Cell 5 : Cross-validation ────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""MODELS = {
    'GBR'   : GradientBoostingRegressor(
                  n_estimators=150, max_depth=4, learning_rate=0.08,
                  subsample=0.8, min_samples_leaf=10, random_state=42),
    'RF'    : RandomForestRegressor(
                  n_estimators=150, max_depth=7, min_samples_leaf=10,
                  n_jobs=-1, random_state=42),
    'Ridge' : Ridge(alpha=1.0),
}

kf = KFold(n_splits=3, shuffle=True, random_state=42)

def mape(y_true, y_pred):
    mask = y_true > 0.5
    return 100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def run_cv(model, X, y, kf):
    oof = np.zeros(len(y))
    for tr, va in kf.split(X):
        model.fit(X[tr], y[tr])
        oof[va] = model.predict(X[va])
    return (r2_score(y, oof),
            mean_absolute_error(y, oof),
            np.sqrt(mean_squared_error(y, oof)),
            mape(y, oof), oof)

results = {}
for tgt, y, tgt_name in [('arpu', y_arpu,'ARPU'), ('facture', y_facture, 'Facture')]:
    print(f"\\n=== {tgt_name} Futur ===")
    print(f"{'Modele':<8} | {'R2':>6} | {'MAE':>6} | {'RMSE':>6} | {'MAPE%':>6}")
    print("-" * 45)
    for name, model in MODELS.items():
        r2, mae, rmse, pct, oof = run_cv(model, X_sc, y, kf)
        results.setdefault(name, {})[tgt] = dict(r2=r2,mae=mae,rmse=rmse,mape=pct,oof=oof)
        print(f"{name:<8} | {r2:>6.4f} | {mae:>6.2f} | {rmse:>6.2f} | {pct:>5.1f}%")

BEST_A = max(results, key=lambda k: results[k]['arpu']['r2'])
BEST_F = max(results, key=lambda k: results[k]['facture']['r2'])
print(f"\\nMeilleur ARPU    : {BEST_A}   R2={results[BEST_A]['arpu']['r2']:.4f}")
print(f"Meilleur Facture : {BEST_F}   R2={results[BEST_F]['facture']['r2']:.4f}")"""
))

# ── Cell 6 : Comparison plots ────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""model_names = list(MODELS.keys())
colors3 = ['#1565C0','#2E7D32','#E65100']
targets_meta = [('arpu','ARPU Futur (TND)'),('facture','Montant Facture (TND)')]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Prédiction ARPU & Facture — Comparaison 3 Modèles (CV 3-Fold)',
             fontsize=13, fontweight='bold')

for ri, (tgt, tgt_label) in enumerate(targets_meta):
    r2s   = [results[m][tgt]['r2']   for m in model_names]
    maes  = [results[m][tgt]['mae']  for m in model_names]
    mapes = [results[m][tgt]['mape'] for m in model_names]

    for ci, (vals, ylabel, title, ref_line) in enumerate([
        (r2s,  'R²',          f'R² — {tgt_label}', (0.85,'green','Cible R²=0.85')),
        (maes, 'MAE (TND)',   f'MAE — {tgt_label}', None),
        (mapes,'MAPE (%)',    f'MAPE — {tgt_label}', (5,'green','<5% excellent')),
    ]):
        ax = axes[ri][ci]
        bars = ax.bar(model_names, vals, color=colors3, alpha=0.85, edgecolor='white')
        if ref_line:
            ax.axhline(ref_line[0], ls='--', color=ref_line[1], lw=1.5, label=ref_line[2])
            ax.legend(fontsize=9)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=0)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v*1.01 if v>0 else v-0.02,
                    f'{v:.3f}' if ci==0 else f'{v:.2f}' if ci==1 else f'{v:.1f}%',
                    ha='center', va='bottom', fontsize=9)
        if ci == 0: ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig(os.path.join(OUT,'07_model_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 07_model_comparison.png")"""
))

# ── Cell 7 : Predicted vs Actual ────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Prédit vs Réel — OOF 3-Fold', fontsize=13, fontweight='bold')

for ax, tgt, best_m, true_y in zip(
        axes,
        ['arpu','facture'],
        [BEST_A, BEST_F],
        [y_arpu, y_facture]):

    y_pred = results[best_m][tgt]['oof']
    r2v = results[best_m][tgt]['r2']
    maev = results[best_m][tgt]['mae']
    mapev = results[best_m][tgt]['mape']
    tgt_label = 'ARPU Futur' if tgt=='arpu' else 'Facture Future'

    # Hexbin for density on large dataset
    hb = ax.hexbin(true_y, y_pred, gridsize=50, cmap='Blues', mincnt=1)
    plt.colorbar(hb, ax=ax, label='Nombre de clients')

    lims = [min(true_y.min(), y_pred.min())*0.95,
            max(true_y.max(), y_pred.max())*1.05]
    ax.plot(lims, lims, 'r--', lw=2, label='Parfait (y=x)')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(f'{tgt_label} Réel (TND)')
    ax.set_ylabel(f'{tgt_label} Prédit (TND)')
    ax.set_title(f'{best_m} — {tgt_label}\\nR²={r2v:.4f}  MAE={maev:.2f} TND  MAPE={mapev:.1f}%',
                 fontweight='bold')
    ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUT,'07_predicted_vs_actual.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 07_predicted_vs_actual.png")"""
))

# ── Cell 8 : Residual distribution ──────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Distribution des Résidus (Réel − Prédit)', fontsize=13, fontweight='bold')

for ax, tgt, best_m, true_y in zip(
        axes,
        ['arpu','facture'],
        [BEST_A, BEST_F],
        [y_arpu, y_facture]):

    resid = true_y - results[best_m][tgt]['oof']
    pct95 = np.percentile(np.abs(resid), 95)
    tgt_label = 'ARPU' if tgt=='arpu' else 'Facture'

    ax.hist(resid, bins=80, color='#1565C0', alpha=0.75, edgecolor='white')
    ax.axvline(0,           color='red',    lw=2,   ls='--', label='Zero')
    ax.axvline(resid.mean(),color='orange', lw=1.5, ls='-',
               label=f'Moyenne={resid.mean():.2f} TND')
    ax.axvline( pct95, color='green', lw=1.5, ls=':', label=f'P95=±{pct95:.1f} TND')
    ax.axvline(-pct95, color='green', lw=1.5, ls=':')
    ax.set_xlabel(f'Résidu {tgt_label} (TND)')
    ax.set_ylabel('Fréquence')
    ax.set_title(f'Résidus {tgt_label} — std={resid.std():.2f} TND')
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT,'07_residuals.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 07_residuals.png")"""
))

# ── Cell 9 : Train final + segment analysis ──────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""# Final models trained on full data
final_arpu    = MODELS[BEST_A]; final_arpu.fit(X_sc, y_arpu)
final_facture = MODELS[BEST_F]; final_facture.fit(X_sc, y_facture)

arpu_pred    = final_arpu.predict(X_sc)
facture_pred = final_facture.predict(X_sc)

df_preds = df_enc[['client_id','segment','anciennete_mois']].copy()
df_preds['arpu_futur_predit']    = arpu_pred
df_preds['facture_futur_predit'] = facture_pred
df_preds['arpu_futur_reel']      = y_arpu
df_preds['facture_futur_reel']   = y_facture
df_preds['arpu_ecart_tnd']       = arpu_pred - y_arpu
df_preds['facture_ecart_tnd']    = facture_pred - y_facture

# Decode segment
seg_le = ENCODERS['segment']
df_preds['segment_name'] = seg_le.inverse_transform(df_preds['segment'].astype(int))

# Segment stats
seg_stats = (df_preds.groupby('segment_name')
             .agg(
                 n=('client_id','count'),
                 arpu_predit_moy=('arpu_futur_predit','mean'),
                 arpu_reel_moy=('arpu_futur_reel','mean'),
                 facture_predite_moy=('facture_futur_predit','mean'),
                 facture_reelle_moy=('facture_futur_reel','mean'),
             )
             .round(2))
print("Statistiques par Segment :")
print(seg_stats.to_string())"""
))

# ── Cell 10 : Segment + feature importance plots ─────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('ARPU & Facture Prédits par Segment', fontsize=13, fontweight='bold')

segments = seg_stats.index.tolist()
x_pos = np.arange(len(segments)); w = 0.35

for ax, reel_col, pred_col, unit in zip(
        axes,
        ['arpu_reel_moy','facture_reelle_moy'],
        ['arpu_predit_moy','facture_predite_moy'],
        ['ARPU (TND)','Facture (TND)']):
    b1 = ax.bar(x_pos - w/2, seg_stats[reel_col],  w, label='Réel', color='#1565C0', alpha=0.8)
    b2 = ax.bar(x_pos + w/2, seg_stats[pred_col],  w, label='Prédit', color='#2E7D32', alpha=0.8)
    ax.set_xticks(x_pos); ax.set_xticklabels(segments, rotation=15, ha='right')
    ax.set_ylabel(unit); ax.set_title(f'{unit} par Segment — Réel vs Prédit')
    ax.legend()
    for b, v in zip(b2, seg_stats[pred_col]):
        ax.text(b.get_x()+b.get_width()/2, v+0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT,'07_arpu_by_segment.png'), dpi=150, bbox_inches='tight')
plt.show()

# Feature importance
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))
fig2.suptitle('Top 20 Features — Prédiction ARPU & Facture', fontsize=13, fontweight='bold')

for ax, tgt, best_m, color in zip(axes2,
                                   ['arpu','facture'],
                                   [BEST_A, BEST_F],
                                   ['#1565C0','#2E7D32']):
    model_obj = final_arpu if tgt == 'arpu' else final_facture
    tgt_label = 'ARPU Futur' if tgt == 'arpu' else 'Facture Future'
    if hasattr(model_obj, 'feature_importances_'):
        imp = model_obj.feature_importances_
    elif hasattr(model_obj, 'coef_'):
        imp = np.abs(model_obj.coef_)
    else:
        continue
    idx = np.argsort(imp)[::-1][:20]
    bar_colors = ['#B71C1C' if imp[i] > np.percentile(imp,80) else color for i in idx]
    ax.barh(range(len(idx)), imp[idx][::-1], color=bar_colors[::-1], alpha=0.8)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([FEAT_COLS[i] for i in idx[::-1]], fontsize=9)
    ax.set_xlabel('Importance relative')
    ax.set_title(f'{best_m} — {tgt_label}', fontweight='bold')
    print(f"Top 5 {tgt_label}: {[FEAT_COLS[i] for i in idx[:5]]}")

plt.tight_layout()
plt.savefig(os.path.join(OUT,'07_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 07_arpu_by_segment.png, 07_feature_importance.png")"""
))

# ── Cell 11 : Summary + export ───────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""# Table
summary = []
for name in MODELS:
    a = results[name]['arpu']
    f = results[name]['facture']
    summary.append({
        'Modele':      name,
        'ARPU R²':     f"{a['r2']:.4f}",
        'ARPU MAE':    f"{a['mae']:.2f} TND",
        'ARPU MAPE':   f"{a['mape']:.1f}%",
        'Facture R²':  f"{f['r2']:.4f}",
        'Facture MAE': f"{f['mae']:.2f} TND",
        'Facture MAPE':f"{f['mape']:.1f}%",
        'Statut': ('★ ARPU' if name==BEST_A else '') + (' ★ Facture' if name==BEST_F else '')
    })
df_sum = pd.DataFrame(summary)
print("=== Tableau Comparatif Final ===")
print(df_sum.to_string(index=False))

# Table image
fig, ax = plt.subplots(figsize=(16, 2.5))
ax.axis('off')
tbl = ax.table(cellText=df_sum.values, colLabels=df_sum.columns, loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
tbl.auto_set_column_width(col=list(range(len(df_sum.columns))))
for j in range(len(df_sum.columns)):
    tbl[0,j].set_facecolor('#1565C0')
    tbl[0,j].set_text_props(color='white', fontweight='bold')
for i, row in enumerate(df_sum.itertuples()):
    if row.Statut:
        for j in range(len(df_sum.columns)):
            tbl[i+1,j].set_facecolor('#E8F5E9')
plt.title('Objectif 7 — Prédiction ARPU & Montant Facture', fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(OUT,'07_summary_table.png'), dpi=150, bbox_inches='tight')
plt.show()

# JSON
log = {
    'objective': 'ARPU & Facture Regression (Obj 7)',
    'split': {'train': f"DateFK {TRAIN_DATES[0]}-{TRAIN_DATES[-1]}",
              'pred':  f"DateFK {PRED_DATES[0]}-{PRED_DATES[-1]}"},
    'n_features': len(FEAT_COLS),
    'best_model_arpu': BEST_A,
    'best_model_facture': BEST_F,
    'metrics': {name: {
        'arpu':    {k: float(v) for k,v in res['arpu'].items() if k != 'oof'},
        'facture': {k: float(v) for k,v in res['facture'].items() if k != 'oof'},
    } for name, res in results.items()},
    'deployment': {
        'n_clients': len(df_preds),
        'arpu_pred_mean': float(arpu_pred.mean()),
        'facture_pred_mean': float(facture_pred.mean()),
    }
}
with open(os.path.join(OUT,'07_arpu_run_log.json'),'w', encoding='utf-8') as f:
    json.dump(log, f, indent=2, ensure_ascii=False, default=str)
print("Saved: 07_summary_table.png, 07_arpu_run_log.json")
print(f"\\nFINAL:")
print(f"  ARPU futur    R2={results[BEST_A]['arpu']['r2']:.4f}  "
      f"MAE={results[BEST_A]['arpu']['mae']:.2f} TND  MAPE={results[BEST_A]['arpu']['mape']:.1f}%")
print(f"  Facture futur R2={results[BEST_F]['facture']['r2']:.4f}  "
      f"MAE={results[BEST_F]['facture']['mae']:.2f} TND  MAPE={results[BEST_F]['facture']['mape']:.1f}%")"""
))

# ── Write ──────────────────────────────────────────────────────────────────
nb.cells = cells
out_path = r'c:\Users\Admin\ml pfe\07_arpu_prediction.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f"Written: {out_path}")
