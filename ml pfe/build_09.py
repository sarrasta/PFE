import nbformat as nbf
import os

nb    = nbf.v4.new_notebook()
cells = []

# ── Cell 0 : Title ───────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
    "# Objectif 9 — Gain Attendu d'une Action de Rétention\n\n"
    "**But** : Quantifier en TND le gain financier net attendu de cibler chaque client\n"
    "avec une offre de rétention, en combinant les outputs des objectifs précédents.\n\n"
    "**Formule centrale** :\n\n"
    "```\n"
    "Gain_Net(i) = P_churn(i) × P_réponse(i) × ARPU(i) × LTV_horizon - Coût_offre\n"
    "```\n\n"
    "- **P_churn** : propension au churn (proxy Obj 6)  \n"
    "- **P_réponse** : probabilité de réponse à l'offre (Obj 8)  \n"
    "- **ARPU** : revenu mensuel moyen observé (proxy Obj 7)  \n"
    "- **LTV_horizon** : mois de valeur sauvegardée si le client reste  \n"
    "- **Coût_offre** : coût de l'action de rétention (remise, bonus, etc.)\n\n"
    "**Livrables** : Score de gain attendu + courbe de gain cumulé + matrice ROI\n\n"
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
from matplotlib.ticker import FuncFormatter
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import brentq

plt.rcParams['figure.dpi'] = 120
OUT = r'c:\\Users\\Admin\\ml pfe'
np.random.seed(42)

# ── Paramètres métier ─────────────────────────────────────────────────────────
LTV_HORIZON = 12          # mois de revenus sauvés si client retenu
COUT_OFFRE_SCENARIOS = {
    'Legere (5 TND)':   5.0,    # remise ou bonus data
    'Standard (15 TND)': 15.0,  # remise mensuelle offerte
    'Premium (30 TND)': 30.0,   # upgrade ou forfait offert
}
COUT_BASE = 15.0    # scénario principal

print("Imports OK")
print(f"LTV horizon : {LTV_HORIZON} mois")
print(f"Scenarios cout : {COUT_OFFRE_SCENARIOS}")"""
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
    fp."QoSScore"               AS qos,
    fp."DropRatePct"            AS drop_rate,
    fp."Impayes"                AS impayes,
    fp."NPS"                    AS nps,
    dc."Segment"                AS segment,
    dc."TypeAbonnement"         AS type_abo,
    dc."AncienneteMois"         AS anciennete_mois,
    dc."EngagementRestantMois"  AS engagement_restant,
    COALESCE(do_."PrixOffre", 0)      AS prix_offre,
    COALESCE(dg."Region", 'Inconnu') AS region
FROM   public."Fact_performance_client" fp
INNER  JOIN public."dim_client"         dc  ON fp."ClientFK"  = dc."Client_PK"
LEFT   JOIN public."dim_geographique"   dg  ON dc."ClientID"  = dg."ClientID"
LEFT   JOIN public."dim_offre"          do_ ON dc."ClientID"  = do_."ClientID"
WHERE  fp."DateFK" BETWEEN 25 AND 36
ORDER  BY fp."ClientFK", fp."DateFK"
'''
df_raw = pd.read_sql(q, engine)
print(f"Donnees : {df_raw.shape[0]:,} lignes  |  {df_raw['client_id'].nunique():,} clients")
print(f"ARPU moyen : {df_raw['arpu'].mean():.2f} TND")"""
))

# ── Cell 3 : Feature Engineering ─────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""NUM_COLS     = ['arpu','montant_facture','minutes','data_gb','qos','drop_rate','impayes','nps']
ALL_DATES    = sorted(df_raw['mois'].unique())
RECENT_DATES = ALL_DATES[-3:]

df_s = df_raw.copy()
for c in NUM_COLS:
    df_s[c] = pd.to_numeric(df_s[c], errors='coerce').fillna(0)

# Stats de base
df_stats = df_s.groupby('client_id')[NUM_COLS].agg(['mean','std']).fillna(0)
df_stats.columns = [f'{c}_{s}' for c, s in df_stats.columns]

# ARPU récent (proxy Obj 7 : derniers 3 mois = meilleur prédicteur du futur)
df_recent = (df_s[df_s['mois'].isin(RECENT_DATES)]
             .groupby('client_id')[['arpu','montant_facture']].mean().fillna(0)
             .rename(columns={'arpu': 'arpu_recent', 'montant_facture': 'facture_recente'}))

# Features statiques
df_static = (df_s.groupby('client_id')
             [['segment','type_abo','anciennete_mois','engagement_restant',
               'prix_offre','region']].last())
df_static['anciennete_mois']    = pd.to_numeric(df_static['anciennete_mois'],    errors='coerce').fillna(0)
df_static['engagement_restant'] = pd.to_numeric(df_static['engagement_restant'], errors='coerce').fillna(0)
df_static['prix_offre']         = pd.to_numeric(df_static['prix_offre'],         errors='coerce').fillna(25.0)

df_feat = (df_stats
           .join(df_recent, how='left')
           .join(df_static, how='left')
           .fillna(0))

print(f"Feature matrix : {df_feat.shape[0]:,} clients × {df_feat.shape[1]} features")
print(f"ARPU moyen     : {df_feat['arpu_mean'].mean():.2f} TND  (std {df_feat['arpu_mean'].std():.2f})")
print(f"ARPU recent    : {df_feat['arpu_recent'].mean():.2f} TND (proxy Obj 7)")"""
))

# ── Cell 4 : Chargement scores Obj 8 + calcul propension proxy ───────────────
cells.append(nbf.v4.new_code_cell(
"""# ── Scores de réponse Obj 8 ──────────────────────────────────────────────────
log8_path = os.path.join(OUT, '08_offer_response_run_log.json')
with open(log8_path) as f:
    log8 = json.load(f)
response_scores = {int(k): float(v) for k, v in log8['response_scores'].items()}
df_feat['p_reponse'] = df_feat.index.map(lambda x: response_scores.get(x, 0.4))
print(f"Scores reponse (Obj 8) charges : {len(response_scores):,} clients")
print(f"  mean={df_feat['p_reponse'].mean():.3f}  std={df_feat['p_reponse'].std():.3f}")

# ── Propension churn proxy (Obj 6 — meme formule que Obj 8) ──────────────────
def norm01(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-9)

arpu_n    = norm01(df_feat['arpu_mean'])
impayes_n = norm01(df_feat['impayes_mean'])
drop_n    = norm01(df_feat['drop_rate_mean'])
qos_n     = norm01(df_feat['qos_mean'])
nps_n     = norm01(df_feat['nps_mean'])

df_feat['p_churn'] = 1 / (1 + np.exp(-(
    - 0.8 * arpu_n
    + 0.9 * impayes_n
    + 0.6 * drop_n
    - 0.5 * qos_n
    - 0.4 * nps_n
)))
print(f"Propension churn proxy : mean={df_feat['p_churn'].mean():.3f}  std={df_feat['p_churn'].std():.3f}")

# ARPU predit proxy : moyenne recente (R²~0.96 dans Obj 7)
df_feat['arpu_predit'] = df_feat['arpu_recent'].clip(lower=1.0)
print(f"ARPU predit proxy      : mean={df_feat['arpu_predit'].mean():.2f}  std={df_feat['arpu_predit'].std():.2f}")"""
))

# ── Cell 5 : Calcul du Gain Attendu ──────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""# ─────────────────────────────────────────────────────────────────────────────
# FORMULE CENTRALE :
#   Gain_Brut(i) = P_churn(i) × P_reponse(i) × ARPU_predit(i) × LTV_horizon
#   Gain_Net(i)  = Gain_Brut(i) - Cout_offre
#
# Interprétation :
#   - P_churn × P_reponse = probabilité que l'action soit utile ET acceptée
#   - ARPU × LTV = revenus préservés si le client est retenu
#   - Gain_Net > 0 → cibler ce client est rentable
# ─────────────────────────────────────────────────────────────────────────────

CLV = df_feat['arpu_predit'] * LTV_HORIZON   # valeur sur 12 mois (TND)
prob_utile = df_feat['p_churn'] * df_feat['p_reponse']  # prob que l'action soit efficace

df_feat['clv_12m']    = CLV
df_feat['prob_utile'] = prob_utile
df_feat['gain_brut']  = prob_utile * CLV

# Gain net pour chaque scénario
for label, cout in COUT_OFFRE_SCENARIOS.items():
    col = 'gain_net_' + label.split()[0].lower()
    df_feat[col] = df_feat['gain_brut'] - cout

# Scénario de base : Standard 15 TND
df_feat['gain_net']   = df_feat['gain_brut'] - COUT_BASE
df_feat['roi']        = df_feat['gain_net'] / COUT_BASE
df_feat['cible']      = (df_feat['gain_net'] > 0).astype(int)

print("=== Distribution du Gain Attendu (scenario Standard 15 TND) ===")
print(f"Gain brut   : mean={df_feat['gain_brut'].mean():.2f} TND  std={df_feat['gain_brut'].std():.2f}")
print(f"Gain net    : mean={df_feat['gain_net'].mean():.2f} TND   std={df_feat['gain_net'].std():.2f}")
print(f"ROI moyen   : {df_feat['roi'].mean():.2f}x  (gain_net / cout)")
print()
n_cible = df_feat['cible'].sum()
gain_total = df_feat[df_feat['cible']==1]['gain_net'].sum()
print(f"Clients ROI-positifs : {n_cible:,} ({n_cible/len(df_feat):.1%})")
print(f"Gain total potentiel : {gain_total:,.0f} TND")
print()
print("Percentiles gain net (TND) :")
for p in [10,25,50,75,90,95,99]:
    print(f"  P{p:02d} : {np.percentile(df_feat['gain_net'], p):.2f}")"""
))

# ── Cell 6 : EDA composantes ──────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""fig, axes = plt.subplots(2, 3, figsize=(17, 9))

# A — Distribution P_churn
axes[0,0].hist(df_feat['p_churn'], bins=50, color='#c0392b', alpha=0.85, edgecolor='white', lw=0.3)
axes[0,0].set_title('Distribution P(churn)', fontweight='bold')
axes[0,0].set_xlabel('Propension churn'); axes[0,0].set_ylabel('Clients')

# B — Distribution P_reponse
axes[0,1].hist(df_feat['p_reponse'], bins=50, color='#2980b9', alpha=0.85, edgecolor='white', lw=0.3)
axes[0,1].set_title('Distribution P(reponse offre)', fontweight='bold')
axes[0,1].set_xlabel('Probabilite de reponse'); axes[0,1].set_ylabel('Clients')

# C — Distribution ARPU predit
axes[0,2].hist(df_feat['arpu_predit'], bins=50, color='#27ae60', alpha=0.85, edgecolor='white', lw=0.3)
axes[0,2].axvline(df_feat['arpu_predit'].mean(), color='red', ls='--', lw=2,
                   label=f"moy={df_feat['arpu_predit'].mean():.1f} TND")
axes[0,2].set_title('Distribution ARPU predit (TND)', fontweight='bold')
axes[0,2].set_xlabel('ARPU (TND)'); axes[0,2].set_ylabel('Clients')
axes[0,2].legend(fontsize=9)

# D — Distribution Gain Brut
axes[1,0].hist(df_feat['gain_brut'], bins=50, color='#8e44ad', alpha=0.85, edgecolor='white', lw=0.3)
axes[1,0].axvline(COUT_BASE, color='red', ls='--', lw=2, label=f'Cout={COUT_BASE} TND')
axes[1,0].set_title('Gain Brut attendu (TND)', fontweight='bold')
axes[1,0].set_xlabel('Gain brut (TND)'); axes[1,0].set_ylabel('Clients')
axes[1,0].legend(fontsize=9)

# E — Distribution Gain Net
n_pos = (df_feat['gain_net'] > 0).sum()
n_neg = (df_feat['gain_net'] <= 0).sum()
axes[1,1].hist(df_feat[df_feat['gain_net'] > 0]['gain_net'], bins=40,
               color='#27ae60', alpha=0.85, edgecolor='white', lw=0.3,
               label=f'ROI+ : {n_pos:,} clients')
axes[1,1].hist(df_feat[df_feat['gain_net'] <= 0]['gain_net'], bins=20,
               color='#e74c3c', alpha=0.85, edgecolor='white', lw=0.3,
               label=f'ROI- : {n_neg:,} clients')
axes[1,1].axvline(0, color='black', lw=2, ls='-')
axes[1,1].set_title(f'Gain Net (scenario Standard {COUT_BASE} TND)', fontweight='bold')
axes[1,1].set_xlabel('Gain net (TND)'); axes[1,1].set_ylabel('Clients')
axes[1,1].legend(fontsize=9)

# F — Prob utile vs ARPU (hexbin)
hb = axes[1,2].hexbin(df_feat['arpu_predit'], df_feat['prob_utile'],
                        gridsize=30, cmap='YlOrRd', mincnt=1)
plt.colorbar(hb, ax=axes[1,2], label='Densite')
axes[1,2].set_title('P(utile) vs ARPU predit', fontweight='bold')
axes[1,2].set_xlabel('ARPU (TND)'); axes[1,2].set_ylabel('P_churn x P_reponse')

plt.suptitle('Objectif 9 — Composantes du Gain Attendu (10 000 clients)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, '09_gain_components.png'), dpi=120, bbox_inches='tight')
plt.show()
print("09_gain_components.png")"""
))

# ── Cell 7 : Courbe de Gain Cumulé ────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""# ── Courbe de Gain Cumulé (Cumulative Gain Curve) ────────────────────────────
# Trier par gain_net décroissant, calculer le gain cumulé normalisé

df_sorted = df_feat.sort_values('gain_net', ascending=False).reset_index(drop=True)
n = len(df_sorted)

cum_gain_model  = df_sorted['gain_net'].clip(lower=0).cumsum()
cum_gain_model  = cum_gain_model / cum_gain_model.iloc[-1]  # normalisé 0-1

# Baseline : ciblage aléatoire
cum_gain_random = np.linspace(0, 1, n)

# Courbe parfaite (Oracle) : trier par gain réel, connaître parfaitement les churners
df_oracle = df_feat.sort_values('gain_brut', ascending=False).reset_index(drop=True)
cum_gain_oracle = df_oracle['gain_net'].clip(lower=0).cumsum()
cum_gain_oracle = cum_gain_oracle / cum_gain_oracle.iloc[-1]

pct_pop = np.linspace(0, 100, n)

# Calcul du Lift à différents seuils
lifts = {}
for pct in [10, 20, 30, 50]:
    idx = int(n * pct / 100) - 1
    model_val = cum_gain_model.iloc[idx]
    random_val = cum_gain_random[idx]
    lifts[pct] = model_val / random_val if random_val > 0 else 1.0

print("Lift du modele de gain :")
for pct, lift in lifts.items():
    idx = int(n * pct / 100) - 1
    gain_tnd = df_sorted.iloc[:idx+1]['gain_net'].clip(lower=0).sum()
    print(f"  Top {pct:2d}% cibles ({int(n*pct/100):,} clients) : "
          f"lift={lift:.2f}x, gain={gain_tnd:,.0f} TND")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Courbe de gain cumulé
axes[0].plot(pct_pop, cum_gain_model * 100, color='#2980b9', lw=2.5,
             label='Modele (tri par gain attendu)')
axes[0].plot(pct_pop, cum_gain_oracle * 100, color='#27ae60', lw=2, ls='--',
             label='Oracle (parfait)')
axes[0].plot(pct_pop, cum_gain_random * 100, color='gray', lw=1.5, ls=':',
             label='Aleatoire (baseline)')
# Annotations lift points
for pct in [20, 50]:
    idx = int(n * pct / 100) - 1
    axes[0].annotate(f'Top {pct}%\\n={cum_gain_model.iloc[idx]*100:.0f}%',
                     xy=(pct, cum_gain_model.iloc[idx]*100),
                     xytext=(pct+5, cum_gain_model.iloc[idx]*100 - 10),
                     fontsize=8, arrowprops=dict(arrowstyle='->', color='navy'), color='navy')
axes[0].fill_between(pct_pop, cum_gain_random*100, cum_gain_model*100,
                      alpha=0.1, color='blue', label='Zone de gain extra')
axes[0].set_title('Courbe de Gain Cumule', fontweight='bold', fontsize=12)
axes[0].set_xlabel('% Clients cibles (tris par gain attendu decroissant)')
axes[0].set_ylabel('% du gain total capture')
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, 100); axes[0].set_ylim(0, 103)

# Courbe de lift
pct_axis   = np.arange(1, 101)
lift_model = np.array([cum_gain_model.iloc[int(n*p/100)-1] / (p/100)
                        if int(n*p/100) > 0 else 1
                        for p in pct_axis])
axes[1].plot(pct_axis, lift_model, color='#e74c3c', lw=2.5,
             label='Lift modele')
axes[1].axhline(1.0, color='gray', lw=1.5, ls='--', label='Baseline (1.0x)')
axes[1].axhline(2.0, color='orange', lw=1, ls=':', alpha=0.7)
axes[1].fill_between(pct_axis, 1.0, lift_model,
                      where=(lift_model > 1), alpha=0.15, color='red')
axes[1].set_title('Courbe de Lift (vs ciblage aleatoire)', fontweight='bold', fontsize=12)
axes[1].set_xlabel('% Clients cibles')
axes[1].set_ylabel('Lift (x fois meilleur que aleatoire)')
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, 100); axes[1].set_ylim(0, max(lift_model)*1.1)

plt.tight_layout()
plt.savefig(os.path.join(OUT, '09_gain_curve.png'), dpi=120, bbox_inches='tight')
plt.show()
print("09_gain_curve.png")"""
))

# ── Cell 8 : Analyse des scénarios de coût ───────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""# ── Analyse des 3 scénarios d'offre ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

colors_sc = ['#27ae60', '#2980b9', '#8e44ad']
scenario_cols = {
    'Legere':   ('gain_net_legere', 5.0,  '#27ae60'),
    'Standard': ('gain_net_standard', 15.0, '#2980b9'),
    'Premium':  ('gain_net_premium', 30.0, '#8e44ad'),
}

# A — Clients rentables par scénario
n_rentable = {}
gain_total_sc = {}
for label, (col, cout, c) in scenario_cols.items():
    mask = df_feat[col] > 0
    n_rentable[label] = mask.sum()
    gain_total_sc[label] = df_feat[mask][col].sum()

axes[0].bar(list(n_rentable.keys()), list(n_rentable.values()),
            color=[v[2] for v in scenario_cols.values()], alpha=0.85)
axes[0].set_title('Clients ROI-positifs par scenario', fontweight='bold')
axes[0].set_ylabel('Nombre de clients')
for i, (label, n) in enumerate(n_rentable.items()):
    axes[0].text(i, n + 50, f'{n:,}\\n({n/len(df_feat):.0%})',
                 ha='center', fontsize=10, fontweight='bold')

# B — Gain total par scénario (en kTND)
axes[1].bar(list(gain_total_sc.keys()),
            [v/1000 for v in gain_total_sc.values()],
            color=[v[2] for v in scenario_cols.values()], alpha=0.85)
axes[1].set_title('Gain total potentiel par scenario', fontweight='bold')
axes[1].set_ylabel('Gain net total (kTND)')
for i, (label, g) in enumerate(gain_total_sc.items()):
    axes[1].text(i, g/1000 + 0.5, f'{g/1000:,.0f} kTND',
                 ha='center', fontsize=10, fontweight='bold')

# C — Distribution comparée gain net (3 scénarios)
for (label, (col, cout, c)) in scenario_cols.items():
    vals = df_feat[col][df_feat[col] > 0]
    axes[2].hist(vals, bins=40, alpha=0.5, color=c,
                 label=f'{label} ({len(vals):,} cibles)', edgecolor='white', lw=0.3)
axes[2].axvline(0, color='black', lw=1.5)
axes[2].set_title('Distribution gain net (clients ROI+)', fontweight='bold')
axes[2].set_xlabel('Gain net (TND)'); axes[2].set_ylabel('Clients')
axes[2].legend(fontsize=9)

plt.suptitle('Objectif 9 — Analyse par Scenario de Cout Offre', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, '09_scenarios.png'), dpi=120, bbox_inches='tight')
plt.show()
print("09_scenarios.png")"""
))

# ── Cell 9 : Heatmap sensibilité coût × LTV ──────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""# ── Sensibilité : coût × horizon LTV ────────────────────────────────────────
couts   = [5, 10, 15, 20, 25, 30, 40, 50]
horizons = [3, 6, 9, 12, 18, 24]

matrix_gain   = np.zeros((len(horizons), len(couts)))
matrix_cibles = np.zeros((len(horizons), len(couts)))

for i, ltvh in enumerate(horizons):
    for j, cout in enumerate(couts):
        gb = df_feat['prob_utile'] * df_feat['arpu_predit'] * ltvh
        gn = gb - cout
        matrix_gain[i,j]   = gn[gn>0].sum() / 1000  # kTND
        matrix_cibles[i,j] = (gn > 0).sum()

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Heatmap gain total
im0 = axes[0].imshow(matrix_gain, cmap='RdYlGn', aspect='auto')
plt.colorbar(im0, ax=axes[0], label='Gain total net (kTND)')
axes[0].set_xticks(range(len(couts)));   axes[0].set_xticklabels([f'{c} TND' for c in couts])
axes[0].set_yticks(range(len(horizons))); axes[0].set_yticklabels([f'{h} mois' for h in horizons])
axes[0].set_title('Gain total potentiel (kTND)\\npar (horizon LTV, cout offre)',
                   fontweight='bold')
axes[0].set_xlabel('Cout offre (TND)'); axes[0].set_ylabel('Horizon LTV (mois)')
for i in range(len(horizons)):
    for j in range(len(couts)):
        axes[0].text(j, i, f'{matrix_gain[i,j]:.0f}',
                     ha='center', va='center', fontsize=7.5,
                     color='black' if matrix_gain[i,j] < matrix_gain.max()*0.65 else 'white')
# Entourer le scenario de base
rect0 = plt.Rectangle((couts.index(15)-0.5, horizons.index(12)-0.5),
                         1, 1, fill=False, edgecolor='blue', lw=3)
axes[0].add_patch(rect0)

# Heatmap nombre de clients ciblés
im1 = axes[1].imshow(matrix_cibles, cmap='Blues', aspect='auto')
plt.colorbar(im1, ax=axes[1], label='Clients ROI-positifs')
axes[1].set_xticks(range(len(couts)));   axes[1].set_xticklabels([f'{c} TND' for c in couts])
axes[1].set_yticks(range(len(horizons))); axes[1].set_yticklabels([f'{h} mois' for h in horizons])
axes[1].set_title('Clients ROI-positifs\\npar (horizon LTV, cout offre)',
                   fontweight='bold')
axes[1].set_xlabel('Cout offre (TND)'); axes[1].set_ylabel('Horizon LTV (mois)')
for i in range(len(horizons)):
    for j in range(len(couts)):
        axes[1].text(j, i, f'{int(matrix_cibles[i,j]):,}',
                     ha='center', va='center', fontsize=7,
                     color='black' if matrix_cibles[i,j] < matrix_cibles.max()*0.7 else 'white')
rect1 = plt.Rectangle((couts.index(15)-0.5, horizons.index(12)-0.5),
                         1, 1, fill=False, edgecolor='blue', lw=3)
axes[1].add_patch(rect1)
axes[1].text(couts.index(15), horizons.index(12)+0.4, 'Base',
              ha='center', fontsize=8, color='blue', fontweight='bold')

plt.suptitle('Objectif 9 — Sensibilite Cout x LTV (cadre bleu = scenario de base)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, '09_sensitivity.png'), dpi=120, bbox_inches='tight')
plt.show()
print("09_sensitivity.png")"""
))

# ── Cell 10 : Analyse par segment ─────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""# ── Analyse par segment ──────────────────────────────────────────────────────
seg_stats = (df_feat.groupby('segment')
             .agg(
                 clients      =('gain_net','count'),
                 gain_net_moy =('gain_net','mean'),
                 gain_net_tot =('gain_net',lambda x: x[x>0].sum()),
                 arpu_moy     =('arpu_predit','mean'),
                 p_churn_moy  =('p_churn','mean'),
                 p_reponse_moy=('p_reponse','mean'),
                 pct_roi_pos  =('cible','mean'),
             )
             .sort_values('gain_net_moy', ascending=False))

print("=== Analyse gain attendu par segment ===")
print(seg_stats.to_string())

# Visualisation
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
segs = seg_stats.index.tolist()
colors_seg = ['#e74c3c','#e67e22','#27ae60','#3498db','#8e44ad','#95a5a6']

# Gain net moyen
axes[0].bar(segs, seg_stats['gain_net_moy'],
            color=colors_seg[:len(segs)], alpha=0.85)
axes[0].axhline(0, color='black', lw=1)
axes[0].set_title('Gain net moyen par client (TND)', fontweight='bold')
axes[0].set_ylabel('Gain net moy. (TND)')
axes[0].tick_params(axis='x', rotation=20)
for i, (s, v) in enumerate(zip(segs, seg_stats['gain_net_moy'])):
    axes[0].text(i, v + 0.1 if v >= 0 else v - 0.3,
                 f'{v:.1f}', ha='center', fontsize=9, fontweight='bold')

# % clients ROI+
axes[1].bar(segs, seg_stats['pct_roi_pos'] * 100,
            color=colors_seg[:len(segs)], alpha=0.85)
axes[1].set_title('% Clients ROI-positifs par segment', fontweight='bold')
axes[1].set_ylabel('% Clients (gain_net > 0)')
axes[1].tick_params(axis='x', rotation=20)
for i, (s, v) in enumerate(zip(segs, seg_stats['pct_roi_pos'])):
    axes[1].text(i, v*100 + 0.5, f'{v:.0%}', ha='center', fontsize=9, fontweight='bold')

# Gain total par segment (kTND)
axes[2].bar(segs, seg_stats['gain_net_tot'] / 1000,
            color=colors_seg[:len(segs)], alpha=0.85)
axes[2].set_title('Gain total (kTND) par segment', fontweight='bold')
axes[2].set_ylabel('Gain net total (kTND)')
axes[2].tick_params(axis='x', rotation=20)
for i, (s, v) in enumerate(zip(segs, seg_stats['gain_net_tot'])):
    axes[2].text(i, v/1000 + 0.3, f'{v/1000:.0f}k', ha='center', fontsize=9, fontweight='bold')

plt.suptitle('Objectif 9 — Gain Attendu par Segment Client', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, '09_gain_by_segment.png'), dpi=120, bbox_inches='tight')
plt.show()
print("09_gain_by_segment.png")"""
))

# ── Cell 11 : Matrice de ciblage Obj 6 × Obj 8 colorée par gain ──────────────
cells.append(nbf.v4.new_code_cell(
"""# ── Visualisation 3D : propension × réponse × gain (couleur) ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# A — Scatter propension × réponse, coloré par gain_net
sc = axes[0].scatter(df_feat['p_churn'], df_feat['p_reponse'],
                      c=df_feat['gain_net'], cmap='RdYlGn',
                      s=df_feat['arpu_predit'] * 0.3 + 5,
                      alpha=0.4, vmin=-30, vmax=df_feat['gain_net'].quantile(0.95))
plt.colorbar(sc, ax=axes[0], label='Gain net (TND)')
axes[0].axhline(0.5, color='blue', ls='--', lw=1, alpha=0.6)
axes[0].axvline(0.5, color='blue', ls='--', lw=1, alpha=0.6)
# Zone optimale
from matplotlib.patches import Rectangle
rect_opt = Rectangle((0.35, 0.45), 0.30, 0.55, fill=True, facecolor='gold',
                        alpha=0.12, edgecolor='darkgoldenrod', lw=2.5, ls='--')
axes[0].add_patch(rect_opt)
axes[0].text(0.50, 0.74, 'Zone ROI+\\nOptimale', ha='center', fontsize=9,
              color='darkgoldenrod', fontweight='bold')
axes[0].set_title('Gain net : Propension x Reponse\\n(taille = ARPU)',
                   fontweight='bold')
axes[0].set_xlabel('Propension churn (P_churn)')
axes[0].set_ylabel('Probabilite reponse (P_reponse)')

# B — Top quartile clients par gain attendu
topQ = df_feat.nlargest(int(len(df_feat)*0.25), 'gain_net')
botQ = df_feat.nsmallest(int(len(df_feat)*0.25), 'gain_net')

labels_comp = ['P_churn moy', 'P_reponse moy', 'ARPU moy', 'Gain net moy']
vals_top = [topQ['p_churn'].mean(), topQ['p_reponse'].mean(),
             topQ['arpu_predit'].mean() / df_feat['arpu_predit'].max(),
             topQ['gain_net'].mean() / df_feat['gain_net'].max()]
vals_bot = [botQ['p_churn'].mean(), botQ['p_reponse'].mean(),
             botQ['arpu_predit'].mean() / df_feat['arpu_predit'].max(),
             botQ['gain_net'].mean() / df_feat['gain_net'].max()]

x = np.arange(len(labels_comp))
axes[1].bar(x - 0.2, vals_top, 0.35, color='#27ae60', alpha=0.85,
             label=f'Top 25% gain ({len(topQ):,} clients)')
axes[1].bar(x + 0.2, vals_bot, 0.35, color='#e74c3c', alpha=0.85,
             label=f'Bot 25% gain ({len(botQ):,} clients)')
axes[1].set_xticks(x); axes[1].set_xticklabels(labels_comp, rotation=10)
axes[1].set_title('Profil Top vs Bot quartile (normalise)', fontweight='bold')
axes[1].set_ylabel('Valeur normalisee [0,1]')
axes[1].legend(fontsize=9)
axes[1].axhline(0, color='black', lw=0.5)

plt.suptitle('Objectif 9 — Carte de Ciblage : Qui Cibler en Priorite ?',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, '09_targeting_map.png'), dpi=120, bbox_inches='tight')
plt.show()
print("09_targeting_map.png")"""
))

# ── Cell 12 : Tableau récapitulatif + rapport final ───────────────────────────
cells.append(nbf.v4.new_code_cell(
"""# ── Tableau récapitulatif des scénarios ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis('off')

headers = ['Scenario', 'Cout offre', 'Clients ROI+', '% Base', 'Gain total net', 'Gain moy/client', 'ROI moyen']
rows = []
for label, (col, cout, c) in scenario_cols.items():
    mask = df_feat[col] > 0
    n    = mask.sum()
    gtot = df_feat[mask][col].sum()
    gmoy = df_feat[mask][col].mean()
    roi_m = gmoy / cout
    rows.append([
        label,
        f'{cout:.0f} TND',
        f'{n:,}',
        f'{n/len(df_feat):.0%}',
        f'{gtot:,.0f} TND',
        f'{gmoy:.1f} TND',
        f'{roi_m:.1f}x',
    ])

tbl = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 1.9)
for j in range(len(headers)):
    tbl[0,j].set_facecolor('#dae8fc')
tbl[2,0].set_facecolor('#d5e8d4')  # Standard row highlighted

ax.set_title('Objectif 9 — Tableau Comparatif des Scenarios de Retention',
             fontweight='bold', fontsize=12, y=0.88)
plt.savefig(os.path.join(OUT, '09_summary_table.png'), dpi=120, bbox_inches='tight')
plt.show()
print("09_summary_table.png")

# ── Rapport final ─────────────────────────────────────────────────────────────
print("\\n" + "="*68)
print("OBJECTIF 9 — GAIN ATTENDU D'UNE ACTION DE RETENTION")
print("="*68)
print(f"Formule : G = P_churn x P_reponse x ARPU x {LTV_HORIZON} mois - Cout_offre")
print()
print("SCENARIO STANDARD (15 TND) :")
mask_std = df_feat['gain_net'] > 0
n_std    = mask_std.sum()
g_std    = df_feat[mask_std]['gain_net'].sum()
g_moy    = df_feat[mask_std]['gain_net'].mean()
print(f"  Clients ROI-positifs     : {n_std:,}  ({n_std/len(df_feat):.1%} de la base)")
print(f"  Gain total potentiel     : {g_std:,.0f} TND")
print(f"  Gain moyen par client    : {g_moy:.1f} TND")
print(f"  ROI moyen sur cout 15 TND: {g_moy/15:.1f}x")
print()
print("LIFT DU MODELE (vs ciblage aleatoire) :")
for pct, lift in lifts.items():
    print(f"  Top {pct:2d}% cibles -> lift = {lift:.2f}x")
print()
print("FORMULE DE CIBLAGE OPTIMAL :")
print(f"  gain_net = P_churn x P_reponse x ARPU x {LTV_HORIZON} - 15")
print(f"  Cible si : P_churn x P_reponse > 15 / (ARPU x {LTV_HORIZON})")
print(f"  Pour ARPU=20 TND : seuil = {15/(20*LTV_HORIZON):.3f}")
print(f"  Pour ARPU=30 TND : seuil = {15/(30*LTV_HORIZON):.3f}")
print("="*68)"""
))

# ── Cell 13 : Sauvegarde JSON ─────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""log9 = {
    'objective': 'Gain attendu action de retention',
    'formula':   'P_churn x P_reponse x ARPU x LTV - Cout_offre',
    'parameters': {
        'ltv_horizon_mois': LTV_HORIZON,
        'cout_base_tnd':    COUT_BASE,
    },
    'results_standard_15tnd': {
        'n_clients_roi_positifs': int(mask_std.sum()),
        'pct_roi_positifs':       float(mask_std.mean()),
        'gain_total_tnd':         float(df_feat[mask_std]['gain_net'].sum()),
        'gain_moyen_tnd':         float(df_feat[mask_std]['gain_net'].mean()),
        'roi_moyen':              float(df_feat[mask_std]['gain_net'].mean() / 15),
    },
    'lift_at_pct': {f'top_{k}pct': float(v) for k, v in lifts.items()},
    'scenarios': {
        label: {
            'cout': cout,
            'n_roi_positifs': int((df_feat[col] > 0).sum()),
            'gain_total_tnd': float(df_feat[df_feat[col]>0][col].sum()),
        }
        for label, (col, cout, _) in scenario_cols.items()
    },
    'gain_scores': {
        str(int(cid)): float(g)
        for cid, g in zip(df_feat.index, df_feat['gain_net'])
    }
}

log_path = os.path.join(OUT, '09_retention_gain_run_log.json')
with open(log_path, 'w', encoding='utf-8') as f:
    json.dump(log9, f, ensure_ascii=False, indent=2, default=str)
print(f"Log sauvegarde : {log_path}")"""
))

nb['cells'] = cells

OUT_PATH = r'c:\Users\Admin\ml pfe\09_retention_gain.ipynb'
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f"Written: {OUT_PATH}")
