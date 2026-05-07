"""
NALCO HEMM Predictive Maintenance — Clean ML Pipeline (No Leakage)
===================================================================
Only real-time sensor + operational features are used as input.
Post-failure/maintenance outcome columns are excluded.
Models:
  A. Failure Classification  (RF + GB + Voting Ensemble)
  B. Days-to-Next-Failure Regression  (RF Regressor)
  C. Maintenance Priority Classification (RF Multi-class)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               VotingClassifier, RandomForestRegressor,
                               GradientBoostingRegressor)
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                              roc_curve, mean_absolute_error, mean_squared_error, r2_score,
                              accuracy_score, precision_score, recall_score, f1_score)
from sklearn.utils.class_weight import compute_class_weight

import os

# ─── AUTO-DETECT PATH (works on both local PC and Claude) ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder where this script lives
DATA_PATH = os.path.join(BASE_DIR, "data", "1778086522685_HEMM_Dataset_FINAL.csv")
OUT = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT, exist_ok=True)

print("=" * 65)
print("  NALCO HEMM PREDICTIVE MAINTENANCE — ML PIPELINE (Clean)")
print("=" * 65)
print(f"\n  Script folder : {BASE_DIR}")
print(f"  Dataset path  : {DATA_PATH}")
print(f"  Output folder : {OUT}")

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"\n❌ Dataset not found at: {DATA_PATH}"
        f"\n   Please place your CSV file inside the 'data' folder next to this script."
        f"\n   Expected structure:"
        f"\n     nalco-hemm-predictive-maintenance/"
        f"\n     ├── data/"
        f"\n     │   └── 1778086522685_HEMM_Dataset_FINAL.csv"
        f"\n     ├── outputs/"
        f"\n     └── hemm_ml_pipeline.py"
    )

df = pd.read_csv(DATA_PATH)
print(f"\n✔ Dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────────────────────────
# 2. FEATURE SELECTION — REAL-TIME SENSOR + OPERATIONAL ONLY
# ─────────────────────────────────────────────────────────────
# These are features known BEFORE a failure occurs — no leakage
SENSOR_FEATURES = [
    # Sensor readings
    'Engine_Temp_C', 'Oil_Pressure_bar', 'Vibration_mms',
    'Fuel_Consumption_Lhr', 'Tyre_Pressure_PSI', 'Coolant_Level',
    'Battery_Voltage_V', 'Hydraulic_Pressure_bar', 'Exhaust_Temp_C', 'RPM',
    # Status flags derived from sensor limits (healthy to use)
    'Engine_Temp_C_Status_enc', 'Oil_Pressure_bar_Status_enc',
    'Vibration_mms_Status_enc', 'Fuel_Consumption_Lhr_Status_enc',
    'Tyre_Pressure_PSI_Status_enc', 'Coolant_Level_Status_enc',
    'Battery_Voltage_V_Status_enc', 'Hydraulic_Pressure_bar_Status_enc',
    'Exhaust_Temp_C_Status_enc', 'RPM_Status_enc',
    # Derived ratios
    'Temp_Oil_Ratio', 'Vib_RPM_Ratio',
    # Operational
    'Operating_Hours', 'Load_Cycles_per_Day',
    'Operator_Experience_Yr', 'Equipment_Type_enc',
    'Shift_enc', 'Road_Condition_enc',
    'Year', 'Month', 'Day_of_Week',
    # Component condition scores (inspection-based, available pre-failure)
    'Engine_Condition_enc', 'Tyre_Condition_enc',
    'Hydraulic_Condition_enc', 'Brake_Condition_enc',
    'Electrical_Condition_enc', 'Fuel_System_Condition_enc',
    'Transmission_Condition_enc', 'Cooling_System_Condition_enc',
    # Life remaining
    'Engine_Life_Remaining_pct', 'Tyre_Life_Remaining_pct',
    'Hydraulic_Life_Remaining_pct', 'Battery_Life_Remaining_pct',
    'Brake_Life_Remaining_pct',
    # Maintenance history (past info, not future)
    'Days_Since_Last_Maintenance', 'Last_Maintenance_Type_enc',
    'PM_Interval_Hours', 'Next_PM_Due_Hours', 'Maintenance_Team_enc',
]

# Replacement flags (pre-failure inspection)
REPLACE_FLAGS = [c for c in df.columns if 'Replacement_Needed_enc' in c]
SENSOR_FEATURES += REPLACE_FLAGS

# Targets
TARGET_CLF  = 'Failure'
TARGET_REG  = 'Days_to_Next_Failure'
TARGET_PRI  = 'Maintenance_Priority_enc'

# Keep only existing columns
FEATURE_COLS = [f for f in SENSOR_FEATURES if f in df.columns]
print(f"  Features used (no leakage): {len(FEATURE_COLS)}")

X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
y_class = df[TARGET_CLF]
y_reg   = df[TARGET_REG]
y_pri   = df[TARGET_PRI]

print(f"  Failure rate: {y_class.mean()*100:.1f}%  ({y_class.sum()} of {len(y_class)})")

# ─────────────────────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
(X_tr, X_te,
 yc_tr, yc_te,
 yr_tr, yr_te,
 yp_tr, yp_te) = train_test_split(X, y_class, y_reg, y_pri,
                                   test_size=0.2, random_state=42,
                                   stratify=y_class)

print(f"  Train: {len(X_tr)}  Test: {len(X_te)}")

# ─────────────────────────────────────────────────────────────
# MODEL A — FAILURE CLASSIFICATION
# ─────────────────────────────────────────────────────────────
print("\n" + "─"*55)
print("  MODEL A — Failure Classification (Binary)")
print("─"*55)

cw = compute_class_weight('balanced', classes=np.array([0,1]), y=yc_tr)
cw_dict = {0: cw[0], 1: cw[1]}

rf_clf = RandomForestClassifier(
    n_estimators=300, max_depth=15, min_samples_split=4,
    min_samples_leaf=2, class_weight=cw_dict, random_state=42, n_jobs=-1)

gb_clf = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.08, max_depth=6,
    subsample=0.8, min_samples_leaf=2, random_state=42)

ensemble_clf = VotingClassifier(
    estimators=[('rf', rf_clf), ('gb', gb_clf)],
    voting='soft', weights=[2, 1])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results_clf = {}
for name, model in [("Random Forest", rf_clf),
                    ("Gradient Boosting", gb_clf),
                    ("Ensemble", ensemble_clf)]:
    cv_scores = cross_val_score(model, X_tr, yc_tr, cv=cv, scoring='roc_auc')
    model.fit(X_tr, yc_tr)
    yp  = model.predict(X_te)
    ypr = model.predict_proba(X_te)[:, 1]
    results_clf[name] = {
        'acc':      accuracy_score(yc_te, yp),
        'precision': precision_score(yc_te, yp),
        'recall':   recall_score(yc_te, yp),
        'f1':       f1_score(yc_te, yp),
        'auc':      roc_auc_score(yc_te, ypr),
        'cv_auc':   cv_scores,
        'cm':       confusion_matrix(yc_te, yp),
        'proba':    ypr,
        'report':   classification_report(yc_te, yp)
    }
    print(f"\n  [{name}]")
    print(f"    CV AUC : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"    Test   : Acc={results_clf[name]['acc']:.4f}  "
          f"AUC={results_clf[name]['auc']:.4f}  "
          f"F1={results_clf[name]['f1']:.4f}  "
          f"Recall={results_clf[name]['recall']:.4f}")

# ─────────────────────────────────────────────────────────────
# MODEL B — DAYS-TO-NEXT-FAILURE REGRESSION
# ─────────────────────────────────────────────────────────────
print("\n" + "─"*55)
print("  MODEL B — Days-to-Next-Failure Regression")
print("─"*55)

rf_reg = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
gb_reg = GradientBoostingRegressor(n_estimators=200, learning_rate=0.08,
                                    max_depth=5, subsample=0.8, random_state=42)

results_reg = {}
for name, model in [("Random Forest", rf_reg), ("Gradient Boosting", gb_reg)]:
    model.fit(X_tr, yr_tr)
    yp = model.predict(X_te)
    results_reg[name] = {
        'mae':  mean_absolute_error(yr_te, yp),
        'rmse': np.sqrt(mean_squared_error(yr_te, yp)),
        'r2':   r2_score(yr_te, yp),
        'pred': yp
    }
    print(f"  [{name}]  MAE={results_reg[name]['mae']:.2f} days  "
          f"RMSE={results_reg[name]['rmse']:.2f} days  R²={results_reg[name]['r2']:.4f}")

best_reg_name = max(results_reg, key=lambda n: results_reg[n]['r2'])
best_reg_pred = results_reg[best_reg_name]['pred']

# ─────────────────────────────────────────────────────────────
# MODEL C — MAINTENANCE PRIORITY (Multi-class)
# ─────────────────────────────────────────────────────────────
print("\n" + "─"*55)
print("  MODEL C — Maintenance Priority (Multi-class)")
print("─"*55)

# Re-define target without using Maintenance_Priority_enc as feature
# (it IS the target here, so it was already excluded from FEATURE_COLS above)
rf_pri = RandomForestClassifier(n_estimators=300, max_depth=15,
                                 class_weight='balanced', random_state=42, n_jobs=-1)
rf_pri.fit(X_tr, yp_tr)
yp_pri_pred = rf_pri.predict(X_te)
pri_acc  = accuracy_score(yp_te, yp_pri_pred)
pri_f1   = f1_score(yp_te, yp_pri_pred, average='weighted')
print(f"  [RF Multi-class]  Accuracy={pri_acc:.4f}  Weighted-F1={pri_f1:.4f}")
print(classification_report(yp_te, yp_pri_pred))

# ─────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────
fi = pd.Series(rf_clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
top20 = fi.head(20)

# ─────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────
COLORS = {
    'bg': '#0d1117', 'card': '#161b22', 'border': '#30363d',
    'accent1': '#58a6ff', 'accent2': '#ff7b72', 'accent3': '#3fb950',
    'accent4': '#d2a8ff', 'accent5': '#ffa657', 'text': '#e6edf3', 'muted': '#8b949e'
}

def style_ax(ax, title, fs=11):
    ax.set_facecolor(COLORS['card'])
    ax.tick_params(colors=COLORS['muted'], labelsize=8)
    ax.xaxis.label.set_color(COLORS['muted'])
    ax.yaxis.label.set_color(COLORS['muted'])
    ax.set_title(title, fontsize=fs, fontweight='bold', color=COLORS['text'], pad=10)
    for spine in ax.spines.values():
        spine.set_color(COLORS['border'])

fig = plt.figure(figsize=(26, 30))
fig.patch.set_facecolor(COLORS['bg'])
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.32)

CLF_COLORS = [COLORS['accent1'], COLORS['accent2'], COLORS['accent3']]
model_names = list(results_clf.keys())

# ── A1: Bar chart — CV AUC + Test AUC
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, 'Model A — CV vs Test AUC')
x = np.arange(len(model_names)); w = 0.35
cv_means = [results_clf[n]['cv_auc'].mean() for n in model_names]
test_aucs = [results_clf[n]['auc'] for n in model_names]
b1 = ax1.bar(x - w/2, cv_means, w, label='CV AUC', color=COLORS['accent1'], alpha=0.85)
b2 = ax1.bar(x + w/2, test_aucs, w, label='Test AUC', color=COLORS['accent3'], alpha=0.85)
ax1.set_xticks(x); ax1.set_xticklabels(model_names, rotation=12, fontsize=7.5, color=COLORS['muted'])
ymin = max(0, min(cv_means + test_aucs) - 0.05)
ax1.set_ylim(ymin, 1.0)
ax1.legend(facecolor=COLORS['card'], labelcolor=COLORS['text'], fontsize=8, framealpha=0.8)
for bar in list(b1)+list(b2):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
             f'{bar.get_height():.3f}', ha='center', va='bottom',
             color=COLORS['text'], fontsize=6.5)
ax1.set_ylabel('AUC Score')

# ── A2: ROC Curves
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, 'Model A — ROC Curves')
for i, (name, res) in enumerate(results_clf.items()):
    fpr, tpr, _ = roc_curve(yc_te, res['proba'])
    ax2.plot(fpr, tpr, color=CLF_COLORS[i], lw=2.2,
             label=f"{name} ({res['auc']:.3f})")
ax2.plot([0,1],[0,1],'--', color=COLORS['muted'], lw=1, alpha=0.5)
ax2.fill_between(fpr, tpr, alpha=0.05, color=COLORS['accent3'])
ax2.set_xlabel('False Positive Rate'); ax2.set_ylabel('True Positive Rate')
ax2.legend(facecolor=COLORS['card'], labelcolor=COLORS['text'], fontsize=7.5, framealpha=0.8)

# ── A3: Confusion Matrix — Ensemble
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3, 'Ensemble — Confusion Matrix')
cm = results_clf['Ensemble']['cm']
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
            linewidths=1.5, linecolor=COLORS['border'],
            annot_kws={'color': COLORS['text'], 'size': 15, 'weight': 'bold'})
ax3.set_xlabel('Predicted', color=COLORS['muted'])
ax3.set_ylabel('Actual', color=COLORS['muted'])
ax3.set_xticklabels(['No Failure','Failure'], color=COLORS['muted'], fontsize=8)
ax3.set_yticklabels(['No Failure','Failure'], color=COLORS['muted'], fontsize=8, rotation=0)

# ── B1: Metrics comparison bar
ax_m = fig.add_subplot(gs[1, 0])
style_ax(ax_m, 'Model A — Classification Metrics')
metrics = ['acc','precision','recall','f1']
metric_labels = ['Accuracy','Precision','Recall','F1']
x = np.arange(len(metrics)); w = 0.25
for i, (name, col) in enumerate(zip(model_names, CLF_COLORS)):
    vals = [results_clf[name][m] for m in metrics]
    bars = ax_m.bar(x + (i-1)*w, vals, w, label=name, color=col, alpha=0.85)
ax_m.set_xticks(x); ax_m.set_xticklabels(metric_labels, fontsize=8, color=COLORS['muted'])
ax_m.set_ylim(0, 1.15)
ax_m.legend(facecolor=COLORS['card'], labelcolor=COLORS['text'], fontsize=7, framealpha=0.8)
ax_m.set_ylabel('Score')

# ── B2: Feature Importance Top-20
ax4 = fig.add_subplot(gs[1, 1:])
style_ax(ax4, 'Top 20 Feature Importances (Random Forest — No Leakage)')
palette = [COLORS['accent3']]*5 + [COLORS['accent1']]*5 + [COLORS['accent4']]*5 + [COLORS['accent5']]*5
bars = ax4.barh(range(20), top20.values, color=palette, alpha=0.9, height=0.7)
ax4.set_yticks(range(20))
ax4.set_yticklabels(top20.index, fontsize=8.5, color=COLORS['text'])
ax4.invert_yaxis()
ax4.set_xlabel('Importance Score', color=COLORS['muted'])
for bar, val in zip(bars, top20.values):
    ax4.text(val + 0.0003, bar.get_y()+bar.get_height()/2,
             f'{val:.4f}', va='center', fontsize=7, color=COLORS['muted'])

# ── C1: Regression — Actual vs Predicted
ax5 = fig.add_subplot(gs[2, 0])
style_ax(ax5, f'Model B — Actual vs Predicted Days\n({best_reg_name})')
ax5.scatter(yr_te, best_reg_pred, alpha=0.35, s=14, color=COLORS['accent4'], edgecolors='none')
mn, mx = int(yr_te.min()), int(yr_te.max())
ax5.plot([mn,mx],[mn,mx],'--', color=COLORS['accent1'], lw=1.8, label='Perfect')
r2v = results_reg[best_reg_name]['r2']
maev = results_reg[best_reg_name]['mae']
ax5.text(0.05, 0.90, f'R²  = {r2v:.4f}\nMAE = {maev:.1f} days',
         transform=ax5.transAxes, color=COLORS['text'], fontsize=9,
         bbox=dict(boxstyle='round,pad=0.45', facecolor=COLORS['bg'], alpha=0.85))
ax5.set_xlabel('Actual Days-to-Failure'); ax5.set_ylabel('Predicted Days-to-Failure')
ax5.legend(facecolor=COLORS['card'], labelcolor=COLORS['text'], fontsize=8)

# ── C2: Residuals
ax6 = fig.add_subplot(gs[2, 1])
style_ax(ax6, 'Model B — Residuals Distribution')
residuals = yr_te.values - best_reg_pred
ax6.hist(residuals, bins=45, color=COLORS['accent1'], alpha=0.8, edgecolor='none')
ax6.axvline(0, color=COLORS['accent2'], lw=2, linestyle='--', label='Zero error')
ax6.axvline(residuals.mean(), color=COLORS['accent3'], lw=1.5,
            linestyle=':', label=f'Mean={residuals.mean():.1f}')
ax6.set_xlabel('Prediction Error (days)'); ax6.set_ylabel('Count')
ax6.legend(facecolor=COLORS['card'], labelcolor=COLORS['text'], fontsize=7.5)

# ── C3: Priority Confusion Matrix
ax7 = fig.add_subplot(gs[2, 2])
style_ax(ax7, 'Model C — Priority Confusion Matrix')
cm_pri = confusion_matrix(yp_te, yp_pri_pred)
sns.heatmap(cm_pri, annot=True, fmt='d', cmap='Purples', ax=ax7,
            linewidths=1.5, linecolor=COLORS['border'],
            annot_kws={'color': COLORS['text'], 'size': 12, 'weight': 'bold'})
ax7.set_xlabel('Predicted Priority', color=COLORS['muted'])
ax7.set_ylabel('Actual Priority', color=COLORS['muted'])
ax7.tick_params(colors=COLORS['muted'])

# Suptitle
fig.suptitle('NALCO HEMM Predictive Maintenance — ML Model Dashboard',
             fontsize=19, fontweight='bold', color=COLORS['text'], y=0.995)

plt.savefig(f'{OUT}/hemm_ml_results.png', dpi=150, bbox_inches='tight',
            facecolor=COLORS['bg'])
plt.close()
print("\n✔ Dashboard chart saved.")

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
ens = results_clf['Ensemble']
summary = f"""
╔══════════════════════════════════════════════════════════════════╗
║       NALCO HEMM PREDICTIVE MAINTENANCE — FINAL ML REPORT       ║
╚══════════════════════════════════════════════════════════════════╝

Dataset      : 2,000 records × {len(FEATURE_COLS)} clean features (no leakage)
Failure Rate : 17.8%  (class-weighted training used)
Train/Test   : 80% / 20%  |  5-Fold Cross-Validation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL A — FAILURE CLASSIFICATION (Binary)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Model                 AUC       Accuracy   Recall   F1-Score
  ─────────────────     ──────    ──────     ──────   ──────
  Random Forest       {results_clf['Random Forest']['auc']:.4f}    {results_clf['Random Forest']['acc']:.4f}     {results_clf['Random Forest']['recall']:.4f}   {results_clf['Random Forest']['f1']:.4f}
  Gradient Boosting   {results_clf['Gradient Boosting']['auc']:.4f}    {results_clf['Gradient Boosting']['acc']:.4f}     {results_clf['Gradient Boosting']['recall']:.4f}   {results_clf['Gradient Boosting']['f1']:.4f}
  Voting Ensemble ★  {ens['auc']:.4f}    {ens['acc']:.4f}     {ens['recall']:.4f}   {ens['f1']:.4f}

  CV AUC (Ensemble): {ens['cv_auc'].mean():.4f} ± {ens['cv_auc'].std():.4f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL B — DAYS-TO-NEXT-FAILURE REGRESSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Model                   R²        MAE (days)   RMSE (days)
  ─────────────────       ──────    ──────       ──────
  Random Forest Reg     {results_reg['Random Forest']['r2']:.4f}    {results_reg['Random Forest']['mae']:.2f}         {results_reg['Random Forest']['rmse']:.2f}
  Gradient Boosting Reg {results_reg['Gradient Boosting']['r2']:.4f}    {results_reg['Gradient Boosting']['mae']:.2f}         {results_reg['Gradient Boosting']['rmse']:.2f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL C — MAINTENANCE PRIORITY (Multi-class RF)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Accuracy    : {pri_acc:.4f}
  Weighted F1 : {pri_f1:.4f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOP 10 PREDICTIVE FEATURES (no-leakage model)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
for i, (feat, score) in enumerate(top20.head(10).items(), 1):
    summary += f"  {i:2d}. {feat:<40s}  {score:.5f}\n"

summary += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INTERPRETATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Model A predicts WILL the equipment fail (binary alert)
• Model B predicts HOW MANY DAYS until the next failure
• Model C predicts WHAT PRIORITY the maintenance should be
• Use Recall as primary metric (catching all failures is critical)
"""

print(summary)
with open(f'{OUT}/hemm_ml_summary.txt', 'w') as f:
    f.write(summary)

print(f"\n✔ Summary : {OUT}/hemm_ml_summary.txt")
print(f"✔ Charts  : {OUT}/hemm_ml_results.png")
print("\n✅  All models trained and evaluated successfully!")
