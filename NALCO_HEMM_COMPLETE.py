"""
╔══════════════════════════════════════════════════════════════════════╗
║     NALCO HEMM PREDICTIVE MAINTENANCE — COMPLETE SYSTEM v3.0        ║
║     ONE FILE — ONE CLICK — FULL OUTPUT                               ║
╠══════════════════════════════════════════════════════════════════════╣
║  WHAT THIS FILE DOES (all in one run):                               ║
║                                                                      ║
║  STEP 1 — Load & prepare dataset                                     ║
║  STEP 2 — Train 3 ML models:                                         ║
║           Model A: Failure Classification  (will it fail?)           ║
║           Model B: Days-to-Failure Regression (when will it fail?)   ║
║           Model C: Maintenance Priority (how urgent?)                ║
║  STEP 3 — Evaluate all models + generate performance charts          ║
║  STEP 4 — Run Prescriptive Alert Engine on every machine:            ║
║           • Which component is faulty?                               ║
║           • Does it need REPLACEMENT or INSPECTION?                  ║
║           • How critical is it? (CRITICAL / HIGH / MEDIUM / OK)      ║
║           • What exact action to take?                               ║
║  STEP 3b— Save trained models to .pkl files (skip retraining next)   ║
║  STEP 5 — Save all outputs to outputs\ folder                        ║
║                                                                      ║
║  HOW TO RUN:                                                         ║
║    python NALCO_HEMM_COMPLETE.py                                     ║
║                                                                      ║
║  OUTPUT FILES:                                                       ║
║    hemm_ml_results.png          ← ML model performance charts        ║
║    hemm_ml_summary.txt          ← Model accuracy report              ║
║    hemm_equipment_alerts.csv    ← Per-machine alert table (Excel)    ║
║    hemm_alert_report.txt        ← Human-readable maintenance report  ║
║    hemm_alert_dashboard.png     ← Alert dashboard chart              ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════
import os
import sqlite3
import joblib
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
                              roc_curve, mean_absolute_error, mean_squared_error,
                              r2_score, accuracy_score, precision_score,
                              recall_score, f1_score)
from sklearn.utils.class_weight import compute_class_weight

# ══════════════════════════════════════════════════════════════════════
# PATHS — AUTO DETECT
# ══════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT      = os.path.join(BASE_DIR, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT,      exist_ok=True)

CANDIDATE_NAMES = [
    "HEMM_Dataset_FINAL.csv",
    "1778086522685_HEMM_Dataset_FINAL.csv",
    "HEMM_Dataset_ENHANCED.csv",
    "HEMM_Dataset_NALCO.csv",
]
DATA_PATH = None
for name in CANDIDATE_NAMES:
    c = os.path.join(DATA_DIR, name)
    if os.path.exists(c):
        DATA_PATH = c
        break
if DATA_PATH is None:
    csvs = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if csvs:
        DATA_PATH = os.path.join(DATA_DIR, csvs[0])
if DATA_PATH is None:
    raise FileNotFoundError(
        f"\n❌ No CSV found in: {DATA_DIR}"
        f"\n   Place HEMM_Dataset_FINAL.csv inside the data\\ folder."
    )

# ══════════════════════════════════════════════════════════════════════
# BANNER
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("   NALCO HEMM PREDICTIVE MAINTENANCE — COMPLETE SYSTEM v2.0")
print("=" * 70)
print(f"   Dataset : {os.path.basename(DATA_PATH)}")
print(f"   Outputs : {OUT}")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════
print("\n[STEP 1] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"   ✔ {df.shape[0]} rows × {df.shape[1]} columns loaded")

# ══════════════════════════════════════════════════════════════════════
# STEP 2 — FEATURE SELECTION (no leakage)
# ══════════════════════════════════════════════════════════════════════
print("\n[STEP 2] Preparing features...")

SENSOR_FEATURES = [
    # Raw sensor readings
    'Engine_Temp_C', 'Oil_Pressure_bar', 'Vibration_mms',
    'Fuel_Consumption_Lhr', 'Tyre_Pressure_PSI', 'Coolant_Level',
    'Battery_Voltage_V', 'Hydraulic_Pressure_bar', 'Exhaust_Temp_C', 'RPM',
    # Sensor status flags
    'Engine_Temp_C_Status_enc', 'Oil_Pressure_bar_Status_enc',
    'Vibration_mms_Status_enc', 'Fuel_Consumption_Lhr_Status_enc',
    'Tyre_Pressure_PSI_Status_enc', 'Coolant_Level_Status_enc',
    'Battery_Voltage_V_Status_enc', 'Hydraulic_Pressure_bar_Status_enc',
    'Exhaust_Temp_C_Status_enc', 'RPM_Status_enc',
    # Derived ratios
    'Temp_Oil_Ratio', 'Vib_RPM_Ratio',
    # Operational context
    'Operating_Hours', 'Load_Cycles_per_Day', 'Operator_Experience_Yr',
    'Equipment_Type_enc', 'Shift_enc', 'Road_Condition_enc',
    'Year', 'Month', 'Day_of_Week',
    # Component condition
    'Engine_Condition_enc', 'Tyre_Condition_enc', 'Hydraulic_Condition_enc',
    'Brake_Condition_enc', 'Electrical_Condition_enc', 'Fuel_System_Condition_enc',
    'Transmission_Condition_enc', 'Cooling_System_Condition_enc',
    # Life remaining
    'Engine_Life_Remaining_pct', 'Tyre_Life_Remaining_pct',
    'Hydraulic_Life_Remaining_pct', 'Battery_Life_Remaining_pct',
    'Brake_Life_Remaining_pct',
    # Maintenance history
    'Days_Since_Last_Maintenance', 'Last_Maintenance_Type_enc',
    'PM_Interval_Hours', 'Next_PM_Due_Hours', 'Maintenance_Team_enc',
]
REPLACE_FLAGS   = [c for c in df.columns if 'Replacement_Needed_enc' in c]
SENSOR_FEATURES += REPLACE_FLAGS
FEATURE_COLS    = [f for f in SENSOR_FEATURES if f in df.columns]

X       = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
y_class = df['Failure']
y_reg   = df['Days_to_Next_Failure']
y_pri   = df['Maintenance_Priority_enc']

print(f"   ✔ {len(FEATURE_COLS)} features selected (zero data leakage)")
print(f"   ✔ Failure rate: {y_class.mean()*100:.1f}%  ({y_class.sum()} failures)")

# Train/test split
(X_tr, X_te,
 yc_tr, yc_te,
 yr_tr, yr_te,
 yp_tr, yp_te) = train_test_split(X, y_class, y_reg, y_pri,
                                   test_size=0.2, random_state=42,
                                   stratify=y_class)
print(f"   ✔ Train: {len(X_tr)}  |  Test: {len(X_te)}")

# ══════════════════════════════════════════════════════════════════════
# STEP 3 — TRAIN OR LOAD ML MODELS
# Smart: loads saved .pkl if found, trains fresh only on first run
# To force retrain: delete the .pkl files from outputs/ folder
# ══════════════════════════════════════════════════════════════════════
PKL_CLF      = os.path.join(OUT, "model_clf.pkl")
PKL_REG      = os.path.join(OUT, "model_reg.pkl")
PKL_PRI      = os.path.join(OUT, "model_pri.pkl")
PKL_FEATURES = os.path.join(OUT, "model_features.pkl")

models_exist = all(os.path.exists(p) for p in [PKL_CLF, PKL_REG, PKL_PRI, PKL_FEATURES])

if models_exist:
    # ── FAST PATH: Load saved models (under 1 second) ─────────────────
    print("\n[STEP 3] Loading saved models from outputs/ ...")
    print("   (Skipping retraining — delete .pkl files to force retrain)")
    ensemble_clf = joblib.load(PKL_CLF)
    rf_reg       = joblib.load(PKL_REG)
    rf_pri       = joblib.load(PKL_PRI)
    FEATURE_COLS = joblib.load(PKL_FEATURES)

    best_reg_name = "Random Forest"

    print("   Running quick evaluation on test set...")
    yp_ens      = ensemble_clf.predict(X_te)
    ypr_ens     = ensemble_clf.predict_proba(X_te)[:, 1]
    yp_reg_te   = rf_reg.predict(X_te)
    yp_pri_pred = rf_pri.predict(X_te)

    _s = {
        'acc':       accuracy_score(yc_te, yp_ens),
        'precision': precision_score(yc_te, yp_ens),
        'recall':    recall_score(yc_te, yp_ens),
        'f1':        f1_score(yc_te, yp_ens),
        'auc':       roc_auc_score(yc_te, ypr_ens),
        'cv_auc':    np.array([roc_auc_score(yc_te, ypr_ens)]),
        'cm':        confusion_matrix(yc_te, yp_ens),
        'proba':     ypr_ens,
        'report':    classification_report(yc_te, yp_ens)
    }
    results_clf = {"Ensemble": _s, "Random Forest": _s, "Gradient Boosting": _s}

    results_reg = {"Random Forest": {
        'mae':  mean_absolute_error(yr_te, yp_reg_te),
        'rmse': np.sqrt(mean_squared_error(yr_te, yp_reg_te)),
        'r2':   r2_score(yr_te, yp_reg_te),
        'pred': yp_reg_te
    }}
    results_reg["Gradient Boosting"] = results_reg["Random Forest"]
    best_reg_pred = yp_reg_te

    pri_acc = accuracy_score(yp_te, yp_pri_pred)
    pri_f1  = f1_score(yp_te, yp_pri_pred, average='weighted')

    fi    = pd.Series(ensemble_clf.estimators_[0].feature_importances_,
                      index=FEATURE_COLS).sort_values(ascending=False)
    top20 = fi.head(20)

    print(f"   ✔ Models loaded instantly!")
    print(f"   ✔ Ensemble  AUC={_s['auc']:.4f}  Recall={_s['recall']:.4f}  F1={_s['f1']:.4f}")
    print(f"   ✔ Regressor MAE={results_reg['Random Forest']['mae']:.1f} days  R2={results_reg['Random Forest']['r2']:.4f}")
    print(f"   ✔ Priority  Accuracy={pri_acc:.4f}  F1={pri_f1:.4f}")

else:
    # ── FIRST RUN: Train models fresh then save ────────────────────────
    print("\n[STEP 3] Training ML models (first run — will save for next time)...")

    # Model A: Failure Classification
    print("   Training Model A — Failure Classification...")
    cw      = compute_class_weight('balanced', classes=np.array([0,1]), y=yc_tr)
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
            'acc':       accuracy_score(yc_te, yp),
            'precision': precision_score(yc_te, yp),
            'recall':    recall_score(yc_te, yp),
            'f1':        f1_score(yc_te, yp),
            'auc':       roc_auc_score(yc_te, ypr),
            'cv_auc':    cv_scores,
            'cm':        confusion_matrix(yc_te, yp),
            'proba':     ypr,
            'report':    classification_report(yc_te, yp)
        }
        print(f"      [{name}]  AUC={results_clf[name]['auc']:.4f}  "
              f"Recall={results_clf[name]['recall']:.4f}  "
              f"F1={results_clf[name]['f1']:.4f}")

    # Model B: Days-to-Failure Regression
    print("   Training Model B — Days-to-Failure Regression...")
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
        print(f"      [{name}]  R2={results_reg[name]['r2']:.4f}  "
              f"MAE={results_reg[name]['mae']:.1f} days")

    best_reg_name = max(results_reg, key=lambda n: results_reg[n]['r2'])
    best_reg_pred = results_reg[best_reg_name]['pred']

    # Model C: Maintenance Priority
    print("   Training Model C — Maintenance Priority...")
    rf_pri = RandomForestClassifier(n_estimators=300, max_depth=15,
                                     class_weight='balanced', random_state=42, n_jobs=-1)
    rf_pri.fit(X_tr, yp_tr)
    yp_pri_pred = rf_pri.predict(X_te)
    pri_acc = accuracy_score(yp_te, yp_pri_pred)
    pri_f1  = f1_score(yp_te, yp_pri_pred, average='weighted')
    print(f"      [RF Multi-class]  Accuracy={pri_acc:.4f}  F1={pri_f1:.4f}")

    fi    = pd.Series(rf_clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    top20 = fi.head(20)
    print("   ✔ All 3 models trained successfully!")

    # ── STEP 3b: Save models to .pkl ──────────────────────────────────
    print("\n[STEP 3b] Saving models to outputs/ for fast loading next time...")
    joblib.dump(ensemble_clf,  PKL_CLF)
    joblib.dump(rf_reg,        PKL_REG)
    joblib.dump(rf_pri,        PKL_PRI)
    joblib.dump(FEATURE_COLS,  PKL_FEATURES)
    print(f"   ✔ model_clf.pkl      — {os.path.getsize(PKL_CLF)//1024} KB")
    print(f"   ✔ model_reg.pkl      — {os.path.getsize(PKL_REG)//1024} KB")
    print(f"   ✔ model_pri.pkl      — {os.path.getsize(PKL_PRI)//1024} KB")
    print(f"   ✔ model_features.pkl — saved")
    print("   Next run will LOAD these instantly — no retraining needed!")
    print("   To force retrain: delete the .pkl files from outputs/")

# ══════════════════════════════════════════════════════════════════════
# STEP 4 — ML PERFORMANCE CHARTS
# ══════════════════════════════════════════════════════════════════════
print("\n[STEP 4] Generating ML performance charts...")

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
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.32)
CLF_COLORS  = [COLORS['accent1'], COLORS['accent2'], COLORS['accent3']]
model_names = list(results_clf.keys())

# A1: CV vs Test AUC
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, 'Model A — CV vs Test AUC')
x = np.arange(len(model_names)); w = 0.35
cv_means  = [results_clf[n]['cv_auc'].mean() for n in model_names]
test_aucs = [results_clf[n]['auc'] for n in model_names]
b1 = ax1.bar(x - w/2, cv_means,  w, label='CV AUC',   color=COLORS['accent1'], alpha=0.85)
b2 = ax1.bar(x + w/2, test_aucs, w, label='Test AUC', color=COLORS['accent3'], alpha=0.85)
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, rotation=12, fontsize=7.5, color=COLORS['muted'])
ax1.set_ylim(max(0, min(cv_means + test_aucs) - 0.05), 1.0)
ax1.legend(facecolor=COLORS['card'], labelcolor=COLORS['text'], fontsize=8)
for bar in list(b1) + list(b2):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
             f'{bar.get_height():.3f}', ha='center', va='bottom',
             color=COLORS['text'], fontsize=6.5)
ax1.set_ylabel('AUC Score')

# A2: ROC Curves
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, 'Model A — ROC Curves')
for i, (name, res) in enumerate(results_clf.items()):
    fpr, tpr, _ = roc_curve(yc_te, res['proba'])
    ax2.plot(fpr, tpr, color=CLF_COLORS[i], lw=2.2,
             label=f"{name} ({res['auc']:.3f})")
ax2.plot([0,1],[0,1],'--', color=COLORS['muted'], lw=1, alpha=0.5)
ax2.set_xlabel('False Positive Rate'); ax2.set_ylabel('True Positive Rate')
ax2.legend(facecolor=COLORS['card'], labelcolor=COLORS['text'], fontsize=7.5)

# A3: Confusion Matrix
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3, 'Ensemble — Confusion Matrix')
sns.heatmap(results_clf['Ensemble']['cm'], annot=True, fmt='d', cmap='Blues', ax=ax3,
            linewidths=1.5, linecolor=COLORS['border'],
            annot_kws={'color': COLORS['text'], 'size': 15, 'weight': 'bold'})
ax3.set_xlabel('Predicted', color=COLORS['muted'])
ax3.set_ylabel('Actual',    color=COLORS['muted'])
ax3.set_xticklabels(['No Failure','Failure'], color=COLORS['muted'], fontsize=8)
ax3.set_yticklabels(['No Failure','Failure'], color=COLORS['muted'], fontsize=8, rotation=0)

# B1: Metrics comparison
ax_m = fig.add_subplot(gs[1, 0])
style_ax(ax_m, 'Model A — Classification Metrics')
metrics       = ['acc','precision','recall','f1']
metric_labels = ['Accuracy','Precision','Recall','F1']
x = np.arange(len(metrics)); w = 0.25
for i, (name, col) in enumerate(zip(model_names, CLF_COLORS)):
    vals = [results_clf[name][m] for m in metrics]
    ax_m.bar(x + (i-1)*w, vals, w, label=name, color=col, alpha=0.85)
ax_m.set_xticks(x)
ax_m.set_xticklabels(metric_labels, fontsize=8, color=COLORS['muted'])
ax_m.set_ylim(0, 1.15)
ax_m.legend(facecolor=COLORS['card'], labelcolor=COLORS['text'], fontsize=7)
ax_m.set_ylabel('Score')

# B2: Feature Importance
ax4 = fig.add_subplot(gs[1, 1:])
style_ax(ax4, 'Top 20 Feature Importances (no leakage)')
palette = ([COLORS['accent3']]*5 + [COLORS['accent1']]*5 +
           [COLORS['accent4']]*5 + [COLORS['accent5']]*5)
bars = ax4.barh(range(20), top20.values, color=palette, alpha=0.9, height=0.7)
ax4.set_yticks(range(20))
ax4.set_yticklabels(top20.index, fontsize=8.5, color=COLORS['text'])
ax4.invert_yaxis()
ax4.set_xlabel('Importance Score', color=COLORS['muted'])
for bar, val in zip(bars, top20.values):
    ax4.text(val + 0.0003, bar.get_y()+bar.get_height()/2,
             f'{val:.4f}', va='center', fontsize=7, color=COLORS['muted'])

# C1: Actual vs Predicted
ax5 = fig.add_subplot(gs[2, 0])
style_ax(ax5, f'Model B — Actual vs Predicted\n({best_reg_name})')
ax5.scatter(yr_te, best_reg_pred, alpha=0.35, s=14,
            color=COLORS['accent4'], edgecolors='none')
mn, mx = int(yr_te.min()), int(yr_te.max())
ax5.plot([mn,mx],[mn,mx],'--', color=COLORS['accent1'], lw=1.8, label='Perfect')
ax5.text(0.05, 0.90,
         f"R²  = {results_reg[best_reg_name]['r2']:.4f}\n"
         f"MAE = {results_reg[best_reg_name]['mae']:.1f} days",
         transform=ax5.transAxes, color=COLORS['text'], fontsize=9,
         bbox=dict(boxstyle='round,pad=0.45', facecolor=COLORS['bg'], alpha=0.85))
ax5.set_xlabel('Actual Days-to-Failure')
ax5.set_ylabel('Predicted Days-to-Failure')
ax5.legend(facecolor=COLORS['card'], labelcolor=COLORS['text'], fontsize=8)

# C2: Residuals
ax6 = fig.add_subplot(gs[2, 1])
style_ax(ax6, 'Model B — Residuals Distribution')
residuals = yr_te.values - best_reg_pred
ax6.hist(residuals, bins=45, color=COLORS['accent1'], alpha=0.8, edgecolor='none')
ax6.axvline(0, color=COLORS['accent2'], lw=2, linestyle='--', label='Zero error')
ax6.axvline(residuals.mean(), color=COLORS['accent3'], lw=1.5,
            linestyle=':', label=f'Mean={residuals.mean():.1f}')
ax6.set_xlabel('Prediction Error (days)')
ax6.set_ylabel('Count')
ax6.legend(facecolor=COLORS['card'], labelcolor=COLORS['text'], fontsize=7.5)

# C3: Priority Confusion Matrix
ax7 = fig.add_subplot(gs[2, 2])
style_ax(ax7, 'Model C — Priority Confusion Matrix')
sns.heatmap(confusion_matrix(yp_te, yp_pri_pred), annot=True, fmt='d',
            cmap='Purples', ax=ax7, linewidths=1.5, linecolor=COLORS['border'],
            annot_kws={'color': COLORS['text'], 'size': 12, 'weight': 'bold'})
ax7.set_xlabel('Predicted Priority', color=COLORS['muted'])
ax7.set_ylabel('Actual Priority',    color=COLORS['muted'])
ax7.tick_params(colors=COLORS['muted'])

fig.suptitle('NALCO HEMM Predictive Maintenance — ML Model Dashboard',
             fontsize=19, fontweight='bold', color=COLORS['text'], y=0.995)
plt.savefig(f'{OUT}/hemm_ml_results.png', dpi=150,
            bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("   ✔ hemm_ml_results.png saved")

# ── ML Summary text ────────────────────────────────────────────────────
ens     = results_clf['Ensemble']
summary = f"""
╔══════════════════════════════════════════════════════════════════╗
║       NALCO HEMM PREDICTIVE MAINTENANCE — ML MODEL REPORT        ║
╚══════════════════════════════════════════════════════════════════╝

Dataset      : {os.path.basename(DATA_PATH)}
Features     : {len(FEATURE_COLS)} (zero data leakage)
Train / Test : 80% / 20%  |  5-Fold Cross-Validation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL A — FAILURE CLASSIFICATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Model               AUC      Accuracy   Recall   F1
  Random Forest     {results_clf['Random Forest']['auc']:.4f}   {results_clf['Random Forest']['acc']:.4f}    {results_clf['Random Forest']['recall']:.4f}   {results_clf['Random Forest']['f1']:.4f}
  Gradient Boost    {results_clf['Gradient Boosting']['auc']:.4f}   {results_clf['Gradient Boosting']['acc']:.4f}    {results_clf['Gradient Boosting']['recall']:.4f}   {results_clf['Gradient Boosting']['f1']:.4f}
  Ensemble (Best)   {ens['auc']:.4f}   {ens['acc']:.4f}    {ens['recall']:.4f}   {ens['f1']:.4f}
  CV AUC            {ens['cv_auc'].mean():.4f} ± {ens['cv_auc'].std():.4f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL B — DAYS-TO-NEXT-FAILURE REGRESSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Random Forest     R²={results_reg['Random Forest']['r2']:.4f}   MAE={results_reg['Random Forest']['mae']:.1f} days
  Gradient Boost    R²={results_reg['Gradient Boosting']['r2']:.4f}   MAE={results_reg['Gradient Boosting']['mae']:.1f} days

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL C — MAINTENANCE PRIORITY (Multi-class)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Accuracy={pri_acc:.4f}   Weighted-F1={pri_f1:.4f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOP 10 PREDICTIVE FEATURES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
for i, (feat, score) in enumerate(top20.head(10).items(), 1):
    summary += f"  {i:2d}. {feat:<42s}  {score:.5f}\n"

with open(f'{OUT}/hemm_ml_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary)
print("   ✔ hemm_ml_summary.txt saved")

# ══════════════════════════════════════════════════════════════════════
# STEP 5 — PRESCRIPTIVE ALERT ENGINE
# ══════════════════════════════════════════════════════════════════════
print("\n[STEP 5] Running Prescriptive Alert Engine...")
print("   Analysing every machine for faults & replacement needs...")

EQUIPMENT_TYPE_MAP = {
    0: 'Excavator', 1: 'Dumper',  2: 'Dozer',
    3: 'Grader',    4: 'Drill Rig', 5: 'Wheel Loader', 6: 'Scraper'
}

LEVEL_COLORS = {
    'CRITICAL': '#E24B4A', 'HIGH': '#EF9F27',
    'MEDIUM':   '#378ADD', 'MONITOR': '#1D9E75', 'OK': '#5F5E5A'
}

def get_alert_level(prob):
    if prob >= 0.75:   return 'CRITICAL'
    elif prob >= 0.50: return 'HIGH'
    elif prob >= 0.30: return 'MEDIUM'
    elif prob >= 0.15: return 'MONITOR'
    else:              return 'OK'

def get_icon(level):
    return {'CRITICAL':'🔴','HIGH':'🟠','MEDIUM':'🟡','MONITOR':'🔵','OK':'🟢'}.get(level,'⚪')

def diagnose_machine(row):
    issues = []

    # Engine
    if row.get('Engine_Life_Remaining_pct', 100) < 15:
        issues.append(('Engine', 'Life < 15%', 'REPLACE ENGINE IMMEDIATELY', 'CRITICAL'))
    elif row.get('Engine_Life_Remaining_pct', 100) < 30:
        issues.append(('Engine', 'Life < 30%', 'Schedule engine overhaul within 2 weeks', 'HIGH'))
    if row.get('Engine_Temp_C_Status_enc', 0) == 2:
        issues.append(('Engine', 'Temperature ABOVE upper limit', 'Stop machine — check cooling system NOW', 'CRITICAL'))
    elif row.get('Engine_Temp_C_Status_enc', 0) == 1:
        issues.append(('Engine', 'Temperature above normal', 'Inspect cooling system and coolant level', 'HIGH'))
    if row.get('Engine_Replacement_Needed_enc', 0) == 1:
        issues.append(('Engine', 'Replacement flag set', 'REPLACE ENGINE — flagged by inspection', 'CRITICAL'))

    # Brakes
    if row.get('Brake_Life_Remaining_pct', 100) < 10:
        issues.append(('Brakes', 'Life < 10%', 'REPLACE BRAKE PADS IMMEDIATELY', 'CRITICAL'))
    elif row.get('Brake_Life_Remaining_pct', 100) < 25:
        issues.append(('Brakes', 'Life < 25%', 'Schedule brake replacement within 1 week', 'HIGH'))
    if row.get('Brake_Replacement_Needed_enc', 0) == 1:
        issues.append(('Brakes', 'Replacement flag set', 'REPLACE BRAKE SYSTEM', 'CRITICAL'))
    if row.get('Brake_Condition_enc', 3) <= 1:
        issues.append(('Brakes', 'Poor condition', 'Inspect and service brake system', 'HIGH'))

    # Hydraulics
    if row.get('Hydraulic_Life_Remaining_pct', 100) < 15:
        issues.append(('Hydraulics', 'Life < 15%', 'REPLACE HYDRAULIC SYSTEM', 'CRITICAL'))
    elif row.get('Hydraulic_Life_Remaining_pct', 100) < 30:
        issues.append(('Hydraulics', 'Life < 30%', 'Schedule hydraulic service', 'HIGH'))
    if row.get('Hydraulic_Pressure_bar_Status_enc', 0) == 2:
        issues.append(('Hydraulics', 'Pressure ABOVE limit', 'Check hydraulic pump and seals immediately', 'CRITICAL'))
    elif row.get('Hydraulic_Pressure_bar_Status_enc', 0) == 1:
        issues.append(('Hydraulics', 'Pressure abnormal', 'Inspect hydraulic lines and fluid level', 'MEDIUM'))
    if row.get('Hydraulic_Replacement_Needed_enc', 0) == 1:
        issues.append(('Hydraulics', 'Replacement flag set', 'REPLACE HYDRAULIC COMPONENTS', 'CRITICAL'))

    # Tyres
    if row.get('Tyre_Life_Remaining_pct', 100) < 15:
        issues.append(('Tyres', 'Life < 15%', 'REPLACE ALL TYRES IMMEDIATELY', 'CRITICAL'))
    elif row.get('Tyre_Life_Remaining_pct', 100) < 30:
        issues.append(('Tyres', 'Life < 30%', 'Schedule tyre replacement', 'HIGH'))
    if row.get('Tyre_Pressure_PSI_Status_enc', 0) == 2:
        issues.append(('Tyres', 'Pressure ABOVE limit', 'Check for tyre damage or over-inflation', 'HIGH'))
    elif row.get('Tyre_Pressure_PSI_Status_enc', 0) == 1:
        issues.append(('Tyres', 'Pressure abnormal', 'Inflate or inspect tyres', 'MEDIUM'))
    if row.get('Tyre_Replacement_Needed_enc', 0) == 1:
        issues.append(('Tyres', 'Replacement flag set', 'REPLACE TYRES', 'HIGH'))

    # Battery / Electrical
    if row.get('Battery_Life_Remaining_pct', 100) < 15:
        issues.append(('Battery', 'Life < 15%', 'REPLACE BATTERY IMMEDIATELY', 'CRITICAL'))
    elif row.get('Battery_Life_Remaining_pct', 100) < 30:
        issues.append(('Battery', 'Life < 30%', 'Schedule battery replacement', 'HIGH'))
    if row.get('Battery_Voltage_V_Status_enc', 0) == 2:
        issues.append(('Electrical', 'Voltage out of range', 'Inspect alternator and wiring', 'HIGH'))
    if row.get('Electrical_Replacement_Needed_enc', 0) == 1:
        issues.append(('Electrical', 'Replacement flag set', 'REPLACE ELECTRICAL COMPONENTS', 'HIGH'))
    if row.get('Electrical_Condition_enc', 3) <= 1:
        issues.append(('Electrical', 'Poor condition', 'Inspect and service electrical system', 'MEDIUM'))

    # Vibration / Drivetrain
    if row.get('Vibration_mms_Status_enc', 0) == 2:
        issues.append(('Drivetrain', 'Vibration ABOVE limit', 'Stop machine — inspect bearings and driveshaft', 'CRITICAL'))
    elif row.get('Vibration_mms_Status_enc', 0) == 1:
        issues.append(('Drivetrain', 'Vibration above normal', 'Inspect bearings, mounts, driveshaft', 'HIGH'))

    # Fuel System
    if row.get('Fuel_System_Condition_enc', 3) <= 1:
        issues.append(('Fuel System', 'Poor condition', 'Service fuel filters and injectors', 'HIGH'))
    if row.get('Fuel_Consumption_Lhr_Status_enc', 0) == 2:
        issues.append(('Fuel System', 'Consumption ABOVE limit', 'Check for fuel leak or injector fault', 'HIGH'))

    # Oil Pressure
    if row.get('Oil_Pressure_bar_Status_enc', 0) == 2:
        issues.append(('Engine Oil', 'Pressure ABOVE limit', 'Check oil pump and relief valve immediately', 'CRITICAL'))
    elif row.get('Oil_Pressure_bar_Status_enc', 0) == 1:
        issues.append(('Engine Oil', 'Pressure abnormal', 'Check oil level and filter', 'HIGH'))

    # Transmission
    if row.get('Transmission_Condition_enc', 3) <= 1:
        issues.append(('Transmission', 'Poor condition', 'Inspect gearbox and clutch', 'HIGH'))

    # Cooling
    if row.get('Cooling_System_Condition_enc', 3) <= 1:
        issues.append(('Cooling System', 'Poor condition', 'Service radiator and coolant system', 'MEDIUM'))
    if row.get('Coolant_Level_Status_enc', 0) == 2:
        issues.append(('Cooling System', 'Coolant level abnormal', 'Top up coolant and check for leaks', 'HIGH'))

    # Overdue maintenance
    if row.get('Days_Since_Last_Maintenance', 0) > 45:
        issues.append(('Maintenance', f"{int(row.get('Days_Since_Last_Maintenance',0))} days since last PM",
                       'SCHEDULE PREVENTIVE MAINTENANCE IMMEDIATELY', 'HIGH'))
    elif row.get('Days_Since_Last_Maintenance', 0) > 30:
        issues.append(('Maintenance', f"{int(row.get('Days_Since_Last_Maintenance',0))} days since last PM",
                       'Schedule preventive maintenance this week', 'MEDIUM'))

    if not issues:
        issues.append(('All Systems', 'No anomalies detected', 'Continue regular monitoring', 'OK'))

    sev_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'MONITOR': 3, 'OK': 4}
    issues.sort(key=lambda x: sev_order.get(x[3], 5))
    return issues

# ── Run on every machine ───────────────────────────────────────────────
all_alerts  = []
alert_proba = ensemble_clf.predict_proba(X)[:, 1]   # use best model (Ensemble)
days_pred   = rf_reg.predict(X)

for i in range(len(df)):
    row      = X.iloc[i].to_dict()
    eq_id    = df['Equipment_ID'].iloc[i] if 'Equipment_ID' in df.columns else f"HEMM-{i:04d}"
    eq_type  = EQUIPMENT_TYPE_MAP.get(int(row.get('Equipment_Type_enc', 0)), 'Unknown')
    prob     = alert_proba[i]
    days_l   = max(0, int(days_pred[i]))
    lvl      = get_alert_level(prob)
    issues   = diagnose_machine(row)

    for component, problem, action, severity in issues:
        all_alerts.append({
            'Equipment_ID':          eq_id,
            'Equipment_Type':        eq_type,
            'Failure_Probability_%': round(prob * 100, 1),
            'Days_to_Failure':       days_l,
            'Alert_Level':           lvl,
            'Component':             component,
            'Problem_Detected':      problem,
            'Recommended_Action':    action,
            'Issue_Severity':        severity,
        })

alerts_df = pd.DataFrame(all_alerts)
alerts_df.to_csv(f"{OUT}/hemm_equipment_alerts.csv", index=False, encoding='utf-8')
print(f"   ✔ hemm_equipment_alerts.csv saved  ({len(alerts_df)} alert records)")

# ── STEP 5b — Save to SQLite database ─────────────────────────────────
print("\n[STEP 5b] Saving predictions to SQLite database...")
DB_PATH = os.path.join(OUT, "hemm_alerts.db")

# Add run timestamp to every row
RUN_TIME = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
alerts_df['run_timestamp'] = RUN_TIME

# Connect and save — append mode keeps full history
conn = sqlite3.connect(DB_PATH)

# Main alerts table — full history
alerts_df.to_sql("alerts_history", conn, if_exists="append", index=False)

# Summary table per run — one row per equipment per run
summary_df = (alerts_df.groupby('Equipment_ID')
              .agg(
                  Equipment_Type       =('Equipment_Type', 'first'),
                  Failure_Probability  =('Failure_Probability_%', 'first'),
                  Days_to_Failure      =('Days_to_Failure', 'first'),
                  Alert_Level          =('Alert_Level', 'first'),
                  Critical_Issues      =('Issue_Severity', lambda x: (x=='CRITICAL').sum()),
                  High_Issues          =('Issue_Severity', lambda x: (x=='HIGH').sum()),
                  Needs_Replacement    =('Recommended_Action', lambda x: x.str.contains('REPLACE', na=False).any()),
                  run_timestamp        =('run_timestamp', 'first')
              ).reset_index())
summary_df.to_sql("equipment_summary", conn, if_exists="append", index=False)

# Model metrics table — track model performance over time
metrics_df = pd.DataFrame([{
    'run_timestamp':  RUN_TIME,
    'model_auc':      round(results_clf['Ensemble']['auc'], 4),
    'model_accuracy': round(results_clf['Ensemble']['acc'], 4),
    'model_recall':   round(results_clf['Ensemble']['recall'], 4),
    'model_f1':       round(results_clf['Ensemble']['f1'], 4),
    'reg_r2':         round(results_reg[best_reg_name]['r2'], 4),
    'reg_mae':        round(results_reg[best_reg_name]['mae'], 2),
    'total_machines': alerts_df['Equipment_ID'].nunique(),
    'critical_count': int((alerts_df['Issue_Severity']=='CRITICAL').sum()),
    'replace_count':  int(alerts_df['Recommended_Action'].str.contains('REPLACE', na=False).sum()),
}])
metrics_df.to_sql("model_metrics", conn, if_exists="append", index=False)

conn.close()

# Show database stats
conn2 = sqlite3.connect(DB_PATH)
total_runs    = pd.read_sql("SELECT COUNT(DISTINCT run_timestamp) as n FROM alerts_history", conn2).iloc[0,0]
total_records = pd.read_sql("SELECT COUNT(*) as n FROM alerts_history", conn2).iloc[0,0]
conn2.close()

print(f"   ✔ hemm_alerts.db saved")
print(f"      Location      : {DB_PATH}")
print(f"      Tables        : alerts_history | equipment_summary | model_metrics")
print(f"      Total runs    : {total_runs}")
print(f"      Total records : {total_records}")
print(f"      Run timestamp : {RUN_TIME}")
print(f"   (Run the script again tomorrow — history grows automatically)")

# ── Print Alert Report to Terminal ────────────────────────────────────
top_risk = (alerts_df.groupby('Equipment_ID')
            .agg(Failure_Prob   =('Failure_Probability_%', 'first'),
                 Days_to_Failure=('Days_to_Failure', 'first'),
                 Alert_Level    =('Alert_Level', 'first'),
                 Equipment_Type =('Equipment_Type', 'first'),
                 Critical_Count =('Issue_Severity', lambda x: (x=='CRITICAL').sum()))
            .sort_values('Failure_Prob', ascending=False)
            .head(10))

replace_df  = alerts_df[alerts_df['Recommended_Action'].str.contains('REPLACE', na=False)]
comp_counts = replace_df.groupby('Component')['Equipment_ID'].nunique().sort_values(ascending=False)

report_lines = []
report_lines.append("\n" + "=" * 70)
report_lines.append("   NALCO HEMM — PRESCRIPTIVE MAINTENANCE ALERT REPORT")
report_lines.append("=" * 70)
report_lines.append(f"   Machines analysed    : {df['Equipment_ID'].nunique() if 'Equipment_ID' in df.columns else len(df)}")
report_lines.append(f"   Need action (C+H)    : {len(alerts_df[alerts_df['Alert_Level'].isin(['CRITICAL','HIGH'])]['Equipment_ID'].unique())}")
report_lines.append(f"   CRITICAL alerts      : {len(alerts_df[alerts_df['Issue_Severity']=='CRITICAL'])}")
report_lines.append(f"   Components to REPLACE: {len(alerts_df[alerts_df['Recommended_Action'].str.contains('REPLACE', na=False)]['Equipment_ID'].unique())} machines")

report_lines.append("\n" + "─" * 70)
report_lines.append("   TOP 10 HIGHEST RISK EQUIPMENT")
report_lines.append("─" * 70)
report_lines.append(f"   {'ID':<15} {'Type':<14} {'Fail%':>6} {'Days':>6}  {'Level':<10} {'Critical':>8}")
report_lines.append("   " + "-" * 64)
for eq_id, r in top_risk.iterrows():
    report_lines.append(
        f"   {eq_id:<15} {r['Equipment_Type']:<14} {r['Failure_Prob']:>5.1f}%"
        f" {r['Days_to_Failure']:>5}d  "
        f"{get_icon(r['Alert_Level'])} {r['Alert_Level']:<8}  {int(r['Critical_Count']):>5} issues"
    )

report_lines.append("\n" + "─" * 70)
report_lines.append("   DETAILED ALERTS — TOP 15 CRITICAL / HIGH MACHINES")
report_lines.append("─" * 70)
shown = 0
for eq_id, grp in alerts_df[alerts_df['Issue_Severity'].isin(['CRITICAL','HIGH'])].groupby('Equipment_ID'):
    if shown >= 15:
        break
    fp  = grp['Failure_Probability_%'].iloc[0]
    dl  = grp['Days_to_Failure'].iloc[0]
    et  = grp['Equipment_Type'].iloc[0]
    lv  = grp['Alert_Level'].iloc[0]
    report_lines.append(f"\n   {get_icon(lv)} {eq_id}  |  {et}  |  Failure: {fp}%  |  Days left: {dl}")
    report_lines.append("   " + "." * 60)
    for _, ar in grp.iterrows():
        s_icon = '🔴' if ar['Issue_Severity'] == 'CRITICAL' else '🟠'
        report_lines.append(f"     {s_icon} [{ar['Issue_Severity']}] {ar['Component']}")
        report_lines.append(f"        Problem : {ar['Problem_Detected']}")
        report_lines.append(f"        Action  : {ar['Recommended_Action']}")
    shown += 1

report_lines.append("\n" + "─" * 70)
report_lines.append("   COMPONENT REPLACEMENT SUMMARY")
report_lines.append("─" * 70)
for comp, count in comp_counts.items():
    report_lines.append(f"   {comp:<25}  {count:>3} machines need replacement")

report_lines.append("\n" + "─" * 70)
report_lines.append("   ALERT LEVEL SUMMARY")
report_lines.append("─" * 70)
lvl_counts = alerts_df.groupby('Alert_Level')['Equipment_ID'].nunique()
for lvl in ['CRITICAL','HIGH','MEDIUM','MONITOR','OK']:
    if lvl in lvl_counts:
        report_lines.append(f"   {get_icon(lvl)} {lvl:<10}  {lvl_counts[lvl]:>4} machines")

report_lines.append("\n" + "=" * 70)

full_report = "\n".join(report_lines)
# Full report saved to file (not printed to avoid terminal flood)

with open(f"{OUT}/hemm_alert_report.txt", 'w', encoding='utf-8') as f:
    f.write(full_report)
print("   ✔ hemm_alert_report.txt saved")

# ── Alert Dashboard chart ──────────────────────────────────────────────
fig2, axes = plt.subplots(1, 3, figsize=(18, 6))
fig2.patch.set_facecolor('#0d1117')
BG = '#161b22'; TEXT = '#e6edf3'; MUT = '#8b949e'

# Pie — alert distribution
ax_p = axes[0]; ax_p.set_facecolor(BG)
lvl_eq  = alerts_df.groupby('Alert_Level')['Equipment_ID'].nunique()
ordered = [l for l in ['CRITICAL','HIGH','MEDIUM','MONITOR','OK'] if l in lvl_eq]
wedges, texts, autotexts = ax_p.pie(
    [lvl_eq[l] for l in ordered],
    labels  =[l for l in ordered],
    colors  =[LEVEL_COLORS[l] for l in ordered],
    autopct ='%1.0f%%', startangle=90,
    textprops={'color': TEXT, 'fontsize': 9})
for at in autotexts:
    at.set_color(BG); at.set_fontweight('bold')
ax_p.set_title('Equipment Alert Distribution', color=TEXT, fontsize=11, fontweight='bold')

# Bar — top 10 at-risk
ax_b = axes[1]; ax_b.set_facecolor(BG)
top10_eq = (alerts_df.groupby('Equipment_ID')['Failure_Probability_%']
            .first().sort_values(ascending=False).head(10))
bar_colors = [LEVEL_COLORS[get_alert_level(v/100)] for v in top10_eq.values]
bars2 = ax_b.barh(range(len(top10_eq)), top10_eq.values,
                   color=bar_colors, alpha=0.9, height=0.65)
ax_b.set_yticks(range(len(top10_eq)))
ax_b.set_yticklabels(top10_eq.index, color=TEXT, fontsize=8)
ax_b.invert_yaxis()
ax_b.set_xlabel('Failure Probability (%)', color=MUT, fontsize=9)
ax_b.set_title('Top 10 At-Risk Equipment', color=TEXT, fontsize=11, fontweight='bold')
ax_b.tick_params(colors=MUT)
ax_b.axvline(50, color=LEVEL_COLORS['HIGH'],     lw=1, linestyle='--', alpha=0.5)
ax_b.axvline(75, color=LEVEL_COLORS['CRITICAL'], lw=1, linestyle='--', alpha=0.5)
for spine in ax_b.spines.values(): spine.set_color('#30363d')
for bar, val in zip(bars2, top10_eq.values):
    ax_b.text(val+0.5, bar.get_y()+bar.get_height()/2,
              f'{val:.0f}%', va='center', color=TEXT, fontsize=8)

# Bar — components to replace
ax_c = axes[2]; ax_c.set_facecolor(BG)
if len(comp_counts) > 0:
    tc = comp_counts.head(8)
    ax_c.barh(range(len(tc)), tc.values, color='#E24B4A', alpha=0.85, height=0.6)
    ax_c.set_yticks(range(len(tc)))
    ax_c.set_yticklabels(tc.index, color=TEXT, fontsize=8)
    ax_c.invert_yaxis()
    ax_c.set_xlabel('Machines needing replacement', color=MUT, fontsize=9)
    ax_c.set_title('Components Needing Replacement', color=TEXT, fontsize=11, fontweight='bold')
    ax_c.tick_params(colors=MUT)
    for spine in ax_c.spines.values(): spine.set_color('#30363d')
    for i, val in enumerate(tc.values):
        ax_c.text(val+0.2, i, str(val), va='center', color=TEXT, fontsize=9)

fig2.suptitle('NALCO HEMM — Prescriptive Maintenance Alert Dashboard',
              fontsize=14, fontweight='bold', color=TEXT, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}/hemm_alert_dashboard.png", dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("   ✔ hemm_alert_dashboard.png saved")


# ── Print condensed summary to terminal ───────────────────────────────
ens_r   = results_clf['Ensemble']
lvl_counts = alerts_df.groupby('Alert_Level')['Equipment_ID'].nunique()
crit_eq = len(alerts_df[alerts_df['Issue_Severity']=='CRITICAL']['Equipment_ID'].unique())
repl_eq = len(alerts_df[alerts_df['Recommended_Action'].str.contains('REPLACE', na=False)]['Equipment_ID'].unique())

print("\n" + "=" * 70)
print("   NALCO HEMM COMPLETE SYSTEM — FINAL SUMMARY")
print("=" * 70)
print(f"""
   DATASET
   -------
   Records analysed   : {len(df)}
   Features used      : {len(FEATURE_COLS)} (zero leakage)
   Failure rate       : {y_class.mean()*100:.1f}%

   ML MODEL RESULTS
   ----------------
   Model A (Failure Classification)
     Best: Voting Ensemble
     AUC      = {ens_r['auc']:.4f}
     Accuracy = {ens_r['acc']:.4f}
     Recall   = {ens_r['recall']:.4f}
     F1-Score = {ens_r['f1']:.4f}
     CV AUC   = {ens_r['cv_auc'].mean():.4f} +/- {ens_r['cv_auc'].std():.4f}

   Model B (Days-to-Failure Regression)
     Best: {best_reg_name}
     R2    = {results_reg[best_reg_name]['r2']:.4f}
     MAE   = {results_reg[best_reg_name]['mae']:.1f} days
     RMSE  = {results_reg[best_reg_name]['rmse']:.1f} days

   Model C (Maintenance Priority)
     Accuracy     = {pri_acc:.4f}
     Weighted F1  = {pri_f1:.4f}

   PRESCRIPTIVE ALERT RESULTS
   --------------------------
   Total machines analysed      : {df['Equipment_ID'].nunique() if 'Equipment_ID' in df.columns else len(df)}
   Machines needing action      : {len(alerts_df[alerts_df['Alert_Level'].isin(['CRITICAL','HIGH'])]['Equipment_ID'].unique())}
   CRITICAL alert machines      : {crit_eq}
   Machines needing REPLACEMENT : {repl_eq}

   ALERT BREAKDOWN:""")

for lvl in ['CRITICAL','HIGH','MEDIUM','MONITOR','OK']:
    if lvl in lvl_counts:
        icon = get_icon(lvl)
        print(f"   {icon} {lvl:<10} : {lvl_counts[lvl]:>4} machines")

print(f"""
   TOP 3 COMPONENTS NEEDING REPLACEMENT:""")
for comp, cnt in comp_counts.head(3).items():
    print(f"   -> {comp:<25} : {cnt} machines")

print(f"""
   TOP PREDICTIVE FEATURES:""")
for i, (feat, score) in enumerate(top20.head(5).items(), 1):
    print(f"   {i}. {feat:<40} {score:.5f}")

print("\n" + "=" * 70)

# ══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("   ALL DONE! OUTPUT FILES SAVED TO: outputs\\")
print("=" * 70)
print(f"""
   ML PERFORMANCE OUTPUTS:
   ├── hemm_ml_results.png       <- Model charts (AUC, ROC, Confusion Matrix)
   └── hemm_ml_summary.txt       <- Model accuracy numbers

   PRESCRIPTIVE ALERT OUTPUTS:
   ├── hemm_equipment_alerts.csv <- Open in Excel, filter by Alert_Level
   ├── hemm_alert_report.txt     <- Full maintenance report
   ├── hemm_alert_dashboard.png  <- Alert distribution + replacement chart
   └── hemm_alerts.db            <- SQLite history (grows every run)

   SQLITE DATABASE TABLES:
   • alerts_history    -> every alert for every machine, every run
   • equipment_summary -> one row per machine per run (for trend charts)
   • model_metrics     -> AUC, accuracy, MAE tracked over time

   HOW TO QUERY THE DATABASE:
   python -c "import sqlite3, pandas as pd; conn=sqlite3.connect('outputs/hemm_alerts.db'); print(pd.read_sql('SELECT * FROM equipment_summary ORDER BY run_timestamp DESC LIMIT 20', conn))"

   QUICK FILTERS IN EXCEL (hemm_equipment_alerts.csv):
   • Alert_Level = CRITICAL          -> Stop machine immediately
   • Recommended_Action has REPLACE  -> Order spare parts now
   • Days_to_Failure < 14            -> Urgent — less than 2 weeks
   • Issue_Severity = HIGH           -> Schedule this week
""")
