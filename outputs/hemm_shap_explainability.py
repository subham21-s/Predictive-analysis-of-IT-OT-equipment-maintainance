"""
NALCO HEMM — SHAP Explainability Module
========================================
Add this file to your project folder and run it AFTER hemm_ml_pipeline.py

What SHAP does:
  - Explains WHY the model predicted failure for each equipment
  - Shows which sensors contributed most to each prediction
  - Generates summary plots, force plots, and waterfall charts
  - Produces a text report with plain-English explanations

How to run:
  1. Run hemm_ml_pipeline.py first (trains and saves models)
  2. Then run: python hemm_shap_explainability.py

Requirements:
  pip install shap
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Try importing SHAP ──────────────────────────────────────────
try:
    import shap
    print("✔ SHAP imported successfully")
except ImportError:
    print("❌ SHAP not installed. Run: pip install shap")
    print("   Then run this script again.")
    exit(1)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

# ── Paths ───────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
OUT       = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT, exist_ok=True)

# ── Auto-find CSV ───────────────────────────────────────────────
CANDIDATE_NAMES = [
    "HEMM_Dataset_FINAL.csv",
    "1778086522685_HEMM_Dataset_FINAL.csv",
    "HEMM_Dataset_ENHANCED.csv",
    "HEMM_Dataset_NALCO.csv",
]
DATA_PATH = None
for name in CANDIDATE_NAMES:
    candidate = os.path.join(DATA_DIR, name)
    if os.path.exists(candidate):
        DATA_PATH = candidate
        break
if DATA_PATH is None:
    csvs = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if csvs:
        DATA_PATH = os.path.join(DATA_DIR, csvs[0])
if DATA_PATH is None:
    raise FileNotFoundError(f"No CSV found in {DATA_DIR}")

print("=" * 65)
print("  NALCO HEMM — SHAP EXPLAINABILITY ANALYSIS")
print("=" * 65)
print(f"\n  Dataset : {os.path.basename(DATA_PATH)}")
print(f"  Outputs : {OUT}\n")

# ───────────────────────────────────────────────────────────────
# 1. LOAD DATA (same as main pipeline)
# ───────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

SENSOR_COLS = [
    'Engine_Temp_C', 'Oil_Pressure_bar', 'Vibration_mms',
    'Fuel_Consumption_Lhr', 'Tyre_Pressure_PSI', 'Coolant_Level',
    'Battery_Voltage_V', 'Hydraulic_Pressure_bar', 'Exhaust_Temp_C', 'RPM',
    'Engine_Temp_C_Status_enc', 'Oil_Pressure_bar_Status_enc',
    'Vibration_mms_Status_enc', 'Fuel_Consumption_Lhr_Status_enc',
    'Tyre_Pressure_PSI_Status_enc', 'Coolant_Level_Status_enc',
    'Battery_Voltage_V_Status_enc', 'Hydraulic_Pressure_bar_Status_enc',
    'Exhaust_Temp_C_Status_enc', 'RPM_Status_enc',
    'Temp_Oil_Ratio', 'Vib_RPM_Ratio',
    'Operating_Hours', 'Load_Cycles_per_Day',
    'Operator_Experience_Yr', 'Equipment_Type_enc',
    'Shift_enc', 'Road_Condition_enc',
    'Year', 'Month', 'Day_of_Week',
    'Engine_Condition_enc', 'Tyre_Condition_enc',
    'Hydraulic_Condition_enc', 'Brake_Condition_enc',
    'Electrical_Condition_enc', 'Fuel_System_Condition_enc',
    'Transmission_Condition_enc', 'Cooling_System_Condition_enc',
    'Engine_Life_Remaining_pct', 'Tyre_Life_Remaining_pct',
    'Hydraulic_Life_Remaining_pct', 'Battery_Life_Remaining_pct',
    'Brake_Life_Remaining_pct',
    'Days_Since_Last_Maintenance', 'Last_Maintenance_Type_enc',
    'PM_Interval_Hours', 'Next_PM_Due_Hours', 'Maintenance_Team_enc',
]
REPLACE_FLAGS = [c for c in df.columns if 'Replacement_Needed_enc' in c]
SENSOR_COLS  += REPLACE_FLAGS
FEATURE_COLS  = [f for f in SENSOR_COLS if f in df.columns]

X       = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
y       = df['Failure']

# ───────────────────────────────────────────────────────────────
# 2. TRAIN MODEL (RandomForest — SHAP works best with trees)
# ───────────────────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

cw = compute_class_weight('balanced', classes=np.array([0,1]), y=y_tr)
model = RandomForestClassifier(
    n_estimators=300, max_depth=15, min_samples_split=4,
    min_samples_leaf=2, class_weight={0: cw[0], 1: cw[1]},
    random_state=42, n_jobs=-1)
model.fit(X_tr, y_tr)
print(f"✔ Model trained on {len(X_tr)} samples")

# ───────────────────────────────────────────────────────────────
# 3. SHAP — COMPUTE VALUES
# ───────────────────────────────────────────────────────────────
print("\n  Computing SHAP values (this takes ~30 seconds)...")

# TreeExplainer is the fastest for RandomForest
explainer   = shap.TreeExplainer(model)

# Use test set for SHAP (unseen data = honest explanation)
# Use a sample of 200 rows for speed — enough for good plots
SHAP_SAMPLE = min(200, len(X_te))
X_shap      = X_te.iloc[:SHAP_SAMPLE].reset_index(drop=True)
y_shap      = y_te.iloc[:SHAP_SAMPLE].reset_index(drop=True)

shap_values = explainer.shap_values(X_shap)

# shap_values is a list: [class_0_values, class_1_values]
# We care about class 1 (Failure = True)
sv_failure  = shap_values[1]   # shape: (SHAP_SAMPLE, n_features)

print(f"  SHAP values computed for {SHAP_SAMPLE} test samples")
print(f"  Shape: {sv_failure.shape}  (samples × features)")

# ───────────────────────────────────────────────────────────────
# 4. PLOT 1 — SHAP Summary Bar (Global Feature Importance)
# ───────────────────────────────────────────────────────────────
print("\n  Generating plots...")

plt.figure(figsize=(10, 8))
shap.summary_plot(
    sv_failure,
    X_shap,
    plot_type="bar",
    max_display=20,
    show=False,
    color='#38bdf8'
)
plt.title("SHAP — Global Feature Importance (Top 20)\nNALCO HEMM Failure Prediction",
          fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f"{OUT}/shap_summary_bar.png", dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print("  ✔ shap_summary_bar.png saved")

# ───────────────────────────────────────────────────────────────
# 5. PLOT 2 — SHAP Summary Dot (Feature Impact Direction)
# ───────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 9))
shap.summary_plot(
    sv_failure,
    X_shap,
    max_display=20,
    show=False
)
plt.title("SHAP — Feature Impact on Failure Prediction\n"
          "Red = high sensor value increases failure risk | Blue = low value",
          fontsize=11, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f"{OUT}/shap_summary_dot.png", dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print("  ✔ shap_summary_dot.png saved")

# ───────────────────────────────────────────────────────────────
# 6. PLOT 3 — SHAP Waterfall (Single Equipment Explanation)
# ───────────────────────────────────────────────────────────────
# Find the highest-risk equipment in the test set
proba       = model.predict_proba(X_shap)[:, 1]
high_risk_i = int(np.argmax(proba))   # index of most likely failure
low_risk_i  = int(np.argmin(proba))   # index of safest equipment

for label, idx in [("HIGH_RISK", high_risk_i), ("LOW_RISK", low_risk_i)]:
    plt.figure(figsize=(10, 7))
    # Waterfall plot uses Explanation object
    exp = shap.Explanation(
        values         = sv_failure[idx],
        base_values    = explainer.expected_value[1],
        data           = X_shap.iloc[idx].values,
        feature_names  = FEATURE_COLS
    )
    shap.waterfall_plot(exp, max_display=15, show=False)
    risk_pct = proba[idx] * 100
    plt.title(f"SHAP Waterfall — {label} Equipment\n"
              f"Failure Probability: {risk_pct:.1f}%  |  "
              f"Equipment Type enc={X_shap.iloc[idx].get('Equipment_Type_enc', 'N/A')}",
              fontsize=11, fontweight='bold', pad=12)
    plt.tight_layout()
    fname = f"{OUT}/shap_waterfall_{label.lower()}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✔ shap_waterfall_{label.lower()}.png saved")

# ───────────────────────────────────────────────────────────────
# 7. PLOT 4 — SHAP Dependence Plot (Top 2 features)
# ───────────────────────────────────────────────────────────────
# Find top 2 most important features by mean |SHAP|
mean_abs_shap = np.abs(sv_failure).mean(axis=0)
top2_idx      = np.argsort(mean_abs_shap)[::-1][:2]
top2_features = [FEATURE_COLS[i] for i in top2_idx]

for feat in top2_features:
    plt.figure(figsize=(8, 5))
    shap.dependence_plot(
        feat,
        sv_failure,
        X_shap,
        show=False,
        alpha=0.6
    )
    plt.title(f"SHAP Dependence — {feat}\n"
              f"How {feat} value affects failure prediction",
              fontsize=11, fontweight='bold')
    plt.tight_layout()
    safe_name = feat.replace(' ', '_').replace('/', '_')
    plt.savefig(f"{OUT}/shap_dependence_{safe_name}.png", dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✔ shap_dependence_{safe_name}.png saved")

# ───────────────────────────────────────────────────────────────
# 8. TEXT REPORT — Plain English Explanation
# ───────────────────────────────────────────────────────────────
mean_shap_per_feature = pd.Series(
    mean_abs_shap, index=FEATURE_COLS
).sort_values(ascending=False)

top10 = mean_shap_per_feature.head(10)

# For high-risk equipment: which features pushed it toward failure?
hri_shap   = sv_failure[high_risk_i]
hri_data   = X_shap.iloc[high_risk_i]
hri_series = pd.Series(hri_shap, index=FEATURE_COLS).sort_values(ascending=False)
top5_push  = hri_series.head(5)    # features pushing TOWARD failure
top5_pull  = hri_series.tail(5)    # features pushing AWAY from failure

report = f"""
╔══════════════════════════════════════════════════════════════════╗
║          NALCO HEMM — SHAP EXPLAINABILITY REPORT                ║
╚══════════════════════════════════════════════════════════════════╝

SHAP = SHapley Additive exPlanations
Each value tells you HOW MUCH a feature contributed to the prediction.
Positive SHAP = pushes toward FAILURE
Negative SHAP = pushes toward SAFE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GLOBAL — TOP 10 MOST IMPORTANT FEATURES (across all equipment)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Rank  Feature                                  Mean |SHAP|
  ────  ───────────────────────────────────────  ──────────
"""
for i, (feat, val) in enumerate(top10.items(), 1):
    report += f"  {i:3d}.  {feat:<40s}  {val:.5f}\n"

report += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HIGH-RISK EQUIPMENT EXPLANATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Failure Probability : {proba[high_risk_i]*100:.1f}%
  Base Rate           : {explainer.expected_value[1]*100:.1f}%  (average failure rate)

  Factors PUSHING toward failure (+SHAP):
"""
for feat, val in top5_push.items():
    actual = hri_data[feat]
    report += f"    + {feat:<40s}  SHAP={val:+.4f}  Value={actual:.3f}\n"

report += "\n  Factors PROTECTING against failure (-SHAP):\n"
for feat, val in top5_pull.items():
    actual = hri_data[feat]
    report += f"    - {feat:<40s}  SHAP={val:+.4f}  Value={actual:.3f}\n"

report += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO READ THE CHARTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  shap_summary_bar.png
    Bar chart of average |SHAP| value per feature.
    Longer bar = more important for predicting failure.

  shap_summary_dot.png
    Dot plot — each dot is one equipment reading.
    RED dots  = high sensor value   BLUE = low sensor value
    X-axis    = SHAP value (positive = increases failure risk)
    Pattern   : If red dots are on the RIGHT, high values = dangerous.
                If blue dots are on the RIGHT, low values = dangerous.

  shap_waterfall_high_risk.png
    Step-by-step breakdown for the MOST at-risk equipment.
    Shows exactly which sensors triggered the failure alert.

  shap_waterfall_low_risk.png
    Same breakdown for the SAFEST equipment.
    Good comparison — what does a healthy machine look like?

  shap_dependence_*.png
    How a single sensor's value relates to failure risk.
    Upward trend = higher sensor value = more dangerous.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INDUSTRY USE CASE AT NALCO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Without SHAP: "Machine HEMM-D004 will fail in 12 days."
  With SHAP   : "Machine HEMM-D004 will fail in 12 days BECAUSE
                 Vibration is 3.2x above normal AND
                 Brake Life is below 15% AND
                 Engine Temperature trend is rising."

  This is what separates a student project from a production system.
  Maintenance engineers can act on SHAP — they know WHAT to fix.
"""

print(report)

with open(f"{OUT}/shap_report.txt", 'w', encoding='utf-8') as f:
    f.write(report)
print(f"\n  ✔ shap_report.txt saved")

# ───────────────────────────────────────────────────────────────
# 9. BONUS — per-equipment SHAP score (top 5 at-risk machines)
# ───────────────────────────────────────────────────────────────
print("\n" + "─"*55)
print("  TOP 5 HIGHEST-RISK EQUIPMENT IN TEST SET")
print("─"*55)

risk_df = pd.DataFrame({
    'Failure_Probability_%': (proba * 100).round(1),
    'Equipment_Type_enc':     X_shap['Equipment_Type_enc'].values
    if 'Equipment_Type_enc' in X_shap.columns else ['N/A']*SHAP_SAMPLE,
    'Top_Risk_Factor':        [
        FEATURE_COLS[int(np.argmax(sv_failure[i]))]
        for i in range(SHAP_SAMPLE)
    ]
})

top5_risk = risk_df.nlargest(5, 'Failure_Probability_%')
print(top5_risk.to_string(index=True))
top5_risk.to_csv(f"{OUT}/shap_top5_risk_equipment.csv", index=True)
print(f"\n  ✔ shap_top5_risk_equipment.csv saved")

print("\n" + "="*65)
print("  SHAP ANALYSIS COMPLETE")
print("="*65)
print(f"""
  Files saved to: {OUT}
  ├── shap_summary_bar.png          (global feature importance)
  ├── shap_summary_dot.png          (feature impact direction)
  ├── shap_waterfall_high_risk.png  (why THIS machine will fail)
  ├── shap_waterfall_low_risk.png   (why this machine is safe)
  ├── shap_dependence_*.png         (sensor vs risk plots)
  ├── shap_report.txt               (plain-English explanations)
  └── shap_top5_risk_equipment.csv  (most at-risk machines)
""")
