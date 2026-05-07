"""
============================================================
NALCO INTERNSHIP PROJECT
Predictive Analysis of IT/OT Equipment Maintenance
Complete ML Model - Random Forest (Best Model)
============================================================
"""

# ─────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
os.makedirs("models",  exist_ok=True)
os.makedirs("graphs",  exist_ok=True)
os.makedirs("reports", exist_ok=True)

print("=" * 65)
print("   NALCO PREDICTIVE MAINTENANCE — COMPLETE ML MODEL")
print("=" * 65)


# ─────────────────────────────────────────────
# 2. LOAD DATASET
# ─────────────────────────────────────────────
print("\n[1/9] Loading Dataset ...")

df = pd.read_csv("HEMM_Dataset_ENHANCED.csv")

# Fix date
df["Date"] = pd.to_datetime(df["Date"])
df["Year"]        = df["Date"].dt.year
df["Month"]       = df["Date"].dt.month
df["Day_of_Week"] = df["Date"].dt.dayofweek

# Fill missing text columns
df["Failure_Type"]      = df["Failure_Type"].fillna("None")
df["Failure_Component"] = df["Failure_Component"].fillna("None")

print(f"   Rows    : {df.shape[0]}")
print(f"   Columns : {df.shape[1]}")
print(f"   Failure Rate : {df['Failure'].mean()*100:.2f}%")


# ─────────────────────────────────────────────
# 3. ENCODE CATEGORICAL COLUMNS
# ─────────────────────────────────────────────
print("\n[2/9] Encoding Categorical Columns ...")

# Sensor status  →  Normal=0 | Above Normal=1 | Below Normal=-1
status_map = {"Normal": 0, "Above Normal": 1, "Below Normal": -1}
status_cols = [
    "Engine_Temp_C_Status", "Oil_Pressure_bar_Status",
    "Vibration_mms_Status", "Fuel_Consumption_Lhr_Status",
    "Tyre_Pressure_PSI_Status", "Coolant_Level_Status",
    "Battery_Voltage_V_Status", "Hydraulic_Pressure_bar_Status",
    "Exhaust_Temp_C_Status", "RPM_Status",
]
for col in status_cols:
    df[col + "_enc"] = df[col].map(status_map)

# Component condition  →  Good=0 | Warning=1 | Critical=2
cond_map = {"Good": 0, "Warning": 1, "Critical": 2}
cond_cols = [
    "Engine_Condition", "Tyre_Condition", "Hydraulic_Condition",
    "Brake_Condition", "Electrical_Condition", "Fuel_System_Condition",
    "Transmission_Condition", "Cooling_System_Condition",
]
for col in cond_cols:
    df[col + "_enc"] = df[col].map(cond_map)

# Replacement needed  →  No=0 | Monitor=1 | Yes=2
repl_map = {"No": 0, "Monitor": 1, "Yes": 2}
repl_cols = [
    "Engine_Replacement_Needed", "Tyre_Replacement_Needed",
    "Hydraulic_Replacement_Needed", "Brake_Replacement_Needed",
    "Electrical_Replacement_Needed",
]
for col in repl_cols:
    df[col + "_enc"] = df[col].map(repl_map)

print("   Done.")


# ─────────────────────────────────────────────
# 4. FEATURE SELECTION
# ─────────────────────────────────────────────
print("\n[3/9] Selecting Features ...")

FEATURES = [
    # Equipment
    "Equipment_Type_enc", "Shift_enc", "Road_Condition_enc",
    "Operator_Experience_Yr",

    # Sensor readings
    "Engine_Temp_C", "Oil_Pressure_bar", "Vibration_mms",
    "Fuel_Consumption_Lhr", "Tyre_Pressure_PSI", "Coolant_Level",
    "Battery_Voltage_V", "Hydraulic_Pressure_bar",
    "Exhaust_Temp_C", "RPM",

    # Sensor status (encoded)
    "Engine_Temp_C_Status_enc", "Oil_Pressure_bar_Status_enc",
    "Vibration_mms_Status_enc", "Coolant_Level_Status_enc",
    "Battery_Voltage_V_Status_enc", "Hydraulic_Pressure_bar_Status_enc",

    # Operational
    "Operating_Hours", "Load_Cycles_per_Day",
    "Temp_Oil_Ratio", "Vib_RPM_Ratio",
    "Health_Score", "High_Risk",
    "Year", "Month", "Day_of_Week",

    # Maintenance
    "PM_Interval_Hours", "Next_PM_Due_Hours",
    "Maintenance_Cost_INR", "Estimated_Downtime_Hrs",
    "Days_Since_Last_Maintenance",

    # Component conditions (encoded)
    "Engine_Condition_enc", "Tyre_Condition_enc",
    "Hydraulic_Condition_enc", "Brake_Condition_enc",
    "Electrical_Condition_enc", "Fuel_System_Condition_enc",
    "Transmission_Condition_enc", "Cooling_System_Condition_enc",

    # Replacement needed (encoded)
    "Engine_Replacement_Needed_enc", "Tyre_Replacement_Needed_enc",
    "Hydraulic_Replacement_Needed_enc", "Brake_Replacement_Needed_enc",
    "Electrical_Replacement_Needed_enc",

    # Life remaining
    "Engine_Life_Remaining_pct", "Tyre_Life_Remaining_pct",
    "Hydraulic_Life_Remaining_pct", "Battery_Life_Remaining_pct",
    "Brake_Life_Remaining_pct",
]

TARGET = "Failure"

X = df[FEATURES].fillna(0)
y = df[TARGET]

print(f"   Features : {len(FEATURES)}")
print(f"   Samples  : {len(X)}")


# ─────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
print("\n[4/9] Splitting Data (80% train / 20% test) ...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"   Train : {X_train.shape[0]} samples  |  Failure rate: {y_train.mean()*100:.1f}%")
print(f"   Test  : {X_test.shape[0]} samples  |  Failure rate: {y_test.mean()*100:.1f}%")


# ─────────────────────────────────────────────
# 6. TRAIN RANDOM FOREST (with GridSearchCV)
# ─────────────────────────────────────────────
print("\n[5/9] Training Random Forest with Hyperparameter Tuning ...")
print("   (This may take 1-2 minutes ...)\n")

param_grid = {
    "n_estimators"    : [100, 200],
    "max_depth"       : [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf" : [1, 2],
    "class_weight"    : ["balanced"],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1,
)
grid_search.fit(X_train, y_train)

best_model  = grid_search.best_estimator_
best_params = grid_search.best_params_

print(f"\n   Best Parameters : {best_params}")


# ─────────────────────────────────────────────
# 7. EVALUATE MODEL
# ─────────────────────────────────────────────
print("\n[6/9] Evaluating Model ...")

y_pred       = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_pred_proba)

# Cross-validation score
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring="f1")

print(f"\n   ┌─────────────────────────────────┐")
print(f"   │  Accuracy  : {acc*100:6.2f}%             │")
print(f"   │  Precision : {prec*100:6.2f}%             │")
print(f"   │  Recall    : {rec*100:6.2f}%             │")
print(f"   │  F1-Score  : {f1*100:6.2f}%             │")
print(f"   │  ROC-AUC   : {auc*100:6.2f}%             │")
print(f"   │  CV F1     : {cv_scores.mean()*100:6.2f}% ± {cv_scores.std()*100:.2f}%  │")
print(f"   └─────────────────────────────────┘")

print("\n   Classification Report:")
print("   " + "-"*55)
print(classification_report(y_test, y_pred,
      target_names=["No Failure", "Failure"],
      digits=4))


# ─────────────────────────────────────────────
# 8. VISUALIZATIONS
# ─────────────────────────────────────────────
print("[7/9] Generating Visualizations ...")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("NALCO Predictive Maintenance — Random Forest Results",
             fontsize=15, fontweight="bold", y=1.01)

# ── 8a. Confusion Matrix ──────────────────────
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0],
            xticklabels=["No Failure", "Failure"],
            yticklabels=["No Failure", "Failure"],
            annot_kws={"size": 14})
axes[0, 0].set_title("Confusion Matrix", fontsize=13, fontweight="bold")
axes[0, 0].set_ylabel("Actual",    fontsize=11)
axes[0, 0].set_xlabel("Predicted", fontsize=11)

# ── 8b. ROC Curve ────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0, 1].plot(fpr, tpr, color="darkorange", lw=2,
                label=f"AUC = {auc:.4f}")
axes[0, 1].plot([0, 1], [0, 1], "navy", lw=1.5, linestyle="--",
                label="Random")
axes[0, 1].set_xlim([0, 1]); axes[0, 1].set_ylim([0, 1.05])
axes[0, 1].set_xlabel("False Positive Rate", fontsize=11)
axes[0, 1].set_ylabel("True Positive Rate",  fontsize=11)
axes[0, 1].set_title("ROC Curve", fontsize=13, fontweight="bold")
axes[0, 1].legend(loc="lower right")
axes[0, 1].grid(alpha=0.3)

# ── 8c. Feature Importance (Top 15) ──────────
feat_imp = pd.Series(best_model.feature_importances_,
                     index=FEATURES).sort_values(ascending=False).head(15)
feat_imp[::-1].plot(kind="barh", ax=axes[1, 0], color="steelblue")
axes[1, 0].set_title("Top 15 Feature Importances",
                      fontsize=13, fontweight="bold")
axes[1, 0].set_xlabel("Importance", fontsize=11)

# ── 8d. Metrics Bar Chart ────────────────────
metrics = {"Accuracy": acc, "Precision": prec,
           "Recall": rec, "F1-Score": f1, "ROC-AUC": auc}
colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
bars = axes[1, 1].bar(metrics.keys(), metrics.values(), color=colors)
axes[1, 1].set_ylim(0, 1.15)
axes[1, 1].set_title("Model Performance Metrics",
                      fontsize=13, fontweight="bold")
axes[1, 1].set_ylabel("Score", fontsize=11)
for bar, val in zip(bars, metrics.values()):
    axes[1, 1].text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val*100:.1f}%", ha="center",
                    fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("graphs/model_results.png", dpi=300, bbox_inches="tight")
plt.show()
print("   Saved → graphs/model_results.png")


# ─────────────────────────────────────────────
# 9. SAVE MODEL & ARTIFACTS
# ─────────────────────────────────────────────
print("\n[8/9] Saving Model and Artifacts ...")

joblib.dump(best_model, "models/random_forest_model.pkl")
joblib.dump(FEATURES,   "models/feature_names.pkl")

report_text = (
    "NALCO Predictive Maintenance — Model Report\n"
    "=" * 55 + "\n"
    f"Model        : Random Forest (GridSearchCV tuned)\n"
    f"Best Params  : {best_params}\n\n"
    f"Accuracy     : {acc*100:.2f}%\n"
    f"Precision    : {prec*100:.2f}%\n"
    f"Recall       : {rec*100:.2f}%\n"
    f"F1-Score     : {f1*100:.2f}%\n"
    f"ROC-AUC      : {auc*100:.2f}%\n"
    f"CV F1 (5-fold): {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%\n\n"
    "Confusion Matrix:\n"
    f"  True Negatives  : {cm[0,0]}\n"
    f"  False Positives : {cm[0,1]}\n"
    f"  False Negatives : {cm[1,0]}\n"
    f"  True Positives  : {cm[1,1]}\n\n"
    "Classification Report:\n"
    + classification_report(y_test, y_pred,
      target_names=["No Failure", "Failure"], digits=4)
)
with open("reports/model_report.txt", "w") as f:
    f.write(report_text)

print("   Saved → models/random_forest_model.pkl")
print("   Saved → models/feature_names.pkl")
print("   Saved → reports/model_report.txt")


# ─────────────────────────────────────────────
# 10. SAMPLE PREDICTIONS
# ─────────────────────────────────────────────
print("\n[9/9] Sample Predictions on Test Data ...")
print("-" * 65)

sample_idx   = X_test.head(8).index
sample_X     = X_test.loc[sample_idx]
sample_proba = best_model.predict_proba(sample_X)[:, 1]
sample_pred  = best_model.predict(sample_X)
sample_actual = y_test.loc[sample_idx].values

for i, (pred, prob, actual) in enumerate(
        zip(sample_pred, sample_proba, sample_actual)):

    equip_id   = df.loc[sample_idx[i], "Equipment_ID"]
    equip_type = df.loc[sample_idx[i], "Equipment_Type"]
    health     = df.loc[sample_idx[i], "Health_Score"]

    risk = "🔴 HIGH"   if prob >= 0.70 else \
           "🟡 MEDIUM" if prob >= 0.30 else "🟢 LOW"

    status = "⚠️  FAILURE"    if pred == 1 else "✓  NO FAILURE"
    match  = "✓ CORRECT" if pred == actual else "✗ WRONG"

    print(f"\n  Equipment : {equip_id} ({equip_type})")
    print(f"  Health    : {health:.1f}  |  Prediction: {status}  |  {match}")
    print(f"  Probability: {prob*100:.1f}%  |  Risk: {risk}")

    if prob >= 0.70:
        print("  → ACTION: Schedule IMMEDIATE maintenance!")
    elif prob >= 0.30:
        print("  → ACTION: Plan preventive maintenance soon.")
    else:
        print("  → ACTION: Continue normal operation.")

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("   TRAINING COMPLETE!")
print("=" * 65)
print(f"   Best Model  : Random Forest (Tuned)")
print(f"   Accuracy    : {acc*100:.2f}%")
print(f"   F1-Score    : {f1*100:.2f}%")
print(f"   ROC-AUC     : {auc*100:.2f}%")
print(f"   CV F1       : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
print()
print("   Saved Files:")
print("   ├── models/random_forest_model.pkl")
print("   ├── models/feature_names.pkl")
print("   ├── graphs/model_results.png")
print("   └── reports/model_report.txt")
print("=" * 65)
print("   Ready for deployment! 🚀")
print("=" * 65)
