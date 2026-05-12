"""
NALCO HEMM — Live Predictive Maintenance Dashboard
====================================================
Auto-refreshes every 30 seconds when data changes.

HOW TO RUN:
    pip install streamlit watchdog
    streamlit run hemm_dashboard.py

Then open browser: http://localhost:8501
"""

import streamlit as st
import sqlite3
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               VotingClassifier, RandomForestRegressor)
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="NALCO HEMM Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

* { font-family: 'Inter', sans-serif; }
.stApp { background: #0a0e1a; color: #e2e8f0; }

.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2035 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-val  { font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 700; }
.metric-label{ font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: .08em; margin-top: 4px; }

.alert-critical { background:#2d1515; border-left:4px solid #ef4444; border-radius:8px; padding:14px 18px; margin:6px 0; }
.alert-high     { background:#2d1f0f; border-left:4px solid #f97316; border-radius:8px; padding:14px 18px; margin:6px 0; }
.alert-medium   { background:#1a2236; border-left:4px solid #3b82f6; border-radius:8px; padding:14px 18px; margin:6px 0; }
.alert-ok       { background:#0f2d1a; border-left:4px solid #22c55e; border-radius:8px; padding:14px 18px; margin:6px 0; }

.section-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: .15em;
    text-transform: uppercase;
    color: #3b82f6;
    margin: 24px 0 12px;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 6px;
}
.badge-critical { background:#ef4444; color:white; border-radius:4px; padding:2px 8px; font-size:11px; font-weight:600; }
.badge-high     { background:#f97316; color:white; border-radius:4px; padding:2px 8px; font-size:11px; font-weight:600; }
.badge-medium   { background:#3b82f6; color:white; border-radius:4px; padding:2px 8px; font-size:11px; font-weight:600; }
.badge-ok       { background:#22c55e; color:white; border-radius:4px; padding:2px 8px; font-size:11px; font-weight:600; }

div[data-testid="stSidebar"] { background: #060912; border-right: 1px solid #1e293b; }
h1,h2,h3 { color: #f1f5f9 !important; }
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT      = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT, exist_ok=True)

CANDIDATE_NAMES = [
    "HEMM_Dataset_FINAL.csv",
    "1778086522685_HEMM_Dataset_FINAL.csv",
    "HEMM_Dataset_ENHANCED.csv",
    "HEMM_Dataset_NALCO.csv",
]

def find_csv():
    for name in CANDIDATE_NAMES:
        c = os.path.join(DATA_DIR, name)
        if os.path.exists(c):
            return c
    csvs = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")] if os.path.exists(DATA_DIR) else []
    return os.path.join(DATA_DIR, csvs[0]) if csvs else None

def get_file_mtime(path):
    return os.path.getmtime(path) if path and os.path.exists(path) else 0

# ── Feature columns ───────────────────────────────────────────────────
SENSOR_FEATURES = [
    'Engine_Temp_C', 'Oil_Pressure_bar', 'Vibration_mms',
    'Fuel_Consumption_Lhr', 'Tyre_Pressure_PSI', 'Coolant_Level',
    'Battery_Voltage_V', 'Hydraulic_Pressure_bar', 'Exhaust_Temp_C', 'RPM',
    'Engine_Temp_C_Status_enc', 'Oil_Pressure_bar_Status_enc',
    'Vibration_mms_Status_enc', 'Fuel_Consumption_Lhr_Status_enc',
    'Tyre_Pressure_PSI_Status_enc', 'Coolant_Level_Status_enc',
    'Battery_Voltage_V_Status_enc', 'Hydraulic_Pressure_bar_Status_enc',
    'Exhaust_Temp_C_Status_enc', 'RPM_Status_enc',
    'Temp_Oil_Ratio', 'Vib_RPM_Ratio',
    'Operating_Hours', 'Load_Cycles_per_Day', 'Operator_Experience_Yr',
    'Equipment_Type_enc', 'Shift_enc', 'Road_Condition_enc',
    'Year', 'Month', 'Day_of_Week',
    'Engine_Condition_enc', 'Tyre_Condition_enc', 'Hydraulic_Condition_enc',
    'Brake_Condition_enc', 'Electrical_Condition_enc', 'Fuel_System_Condition_enc',
    'Transmission_Condition_enc', 'Cooling_System_Condition_enc',
    'Engine_Life_Remaining_pct', 'Tyre_Life_Remaining_pct',
    'Hydraulic_Life_Remaining_pct', 'Battery_Life_Remaining_pct',
    'Brake_Life_Remaining_pct',
    'Days_Since_Last_Maintenance', 'Last_Maintenance_Type_enc',
    'PM_Interval_Hours', 'Next_PM_Due_Hours', 'Maintenance_Team_enc',
]

EQ_TYPE_MAP = {0:'Excavator',1:'Dumper',2:'Dozer',
               3:'Grader',4:'Drill Rig',5:'Wheel Loader',6:'Scraper'}

def get_alert_level(p):
    if p>=0.75: return 'CRITICAL'
    elif p>=0.50: return 'HIGH'
    elif p>=0.30: return 'MEDIUM'
    elif p>=0.15: return 'MONITOR'
    else: return 'OK'

def diagnose_machine(row):
    issues = []
    if row.get('Engine_Life_Remaining_pct',100) < 15:
        issues.append(('Engine','Life < 15%','REPLACE ENGINE IMMEDIATELY','CRITICAL'))
    elif row.get('Engine_Life_Remaining_pct',100) < 30:
        issues.append(('Engine','Life < 30%','Schedule engine overhaul','HIGH'))
    if row.get('Engine_Temp_C_Status_enc',0) == 2:
        issues.append(('Engine','Temp ABOVE limit','Stop machine — check cooling NOW','CRITICAL'))
    if row.get('Engine_Replacement_Needed_enc',0) == 1:
        issues.append(('Engine','Replacement flag','REPLACE ENGINE','CRITICAL'))
    if row.get('Brake_Life_Remaining_pct',100) < 10:
        issues.append(('Brakes','Life < 10%','REPLACE BRAKE PADS IMMEDIATELY','CRITICAL'))
    elif row.get('Brake_Life_Remaining_pct',100) < 25:
        issues.append(('Brakes','Life < 25%','Schedule brake replacement','HIGH'))
    if row.get('Brake_Replacement_Needed_enc',0) == 1:
        issues.append(('Brakes','Replacement flag','REPLACE BRAKE SYSTEM','CRITICAL'))
    if row.get('Hydraulic_Life_Remaining_pct',100) < 15:
        issues.append(('Hydraulics','Life < 15%','REPLACE HYDRAULIC SYSTEM','CRITICAL'))
    elif row.get('Hydraulic_Life_Remaining_pct',100) < 30:
        issues.append(('Hydraulics','Life < 30%','Schedule hydraulic service','HIGH'))
    if row.get('Hydraulic_Pressure_bar_Status_enc',0) == 2:
        issues.append(('Hydraulics','Pressure ABOVE limit','Check pump and seals NOW','CRITICAL'))
    if row.get('Tyre_Life_Remaining_pct',100) < 15:
        issues.append(('Tyres','Life < 15%','REPLACE ALL TYRES IMMEDIATELY','CRITICAL'))
    elif row.get('Tyre_Life_Remaining_pct',100) < 30:
        issues.append(('Tyres','Life < 30%','Schedule tyre replacement','HIGH'))
    if row.get('Battery_Life_Remaining_pct',100) < 15:
        issues.append(('Battery','Life < 15%','REPLACE BATTERY IMMEDIATELY','CRITICAL'))
    elif row.get('Battery_Life_Remaining_pct',100) < 30:
        issues.append(('Battery','Life < 30%','Schedule battery replacement','HIGH'))
    if row.get('Vibration_mms_Status_enc',0) == 2:
        issues.append(('Drivetrain','Vibration ABOVE limit','Stop machine — inspect bearings','CRITICAL'))
    elif row.get('Vibration_mms_Status_enc',0) == 1:
        issues.append(('Drivetrain','Vibration above normal','Inspect bearings and mounts','HIGH'))
    if row.get('Oil_Pressure_bar_Status_enc',0) == 2:
        issues.append(('Engine Oil','Pressure ABOVE limit','Check oil pump immediately','CRITICAL'))
    if row.get('Days_Since_Last_Maintenance',0) > 45:
        issues.append(('Maintenance',f"{int(row.get('Days_Since_Last_Maintenance',0))} days since PM",
                       'SCHEDULE PM IMMEDIATELY','HIGH'))
    if not issues:
        issues.append(('All Systems','No anomalies','Continue regular monitoring','OK'))
    issues.sort(key=lambda x: {'CRITICAL':0,'HIGH':1,'MEDIUM':2,'MONITOR':3,'OK':4}.get(x[3],5))
    return issues

# ── Load & train (cached so it only reruns when CSV changes) ──────────
@st.cache_data(ttl=30)
def load_and_train(data_path, mtime):
    df = pd.read_csv(data_path)
    replace_flags = [c for c in df.columns if 'Replacement_Needed_enc' in c]
    feat_cols = [f for f in SENSOR_FEATURES + replace_flags if f in df.columns]

    X       = df[feat_cols].fillna(df[feat_cols].median())
    y_class = df['Failure']
    y_reg   = df['Days_to_Next_Failure']
    y_pri   = df['Maintenance_Priority_enc']

    X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te, yp_tr, yp_te = train_test_split(
        X, y_class, y_reg, y_pri, test_size=0.2, random_state=42, stratify=y_class)

    # Load saved models if .pkl exists (instant), else train fresh
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PKL_CLF  = os.path.join(BASE_DIR, "outputs", "model_clf.pkl")
    PKL_REG  = os.path.join(BASE_DIR, "outputs", "model_reg.pkl")

    if os.path.exists(PKL_CLF) and os.path.exists(PKL_REG):
        ens       = joblib.load(PKL_CLF)
        reg       = joblib.load(PKL_REG)
        rf_fitted = ens.estimators_[0]   # RF inside VotingClassifier
    else:
        cw  = compute_class_weight('balanced', classes=np.array([0,1]), y=yc_tr)
        rf  = RandomForestClassifier(n_estimators=200, max_depth=15,
                                      class_weight={0:cw[0],1:cw[1]}, random_state=42, n_jobs=-1)
        gb  = GradientBoostingClassifier(n_estimators=150, learning_rate=0.08,
                                          max_depth=5, random_state=42)
        ens = VotingClassifier(estimators=[('rf',rf),('gb',gb)], voting='soft', weights=[2,1])
        ens.fit(X_tr, yc_tr)
        reg = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
        reg.fit(X_tr, yr_tr)
        rf_fitted = ens.estimators_[0]   # access fitted RF from inside ensemble

    yp  = ens.predict(X_te)
    ypr = ens.predict_proba(X_te)[:,1]
    metrics = {
        'auc':      round(roc_auc_score(yc_te, ypr), 4),
        'accuracy': round(accuracy_score(yc_te, yp), 4),
        'recall':   round(recall_score(yc_te, yp), 4),
        'f1':       round(f1_score(yc_te, yp), 4),
    }

    # Feature importance — use fitted RF sub-estimator (fixes NotFittedError)
    fi = pd.Series(rf_fitted.feature_importances_, index=feat_cols).sort_values(ascending=False)

    # Run alert engine on all machines
    alert_proba = ens.predict_proba(X)[:,1]
    days_pred   = reg.predict(X)
    all_alerts  = []
    for i in range(len(df)):
        row     = X.iloc[i].to_dict()
        eq_id   = df['Equipment_ID'].iloc[i] if 'Equipment_ID' in df.columns else f"HEMM-{i:04d}"
        eq_type = EQ_TYPE_MAP.get(int(row.get('Equipment_Type_enc',0)), 'Unknown')
        prob    = alert_proba[i]
        days_l  = max(0, int(days_pred[i]))
        lvl     = get_alert_level(prob)
        for comp, problem, action, sev in diagnose_machine(row):
            all_alerts.append({
                'Equipment_ID':          eq_id,
                'Equipment_Type':        eq_type,
                'Failure_Probability_%': round(prob*100, 1),
                'Days_to_Failure':       days_l,
                'Alert_Level':           lvl,
                'Component':             comp,
                'Problem_Detected':      problem,
                'Recommended_Action':    action,
                'Issue_Severity':        sev,
            })

    alerts_df = pd.DataFrame(all_alerts)
    return df, X, feat_cols, metrics, fi, alerts_df, ens, reg, mtime

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ NALCO HEMM")
    st.markdown("**Predictive Maintenance**")
    st.markdown("---")

    data_path = find_csv()
    if data_path:
        st.success(f"✅ Dataset: `{os.path.basename(data_path)}`")
    else:
        st.error("❌ No CSV found in `data/` folder")
        st.stop()

    st.markdown("### Auto-refresh")
    auto_refresh = st.toggle("Live Auto-refresh (30s)", value=True)
    if auto_refresh:
        st.info("🔄 Dashboard refreshes every 30s when data changes")

    st.markdown("### Filters")
    alert_filter = st.multiselect(
        "Show Alert Levels",
        ['CRITICAL','HIGH','MEDIUM','MONITOR','OK'],
        default=['CRITICAL','HIGH']
    )
    eq_types = list(EQ_TYPE_MAP.values())
    eq_filter = st.multiselect("Equipment Type", eq_types, default=eq_types)

    st.markdown("### View")
    page = st.radio("", ["🏠 Overview", "🚨 Alerts", "📊 ML Models", "🔧 Equipment Detail", "💰 Cost Savings", "🔀 Shift Analysis", "👷 Operator Insights", "🎯 Predict Machine"])
    st.markdown("---")
    st.markdown(f"<small style='color:#475569'>Last checked: {time.strftime('%H:%M:%S')}</small>",
                unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────
mtime = get_file_mtime(data_path)
with st.spinner("🔄 Loading data and training models..."):
    df, X, feat_cols, metrics, fi, alerts_df, ens, reg, _ = load_and_train(data_path, mtime)

# Apply filters
filtered_alerts = alerts_df[
    alerts_df['Alert_Level'].isin(alert_filter) &
    alerts_df['Equipment_Type'].isin(eq_filter)
]

LEVEL_COLORS = {'CRITICAL':'#ef4444','HIGH':'#f97316','MEDIUM':'#3b82f6','MONITOR':'#22c55e','OK':'#64748b'}
ICONS        = {'CRITICAL':'🔴','HIGH':'🟠','MEDIUM':'🟡','MONITOR':'🔵','OK':'🟢'}

# ══════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("# NALCO HEMM — Predictive Maintenance Dashboard")
    st.markdown(f"<small style='color:#64748b'>Dataset: {os.path.basename(data_path)} &nbsp;|&nbsp; {len(df)} records &nbsp;|&nbsp; Updated: {time.strftime('%d %b %Y %H:%M')}</small>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Inline Filters ────────────────────────────────────────────────
    with st.expander("🔍 Filter Overview Data", expanded=False):
        ov_f1, ov_f2, ov_f3, ov_f4 = st.columns(4)
        with ov_f1:
            ov_eq_type = st.multiselect("Equipment Type", sorted(alerts_df['Equipment_Type'].unique()),
                                         default=sorted(alerts_df['Equipment_Type'].unique()), key="ov_eq")
        with ov_f2:
            ov_alert   = st.multiselect("Alert Level", ['CRITICAL','HIGH','MEDIUM','MONITOR','OK'],
                                         default=['CRITICAL','HIGH','MEDIUM','MONITOR','OK'], key="ov_al")
        with ov_f3:
            ov_prob_min, ov_prob_max = st.slider("Failure Probability (%)", 0, 100, (0, 100), key="ov_prob")
        with ov_f4:
            ov_days_max = st.slider("Max Days to Failure", 1, 180, 180, key="ov_days")

    ov_df = alerts_df[
        alerts_df['Equipment_Type'].isin(ov_eq_type) &
        alerts_df['Alert_Level'].isin(ov_alert) &
        (alerts_df['Failure_Probability_%'] >= ov_prob_min) &
        (alerts_df['Failure_Probability_%'] <= ov_prob_max) &
        (alerts_df['Days_to_Failure'] <= ov_days_max)
    ]
    st.caption(f"Showing {ov_df['Equipment_ID'].nunique()} of {alerts_df['Equipment_ID'].nunique()} machines after filters")
    st.markdown("---")

    # KPI Row — based on filtered data
    crit_count   = len(ov_df[ov_df['Issue_Severity']=='CRITICAL']['Equipment_ID'].unique())
    high_count   = len(ov_df[ov_df['Alert_Level']=='HIGH']['Equipment_ID'].unique())
    repl_count   = len(ov_df[ov_df['Recommended_Action'].str.contains('REPLACE',na=False)]['Equipment_ID'].unique())
    total_eq     = ov_df['Equipment_ID'].nunique()
    fail_rate    = round(df['Failure'].mean()*100, 1)
    avg_days     = round(df['Days_to_Next_Failure'].mean(), 1)

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    kpis = [
        (k1, str(crit_count),   "Critical Machines",    "#ef4444"),
        (k2, str(high_count),   "High Risk",            "#f97316"),
        (k3, str(repl_count),   "Need Replacement",     "#eab308"),
        (k4, f"{metrics['auc']}","Model AUC",           "#3b82f6"),
        (k5, f"{fail_rate}%",   "Failure Rate",         "#8b5cf6"),
        (k6, f"{avg_days}d",    "Avg Days to Failure",  "#22c55e"),
    ]
    for col, val, label, color in kpis:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-val' style='color:{color}'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])

    # Alert level distribution
    with col1:
        st.markdown("<div class='section-title'>Alert Level Distribution</div>", unsafe_allow_html=True)
        lvl_counts = ov_df.groupby('Alert_Level')['Equipment_ID'].nunique()
        ordered    = [l for l in ['CRITICAL','HIGH','MEDIUM','MONITOR','OK'] if l in lvl_counts]
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#111827')
        ax.set_facecolor('#111827')
        wedges, texts, autotexts = ax.pie(
            [lvl_counts[l] for l in ordered],
            labels=[f"{l}\n({lvl_counts[l]})" for l in ordered],
            colors=[LEVEL_COLORS[l] for l in ordered],
            autopct='%1.0f%%', startangle=90,
            textprops={'color':'#94a3b8','fontsize':8})
        for at in autotexts:
            at.set_color('#0a0e1a'); at.set_fontweight('bold')
        ax.set_title('Equipment Alert Status', color='#e2e8f0', fontsize=10, pad=10)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Equipment type failure rate
    with col2:
        st.markdown("<div class='section-title'>Failure Rate by Equipment Type</div>", unsafe_allow_html=True)
        if 'Equipment_Type' in df.columns:
            eq_fail = df.groupby('Equipment_Type')['Failure'].mean().sort_values(ascending=True) * 100
        else:
            eq_fail = alerts_df.groupby('Equipment_Type')['Failure_Probability_%'].mean().sort_values(ascending=True)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        fig2.patch.set_facecolor('#111827')
        ax2.set_facecolor('#111827')
        colors_eq = ['#ef4444' if v > 25 else '#f97316' if v > 15 else '#3b82f6' for v in eq_fail.values]
        ax2.barh(eq_fail.index, eq_fail.values, color=colors_eq, alpha=0.85, height=0.6)
        ax2.set_xlabel('Failure Rate (%)', color='#64748b', fontsize=9)
        ax2.tick_params(colors='#94a3b8', labelsize=8)
        for spine in ax2.spines.values(): spine.set_color('#1e293b')
        ax2.set_title('Failure Rate by Equipment Type', color='#e2e8f0', fontsize=10)
        for i, v in enumerate(eq_fail.values):
            ax2.text(v+0.3, i, f'{v:.1f}%', va='center', color='#e2e8f0', fontsize=8)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    # Component replacement summary
    st.markdown("<div class='section-title'>Components Needing Replacement</div>", unsafe_allow_html=True)
    repl_df   = alerts_df[alerts_df['Recommended_Action'].str.contains('REPLACE', na=False)]
    comp_cnt  = repl_df.groupby('Component')['Equipment_ID'].nunique().sort_values(ascending=False)
    fig3, ax3 = plt.subplots(figsize=(12, 3))
    fig3.patch.set_facecolor('#111827')
    ax3.set_facecolor('#111827')
    ax3.bar(comp_cnt.index, comp_cnt.values, color='#ef4444', alpha=0.85, width=0.6)
    ax3.tick_params(colors='#94a3b8', labelsize=9)
    ax3.set_ylabel('Machines', color='#64748b', fontsize=9)
    for spine in ax3.spines.values(): spine.set_color('#1e293b')
    for i, v in enumerate(comp_cnt.values):
        ax3.text(i, v+0.3, str(v), ha='center', color='#e2e8f0', fontsize=9, fontweight='bold')
    st.pyplot(fig3, use_container_width=True)
    plt.close()

    # ── Shift-wise failure snapshot (overview teaser) ──────────────────
    if 'Shift' in df.columns or 'Shift_enc' in df.columns:
        st.markdown("<div class='section-title'>Shift-wise Failure Overview</div>", unsafe_allow_html=True)

        SHIFT_MAP = {0: 'Day', 1: 'Afternoon', 2: 'Night'}
        shift_col = 'Shift' if 'Shift' in df.columns else 'Shift_enc'
        df_s = df.copy()
        if shift_col == 'Shift_enc':
            df_s['Shift_Name'] = df_s['Shift_enc'].map(SHIFT_MAP).fillna('Unknown')
        else:
            df_s['Shift_Name'] = df_s['Shift']

        shift_fail  = df_s.groupby('Shift_Name')['Failure'].agg(['sum','count','mean']).reset_index()
        shift_fail.columns = ['Shift','Failures','Total','Rate']
        shift_fail['Rate_pct'] = (shift_fail['Rate'] * 100).round(1)
        shift_fail = shift_fail.sort_values('Rate_pct', ascending=False)

        worst_shift = shift_fail.iloc[0]['Shift']
        worst_rate  = shift_fail.iloc[0]['Rate_pct']
        total_crit  = len(alerts_df[alerts_df['Issue_Severity']=='CRITICAL'])

        # Banner
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#2d1515,#3d1a1a);border:1px solid #ef4444;
                    border-radius:12px;padding:16px 20px;margin:8px 0;display:flex;
                    align-items:center;justify-content:space-between'>
            <div>
                <div style='font-size:12px;color:#fca5a5;letter-spacing:.1em;
                            text-transform:uppercase;margin-bottom:4px'>Highest failure shift</div>
                <div style='font-size:1.6rem;font-weight:700;color:#ef4444'>{worst_shift} Shift</div>
                <div style='font-size:12px;color:#fca5a5;margin-top:2px'>
                    Failure rate: {worst_rate}% — Recommend extra pre-shift inspection</div>
            </div>
            <div style='text-align:right'>
                <div style='font-size:12px;color:#94a3b8'>Tip: See full analysis in</div>
                <div style='font-size:13px;color:#3b82f6;font-weight:500'>🔀 Shift Analysis page</div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Mini bar chart
        fig_s, ax_s = plt.subplots(figsize=(12, 2.5))
        fig_s.patch.set_facecolor('#111827')
        ax_s.set_facecolor('#111827')
        s_colors = ['#ef4444' if s == worst_shift else '#3b82f6'
                    for s in shift_fail['Shift']]
        bars_s = ax_s.bar(shift_fail['Shift'], shift_fail['Rate_pct'],
                           color=s_colors, alpha=0.9, width=0.4)
        ax_s.set_ylabel('Failure Rate (%)', color='#94a3b8', fontsize=9)
        ax_s.tick_params(colors='#94a3b8', labelsize=9)
        for spine in ax_s.spines.values(): spine.set_color('#1e293b')
        for bar, row in zip(bars_s, shift_fail.itertuples()):
            ax_s.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                      str(row.Rate_pct) + '%\n(' + str(int(row.Failures)) + ' failures)',
                      ha='center', color='#e2e8f0', fontsize=9, fontweight='bold')
        ax_s.set_title('Failure Rate by Shift', color='#e2e8f0', fontsize=10, fontweight='bold')
        st.pyplot(fig_s, use_container_width=True)
        plt.close()

# ══════════════════════════════════════════════════════════════════════
# PAGE: ALERTS
# ══════════════════════════════════════════════════════════════════════
elif page == "🚨 Alerts":
    st.markdown("# 🚨 Equipment Alerts")
    st.markdown("---")

    # ── Inline Filters ────────────────────────────────────────────────
    al_f1, al_f2, al_f3, al_f4 = st.columns([2, 2, 2, 1])
    with al_f1:
        al_search = st.text_input("🔍 Search Equipment ID", placeholder="e.g. HEMM-D004", key="al_srch")
    with al_f2:
        al_eq_type = st.multiselect("Equipment Type",
            sorted(alerts_df['Equipment_Type'].unique()),
            default=sorted(alerts_df['Equipment_Type'].unique()), key="al_eq")
    with al_f3:
        al_sev = st.multiselect("Severity Level",
            ['CRITICAL','HIGH','MEDIUM','MONITOR','OK'],
            default=['CRITICAL','HIGH'], key="al_sev")
    with al_f4:
        al_replace_only = st.checkbox("Replacement only", value=False, key="al_rep")

    al_f5, al_f6 = st.columns(2)
    with al_f5:
        al_prob_range = st.slider("Failure Probability (%)", 0, 100, (0, 100), key="al_prob")
    with al_f6:
        al_days_range = st.slider("Days to Failure", 0, 180, (0, 180), key="al_days")

    # Apply all filters
    filtered_alerts = alerts_df.copy()
    if al_search:
        filtered_alerts = filtered_alerts[filtered_alerts['Equipment_ID'].str.contains(al_search.upper(), na=False)]
    if al_eq_type:
        filtered_alerts = filtered_alerts[filtered_alerts['Equipment_Type'].isin(al_eq_type)]
    if al_sev:
        filtered_alerts = filtered_alerts[filtered_alerts['Issue_Severity'].isin(al_sev)]
    if al_replace_only:
        filtered_alerts = filtered_alerts[filtered_alerts['Recommended_Action'].str.contains('REPLACE', na=False)]
    filtered_alerts = filtered_alerts[
        (filtered_alerts['Failure_Probability_%'] >= al_prob_range[0]) &
        (filtered_alerts['Failure_Probability_%'] <= al_prob_range[1]) &
        (filtered_alerts['Days_to_Failure'] >= al_days_range[0]) &
        (filtered_alerts['Days_to_Failure'] <= al_days_range[1])
    ]

    st.caption(f"Showing **{filtered_alerts['Equipment_ID'].nunique()}** machines matching filters "
               f"| {len(filtered_alerts[filtered_alerts['Issue_Severity']=='CRITICAL'])} CRITICAL alerts "
               f"| {len(filtered_alerts[filtered_alerts['Recommended_Action'].str.contains('REPLACE',na=False)])} replacements needed")
    st.markdown("---")

    # Top risk machines
    top_risk = (filtered_alerts.groupby('Equipment_ID')
                .agg(Prob=('Failure_Probability_%','first'),
                     Days=('Days_to_Failure','first'),
                     Level=('Alert_Level','first'),
                     Type=('Equipment_Type','first'),
                     Issues=('Issue_Severity', lambda x: (x=='CRITICAL').sum()))
                .sort_values('Prob', ascending=False).head(20))

    st.markdown("<div class='section-title'>Top 20 Highest Risk Machines</div>", unsafe_allow_html=True)
    for eq_id, row2 in top_risk.iterrows():
        lvl    = row2['Level']
        icon   = ICONS.get(lvl, '⚪')
        color  = LEVEL_COLORS.get(lvl, '#64748b')
        css    = f"alert-{lvl.lower()}" if lvl in ['CRITICAL','HIGH','MEDIUM'] else 'alert-ok'
        st.markdown(f"""
        <div class='{css}'>
            <b style='color:{color}'>{icon} {eq_id}</b>
            &nbsp;&nbsp;<span style='color:#94a3b8;font-size:13px'>{row2['Type']}</span>
            &nbsp;&nbsp;<span class='badge-{lvl.lower() if lvl in ["CRITICAL","HIGH","MEDIUM","OK"] else "ok"}'>{lvl}</span>
            <span style='float:right;color:#e2e8f0;font-family:monospace'>
                🎯 {row2['Prob']:.1f}% &nbsp;|&nbsp; ⏱ {row2['Days']}d &nbsp;|&nbsp; ⚠️ {int(row2['Issues'])} critical issues
            </span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Detailed Alert Table</div>", unsafe_allow_html=True)

    display_cols = ['Equipment_ID','Equipment_Type','Failure_Probability_%',
                    'Days_to_Failure','Alert_Level','Component','Problem_Detected','Recommended_Action']
    st.dataframe(
        filtered_alerts[display_cols].sort_values('Failure_Probability_%', ascending=False),
        use_container_width=True,
        height=400,
        hide_index=True
    )

    # Download button
    csv_data = filtered_alerts.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Alert Report (CSV)",
        data=csv_data,
        file_name="hemm_alerts_export.csv",
        mime="text/csv"
    )

# ══════════════════════════════════════════════════════════════════════
# PAGE: ML MODELS
# ══════════════════════════════════════════════════════════════════════
elif page == "📊 ML Models":
    st.markdown("# 📊 ML Model Performance")
    st.markdown("---")

    # ── Filters ───────────────────────────────────────────────────────
    with st.expander("🔍 Filter Model Analysis", expanded=False):
        ml_f1, ml_f2, ml_f3 = st.columns(3)
        with ml_f1:
            ml_eq_type = st.multiselect("Equipment Type (for prob distribution)",
                sorted(alerts_df['Equipment_Type'].unique()),
                default=sorted(alerts_df['Equipment_Type'].unique()), key="ml_eq")
        with ml_f2:
            ml_top_n = st.slider("Top N features to show", 5, 20, 15, key="ml_topn")
        with ml_f3:
            ml_prob_min = st.slider("Min failure probability to show (%)", 0, 100, 0, key="ml_prob")

    ml_alerts = alerts_df[alerts_df['Equipment_Type'].isin(ml_eq_type)]
    ml_fi_top = fi.head(ml_top_n)
    st.caption(f"Analysing {ml_alerts['Equipment_ID'].nunique()} machines | Top {ml_top_n} features shown")
    st.markdown("---")

    m1,m2,m3,m4 = st.columns(4)
    model_kpis = [
        (m1, f"{metrics['auc']:.4f}", "AUC Score",    "#3b82f6"),
        (m2, f"{metrics['accuracy']:.4f}", "Accuracy","#22c55e"),
        (m3, f"{metrics['recall']:.4f}",   "Recall",  "#f97316"),
        (m4, f"{metrics['f1']:.4f}",       "F1 Score","#8b5cf6"),
    ]
    for col, val, label, color in model_kpis:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-val' style='color:{color}'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-title'>Top 15 Feature Importances</div>", unsafe_allow_html=True)
        top15 = ml_fi_top
        fig4, ax4 = plt.subplots(figsize=(6, 5))
        fig4.patch.set_facecolor('#111827')
        ax4.set_facecolor('#111827')
        palette = (['#3b82f6']*5 + ['#22c55e']*5 + ['#8b5cf6']*5)
        ax4.barh(range(15), top15.values, color=palette, alpha=0.9, height=0.7)
        ax4.set_yticks(range(15))
        ax4.set_yticklabels(top15.index, color='#e2e8f0', fontsize=8)
        ax4.invert_yaxis()
        ax4.set_xlabel('Importance Score', color='#64748b', fontsize=9)
        ax4.tick_params(colors='#94a3b8')
        for spine in ax4.spines.values(): spine.set_color('#1e293b')
        st.pyplot(fig4, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("<div class='section-title'>Failure Probability Distribution</div>", unsafe_allow_html=True)
        proba_all = ml_alerts.groupby('Equipment_ID')['Failure_Probability_%'].first()
        fig5, ax5 = plt.subplots(figsize=(6, 5))
        fig5.patch.set_facecolor('#111827')
        ax5.set_facecolor('#111827')
        ax5.hist(proba_all, bins=40, color='#3b82f6', alpha=0.8, edgecolor='none')
        ax5.axvline(75, color='#ef4444', lw=2, linestyle='--', label='Critical (75%)')
        ax5.axvline(50, color='#f97316', lw=1.5, linestyle='--', label='High (50%)')
        ax5.axvline(30, color='#3b82f6', lw=1, linestyle=':', label='Medium (30%)')
        ax5.set_xlabel('Failure Probability (%)', color='#64748b', fontsize=9)
        ax5.set_ylabel('Number of Machines', color='#64748b', fontsize=9)
        ax5.tick_params(colors='#94a3b8')
        ax5.legend(facecolor='#1e293b', labelcolor='#e2e8f0', fontsize=8)
        for spine in ax5.spines.values(): spine.set_color('#1e293b')
        st.pyplot(fig5, use_container_width=True)
        plt.close()

    st.markdown("<div class='section-title'>Days to Failure — Distribution by Equipment Type</div>", unsafe_allow_html=True)
    top_risk_eq = (alerts_df.groupby(['Equipment_Type','Equipment_ID'])
                   .agg(Days=('Days_to_Failure','first'), Prob=('Failure_Probability_%','first'))
                   .reset_index())
    fig6, ax6 = plt.subplots(figsize=(12, 3.5))
    fig6.patch.set_facecolor('#111827')
    ax6.set_facecolor('#111827')
    eq_types_list = top_risk_eq['Equipment_Type'].unique()
    colors_list   = ['#ef4444','#f97316','#eab308','#22c55e','#3b82f6','#8b5cf6','#ec4899']
    for i, eq_t in enumerate(eq_types_list):
        sub = top_risk_eq[top_risk_eq['Equipment_Type']==eq_t]
        ax6.scatter(sub['Days'], [i]*len(sub), alpha=0.5, s=20,
                    color=colors_list[i % len(colors_list)])
    ax6.set_yticks(range(len(eq_types_list)))
    ax6.set_yticklabels(eq_types_list, color='#e2e8f0', fontsize=9)
    ax6.set_xlabel('Days to Failure', color='#64748b', fontsize=9)
    ax6.tick_params(colors='#94a3b8')
    for spine in ax6.spines.values(): spine.set_color('#1e293b')
    st.pyplot(fig6, use_container_width=True)
    plt.close()

# ══════════════════════════════════════════════════════════════════════
# PAGE: EQUIPMENT DETAIL
# ══════════════════════════════════════════════════════════════════════
elif page == "🔧 Equipment Detail":
    st.markdown("# 🔧 Equipment Detail View")
    st.markdown("---")

    # Always use FULL alerts_df so ALL machines appear regardless of sidebar filters
    all_eq_ids = sorted(alerts_df['Equipment_ID'].unique().tolist())

    # Inline filter controls
    col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
    with col_f1:
        eq_type_filter = st.selectbox(
            "Filter by Equipment Type",
            ["All Types"] + sorted(alerts_df['Equipment_Type'].unique().tolist())
        )
    with col_f2:
        alert_lvl_filter = st.selectbox(
            "Filter by Alert Level",
            ["All Levels", "CRITICAL", "HIGH", "MEDIUM", "MONITOR", "OK"]
        )
    with col_f3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption(f"Total: {len(all_eq_ids)} machines")

    # Apply inline filters
    filtered_eq_df = alerts_df.copy()
    if eq_type_filter != "All Types":
        filtered_eq_df = filtered_eq_df[filtered_eq_df['Equipment_Type'] == eq_type_filter]
    if alert_lvl_filter != "All Levels":
        filtered_eq_df = filtered_eq_df[filtered_eq_df['Alert_Level'] == alert_lvl_filter]

    eq_list = sorted(filtered_eq_df['Equipment_ID'].unique().tolist())

    if not eq_list:
        st.warning("No machines match the selected filters. Try changing the filters above.")
        st.stop()

    st.caption(f"Showing {len(eq_list)} machines matching filters")
    selected_eq = st.selectbox("Select Equipment ID", eq_list)

    eq_alerts = alerts_df[alerts_df['Equipment_ID'] == selected_eq]

    if len(eq_alerts) > 0:
        prob = eq_alerts['Failure_Probability_%'].iloc[0]
        days = eq_alerts['Days_to_Failure'].iloc[0]
        lvl  = eq_alerts['Alert_Level'].iloc[0]
        eq_t = eq_alerts['Equipment_Type'].iloc[0]
        color = LEVEL_COLORS.get(lvl, '#64748b')
        icon  = ICONS.get(lvl, '⚪')

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"<div class='metric-card'><div class='metric-val' style='color:{color}'>{prob}%</div><div class='metric-label'>Failure Probability</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-card'><div class='metric-val' style='color:#22c55e'>{days}d</div><div class='metric-label'>Days to Failure</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='metric-card'><div class='metric-val' style='color:{color}'>{icon} {lvl}</div><div class='metric-label'>Alert Level</div></div>", unsafe_allow_html=True)
        with c4:
            crit_issues = len(eq_alerts[eq_alerts['Issue_Severity']=='CRITICAL'])
            st.markdown(f"<div class='metric-card'><div class='metric-val' style='color:#ef4444'>{crit_issues}</div><div class='metric-label'>Critical Issues</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-title'>Issues for {selected_eq} ({eq_t})</div>", unsafe_allow_html=True)

        for _, row3 in eq_alerts.iterrows():
            sev   = row3['Issue_Severity']
            comp  = row3['Component']
            prob2 = row3['Problem_Detected']
            act   = row3['Recommended_Action']
            css   = f"alert-{sev.lower()}" if sev in ['CRITICAL','HIGH','MEDIUM'] else 'alert-ok'
            s_icon = ICONS.get(sev,'⚪')
            is_replace = 'REPLACE' in act
            st.markdown(f"""
            <div class='{css}'>
                <b>{s_icon} [{sev}] {comp}</b>
                {'<span style="background:#ef4444;color:white;border-radius:4px;padding:1px 8px;font-size:11px;margin-left:8px">⚠ REPLACEMENT NEEDED</span>' if is_replace else ''}
                <br>
                <span style='color:#94a3b8;font-size:13px'>Problem: {prob2}</span><br>
                <span style='color:#e2e8f0;font-size:13px'>→ <b>{act}</b></span>
            </div>""", unsafe_allow_html=True)

        # ── Current Sensor Readings ──────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-title'>Current Sensor Readings — {selected_eq}</div>", unsafe_allow_html=True)

        SENSOR_LIMITS = {
            'Engine_Temp_C':          {'min': 70,  'max': 100, 'unit': 'C',   'danger': 105},
            'Oil_Pressure_bar':       {'min': 2.5, 'max': 5.0, 'unit': 'bar', 'danger': 5.5},
            'Vibration_mms':          {'min': 0,   'max': 4.5, 'unit': 'mms', 'danger': 5.0},
            'Tyre_Pressure_PSI':      {'min': 80,  'max': 110, 'unit': 'PSI', 'danger': 115},
            'Battery_Voltage_V':      {'min': 11.5,'max': 14.5,'unit': 'V',   'danger': 15.0},
            'Hydraulic_Pressure_bar': {'min': 150, 'max': 280, 'unit': 'bar', 'danger': 290},
        }
        sensor_raw  = list(SENSOR_LIMITS.keys())
        sensor_vals = {}
        eq_idx = df.index[df['Equipment_ID'] == selected_eq].tolist() if 'Equipment_ID' in df.columns else []
        if eq_idx:
            eq_row_data = df.iloc[eq_idx[0]]
            for s in sensor_raw:
                if s in df.columns:
                    sensor_vals[s] = float(eq_row_data[s])

        if sensor_vals:
            cols_s = st.columns(len(sensor_vals))
            for i, (s, v) in enumerate(sensor_vals.items()):
                lim    = SENSOR_LIMITS[s]
                unit   = lim['unit']
                is_hi  = v > lim['max']
                is_lo  = v < lim['min']
                is_dan = v > lim['danger']
                color  = '#ef4444' if is_dan else '#f97316' if (is_hi or is_lo) else '#22c55e'
                status = 'DANGER' if is_dan else 'HIGH' if is_hi else 'LOW' if is_lo else 'OK'
                with cols_s[i]:
                    st.markdown(
                        "<div class='metric-card' style='border-color:" + color + "'>"
                        "<div class='metric-val' style='color:" + color + ";font-size:1.3rem'>"
                        + str(round(v, 1)) + " " + unit + "</div>"
                        "<div class='metric-label'>" + s.replace('_',' ') + "</div>"
                        "<div style='font-size:10px;color:" + color + ";margin-top:3px'>" + status + "</div>"
                        "<div style='font-size:10px;color:#475569'>Normal: "
                        + str(lim['min']) + " - " + str(lim['max']) + " " + unit + "</div>"
                        "</div>",
                        unsafe_allow_html=True
                    )

        # ── 30-Day Sensor Trend Charts ────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-title'>Sensor Trend — Last 30 Days ({selected_eq})</div>", unsafe_allow_html=True)
        st.caption("Rising trend = gradual degradation warning. Dotted lines = safe operating limits.")

        TREND_SENSORS = [s for s in ['Engine_Temp_C','Oil_Pressure_bar','Vibration_mms',
                                      'Hydraulic_Pressure_bar','Battery_Voltage_V','Exhaust_Temp_C']
                         if s in df.columns]

        # Build history: use Date column if available, else simulate
        if 'Date' in df.columns and 'Equipment_ID' in df.columns:
            eq_hist = df[df['Equipment_ID'] == selected_eq].sort_values('Date').tail(30).copy()
            eq_hist['Date'] = pd.to_datetime(eq_hist['Date'])
            use_real = len(eq_hist) >= 3
        else:
            use_real = False

        if not use_real:
            np.random.seed(abs(hash(selected_eq)) % 9999)
            dates_sim = pd.date_range(end=pd.Timestamp.now(), periods=30, freq='D')
            sim = {'Date': dates_sim}
            for s in TREND_SENSORS:
                if s in sensor_vals:
                    base  = sensor_vals[s]
                    trend = 0.018 * base if prob > 60 else -0.008 * base if prob < 20 else 0
                    sim[s] = [base - trend*(29-i) + np.random.normal(0, base*0.018)
                               for i in range(30)]
            eq_hist = pd.DataFrame(sim)

        # Sensor selector in sidebar
        st.sidebar.markdown("### Sensor Trend Selector")
        selected_sensors = st.sidebar.multiselect(
            "Sensors to plot", TREND_SENSORS, default=TREND_SENSORS[:3])

        if not selected_sensors:
            st.info("Select sensors from the sidebar to view trends.")
        else:
            for s in selected_sensors:
                if s not in eq_hist.columns:
                    continue
                lim   = SENSOR_LIMITS.get(s, {})
                unit  = lim.get('unit','')
                s_max = lim.get('max', None)
                s_min = lim.get('min', None)
                s_dan = lim.get('danger', None)

                vals_t  = eq_hist[s].values
                dates_t = eq_hist['Date']

                # Trend direction
                if len(vals_t) >= 5:
                    slope = np.polyfit(range(len(vals_t)), vals_t, 1)[0]
                    if slope > abs(np.mean(vals_t)) * 0.001:
                        trend_txt   = "RISING — Degrading"
                        trend_color = '#ef4444'
                    elif slope < -abs(np.mean(vals_t)) * 0.001:
                        trend_txt   = "FALLING — Improving"
                        trend_color = '#22c55e'
                    else:
                        trend_txt   = "STABLE"
                        trend_color = '#3b82f6'
                else:
                    slope = 0; trend_txt = "Insufficient data"; trend_color = '#94a3b8'

                fig_t, ax_t = plt.subplots(figsize=(13, 3))
                fig_t.patch.set_facecolor('#111827')
                ax_t.set_facecolor('#0f172a')

                # Main line + fill
                ax_t.plot(range(len(vals_t)), vals_t, color='#58a6ff', lw=2, zorder=5)
                ax_t.fill_between(range(len(vals_t)), vals_t, alpha=0.12, color='#58a6ff')

                # Trend line
                if len(vals_t) >= 3:
                    p_fit = np.poly1d(np.polyfit(range(len(vals_t)), vals_t, 1))
                    ax_t.plot(range(len(vals_t)), p_fit(range(len(vals_t))),
                              '--', color=trend_color, lw=1.5, alpha=0.85,
                              label="Trend: " + trend_txt)

                # Limit lines
                if s_max: ax_t.axhline(s_max, color='#f97316', lw=1, linestyle=':', label="Max (" + str(s_max) + ")")
                if s_min: ax_t.axhline(s_min, color='#3b82f6', lw=1, linestyle=':', label="Min (" + str(s_min) + ")")
                if s_dan: ax_t.axhline(s_dan, color='#ef4444', lw=1.5, linestyle='--', label="DANGER (" + str(s_dan) + ")")

                # Danger points
                if s_dan:
                    danger_pts = [i for i, v in enumerate(vals_t) if v > s_dan]
                    if danger_pts:
                        ax_t.scatter(danger_pts, [vals_t[i] for i in danger_pts],
                                     color='#ef4444', s=45, zorder=10, label='Above danger')

                # X-axis dates
                step = max(1, len(dates_t)//6)
                ax_t.set_xticks(range(0, len(dates_t), step))
                ax_t.set_xticklabels([str(dates_t.iloc[i])[:10]
                                       for i in range(0, len(dates_t), step)],
                                      color='#64748b', fontsize=8, rotation=20)
                ax_t.set_ylabel(s.replace('_',' ') + " (" + unit + ")", color='#94a3b8', fontsize=9)
                ax_t.set_title(s.replace('_',' ') + " — 30-Day Trend", color='#e2e8f0', fontsize=10, fontweight='bold')
                ax_t.tick_params(colors='#64748b')
                ax_t.legend(facecolor='#1e293b', labelcolor='#e2e8f0', fontsize=8, loc='upper left')
                for spine in ax_t.spines.values(): spine.set_color('#1e293b')

                col_badge, _ = st.columns([1, 3])
                with col_badge:
                    st.markdown(
                        "<span style='font-size:12px;font-weight:600;color:" + trend_color + "'>"
                        + trend_txt + "</span>",
                        unsafe_allow_html=True)
                st.pyplot(fig_t, use_container_width=True)
                plt.close()

        # ── Component Life Remaining ──────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-title'>Component Life Remaining — {selected_eq}</div>", unsafe_allow_html=True)

        LIFE_COLS = {
            'Engine_Life_Remaining_pct':    'Engine',
            'Brake_Life_Remaining_pct':     'Brakes',
            'Tyre_Life_Remaining_pct':      'Tyres',
            'Hydraulic_Life_Remaining_pct': 'Hydraulics',
            'Battery_Life_Remaining_pct':   'Battery',
        }
        life_vals = {}
        if eq_idx:
            for col_name, label in LIFE_COLS.items():
                if col_name in df.columns:
                    life_vals[label] = float(df.iloc[eq_idx[0]][col_name])

        if life_vals:
            fig_l, ax_l = plt.subplots(figsize=(13, 2.5))
            fig_l.patch.set_facecolor('#111827')
            ax_l.set_facecolor('#111827')
            comps    = list(life_vals.keys())
            vals_l   = list(life_vals.values())
            lc_colors = ['#ef4444' if v < 15 else '#f97316' if v < 30 else '#22c55e' for v in vals_l]
            bars_l   = ax_l.barh(comps, vals_l, color=lc_colors, alpha=0.9, height=0.5)
            ax_l.axvline(15, color='#ef4444', lw=1.5, linestyle='--', label='Critical (<15%)')
            ax_l.axvline(30, color='#f97316', lw=1,   linestyle=':',  label='Warning (<30%)')
            ax_l.set_xlim(0, 100)
            ax_l.set_xlabel('Life Remaining (%)', color='#94a3b8', fontsize=9)
            ax_l.tick_params(colors='#94a3b8')
            ax_l.legend(facecolor='#1e293b', labelcolor='#e2e8f0', fontsize=8)
            ax_l.set_title('Component Life Remaining %', color='#e2e8f0', fontsize=10, fontweight='bold')
            for spine in ax_l.spines.values(): spine.set_color('#30363d')
            for bar, val in zip(bars_l, vals_l):
                ax_l.text(val+1, bar.get_y()+bar.get_height()/2,
                           str(round(val)) + '%', va='center', color='#e2e8f0',
                           fontsize=9, fontweight='bold')
            st.pyplot(fig_l, use_container_width=True)
            plt.close()


# ══════════════════════════════════════════════════════════════════════
# PAGE: COST SAVINGS CALCULATOR
# ══════════════════════════════════════════════════════════════════════
elif page == "💰 Cost Savings":
    st.markdown("# 💰 Cost Savings Calculator")
    st.markdown("Estimated savings by using predictive maintenance vs reactive maintenance at NALCO")
    st.markdown("---")

    # ── Filters ───────────────────────────────────────────────────────
    with st.expander("🔍 Filter Equipment for Cost Analysis", expanded=False):
        cs_f1, cs_f2, cs_f3 = st.columns(3)
        with cs_f1:
            cs_eq_type = st.multiselect("Equipment Type",
                sorted(alerts_df['Equipment_Type'].unique()),
                default=sorted(alerts_df['Equipment_Type'].unique()), key="cs_eq")
        with cs_f2:
            cs_alert = st.multiselect("Alert Level to count as caught early",
                ['CRITICAL','HIGH','MEDIUM'],
                default=['CRITICAL','HIGH'], key="cs_al")
        with cs_f3:
            cs_prob_min = st.slider("Min failure probability (%)", 0, 100, 30, key="cs_prob")

    alerts_df = alerts_df.copy()
    cs_filtered = alerts_df[
        alerts_df['Equipment_Type'].isin(cs_eq_type) &
        alerts_df['Alert_Level'].isin(cs_alert) &
        (alerts_df['Failure_Probability_%'] >= cs_prob_min)
    ]
    st.caption(f"{cs_filtered['Equipment_ID'].nunique()} machines match cost filter criteria")
    st.markdown("---")

    # ── Sidebar inputs for cost assumptions ───────────────────────────
    st.sidebar.markdown("### Cost Assumptions (INR)")
    reactive_repair   = st.sidebar.number_input("Reactive repair cost (₹)", value=250000, step=10000)
    downtime_per_day  = st.sidebar.number_input("Downtime cost per day (₹)", value=80000,  step=5000)
    avg_downtime_days = st.sidebar.number_input("Avg downtime days (reactive)", value=3, step=1)
    preventive_cost   = st.sidebar.number_input("Preventive service cost (₹)", value=40000, step=5000)
    working_days      = st.sidebar.number_input("Working days this month", value=26, step=1)

    # ── Calculate from alert data ──────────────────────────────────────
    # Machines caught early = machines with HIGH or CRITICAL alert (model warned before failure)
    caught_early  = cs_filtered['Equipment_ID'].nunique()
    total_eq      = alerts_df['Equipment_ID'].nunique()
    ok_machines   = alerts_df[alerts_df['Alert_Level']=='OK']['Equipment_ID'].nunique()

    # Cost calculations
    reactive_total    = caught_early * (reactive_repair + downtime_per_day * avg_downtime_days)
    preventive_total  = caught_early * preventive_cost
    total_savings     = reactive_total - preventive_total
    savings_crore     = total_savings / 10_000_000
    savings_lakh      = total_savings / 100_000

    # ROI
    roi_pct = ((total_savings - preventive_total) / preventive_total * 100) if preventive_total > 0 else 0

    # ── KPI cards ─────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-val' style='color:#ef4444'>₹{reactive_total/100000:.1f}L</div>
            <div class='metric-label'>Reactive cost (if no ML)</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-val' style='color:#3b82f6'>₹{preventive_total/100000:.1f}L</div>
            <div class='metric-label'>Predictive cost (with ML)</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-val' style='color:#22c55e'>₹{savings_lakh:.1f}L</div>
            <div class='metric-label'>Estimated savings</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-val' style='color:#f97316'>{roi_pct:.0f}%</div>
            <div class='metric-label'>Return on investment</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Big savings banner ─────────────────────────────────────────────
    if savings_crore >= 1:
        banner_val = f"₹{savings_crore:.2f} Crore"
    else:
        banner_val = f"₹{savings_lakh:.1f} Lakh"

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#0f2d1a,#1a3d2b);border:2px solid #22c55e;
                border-radius:16px;padding:28px;text-align:center;margin:12px 0'>
        <div style='font-size:13px;color:#4ade80;letter-spacing:.15em;text-transform:uppercase;margin-bottom:8px'>
            Estimated savings this month using predictive maintenance
        </div>
        <div style='font-size:3rem;font-weight:700;color:#22c55e;font-family:monospace'>
            {banner_val}
        </div>
        <div style='font-size:13px;color:#86efac;margin-top:8px'>
            {caught_early} machines caught early &nbsp;|&nbsp;
            {total_eq} total machines &nbsp;|&nbsp;
            ROI = {roi_pct:.0f}%
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    # ── Chart 1: Reactive vs Predictive comparison bar ─────────────────
    with col1:
        st.markdown("<div class='section-title'>Reactive vs Predictive Cost Comparison</div>",
                    unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        fig1.patch.set_facecolor('#111827')
        ax1.set_facecolor('#111827')

        categories = ['Reactive\n(No ML)', 'Predictive\n(With ML)', 'Savings']
        values     = [reactive_total/100000, preventive_total/100000, total_savings/100000]
        colors_b   = ['#ef4444', '#3b82f6', '#22c55e']

        bars = ax1.bar(categories, values, color=colors_b, alpha=0.9, width=0.5)
        ax1.set_ylabel('Cost (₹ Lakhs)', color='#94a3b8', fontsize=10)
        ax1.tick_params(colors='#94a3b8')
        ax1.set_title('Cost Comparison (₹ Lakhs)', color='#e2e8f0', fontsize=11, fontweight='bold')
        for spine in ax1.spines.values(): spine.set_color('#30363d')
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                     f'₹{val:.1f}L', ha='center', color='#e2e8f0',
                     fontsize=10, fontweight='bold')
        st.pyplot(fig1, use_container_width=True)
        plt.close()

    # ── Chart 2: Per equipment type savings breakdown ──────────────────
    with col2:
        st.markdown("<div class='section-title'>Savings Breakdown by Equipment Type</div>",
                    unsafe_allow_html=True)
        eq_caught = (alerts_df[alerts_df['Alert_Level'].isin(['CRITICAL','HIGH'])]
                     .groupby('Equipment_Type')['Equipment_ID'].nunique())

        eq_savings = (eq_caught * (reactive_repair + downtime_per_day * avg_downtime_days
                                   - preventive_cost)) / 100000

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        fig2.patch.set_facecolor('#111827')
        ax2.set_facecolor('#111827')
        eq_colors = ['#22c55e' if v > eq_savings.mean() else '#3b82f6'
                     for v in eq_savings.values]
        bars2 = ax2.barh(eq_savings.index, eq_savings.values,
                          color=eq_colors, alpha=0.9, height=0.6)
        ax2.set_xlabel('Savings (₹ Lakhs)', color='#94a3b8', fontsize=10)
        ax2.tick_params(colors='#94a3b8', labelsize=9)
        ax2.set_title('Savings per Equipment Type', color='#e2e8f0',
                       fontsize=11, fontweight='bold')
        for spine in ax2.spines.values(): spine.set_color('#30363d')
        for bar, val in zip(bars2, eq_savings.values):
            ax2.text(val+0.2, bar.get_y()+bar.get_height()/2,
                     f'₹{val:.1f}L', va='center', color='#e2e8f0', fontsize=9)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    # ── Chart 3: Monthly savings trend (simulated over 12 months) ──────
    st.markdown("<div class='section-title'>Projected Annual Savings (12-Month Forecast)</div>",
                unsafe_allow_html=True)

    months      = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']
    # Simulate slight variation each month (+/- 10%)
    np.random.seed(42)
    monthly_savings = [total_savings/100000 * (1 + np.random.uniform(-0.1, 0.15))
                       for _ in range(12)]
    cumulative      = np.cumsum(monthly_savings)

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 4))
    fig3.patch.set_facecolor('#111827')

    # Monthly bar
    ax3a.set_facecolor('#111827')
    bar_colors = ['#22c55e' if v >= total_savings/100000 else '#3b82f6'
                  for v in monthly_savings]
    ax3a.bar(months, monthly_savings, color=bar_colors, alpha=0.9, width=0.6)
    ax3a.axhline(total_savings/100000, color='#f97316', lw=1.5,
                  linestyle='--', label=f'Baseline ₹{total_savings/100000:.1f}L')
    ax3a.set_ylabel('Savings (₹ Lakhs)', color='#94a3b8', fontsize=9)
    ax3a.tick_params(colors='#94a3b8', labelsize=8)
    ax3a.set_title('Monthly Savings Forecast', color='#e2e8f0', fontsize=10, fontweight='bold')
    ax3a.legend(facecolor='#1e293b', labelcolor='#e2e8f0', fontsize=8)
    for spine in ax3a.spines.values(): spine.set_color('#30363d')

    # Cumulative line
    ax3b.set_facecolor('#111827')
    ax3b.plot(months, cumulative, color='#22c55e', lw=2.5, marker='o',
               markersize=5, markerfacecolor='#22c55e')
    ax3b.fill_between(range(12), cumulative, alpha=0.1, color='#22c55e')
    ax3b.set_ylabel('Cumulative Savings (₹ Lakhs)', color='#94a3b8', fontsize=9)
    ax3b.tick_params(colors='#94a3b8', labelsize=8)
    ax3b.set_xticks(range(12)); ax3b.set_xticklabels(months, fontsize=8, color='#94a3b8')
    ax3b.set_title(f'Cumulative Annual Savings: ₹{cumulative[-1]/100:.2f} Crore',
                    color='#e2e8f0', fontsize=10, fontweight='bold')
    for spine in ax3b.spines.values(): spine.set_color('#30363d')

    # Annotate final value
    ax3b.annotate(f'₹{cumulative[-1]:.0f}L',
                   xy=(11, cumulative[-1]),
                   xytext=(-40, -20), textcoords='offset points',
                   color='#22c55e', fontsize=10, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close()

    # ── Detailed breakdown table ───────────────────────────────────────
    st.markdown("<div class='section-title'>Detailed Cost Breakdown</div>",
                unsafe_allow_html=True)

    breakdown_data = {
        'Item': [
            'Machines caught early by ML',
            'Reactive repair cost per machine',
            'Downtime cost per machine',
            'Total reactive cost (if no ML)',
            'Preventive service cost per machine',
            'Total preventive cost (with ML)',
            'NET SAVINGS',
            'Annual projection'
        ],
        'Value (INR)': [
            f"{caught_early} machines",
            f"₹{reactive_repair:,.0f}",
            f"₹{downtime_per_day * avg_downtime_days:,.0f} ({avg_downtime_days} days × ₹{downtime_per_day:,.0f})",
            f"₹{reactive_total:,.0f}  (₹{reactive_total/100000:.1f} Lakhs)",
            f"₹{preventive_cost:,.0f}",
            f"₹{preventive_total:,.0f}  (₹{preventive_total/100000:.1f} Lakhs)",
            f"₹{total_savings:,.0f}  (₹{savings_lakh:.1f} Lakhs = ₹{savings_crore:.2f} Crore)",
            f"₹{total_savings*12:,.0f}  (₹{total_savings*12/10000000:.2f} Crore/year)"
        ]
    }
    breakdown_df = pd.DataFrame(breakdown_data)
    st.dataframe(breakdown_df, use_container_width=True, hide_index=True, height=320)

    # ── Download button ────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    report_text = f"""
NALCO HEMM — COST SAVINGS REPORT
Generated: {time.strftime('%d %b %Y %H:%M')}
{'='*50}
Machines analysed       : {total_eq}
Machines caught early   : {caught_early}
Machines safe (OK)      : {ok_machines}

COST ASSUMPTIONS
Reactive repair/machine : INR {reactive_repair:,.0f}
Downtime cost/day       : INR {downtime_per_day:,.0f}
Avg downtime (reactive) : {avg_downtime_days} days
Preventive cost/machine : INR {preventive_cost:,.0f}

RESULTS
Reactive total cost     : INR {reactive_total:,.0f}
Preventive total cost   : INR {preventive_total:,.0f}
NET SAVINGS THIS MONTH  : INR {total_savings:,.0f} (INR {savings_lakh:.1f} Lakhs)
ANNUAL PROJECTION       : INR {total_savings*12:,.0f} (INR {total_savings*12/10000000:.2f} Crore)
ROI                     : {roi_pct:.0f}%
{'='*50}
"""
    st.download_button(
        label="⬇️ Download Cost Savings Report (TXT)",
        data=report_text,
        file_name="nalco_hemm_cost_savings.txt",
        mime="text/plain"
    )



# ══════════════════════════════════════════════════════════════════════
# PAGE: SHIFT ANALYSIS
# ══════════════════════════════════════════════════════════════════════
elif page == "🔀 Shift Analysis":
    st.markdown("# 🔀 Shift-wise Failure Analysis")
    st.markdown("NALCO operates in 3 shifts — Day, Afternoon, Night. This page shows which shift has the most failures and what to do about it.")
    st.markdown("---")

    # ── Filters ───────────────────────────────────────────────────────
    with st.expander("🔍 Filter Shift Analysis", expanded=False):
        sh_f1, sh_f2, sh_f3 = st.columns(3)
        with sh_f1:
            sh_eq_type = st.multiselect("Equipment Type",
                list(EQ_TYPE_MAP.values()),
                default=list(EQ_TYPE_MAP.values()), key="sh_eq")
        with sh_f2:
            sh_shifts = st.multiselect("Shifts to compare",
                ["Day","Afternoon","Night"],
                default=["Day","Afternoon","Night"], key="sh_sh")
        with sh_f3:
            sh_min_ops = st.slider("Min Operating Hours", 0, 10000, 0, key="sh_hrs")

    shift_col = 'Shift' if 'Shift' in df.columns else 'Shift_enc'
    SHIFT_MAP_SA = {0: 'Day', 1: 'Afternoon', 2: 'Night'}

    if shift_col not in df.columns:
        st.warning("Shift column not found.")
    else:
        df_sa = df.copy()
        if shift_col == 'Shift_enc':
            df_sa['Shift_Name'] = df_sa['Shift_enc'].map(SHIFT_MAP_SA).fillna('Unknown')
        else:
            df_sa['Shift_Name'] = df_sa['Shift']

        if 'Equipment_Type' in df_sa.columns:
            df_sa = df_sa[df_sa['Equipment_Type'].isin(sh_eq_type)]
        if sh_min_ops > 0 and 'Operating_Hours' in df_sa.columns:
            df_sa = df_sa[df_sa['Operating_Hours'] >= sh_min_ops]
        df_sa = df_sa[df_sa['Shift_Name'].isin(sh_shifts)]
        st.caption(f"{len(df_sa)} records after filters | {df_sa['Shift_Name'].nunique()} shifts shown")
    st.markdown("---")

    SHIFT_MAP = {0: 'Day', 1: 'Afternoon', 2: 'Night'}
    shift_col = 'Shift' if 'Shift' in df.columns else 'Shift_enc'

    if shift_col not in df.columns:
        st.warning("Shift column not found in dataset.")
    else:
        df_s = df.copy()
        if shift_col == 'Shift_enc':
            df_s['Shift_Name'] = df_s['Shift_enc'].map(SHIFT_MAP).fillna('Unknown')
        else:
            df_s['Shift_Name'] = df_s['Shift']

        # ── KPI row ───────────────────────────────────────────────────
        shift_stats = df_s.groupby('Shift_Name')['Failure'].agg(
            Failures='sum', Total='count', Rate='mean').reset_index()
        shift_stats['Rate_pct'] = (shift_stats['Rate'] * 100).round(1)
        shift_stats = shift_stats.sort_values('Rate_pct', ascending=False).reset_index(drop=True)

        worst_shift  = shift_stats.iloc[0]['Shift_Name']
        worst_rate   = shift_stats.iloc[0]['Rate_pct']
        best_shift   = shift_stats.iloc[-1]['Shift_Name']
        best_rate    = shift_stats.iloc[-1]['Rate_pct']
        diff_pct     = round(worst_rate - best_rate, 1)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-val' style='color:#ef4444'>{worst_shift}</div>
                <div class='metric-label'>Highest failure shift</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-val' style='color:#ef4444'>{worst_rate}%</div>
                <div class='metric-label'>Failure rate ({worst_shift})</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-val' style='color:#22c55e'>{best_shift}</div>
                <div class='metric-label'>Safest shift</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-val' style='color:#f97316'>+{diff_pct}%</div>
                <div class='metric-label'>{worst_shift} vs {best_shift} gap</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Recommendation banner ─────────────────────────────────────
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);border:1px solid #3b82f6;
                    border-radius:12px;padding:18px 22px;margin:8px 0'>
            <div style='font-size:13px;font-weight:600;color:#93c5fd;margin-bottom:10px'>
                NALCO RECOMMENDATION — Based on shift analysis
            </div>
            <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px'>
                <div style='background:#0f172a;border-radius:8px;padding:12px'>
                    <div style='font-size:11px;color:#64748b;margin-bottom:4px'>ACTION 1</div>
                    <div style='font-size:12px;color:#e2e8f0'>Schedule <b>extra pre-shift inspection</b>
                    for <b style="color:#ef4444">{worst_shift} shift</b> equipment before every shift starts</div>
                </div>
                <div style='background:#0f172a;border-radius:8px;padding:12px'>
                    <div style='font-size:11px;color:#64748b;margin-bottom:4px'>ACTION 2</div>
                    <div style='font-size:12px;color:#e2e8f0'>Pair <b>experienced operators</b> with
                    new operators during <b style="color:#ef4444">{worst_shift} shift</b> to reduce human error</div>
                </div>
                <div style='background:#0f172a;border-radius:8px;padding:12px'>
                    <div style='font-size:11px;color:#64748b;margin-bottom:4px'>ACTION 3</div>
                    <div style='font-size:12px;color:#e2e8f0'>Study <b>{best_shift} shift</b> practices
                    and replicate across all shifts — {best_rate}% failure rate is the benchmark</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        # ── Chart 1: Failure rate per shift (bar) ─────────────────────
        with col1:
            st.markdown("<div class='section-title'>Failure Rate by Shift</div>",
                        unsafe_allow_html=True)
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            fig1.patch.set_facecolor('#111827')
            ax1.set_facecolor('#111827')
            s_colors = ['#ef4444' if s == worst_shift else
                        '#22c55e' if s == best_shift else '#3b82f6'
                        for s in shift_stats['Shift_Name']]
            bars1 = ax1.bar(shift_stats['Shift_Name'], shift_stats['Rate_pct'],
                             color=s_colors, alpha=0.9, width=0.45)
            ax1.set_ylabel('Failure Rate (%)', color='#94a3b8', fontsize=10)
            ax1.tick_params(colors='#94a3b8', labelsize=10)
            ax1.set_title('Failure Rate by Shift', color='#e2e8f0',
                           fontsize=11, fontweight='bold')
            for spine in ax1.spines.values(): spine.set_color('#30363d')
            for bar, row in zip(bars1, shift_stats.itertuples()):
                ax1.text(bar.get_x()+bar.get_width()/2,
                         bar.get_height()+0.2,
                         str(row.Rate_pct) + '%\n(' + str(int(row.Failures)) + ' failures)',
                         ha='center', color='#e2e8f0',
                         fontsize=9, fontweight='bold')
            st.pyplot(fig1, use_container_width=True)
            plt.close()

        # ── Chart 2: Total failures per shift (donut) ─────────────────
        with col2:
            st.markdown("<div class='section-title'>Failure Share by Shift</div>",
                        unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            fig2.patch.set_facecolor('#111827')
            ax2.set_facecolor('#111827')
            s_colors2 = ['#ef4444', '#f97316', '#22c55e']
            wedges, texts, autotexts = ax2.pie(
                shift_stats['Failures'],
                labels=shift_stats['Shift_Name'],
                colors=s_colors2[:len(shift_stats)],
                autopct='%1.1f%%', startangle=90,
                wedgeprops=dict(width=0.6),
                textprops={'color':'#94a3b8','fontsize':10})
            for at in autotexts:
                at.set_color('#0f172a'); at.set_fontweight('bold'); at.set_fontsize(9)
            ax2.set_title('Share of Total Failures by Shift',
                           color='#e2e8f0', fontsize=11, fontweight='bold')
            st.pyplot(fig2, use_container_width=True)
            plt.close()

        # ── Chart 3: Heatmap — Shift × Equipment Type ─────────────────
        st.markdown("<div class='section-title'>Failure Heatmap — Shift × Equipment Type</div>",
                    unsafe_allow_html=True)

        if 'Equipment_Type' in df_s.columns:
            eq_col = 'Equipment_Type'
        elif 'Equipment_Type_enc' in df_s.columns:
            df_s['Equipment_Type'] = df_s['Equipment_Type_enc'].map(
                {0:'Excavator',1:'Dumper',2:'Dozer',
                 3:'Grader',4:'Drill Rig',5:'Wheel Loader',6:'Scraper'})
            eq_col = 'Equipment_Type'
        else:
            eq_col = None

        if eq_col:
            pivot = df_s.pivot_table(values='Failure', index='Shift_Name',
                                      columns=eq_col, aggfunc='mean') * 100
            pivot = pivot.round(1)

            fig3, ax3 = plt.subplots(figsize=(13, 4))
            fig3.patch.set_facecolor('#111827')
            ax3.set_facecolor('#111827')
            im = ax3.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto',
                             vmin=0, vmax=pivot.values.max())
            ax3.set_xticks(range(len(pivot.columns)))
            ax3.set_xticklabels(pivot.columns, color='#e2e8f0', fontsize=10)
            ax3.set_yticks(range(len(pivot.index)))
            ax3.set_yticklabels(pivot.index, color='#e2e8f0', fontsize=10)
            ax3.set_title('Failure Rate % by Shift x Equipment Type (Red=high risk, Green=low risk)',
                           color='#e2e8f0', fontsize=11, fontweight='bold')
            # Add value labels inside cells
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i][j]
                    if not np.isnan(val):
                        ax3.text(j, i, f'{val:.1f}%', ha='center', va='center',
                                  color='white', fontsize=9, fontweight='bold')
            plt.colorbar(im, ax=ax3, label='Failure Rate (%)')
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True)
            plt.close()

        # ── Chart 4: Failures by shift + time of day bar ──────────────
        st.markdown("<div class='section-title'>Failure Count & Average Probability by Shift</div>",
                    unsafe_allow_html=True)

        fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 4))
        fig4.patch.set_facecolor('#111827')

        # Left: failure count bars
        ax4a.set_facecolor('#111827')
        b_colors = ['#ef4444' if s == worst_shift else
                    '#22c55e' if s == best_shift else '#3b82f6'
                    for s in shift_stats['Shift_Name']]
        bars4 = ax4a.bar(shift_stats['Shift_Name'], shift_stats['Failures'],
                          color=b_colors, alpha=0.9, width=0.5)
        ax4a.set_ylabel('Total Failures', color='#94a3b8', fontsize=9)
        ax4a.tick_params(colors='#94a3b8')
        ax4a.set_title('Total Failures per Shift', color='#e2e8f0',
                        fontsize=10, fontweight='bold')
        for spine in ax4a.spines.values(): spine.set_color('#30363d')
        for bar, val in zip(bars4, shift_stats['Failures']):
            ax4a.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                      str(int(val)), ha='center', color='#e2e8f0',
                      fontsize=10, fontweight='bold')

        # Right: avg probability
        ax4b.set_facecolor('#111827')
        if 'Failure_Probability_%' in alerts_df.columns:
            eq_ids_by_shift = {}
            if eq_col and shift_col in df_s.columns:
                for shift in shift_stats['Shift_Name']:
                    ids = df_s[df_s['Shift_Name'] == shift]['Equipment_ID'].unique()                           if 'Equipment_ID' in df_s.columns else []
                    eq_ids_by_shift[shift] = ids

            shift_prob = df_s.groupby('Shift_Name').apply(
                lambda g: g['Failure'].mean() * 100).reset_index()
            shift_prob.columns = ['Shift_Name', 'Avg_Fail_Rate']

            p_colors = ['#ef4444' if s == worst_shift else
                        '#22c55e' if s == best_shift else '#3b82f6'
                        for s in shift_prob['Shift_Name']]
            bars_p = ax4b.bar(shift_prob['Shift_Name'], shift_prob['Avg_Fail_Rate'],
                               color=p_colors, alpha=0.9, width=0.5)
            ax4b.set_ylabel('Avg Failure Rate (%)', color='#94a3b8', fontsize=9)
            ax4b.tick_params(colors='#94a3b8')
            ax4b.set_title('Average Failure Rate per Shift',
                            color='#e2e8f0', fontsize=10, fontweight='bold')
            for spine in ax4b.spines.values(): spine.set_color('#30363d')
            for bar, val in zip(bars_p, shift_prob['Avg_Fail_Rate']):
                ax4b.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                          f'{val:.1f}%', ha='center', color='#e2e8f0',
                          fontsize=10, fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        plt.close()

        # ── Detailed stats table ───────────────────────────────────────
        st.markdown("<div class='section-title'>Shift Statistics Table</div>",
                    unsafe_allow_html=True)
        display_stats = shift_stats.copy()
        display_stats['Failure Rate'] = display_stats['Rate_pct'].apply(lambda x: f"{x}%")
        display_stats['Status'] = display_stats['Shift_Name'].apply(
            lambda s: "⚠ HIGHEST RISK — Extra inspection needed" if s == worst_shift else
                      "✅ SAFEST — Use as benchmark" if s == best_shift else
                      "🔵 MODERATE")
        display_stats = display_stats[['Shift_Name','Failures','Total','Failure Rate','Status']]
        display_stats.columns = ['Shift','Failures','Total Machines','Failure Rate','Recommendation']
        st.dataframe(display_stats, use_container_width=True, hide_index=True)



# ══════════════════════════════════════════════════════════════════════
# PAGE: OPERATOR INSIGHTS
# ══════════════════════════════════════════════════════════════════════
elif page == "👷 Operator Insights":
    st.markdown("# 👷 Operator Experience vs Failure Rate")
    st.markdown("Analysing how operator experience affects equipment failure — key insight for NALCO HR and training decisions.")
    st.markdown("---")

    # ── Filters ───────────────────────────────────────────────────────
    with st.expander("🔍 Filter Operator Analysis", expanded=False):
        op_f1, op_f2, op_f3 = st.columns(3)
        with op_f1:
            op_eq_type_f = st.multiselect("Equipment Type",
                list(EQ_TYPE_MAP.values()),
                default=list(EQ_TYPE_MAP.values()), key="op_eq")
        with op_f2:
            op_exp_range = st.slider("Experience Range (years)", 0, 40, (0, 40), key="op_exp")
        with op_f3:
            op_shift_f = st.multiselect("Shift", ["Day","Afternoon","Night"],
                default=["Day","Afternoon","Night"], key="op_sh")
    st.markdown("---")

    if 'Operator_Experience_Yr' not in df.columns:
        st.warning("Operator_Experience_Yr column not found in dataset.")
    else:
        op_df = df[['Operator_Experience_Yr','Failure']].dropna().copy()
        # Apply experience range filter
        op_df = op_df[(op_df['Operator_Experience_Yr'] >= op_exp_range[0]) &
                      (op_df['Operator_Experience_Yr'] <= op_exp_range[1])]
        st.caption(f"{len(op_df)} records after filters")

        # Add equipment type if available
        if 'Equipment_Type' in df.columns:
            op_df['Equipment_Type'] = df['Equipment_Type']
        elif 'Equipment_Type_enc' in df.columns:
            op_df['Equipment_Type'] = df['Equipment_Type_enc'].map(
                {0:'Excavator',1:'Dumper',2:'Dozer',
                 3:'Grader',4:'Drill Rig',5:'Wheel Loader',6:'Scraper'})

        # Experience bins
        bins   = [0, 2, 5, 10, 15, 50]
        labels = ['0-2 yrs (Novice)', '2-5 yrs (Junior)', '5-10 yrs (Mid)', '10-15 yrs (Senior)', '15+ yrs (Expert)']
        op_df['Exp_Group'] = pd.cut(op_df['Operator_Experience_Yr'],
                                     bins=bins, labels=labels, right=False)

        # Stats per group
        grp_stats = op_df.groupby('Exp_Group', observed=True)['Failure'].agg(
            ['mean','sum','count']).reset_index()
        grp_stats.columns = ['Group','Failure_Rate','Failures','Total']
        grp_stats['Failure_Rate_pct'] = (grp_stats['Failure_Rate'] * 100).round(1)

        # Overall stats
        novice_rate  = grp_stats[grp_stats['Group'].astype(str).str.contains('0-2')]['Failure_Rate_pct'].values
        expert_rate  = grp_stats[grp_stats['Group'].astype(str).str.contains('15')]['Failure_Rate_pct'].values
        novice_rate  = float(novice_rate[0]) if len(novice_rate) else 0
        expert_rate  = float(expert_rate[0]) if len(expert_rate) else 0
        multiplier   = round(novice_rate / expert_rate, 1) if expert_rate > 0 else 0
        worst_grp    = grp_stats.loc[grp_stats['Failure_Rate_pct'].idxmax(), 'Group']
        best_grp     = grp_stats.loc[grp_stats['Failure_Rate_pct'].idxmin(), 'Group']
        corr_val     = op_df['Operator_Experience_Yr'].corr(op_df['Failure'])

        # ── KPI cards ──────────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-val' style='color:#ef4444'>{novice_rate}%</div>
                <div class='metric-label'>Novice failure rate (0-2 yrs)</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-val' style='color:#22c55e'>{expert_rate}%</div>
                <div class='metric-label'>Expert failure rate (15+ yrs)</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-val' style='color:#f97316'>{multiplier}x</div>
                <div class='metric-label'>Novice vs Expert risk ratio</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            corr_color = '#ef4444' if corr_val < -0.1 else '#22c55e' if corr_val > 0.1 else '#3b82f6'
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-val' style='color:{corr_color}'>{corr_val:.3f}</div>
                <div class='metric-label'>Correlation (experience vs failure)</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Recommendation banner ───────────────────────────────────────
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);
                    border:1px solid #3b82f6;border-radius:12px;padding:18px 22px;margin:8px 0'>
            <div style='font-size:13px;font-weight:600;color:#93c5fd;margin-bottom:10px'>
                NALCO HR RECOMMENDATION — Based on operator analysis
            </div>
            <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px'>
                <div style='background:#0f172a;border-radius:8px;padding:12px'>
                    <div style='font-size:11px;color:#64748b;margin-bottom:4px'>ACTION 1</div>
                    <div style='font-size:12px;color:#e2e8f0'>
                        <b>Buddy system</b> — Pair all operators with
                        <b style="color:#22c55e">less than 2 years</b> experience
                        with a Senior (10+ yr) operator on HEMM equipment
                    </div>
                </div>
                <div style='background:#0f172a;border-radius:8px;padding:12px'>
                    <div style='font-size:11px;color:#64748b;margin-bottom:4px'>ACTION 2</div>
                    <div style='font-size:12px;color:#e2e8f0'>
                        <b>Training program</b> — Mandatory pre-operation
                        HEMM safety checklist for operators with
                        <b style="color:#f97316">less than 5 years</b> experience
                    </div>
                </div>
                <div style='background:#0f172a;border-radius:8px;padding:12px'>
                    <div style='font-size:11px;color:#64748b;margin-bottom:4px'>ACTION 3</div>
                    <div style='font-size:12px;color:#e2e8f0'>
                        <b>Night shift restriction</b> — Avoid assigning
                        <b style="color:#ef4444">novice operators (0-2 yr)</b>
                        to Night shift where failure risk is already highest
                    </div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        # ── Chart 1: Failure rate by experience group (bar) ────────────
        with col1:
            st.markdown("<div class='section-title'>Failure Rate by Experience Group</div>",
                        unsafe_allow_html=True)
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            fig1.patch.set_facecolor('#111827')
            ax1.set_facecolor('#111827')

            bar_colors = ['#ef4444' if v == grp_stats['Failure_Rate_pct'].max()
                          else '#22c55e' if v == grp_stats['Failure_Rate_pct'].min()
                          else '#3b82f6'
                          for v in grp_stats['Failure_Rate_pct']]
            bars1 = ax1.bar(grp_stats['Group'].astype(str),
                             grp_stats['Failure_Rate_pct'],
                             color=bar_colors, alpha=0.9, width=0.55)
            ax1.set_ylabel('Failure Rate (%)', color='#94a3b8', fontsize=10)
            ax1.tick_params(colors='#94a3b8', labelsize=8)
            ax1.set_title('Failure Rate by Operator Experience',
                           color='#e2e8f0', fontsize=11, fontweight='bold')
            for spine in ax1.spines.values():
                spine.set_color('#30363d')
            for bar, val in zip(bars1, grp_stats['Failure_Rate_pct']):
                ax1.text(bar.get_x()+bar.get_width()/2,
                         bar.get_height()+0.3,
                         str(val)+'%',
                         ha='center', color='#e2e8f0',
                         fontsize=9, fontweight='bold')
            # Annotate novice vs expert
            ax1.annotate(str(multiplier)+'x higher risk',
                          xy=(0, novice_rate),
                          xytext=(1.5, novice_rate+2),
                          color='#ef4444', fontsize=9, fontweight='bold',
                          arrowprops=dict(arrowstyle='->', color='#ef4444', lw=1.5))
            st.pyplot(fig1, use_container_width=True)
            plt.close()

        # ── Chart 2: Total failures per group (horizontal bar) ─────────
        with col2:
            st.markdown("<div class='section-title'>Total Failures per Experience Group</div>",
                        unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            fig2.patch.set_facecolor('#111827')
            ax2.set_facecolor('#111827')
            bar_colors2 = ['#ef4444' if v == grp_stats['Failures'].max()
                            else '#22c55e' if v == grp_stats['Failures'].min()
                            else '#3b82f6'
                            for v in grp_stats['Failures']]
            bars2 = ax2.barh(grp_stats['Group'].astype(str),
                              grp_stats['Failures'],
                              color=bar_colors2, alpha=0.9, height=0.55)
            ax2.set_xlabel('Number of Failures', color='#94a3b8', fontsize=10)
            ax2.tick_params(colors='#94a3b8', labelsize=8)
            ax2.set_title('Total Failures per Group',
                           color='#e2e8f0', fontsize=11, fontweight='bold')
            for spine in ax2.spines.values():
                spine.set_color('#30363d')
            for bar, row in zip(bars2, grp_stats.itertuples()):
                ax2.text(bar.get_width()+0.3,
                         bar.get_y()+bar.get_height()/2,
                         str(int(row.Failures)) + ' failures (' + str(row.Total) + ' total)',
                         va='center', color='#e2e8f0', fontsize=8)
            st.pyplot(fig2, use_container_width=True)
            plt.close()

        # ── Chart 3: Scatter plot — experience vs failure probability ──
        st.markdown("<div class='section-title'>Scatter Plot — Operator Experience vs Failure Probability</div>",
                    unsafe_allow_html=True)
        st.caption("Each dot = one equipment record. Color = equipment type. Rising dots on left = novice operators cause more failures.")

        # Merge with alert proba
        if 'Equipment_ID' in df.columns:
            proba_map = alerts_df.groupby('Equipment_ID')['Failure_Probability_%'].first()
            scatter_df = df[['Operator_Experience_Yr','Failure']].copy()
            if 'Equipment_ID' in df.columns:
                scatter_df['Equipment_ID'] = df['Equipment_ID']
                scatter_df['Failure_Prob'] = scatter_df['Equipment_ID'].map(proba_map).fillna(
                    df['Failure'] * 100)
            if 'Equipment_Type' in df.columns:
                scatter_df['Equipment_Type'] = df['Equipment_Type']
            elif 'Equipment_Type_enc' in df.columns:
                scatter_df['Equipment_Type'] = df['Equipment_Type_enc'].map(
                    {0:'Excavator',1:'Dumper',2:'Dozer',
                     3:'Grader',4:'Drill Rig',5:'Wheel Loader',6:'Scraper'})
        else:
            scatter_df = df[['Operator_Experience_Yr','Failure']].copy()
            scatter_df['Failure_Prob'] = df['Failure'] * 100
            scatter_df['Equipment_Type'] = 'Unknown'

        fig3, ax3 = plt.subplots(figsize=(13, 5))
        fig3.patch.set_facecolor('#111827')
        ax3.set_facecolor('#0f172a')

        eq_types_uniq = scatter_df['Equipment_Type'].unique() if 'Equipment_Type' in scatter_df.columns else ['Unknown']
        scatter_colors = ['#ef4444','#f97316','#eab308','#22c55e','#3b82f6','#8b5cf6','#ec4899']

        for i, eq_t in enumerate(eq_types_uniq):
            mask = scatter_df['Equipment_Type'] == eq_t
            ax3.scatter(
                scatter_df[mask]['Operator_Experience_Yr'],
                scatter_df[mask]['Failure_Prob'],
                alpha=0.35, s=18,
                color=scatter_colors[i % len(scatter_colors)],
                label=str(eq_t), edgecolors='none'
            )

        # Trend line
        valid = scatter_df.dropna(subset=['Operator_Experience_Yr','Failure_Prob'])
        if len(valid) > 10:
            z   = np.polyfit(valid['Operator_Experience_Yr'], valid['Failure_Prob'], 1)
            p   = np.poly1d(z)
            x_l = np.linspace(valid['Operator_Experience_Yr'].min(),
                               valid['Operator_Experience_Yr'].max(), 100)
            ax3.plot(x_l, p(x_l), '--', color='#ffffff', lw=2,
                      alpha=0.7, label='Trend line')

        # Danger zone shading for novice operators
        ax3.axvspan(0, 2, alpha=0.08, color='#ef4444', label='Novice zone (0-2 yr)')
        ax3.axvspan(2, 5, alpha=0.05, color='#f97316', label='Junior zone (2-5 yr)')

        ax3.set_xlabel('Operator Experience (Years)', color='#94a3b8', fontsize=11)
        ax3.set_ylabel('Failure Probability (%)', color='#94a3b8', fontsize=11)
        ax3.set_title('Operator Experience vs Failure Probability — Left = novice (higher risk), Right = expert (lower risk)', color='#e2e8f0', fontsize=11, fontweight='bold')
        ax3.tick_params(colors='#94a3b8')
        ax3.legend(facecolor='#1e293b', labelcolor='#e2e8f0',
                    fontsize=8, loc='upper right', ncol=2)
        for spine in ax3.spines.values():
            spine.set_color('#1e293b')
        st.pyplot(fig3, use_container_width=True)
        plt.close()

        # ── Chart 4: Experience vs failure by equipment type heatmap ──
        st.markdown("<div class='section-title'>Failure Rate Heatmap — Experience Group x Equipment Type</div>",
                    unsafe_allow_html=True)

        if 'Equipment_Type' in op_df.columns:
            pivot2 = op_df.pivot_table(
                values='Failure', index='Exp_Group',
                columns='Equipment_Type', aggfunc='mean') * 100
            pivot2 = pivot2.round(1)

            fig4, ax4 = plt.subplots(figsize=(13, 4))
            fig4.patch.set_facecolor('#111827')
            ax4.set_facecolor('#111827')
            im = ax4.imshow(pivot2.values, cmap='RdYlGn_r',
                             aspect='auto', vmin=0, vmax=pivot2.values.max())
            ax4.set_xticks(range(len(pivot2.columns)))
            ax4.set_xticklabels(pivot2.columns, color='#e2e8f0', fontsize=9)
            ax4.set_yticks(range(len(pivot2.index)))
            ax4.set_yticklabels([str(g) for g in pivot2.index],
                                  color='#e2e8f0', fontsize=9)
            ax4.set_title(
                'Failure Rate % — Experience Group x Equipment Type (Red=high, Green=low)',
                color='#e2e8f0', fontsize=11, fontweight='bold')
            for i in range(len(pivot2.index)):
                for j in range(len(pivot2.columns)):
                    val = pivot2.values[i][j]
                    if not np.isnan(val):
                        ax4.text(j, i, str(round(val,1))+'%',
                                  ha='center', va='center',
                                  color='white', fontsize=9, fontweight='bold')
            plt.colorbar(im, ax=ax4, label='Failure Rate (%)')
            plt.tight_layout()
            st.pyplot(fig4, use_container_width=True)
            plt.close()

        # ── Stats table ──────────────────────────────────────────────────
        st.markdown("<div class='section-title'>Experience Group Statistics</div>",
                    unsafe_allow_html=True)
        display_grp = grp_stats.copy()
        display_grp['Group'] = display_grp['Group'].astype(str)
        display_grp['Risk vs Expert'] = display_grp['Failure_Rate_pct'].apply(
            lambda x: str(round(x/expert_rate, 1))+'x higher' if expert_rate > 0 else 'N/A')
        display_grp['Recommendation'] = display_grp['Group'].apply(
            lambda g: 'BUDDY SYSTEM + Night restriction' if '0-2' in g
            else 'Mandatory checklist + mentoring' if '2-5' in g
            else 'Standard supervision' if '5-10' in g
            else 'Can mentor juniors' if '10-15' in g
            else 'Senior mentor — assign to critical machines')
        display_grp = display_grp[['Group','Total','Failures','Failure_Rate_pct',
                                    'Risk vs Expert','Recommendation']]
        display_grp.columns = ['Experience Group','Total Records','Failures',
                                'Failure Rate %','Risk vs Expert','Recommendation']
        st.dataframe(display_grp, use_container_width=True, hide_index=True)



# ══════════════════════════════════════════════════════════════════════
# PAGE: PREDICT SINGLE MACHINE
# ══════════════════════════════════════════════════════════════════════
elif page == "🎯 Predict Machine":
    st.markdown("# 🎯 Predict Single Machine")
    st.markdown("Enter live sensor readings for any machine and get an **instant prediction** — will it fail, when, and what to do.")
    st.markdown("---")

    # ── Sensor input form ─────────────────────────────────────────────
    st.markdown("<div class='section-title'>Step 1 — Enter Machine Details</div>", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        eq_type_input = st.selectbox("Equipment Type", 
            ["Dumper","Excavator","Dozer","Grader","Drill Rig","Wheel Loader","Scraper"])
        shift_input   = st.selectbox("Current Shift", ["Day","Afternoon","Night"])
        road_input    = st.selectbox("Road Condition", ["Good","Fair","Poor"])
    with col_b:
        op_exp        = st.number_input("Operator Experience (years)", 0.0, 40.0, 5.0, 0.5)
        op_hours      = st.number_input("Operating Hours (total)", 0, 50000, 5000, 100)
        load_cycles   = st.number_input("Load Cycles per Day", 0, 100, 20, 1)
    with col_c:
        days_since_pm = st.number_input("Days Since Last Maintenance", 0, 365, 30, 1)
        pm_interval   = st.number_input("PM Interval (hours)", 100, 1000, 250, 50)
        next_pm_due   = st.number_input("Next PM Due (hours)", 0, 1000, 180, 10)

    st.markdown("<div class='section-title'>Step 2 — Enter Sensor Readings</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        engine_temp   = st.slider("Engine Temperature (°C)",    50.0,  130.0,  85.0,  0.5)
        oil_pressure  = st.slider("Oil Pressure (bar)",          1.0,    7.0,   3.5,  0.1)
        vibration     = st.slider("Vibration (mm/s)",            0.0,    8.0,   2.5,  0.1)
        rpm_val       = st.slider("RPM",                       500.0, 3000.0,1800.0, 10.0)
    with c2:
        hydraulic_p   = st.slider("Hydraulic Pressure (bar)",  100.0,  320.0, 210.0,  5.0)
        exhaust_temp  = st.slider("Exhaust Temperature (°C)",  150.0,  550.0, 320.0,  5.0)
        fuel_consump  = st.slider("Fuel Consumption (L/hr)",    10.0,   60.0,  28.0,  0.5)
        battery_v     = st.slider("Battery Voltage (V)",        10.0,   16.0,  12.6,  0.1)
    with c3:
        tyre_psi      = st.slider("Tyre Pressure (PSI)",        60.0,  130.0,  95.0,  1.0)
        coolant_lvl   = st.slider("Coolant Level (0-1)",         0.0,    1.0,   0.8,  0.01)
        engine_life   = st.slider("Engine Life Remaining (%)",   0.0,  100.0,  60.0,  1.0)
        brake_life    = st.slider("Brake Life Remaining (%)",    0.0,  100.0,  55.0,  1.0)

    c4, c5 = st.columns(2)
    with c4:
        tyre_life     = st.slider("Tyre Life Remaining (%)",      0.0, 100.0, 65.0, 1.0)
        hydraulic_life= st.slider("Hydraulic Life Remaining (%)", 0.0, 100.0, 70.0, 1.0)
    with c5:
        battery_life  = st.slider("Battery Life Remaining (%)",   0.0, 100.0, 72.0, 1.0)
        brake_replace = st.selectbox("Brake Replacement Needed?", ["No","Yes"])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Predict button ────────────────────────────────────────────────
    predict_btn = st.button("🚀 Run Prediction", type="primary", use_container_width=True)

    if predict_btn:
        # Encode categorical inputs
        eq_enc   = {"Excavator":0,"Dumper":1,"Dozer":2,"Grader":3,
                    "Drill Rig":4,"Wheel Loader":5,"Scraper":6}.get(eq_type_input, 0)
        sh_enc   = {"Day":0,"Afternoon":1,"Night":2}.get(shift_input, 0)
        rd_enc   = {"Good":0,"Fair":1,"Poor":2}.get(road_input, 0)
        br_enc   = 1 if brake_replace == "Yes" else 0

        # Derived features
        temp_oil_ratio = engine_temp / max(oil_pressure, 0.1)
        vib_rpm_ratio  = vibration   / max(rpm_val, 1)

        # Status encodings (0=normal, 1=warning, 2=critical)
        def status_enc(val, low, high, danger_high):
            if val > danger_high: return 2
            elif val > high or val < low: return 1
            return 0

        eng_status  = status_enc(engine_temp,  70, 100, 105)
        oil_status  = status_enc(oil_pressure, 2.5, 5.0, 5.5)
        vib_status  = status_enc(vibration,    0,   4.5, 5.0)
        fuel_status = status_enc(fuel_consump, 15,  45,  52)
        tyre_status = status_enc(tyre_psi,     80, 110, 115)
        cool_status = 2 if coolant_lvl < 0.3 else 1 if coolant_lvl < 0.5 else 0
        bat_status  = status_enc(battery_v,    11.5, 14.5, 15.0)
        hyd_status  = status_enc(hydraulic_p,  150, 280, 290)
        exh_status  = status_enc(exhaust_temp, 200, 450, 500)
        rpm_status  = status_enc(rpm_val,      600, 2500, 2800)

        # Build input row matching FEATURE_COLS order
        input_data = {
            'Engine_Temp_C':                  engine_temp,
            'Oil_Pressure_bar':               oil_pressure,
            'Vibration_mms':                  vibration,
            'Fuel_Consumption_Lhr':           fuel_consump,
            'Tyre_Pressure_PSI':              tyre_psi,
            'Coolant_Level':                  coolant_lvl,
            'Battery_Voltage_V':              battery_v,
            'Hydraulic_Pressure_bar':         hydraulic_p,
            'Exhaust_Temp_C':                 exhaust_temp,
            'RPM':                            rpm_val,
            'Engine_Temp_C_Status_enc':       eng_status,
            'Oil_Pressure_bar_Status_enc':    oil_status,
            'Vibration_mms_Status_enc':       vib_status,
            'Fuel_Consumption_Lhr_Status_enc':fuel_status,
            'Tyre_Pressure_PSI_Status_enc':   tyre_status,
            'Coolant_Level_Status_enc':       cool_status,
            'Battery_Voltage_V_Status_enc':   bat_status,
            'Hydraulic_Pressure_bar_Status_enc': hyd_status,
            'Exhaust_Temp_C_Status_enc':      exh_status,
            'RPM_Status_enc':                 rpm_status,
            'Temp_Oil_Ratio':                 temp_oil_ratio,
            'Vib_RPM_Ratio':                  vib_rpm_ratio,
            'Operating_Hours':                op_hours,
            'Load_Cycles_per_Day':            load_cycles,
            'Operator_Experience_Yr':         op_exp,
            'Equipment_Type_enc':             eq_enc,
            'Shift_enc':                      sh_enc,
            'Road_Condition_enc':             rd_enc,
            'Year':                           pd.Timestamp.now().year,
            'Month':                          pd.Timestamp.now().month,
            'Day_of_Week':                    pd.Timestamp.now().dayofweek,
            'Engine_Condition_enc':           2 if engine_life > 60 else 1 if engine_life > 30 else 0,
            'Tyre_Condition_enc':             2 if tyre_life   > 60 else 1 if tyre_life   > 30 else 0,
            'Hydraulic_Condition_enc':        2 if hydraulic_life > 60 else 1 if hydraulic_life > 30 else 0,
            'Brake_Condition_enc':            2 if brake_life  > 60 else 1 if brake_life  > 30 else 0,
            'Electrical_Condition_enc':       2 if battery_life > 60 else 1 if battery_life > 30 else 0,
            'Fuel_System_Condition_enc':      1,
            'Transmission_Condition_enc':     1,
            'Cooling_System_Condition_enc':   2 if coolant_lvl > 0.6 else 1,
            'Engine_Life_Remaining_pct':      engine_life,
            'Tyre_Life_Remaining_pct':        tyre_life,
            'Hydraulic_Life_Remaining_pct':   hydraulic_life,
            'Battery_Life_Remaining_pct':     battery_life,
            'Brake_Life_Remaining_pct':       brake_life,
            'Days_Since_Last_Maintenance':    days_since_pm,
            'Last_Maintenance_Type_enc':      1,
            'PM_Interval_Hours':              pm_interval,
            'Next_PM_Due_Hours':              next_pm_due,
            'Maintenance_Team_enc':           0,
            'Brake_Replacement_Needed_enc':   br_enc,
        }

        # Build DataFrame aligned to feat_cols
        input_row = pd.DataFrame([{c: input_data.get(c, 0) for c in feat_cols}])

        # Run models
        fail_prob   = float(ens.predict_proba(input_row)[0][1]) * 100
        days_left   = max(0, int(reg.predict(input_row)[0]))
        alert_level = get_alert_level(fail_prob / 100)
        icon        = ICONS.get(alert_level, "⚪")
        color       = LEVEL_COLORS.get(alert_level, "#64748b")

        # ── Result banner ─────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Prediction Result</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#0f172a,#1e293b);
                    border:2px solid {color};border-radius:16px;
                    padding:28px 32px;margin:8px 0'>
            <div style='display:grid;grid-template-columns:2fr 1fr 1fr 1fr;gap:20px;align-items:center'>
                <div>
                    <div style='font-size:12px;color:#64748b;letter-spacing:.1em;
                                text-transform:uppercase;margin-bottom:6px'>Prediction Result</div>
                    <div style='font-size:2.2rem;font-weight:700;color:{color}'>{icon} {alert_level}</div>
                    <div style='font-size:13px;color:#94a3b8;margin-top:6px'>{eq_type_input} — {shift_input} Shift</div>
                </div>
                <div style='text-align:center'>
                    <div style='font-size:12px;color:#64748b;margin-bottom:4px'>Failure Probability</div>
                    <div style='font-size:2rem;font-weight:700;color:{color}'>{fail_prob:.1f}%</div>
                </div>
                <div style='text-align:center'>
                    <div style='font-size:12px;color:#64748b;margin-bottom:4px'>Days to Failure</div>
                    <div style='font-size:2rem;font-weight:700;color:#22c55e'>{days_left}d</div>
                </div>
                <div style='text-align:center'>
                    <div style='font-size:12px;color:#64748b;margin-bottom:4px'>Operator Risk</div>
                    <div style='font-size:1.4rem;font-weight:700;color:{"#ef4444" if op_exp < 2 else "#f97316" if op_exp < 5 else "#22c55e"}'>
                        {"HIGH" if op_exp < 2 else "MEDIUM" if op_exp < 5 else "LOW"}
                    </div>
                    <div style='font-size:11px;color:#64748b'>{op_exp} yrs exp</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        # ── Sensor health cards ───────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Sensor Status at Time of Prediction</div>",
                    unsafe_allow_html=True)

        sensor_checks = [
            ("Engine Temp",     engine_temp,  "°C",  70,  100, 105),
            ("Oil Pressure",    oil_pressure, "bar", 2.5, 5.0, 5.5),
            ("Vibration",       vibration,    "mm/s",0,   4.5, 5.0),
            ("Hydraulic Press", hydraulic_p,  "bar", 150, 280, 290),
            ("Battery Voltage", battery_v,    "V",   11.5,14.5,15.0),
            ("Exhaust Temp",    exhaust_temp, "°C",  200, 450, 500),
        ]
        sc_cols = st.columns(6)
        for i, (name, val, unit, lo, hi, danger) in enumerate(sensor_checks):
            is_dan = val > danger
            is_hi  = val > hi
            is_lo  = val < lo
            c      = "#ef4444" if is_dan else "#f97316" if (is_hi or is_lo) else "#22c55e"
            s      = "DANGER" if is_dan else "HIGH" if is_hi else "LOW" if is_lo else "OK"
            with sc_cols[i]:
                st.markdown(
                    "<div class='metric-card' style='border-color:" + c + "'>"
                    "<div class='metric-val' style='color:" + c + ";font-size:1.2rem'>"
                    + str(round(val,1)) + " " + unit + "</div>"
                    "<div class='metric-label'>" + name + "</div>"
                    "<div style='font-size:10px;color:" + c + ";margin-top:3px'>" + s + "</div>"
                    "</div>", unsafe_allow_html=True)

        # ── Component life cards ──────────────────────────────────────
        st.markdown("<div class='section-title'>Component Life at Time of Prediction</div>",
                    unsafe_allow_html=True)
        life_checks = [
            ("Engine",     engine_life),
            ("Brakes",     brake_life),
            ("Tyres",      tyre_life),
            ("Hydraulics", hydraulic_life),
            ("Battery",    battery_life),
        ]
        lc_cols = st.columns(5)
        for i, (name, val) in enumerate(life_checks):
            c = "#ef4444" if val < 15 else "#f97316" if val < 30 else "#22c55e"
            s = "REPLACE NOW" if val < 15 else "REPLACE SOON" if val < 30 else "OK"
            with lc_cols[i]:
                st.markdown(
                    "<div class='metric-card' style='border-color:" + c + "'>"
                    "<div class='metric-val' style='color:" + c + ";font-size:1.4rem'>"
                    + str(round(val)) + "%</div>"
                    "<div class='metric-label'>" + name + " Life</div>"
                    "<div style='font-size:10px;color:" + c + ";margin-top:3px'>" + s + "</div>"
                    "</div>", unsafe_allow_html=True)

        # ── Prescriptive diagnosis ────────────────────────────────────
        st.markdown("<div class='section-title'>Prescriptive Diagnosis</div>",
                    unsafe_allow_html=True)
        row_dict = {
            'Engine_Life_Remaining_pct':         engine_life,
            'Brake_Life_Remaining_pct':          brake_life,
            'Tyre_Life_Remaining_pct':           tyre_life,
            'Hydraulic_Life_Remaining_pct':      hydraulic_life,
            'Battery_Life_Remaining_pct':        battery_life,
            'Engine_Temp_C_Status_enc':          eng_status,
            'Oil_Pressure_bar_Status_enc':       oil_status,
            'Vibration_mms_Status_enc':          vib_status,
            'Hydraulic_Pressure_bar_Status_enc': hyd_status,
            'Battery_Voltage_V_Status_enc':      bat_status,
            'Fuel_System_Condition_enc':         1,
            'Brake_Condition_enc':               2 if brake_life>60 else 1 if brake_life>30 else 0,
            'Transmission_Condition_enc':        1,
            'Cooling_System_Condition_enc':      2 if coolant_lvl>0.6 else 1,
            'Coolant_Level_Status_enc':          cool_status,
            'Fuel_Consumption_Lhr_Status_enc':   fuel_status,
            'Brake_Replacement_Needed_enc':      br_enc,
            'Engine_Replacement_Needed_enc':     1 if engine_life < 15 else 0,
            'Tyre_Replacement_Needed_enc':       1 if tyre_life < 15 else 0,
            'Hydraulic_Replacement_Needed_enc':  1 if hydraulic_life < 15 else 0,
            'Electrical_Replacement_Needed_enc': 1 if battery_life < 15 else 0,
            'Days_Since_Last_Maintenance':       days_since_pm,
            'Electrical_Condition_enc':          2 if battery_life>60 else 1 if battery_life>30 else 0,
        }
        issues = diagnose_machine(row_dict)

        if not issues or (len(issues)==1 and issues[0][0]=="All Systems"):
            st.markdown("""<div class='alert-ok'>
                <b>🟢 All Systems Normal</b><br>
                <span style='color:#94a3b8'>No immediate action required. Continue regular monitoring.</span>
            </div>""", unsafe_allow_html=True)
        else:
            for comp, problem, action, sev in issues:
                css   = "alert-critical" if sev=="CRITICAL" else "alert-high" if sev=="HIGH" else "alert-medium"
                s_icon = "🔴" if sev=="CRITICAL" else "🟠" if sev=="HIGH" else "🟡"
                is_rep = "REPLACE" in action
                rep_badge = (" <span style='background:#ef4444;color:white;border-radius:3px;"
                             "padding:1px 7px;font-size:11px;margin-left:6px'>REPLACEMENT NEEDED</span>"
                             if is_rep else "")
                st.markdown(
                    "<div class='" + css + "'>"
                    "<b>" + s_icon + " [" + sev + "] " + comp + "</b>" + rep_badge + "<br>"
                    "<span style='color:#94a3b8;font-size:13px'>Problem: " + problem + "</span><br>"
                    "<span style='color:#e2e8f0;font-size:13px'>Action: <b>" + action + "</b></span>"
                    "</div>", unsafe_allow_html=True)

        # ── Save prediction to SQLite ──────────────────────────────────
        db_path = os.path.join(OUT, "hemm_alerts")
        if not os.path.exists(db_path):
            db_path = os.path.join(OUT, "hemm_alerts.db")
        try:
            conn_p = sqlite3.connect(db_path)
            pred_record = pd.DataFrame([{
                'run_timestamp':        pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Equipment_ID':         'MANUAL_INPUT',
                'Equipment_Type':       eq_type_input,
                'Failure_Probability':  round(fail_prob, 2),
                'Days_to_Failure':      days_left,
                'Alert_Level':          alert_level,
                'Shift':                shift_input,
                'Operator_Experience':  op_exp,
                'Critical_Issues':      sum(1 for _,_,_,s in issues if s=='CRITICAL'),
                'Needs_Replacement':    any('REPLACE' in a for _,_,a,_ in issues),
            }])
            pred_record.to_sql("manual_predictions", conn_p, if_exists="append", index=False)
            conn_p.close()
            st.success("Prediction saved to database (manual_predictions table)")
        except Exception as e:
            st.info("Note: Could not save to database — " + str(e))

        # ── Download prediction report ─────────────────────────────────
        report_txt = (
            "NALCO HEMM — SINGLE MACHINE PREDICTION REPORT\n"
            "=" * 50 + "\n"
            "Generated  : " + pd.Timestamp.now().strftime('%d %b %Y %H:%M') + "\n"
            "Equipment  : " + eq_type_input + "\n"
            "Shift      : " + shift_input + "\n"
            "Operator   : " + str(op_exp) + " years experience\n\n"
            "PREDICTION RESULT\n"
            "-" * 30 + "\n"
            "Alert Level        : " + alert_level + "\n"
            "Failure Probability: " + str(round(fail_prob,1)) + "%\n"
            "Days to Failure    : " + str(days_left) + " days\n\n"
            "SENSOR READINGS\n"
            "-" * 30 + "\n"
            "Engine Temp        : " + str(engine_temp) + " C\n"
            "Oil Pressure       : " + str(oil_pressure) + " bar\n"
            "Vibration          : " + str(vibration) + " mm/s\n"
            "Hydraulic Pressure : " + str(hydraulic_p) + " bar\n\n"
            "DIAGNOSIS\n"
            "-" * 30 + "\n"
        )
        for comp, problem, action, sev in issues:
            report_txt += "[" + sev + "] " + comp + " — " + action + "\n"

        st.download_button(
            label="Download Prediction Report",
            data=report_txt,
            file_name="hemm_single_prediction.txt",
            mime="text/plain"
        )


# ── Auto-refresh ──────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(30)
    st.rerun()
