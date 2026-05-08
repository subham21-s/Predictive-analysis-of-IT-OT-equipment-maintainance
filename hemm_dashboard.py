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

    cw  = compute_class_weight('balanced', classes=np.array([0,1]), y=yc_tr)
    rf  = RandomForestClassifier(n_estimators=200, max_depth=15,
                                  class_weight={0:cw[0],1:cw[1]}, random_state=42, n_jobs=-1)
    gb  = GradientBoostingClassifier(n_estimators=150, learning_rate=0.08,
                                      max_depth=5, random_state=42)
    ens = VotingClassifier(estimators=[('rf',rf),('gb',gb)], voting='soft', weights=[2,1])
    ens.fit(X_tr, yc_tr)

    reg = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    reg.fit(X_tr, yr_tr)

    yp  = ens.predict(X_te)
    ypr = ens.predict_proba(X_te)[:,1]
    metrics = {
        'auc':      round(roc_auc_score(yc_te, ypr), 4),
        'accuracy': round(accuracy_score(yc_te, yp), 4),
        'recall':   round(recall_score(yc_te, yp), 4),
        'f1':       round(f1_score(yc_te, yp), 4),
    }

    # Feature importance
    rf.fit(X_tr, yc_tr)
    fi = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=False)

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
    return df, X, feat_cols, metrics, fi, alerts_df, mtime

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
    page = st.radio("", ["🏠 Overview", "🚨 Alerts", "📊 ML Models", "🔧 Equipment Detail"])
    st.markdown("---")
    st.markdown(f"<small style='color:#475569'>Last checked: {time.strftime('%H:%M:%S')}</small>",
                unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────
mtime = get_file_mtime(data_path)
with st.spinner("🔄 Loading data and training models..."):
    df, X, feat_cols, metrics, fi, alerts_df, _ = load_and_train(data_path, mtime)

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

    # KPI Row
    crit_count   = len(alerts_df[alerts_df['Issue_Severity']=='CRITICAL']['Equipment_ID'].unique())
    high_count   = len(alerts_df[alerts_df['Alert_Level']=='HIGH']['Equipment_ID'].unique())
    repl_count   = len(alerts_df[alerts_df['Recommended_Action'].str.contains('REPLACE',na=False)]['Equipment_ID'].unique())
    total_eq     = alerts_df['Equipment_ID'].nunique()
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
        lvl_counts = alerts_df.groupby('Alert_Level')['Equipment_ID'].nunique()
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

# ══════════════════════════════════════════════════════════════════════
# PAGE: ALERTS
# ══════════════════════════════════════════════════════════════════════
elif page == "🚨 Alerts":
    st.markdown("# 🚨 Equipment Alerts")
    st.markdown(f"Showing **{len(filtered_alerts['Equipment_ID'].unique())} machines** matching filters")
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
        top15 = fi.head(15)
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
        proba_all = alerts_df.groupby('Equipment_ID')['Failure_Probability_%'].first()
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

    eq_list = sorted(alerts_df['Equipment_ID'].unique().tolist())
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

        # Sensor gauges
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-title'>Sensor Readings for {selected_eq}</div>", unsafe_allow_html=True)

        sensor_raw = ['Engine_Temp_C','Oil_Pressure_bar','Vibration_mms',
                      'Tyre_Pressure_PSI','Battery_Voltage_V','Hydraulic_Pressure_bar']
        sensor_vals = {}
        eq_idx = df.index[df['Equipment_ID'] == selected_eq].tolist() if 'Equipment_ID' in df.columns else []
        if eq_idx:
            eq_row_data = df.iloc[eq_idx[0]]
            for s in sensor_raw:
                if s in df.columns:
                    sensor_vals[s] = eq_row_data[s]

        if sensor_vals:
            cols = st.columns(len(sensor_vals))
            for i, (s, v) in enumerate(sensor_vals.items()):
                with cols[i]:
                    st.metric(label=s.replace('_',' '), value=f"{v:.1f}")

# ── Auto-refresh ──────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(30)
    st.rerun()
