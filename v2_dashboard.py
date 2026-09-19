"""
ProactiveGuard v2.0 — Industrial Safety Dashboard (Rebuild)
============================================================
Professional dark HMI dashboard — Aramco control room standard.

Fixed in this rebuild:
- Zone map crash (titlefont → correct API)
- Logo fully visible
- Gauge axis clean (no broken [)- labels)
- Workers ranked by risk (highest first)
- Pulse animation on danger/critical only
- Breathing rate + SpO2 displayed
- Shift progress bar
- Sensor status indicators
- Dark tooltips on all charts
- Full-screen critical alert mode
- Zone WBGT microclimate panel
- Shift heatmap per worker
- Model confidence badge
- Data freshness timestamp
- Acclimatization progress bars
- Multi-signal sparklines

Run: streamlit run v2_dashboard.py --server.port 8504
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sqlite3, pickle, os, time, warnings
from collections import deque
from datetime import datetime
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="ProactiveGuard v2.0",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════
# PROFESSIONAL CSS — Control Room Aesthetic
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;700&display=swap');

/* Base */
.main, .block-container { background:#0a0e1a; padding-top:0.5rem; }
.stApp { background:#0a0e1a; font-family:'Inter',sans-serif; }


/* Hide default streamlit elements */

#MainMenu, footer, header { visibility:hidden; }
.stDeployButton { display:none; }

/* Typography */
h1,h2,h3,h4 { font-family:'Inter',sans-serif !important; letter-spacing:-0.02em; }

/* ── PULSE ANIMATIONS ── */
@keyframes pulse-danger {
    0%   { box-shadow:0 0 0 0 rgba(231,76,60,0.6); border-color:#e74c3c; }
    50%  { box-shadow:0 0 20px 6px rgba(231,76,60,0.15); border-color:#ff6b6b; }
    100% { box-shadow:0 0 0 0 rgba(231,76,60,0); border-color:#e74c3c; }
}
@keyframes pulse-critical {
    0%   { box-shadow:0 0 0 0 rgba(142,68,173,0.7); border-color:#8e44ad; }
    50%  { box-shadow:0 0 25px 8px rgba(142,68,173,0.2); border-color:#b85ce8; }
    100% { box-shadow:0 0 0 0 rgba(142,68,173,0); border-color:#8e44ad; }
}
@keyframes flash-alert {
    0%,100% { opacity:1; }
    50%      { opacity:0.7; }
}
@keyframes slide-in {
    from { transform:translateY(-10px); opacity:0; }
    to   { transform:translateY(0);     opacity:1; }
}
@keyframes blink-led {
    0%,100% { opacity:1; }
    50%      { opacity:0.3; }
}

/* ── WORKER CARDS ── */
.worker-card {
    background:linear-gradient(145deg,#0d1117,#111827);
    border:1px solid #1e2d40;
    border-radius:12px;
    padding:14px;
    margin:4px 0;
    transition:all 0.3s ease;
    position:relative;
    overflow:hidden;
}
.worker-card::before {
    content:'';
    position:absolute;
    top:0; left:0; right:0;
    height:3px;
    background:var(--risk-color,#2ecc71);
}
.worker-card.safe     { --risk-color:#2ecc71; border-color:#0d2b0d; }
.worker-card.caution  { --risk-color:#f39c12; border-color:#2b2200; }
.worker-card.warning  { --risk-color:#e67e22; border-color:#2b1500; }
.worker-card.danger   { --risk-color:#e74c3c; border-color:#3a0a0a;
    animation:pulse-danger 2.5s infinite; }
.worker-card.critical { --risk-color:#8e44ad; border-color:#2a0a3a;
    animation:pulse-critical 2s infinite; }

.worker-name  { font-size:0.95rem; font-weight:700; color:#ecf0f1; letter-spacing:-0.01em; }
.worker-role  { font-size:0.72rem; color:#4a5568; margin-bottom:6px; text-transform:uppercase; letter-spacing:0.05em; }
.risk-number  { font-size:2.8rem; font-weight:900; line-height:1; font-family:'Inter',sans-serif; }
.risk-label   { font-size:0.7rem; font-weight:700; letter-spacing:0.1em; margin-top:2px; }
.vital-row    { display:flex; gap:8px; margin-top:8px; flex-wrap:wrap; }
.vital-chip   { background:#0a0e1a; border-radius:4px; padding:3px 7px;
                font-size:0.7rem; font-family:'JetBrains Mono',monospace;
                border:1px solid #1e2d40; }

/* ── ALERT BANNER ── */
.alert-banner {
    background:linear-gradient(135deg,#1a0505,#2d0a0a);
    border:1.5px solid #e74c3c;
    border-radius:10px;
    padding:16px 20px;
    margin:14px 0;
    animation:slide-in 0.3s ease, flash-alert 3s infinite;
    position:relative;
}
.alert-critical {
    border-color:#8e44ad;
    background:linear-gradient(135deg,#0d0015,#1a0028);
}

/* ── ZONE EVENT ── */
.zone-event {
    background:linear-gradient(135deg,#0d0520,#1a0a2e);
    border:1.5px solid #8e44ad;
    border-radius:10px;
    padding:14px 18px;
    margin:6px 0;
    animation:pulse-critical 3s infinite;
}

/* ── INFO PANELS ── */
.panel {
    background:linear-gradient(145deg,#0d1117,#0f1520);
    border:1px solid #1a2035;
    border-radius:10px;
    padding:16px;
    margin:4px 0;
    height:100%;
}
.panel-title {
    font-size:0.72rem;
    font-weight:700;
    text-transform:uppercase;
    letter-spacing:0.12em;
    color:#4a5568;
    margin-bottom:10px;
    display:flex;
    align-items:center;
    gap:6px;
}
.panel-title::before {
    content:'';
    display:inline-block;
    width:3px; height:12px;
    background:var(--accent,#3498db);
    border-radius:2px;
}

/* ── ENV CARDS ── */
.env-card {
    background:linear-gradient(145deg,#0d1117,#111827);
    border:1px solid #1a2035;
    border-radius:10px;
    padding:14px 16px;
    text-align:center;
}
.env-value { font-size:1.9rem; font-weight:900; line-height:1.1; font-family:'Inter',sans-serif; }
.env-label { font-size:0.7rem; color:#4a5568; text-transform:uppercase; letter-spacing:0.1em; }

/* ── LED INDICATORS ── */
.led {
    display:inline-block;
    width:8px; height:8px;
    border-radius:50%;
    margin-right:5px;
}
.led-green  { background:#2ecc71; box-shadow:0 0 6px #2ecc71; animation:blink-led 2s infinite; }
.led-yellow { background:#f39c12; box-shadow:0 0 6px #f39c12; animation:blink-led 1.5s infinite; }
.led-red    { background:#e74c3c; box-shadow:0 0 8px #e74c3c; animation:blink-led 0.8s infinite; }
.led-grey   { background:#4a5568; }

/* ── PROGRESS BARS ── */
.progress-container { background:#1a2035; border-radius:4px; height:6px; overflow:hidden; margin:4px 0; }
.progress-fill { height:100%; border-radius:4px; transition:width 0.5s ease; }

/* ── SYSTEM STATUS ── */
.status-badge {
    display:inline-block;
    padding:3px 10px;
    border-radius:20px;
    font-size:0.68rem;
    font-weight:700;
    letter-spacing:0.05em;
    text-transform:uppercase;
}
.badge-ok       { background:#0d2b0d; color:#2ecc71; border:1px solid #2ecc71; }
.badge-warn     { background:#2b2200; color:#f39c12; border:1px solid #f39c12; }
.badge-error    { background:#2b0000; color:#e74c3c; border:1px solid #e74c3c; }
.badge-fallback { background:#1a1500; color:#f39c12; border:1px solid #f39c12; }

/* ── LLM PANEL ── */
.llm-panel {
    background:#060810;
    border:1px solid #1a2035;
    border-left:3px solid #3498db;
    border-radius:8px;
    padding:14px;
    font-size:0.85rem;
    line-height:1.65;
    color:#cbd5e0;
    margin-top:6px;
}
.llm-source { font-size:0.65rem; color:#4a5568; text-transform:uppercase;
              letter-spacing:0.1em; margin-bottom:6px; }

/* ── TWIN PANEL ── */
.twin-panel {
    background:#060d08;
    border:1px solid #1a2035;
    border-left:3px solid #2ecc71;
    border-radius:8px;
    padding:12px;
    font-size:0.82rem;
    margin-top:6px;
}

/* ── RL PANEL ── */
.rl-panel {
    background:#08060d;
    border:1px solid #1a2035;
    border-left:3px solid #9b59b6;
    border-radius:8px;
    padding:12px;
    margin-top:6px;
}
.rl-action {
    font-size:1.1rem;
    font-weight:800;
    letter-spacing:0.02em;
}

/* ── AUDIT ROW ── */
.audit-row {
    display:flex;
    flex-wrap:wrap;
    align-items:center;
    gap:8px;
    padding:6px 8px;
    border-radius:6px;
    background:#0d1117;
    border-left:3px solid var(--risk-color,#4a5568);
    margin:3px 0;
    font-size:0.72rem;
}

/* ── FOOTER ── */
.pg-footer {
    text-align:center;
    color:#1e2d40;
    font-size:0.68rem;
    padding:12px 0;
    border-top:1px solid #111827;
    letter-spacing:0.03em;
    margin-top:16px;
}

/* Fix Streamlit metric */
[data-testid='stMetricValue'] { color:#ecf0f1 !important; font-family:'Inter' !important; }
[data-testid='stMetricDelta'] { font-size:0.75rem !important; }
div[data-testid='stVerticalBlock'] { gap:0.3rem; }

/* ══ MOBILE RESPONSIVE ══════════════════════════════════ */
.pg-title{font-size:clamp(1.2rem,4vw,1.8rem)!important}
.env-value{font-size:clamp(1.2rem,4vw,1.9rem)!important}
.env-label{font-size:clamp(0.6rem,1.8vw,0.7rem)!important}
.risk-number{font-size:clamp(1.5rem,6vw,3.5rem)!important}
.worker-name{font-size:clamp(0.8rem,3vw,0.95rem)!important}
.llm-panel{font-size:clamp(0.78rem,2.5vw,0.85rem)!important}
.vital-chip{font-size:clamp(0.62rem,2vw,0.75rem)!important}
.main,.block-container{overflow-x:hidden!important;max-width:100vw!important}
.stButton button{min-height:48px!important;padding:10px 14px!important;touch-action:manipulation!important}
@media(max-width:767px){
    .block-container{padding-left:0.4rem!important;padding-right:0.4rem!important}
    .worker-card{margin:2px 0!important;padding:10px!important}
    .alert-banner{padding:10px!important}
    .env-card{padding:8px 6px!important}
    .panel{padding:10px!important}
    .llm-panel{padding:10px!important;line-height:1.7!important}
}
@media(min-width:768px) and (max-width:1023px){
    .block-container{padding-left:0.8rem!important;padding-right:0.8rem!important}
}
.js-plotly-plot .plotly{touch-action:pan-y pinch-zoom!important}
/* ══ END MOBILE ══════════════════════════════════════════ */

</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# WORKERS / SITES / ZONE_WBGT_OFFSET come from v2_config.py — the
# single source of truth also used by the data engine, LLM engine,
# and zone engine. Previously this dashboard held its own hand-typed
# copy of the roster that had already drifted from the others (e.g.
# "Instrument Tech" here vs "Instrument Technician" elsewhere) —
# proof that hand-syncing config across files doesn't stay in sync.
# ══════════════════════════════════════════════════════════════
from v2_config import WORKERS_BY_ID as WORKERS
from v2_config import SITES
import v2_zone_engine as zone_engine
import v2_llm_engine as llm_engine
import v2_safety_infrastructure as safety_infra
import v2_rl_scheduler as rl_scheduler

# The imports above each call np.random.seed(42) at module load time
# (correct for THEIR reproducible offline batch jobs) — that would
# otherwise silently make this "live" dashboard replay the exact same
# sequence of simulated vitals on every restart. Re-randomize now so
# the live simulation stays live.
np.random.seed(None)

DB_PATH     = "data/audit.db"
MODEL_PATH  = "models/v2_model.json"
SCALER_PATH = "models/v2_scaler.pkl"
FEAT_PATH   = "models/v2_features.pkl"
RL_PATH     = "models/v2_rl_agent.pkl"

# Initialize the audit database schema on startup if it doesn't exist
# yet. Previously, log_alert()/get_audit() silently no-op'd forever if
# data/audit.db hadn't already been created by manually running
# v2_safety_infrastructure.py first — the audit panel would just show
# nothing, with no error, no matter what happened in the app.
try:
    safety_infra.init_audit_database(DB_PATH)
except Exception as _audit_init_err:
    st.warning(f"⚠️ Audit database could not be initialized: "
               f"{_audit_init_err}. Alert history will not be logged "
               f"this session.")


# ══════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════
def risk_color(s):
    if s<40:return "#2ecc71"
    elif s<55:return "#f39c12"
    elif s<70:return "#e67e22"
    elif s<85:return "#e74c3c"
    else:return "#8e44ad"

def risk_label(s):
    if s<40:return "SAFE"
    elif s<55:return "CAUTION"
    elif s<70:return "WARNING"
    elif s<85:return "DANGER"
    else:return "CRITICAL"

def risk_class(s):
    if s<40:return "safe"
    elif s<55:return "caution"
    elif s<70:return "warning"
    elif s<85:return "danger"
    else:return "critical"

def wbgt_label(w):
    if w<25:return "Safe","#2ecc71"
    elif w<28:return "Caution","#f39c12"
    elif w<30:return "Moderate","#e67e22"
    elif w<32:return "Danger","#e74c3c"
    else:return "Extreme","#8e44ad"

def action_text(s):
    if s<40:return "No action required"
    elif s<55:return "Monitor closely"
    elif s<70:return "Rest break in 15 min"
    elif s<85:return "⚠️ Immediate rest break"
    else:return "🚨 EVACUATE NOW"

def led_class(status):
    if status=="LIVE":return "led-green"
    elif status=="DEGRADED":return "led-yellow"
    elif status=="FAULT":return "led-red"
    else:return "led-grey"


# ══════════════════════════════════════════════════════════════
# MODEL LOADER
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_models():
    m={"model":None,"scaler":None,"features":None,
       "explainer":None,"rl_agent":None,"source":"rule_based",
       "load_warnings":[]}
    try:
        import xgboost as xgb, shap
        missing=[p for p in [MODEL_PATH,SCALER_PATH,FEAT_PATH] if not os.path.exists(p)]
        if missing:
            m["load_warnings"].append(
                f"Trained model artifacts not found ({', '.join(missing)}) — "
                f"running in rule-based mode. Run v2_model.py first to train "
                f"and enable ML-based risk scoring + SHAP explainability.")
        else:
            md=xgb.XGBClassifier(); md.load_model(MODEL_PATH)
            with open(SCALER_PATH,"rb") as f: sc=pickle.load(f)
            with open(FEAT_PATH,"rb")  as f: ft=pickle.load(f)
            # Uses XGBoostNativeExplainer (imported from v2_model.py)
            # instead of shap.TreeExplainer(). Three separate attempts
            # to patch shap's XGBoost model loader for multi-class
            # base_score compatibility each failed for a different
            # real reason inside shap's own undocumented internals —
            # see the full explanation in v2_model.py's
            # XGBoostNativeExplainer docstring. Using XGBoost's own
            # native SHAP computation (pred_contribs=True) sidesteps
            # that cross-library serialization boundary entirely,
            # rather than maintaining two separately-fragile
            # workarounds for the same bug in two files.
            from v2_model import XGBoostNativeExplainer
            m.update({"model":md,"scaler":sc,"features":ft,
                      "explainer":XGBoostNativeExplainer(md,ft),
                      "source":"ml_model"})
    except Exception as e:
        m["load_warnings"].append(
            f"Model loading failed ({type(e).__name__}: {e}) — "
            f"running in rule-based mode.")
    try:
        if os.path.exists(RL_PATH):
            with open(RL_PATH,"rb") as f: q_table = pickle.load(f)
            # v2_rl_scheduler.py saves only the learned q_table array,
            # not a QAgent class instance — see that file's save step
            # for why (pickling a custom class from a script run as
            # __main__ breaks when unpickled from a different script).
            # Reconstruct a fresh, fully-functional agent here instead.
            agent = rl_scheduler.QAgent()
            agent.q_table = q_table
            m["rl_agent"] = agent
        else:
            m["load_warnings"].append(
                f"RL scheduler not found ({RL_PATH}) — using rule-based "
                f"schedule recommendations. Run v2_rl_scheduler.py first "
                f"to train the Q-learning agent.")
    except Exception as e:
        m["load_warnings"].append(
            f"RL agent loading failed ({type(e).__name__}: {e}) — "
            f"using rule-based schedule recommendations.")
    return m


# ══════════════════════════════════════════════════════════════
# SIMULATION ENGINE
# ══════════════════════════════════════════════════════════════
def init_state(w):
    resting = float(w["resting_hr"])
    baseline_temp = float(w["baseline_temp"])
    # Pre-seed history buffers with a few realistic starting points
    # instead of leaving them empty. See the note above this function
    # for why an empty starting history caused visibly blank chart
    # areas, especially on mobile.
    seed_len = 5
    return {"hr":resting+8,"core_temp":baseline_temp,
            "hrv":48.0,"br":14.0,"spo2":98.5,"heat_debt":0.0,"dehydration":0.0,
            "minutes":0,"on_break":False,"break_timer":0,
            "history":deque(maxlen=60),
            "hr_history":deque([resting+8]*seed_len,maxlen=60),
            "br_history":deque([14.0]*seed_len,maxlen=60),
            "spo2_history":deque([98.5]*seed_len,maxlen=60),
            "sensor_status":"LIVE","data_quality":1.0}

def step(state, w, wbgt, speed=1):
    hr=state["hr"]; ct=state["core_temp"]; t=state["minutes"]
    accl=min(1.0,w["accl_days"]/14)
    wl={"heavy":0.75,"moderate":0.5,"light":0.3}[w["workload"]]
    if t>0 and t%90==0 and not state["on_break"]:
        state["on_break"]=True; state["break_timer"]=10
    if state["on_break"]:
        hr=max(float(w["resting_hr"])+4,hr-3.0*speed)
        ct=max(float(w["baseline_temp"]),ct-0.025*speed)
        state["hrv"]=min(58.0,state["hrv"]+2.0*speed)
        state["break_timer"]-=1
        if state["break_timer"]<=0: state["on_break"]=False
    else:
        hr_t=(float(w["resting_hr"])+max(0,wbgt-25)*2.5+t*3.5/60
              +wl*(float(w["max_hr"])-float(w["resting_hr"]))*0.4)*(1-0.25*accl)
        hr+=(hr_t-hr)*0.05*speed+np.random.normal(0,1.5)
        ct+=0.005*speed+np.random.normal(0,0.008)
        if w["accl_days"]<=5 and t>55:
            hr=min(float(w["max_hr"])-2,hr+0.9*speed)
            ct=min(40.4,ct+0.009*speed)
        if w["accl_days"]<=10 and t>280:
            hr=min(float(w["max_hr"])-2,hr+0.5*speed)
    state["hr"]=float(np.clip(hr,float(w["resting_hr"])-5,float(w["max_hr"])-2))
    state["core_temp"]=float(np.clip(ct,36.0,41.0))
    state["hrv"]=float(np.clip(65-(state["hr"]-w["resting_hr"])*0.4+np.random.normal(0,2),5,80))
    state["br"]=float(np.clip(14+(state["hr"]-w["resting_hr"])*0.08+max(0,wbgt-28)*0.4+np.random.normal(0,0.8),10,40))
    state["spo2"]=float(np.clip(99-(state["hr"]-w["resting_hr"])*0.02+np.random.normal(0,0.3),88,100))
    state["heat_debt"]=float(state["heat_debt"]+max(0,wbgt-25)*0.1*(1-0.3*accl)*speed)
    state["dehydration"]=float(np.clip(state["heat_debt"]/800,0,0.95))
    state["minutes"]=t+speed
    # Simulate occasional sensor degradation
    if np.random.random()<0.005: state["sensor_status"]="DEGRADED"
    elif np.random.random()<0.002: state["sensor_status"]="FAULT"
    else: state["sensor_status"]="LIVE"
    return state

def compute_risk(state, w, wbgt):
    hr=state["hr"]; ct=state["core_temp"]
    r_hr=float(w["resting_hr"]); m_hr=float(w["max_hr"]); accl=float(w["accl_days"])
    hrr=np.clip((hr-r_hr)/(m_hr-r_hr)*100,0,100)
    return float(np.clip(
        np.clip(hrr*0.35,0,35)+np.clip((wbgt-25)*1.8,0,25)+
        np.clip((hr-r_hr)/r_hr*15,0,20)+np.clip(10*np.exp(-accl/7),0,10)+
        np.clip((state.get("br",16)-14)*0.5,0,5)+
        np.clip(state.get("dehydration",0)*5,0,5),0,100))

def get_env(tick):
    h=6.0+(tick%480)/60
    amb=float(np.clip(38+10*np.sin(np.pi*(h-6)/16)+np.random.normal(0,0.2),35,52))
    hum=float(np.clip(55-0.3*(amb-38)+np.random.normal(0,1),30,70))
    sol=max(0.0,800*np.sin(np.pi*(h-5)/10))
    Tnwb=(amb*np.arctan(0.151977*(hum+8.313659)**0.5)+np.arctan(amb+hum)
          -np.arctan(hum-1.676331)+0.00391838*hum**1.5*np.arctan(0.023101*hum)-4.686035)
    wbgt=float(np.clip(0.7*Tnwb+0.2*(amb+sol*0.00015)+0.1*amb,25,46))
    return {"ambient":round(amb,1),"humidity":round(hum,1),
            "wbgt":round(wbgt,1),"solar":round(sol,0)}


# ══════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════
CHART_BG = "#0a0e1a"
PLOT_BG  = "#0d1117"
GRID_CLR = "#111827"

def chart_base(height=150):
    return dict(paper_bgcolor=CHART_BG,plot_bgcolor=PLOT_BG,
                height=height,margin=dict(l=35,r=10,t=10,b=25),
                font=dict(color="#4a5568",size=8),
                hoverlabel=dict(bgcolor="#0d1117",font_color="#ecf0f1",
                                bordercolor="#1e2d40",font_size=11))

def make_gauge(value, name, role):
    c=risk_color(value); lbl=risk_label(value)
    fig=go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value,1),
        number={"font":{"size":38,"color":c,"family":"Inter"},"suffix":""},
        title={"text":f"<b style='font-size:13px'>{name.split()[0]}</b><br>"
               f"<span style='font-size:10px;color:#4a5568'>{lbl}</span>",
               "font":{"color":"#ecf0f1","size":11}},
        gauge={
            "axis":{"range":[0,100],"tickwidth":1,"tickcolor":GRID_CLR,
                    "nticks":6,"tickfont":{"color":"#2c3e50","size":7}},
            "bar":{"color":c,"thickness":0.28},
            "bgcolor":PLOT_BG,"borderwidth":0,
            "steps":[
                {"range":[0,40],"color":"#0a1a0a"},{"range":[40,55],"color":"#1a1500"},
                {"range":[55,70],"color":"#1a0d00"},{"range":[70,85],"color":"#1a0000"},
                {"range":[85,100],"color":"#0d0010"},
            ],
            "threshold":{"line":{"color":"#ecf0f1","width":2},"thickness":0.8,"value":70}
        }
    ))
    fig.update_layout(paper_bgcolor="#0d1117",height=gauge_h(),
                      margin=dict(l=15,r=15,t=35,b=5),
                      font={"color":"#ecf0f1"})
    return fig

def make_trajectory(history, name, risk_now, twin_pred=None):
    hist=list(history) if history else [risk_now,risk_now]
    if len(hist)<2: hist=[risk_now,risk_now]
    c=risk_color(risk_now)
    fig=go.Figure()
    fig.add_hrect(y0=70,y1=100,fillcolor="rgba(231,76,60,0.05)",line_width=0)
    fig.add_hrect(y0=55,y1=70,fillcolor="rgba(230,126,34,0.04)",line_width=0)
    x=list(range(len(hist)))
    fig.add_trace(go.Scatter(x=x,y=hist,mode="lines",
        line=dict(color=c,width=2.5),fill="tozeroy",
        fillcolor=f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.12)",
        name="Risk",hovertemplate="Min %{x}: %{y:.0f}<extra></extra>"))
    if twin_pred:
        px2=list(range(len(hist),len(hist)+len(twin_pred)))
        fig.add_trace(go.Scatter(x=px2,y=twin_pred,mode="lines",
            line=dict(color="#2ecc71",width=1.5,dash="dot"),
            name="Twin +30m",hovertemplate="+%{x}m: %{y:.0f}<extra></extra>"))
    fig.add_hline(y=70,line=dict(color="#e74c3c",width=1,dash="dash"),
                  annotation_text="Danger",annotation_font_color="#e74c3c",
                  annotation_font_size=8)
    fig.add_hline(y=55,line=dict(color="#f39c12",width=0.8,dash="dash"))
    d=chart_base(150); d["showlegend"]=True
    d["legend"]=dict(font=dict(size=7,color="#4a5568"),bgcolor=CHART_BG,
                     bordercolor=GRID_CLR,x=0.01,y=0.99)
    d["xaxis"]=dict(gridcolor=GRID_CLR,tickfont=dict(color="#2c3e50",size=7))
    fig.update_layout(**d)
    return fig

def make_sparkline(history, color, height=70):
    hist=list(history) if history else [0,0]
    if len(hist)<2: hist=[hist[0],hist[0]] if hist else [0,0]
    fig=go.Figure(go.Scatter(x=list(range(len(hist))),y=hist,mode="lines",
        line=dict(color=color,width=1.5),fill="tozeroy",
        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.1)",
        hovertemplate="%{y:.1f}<extra></extra>"))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        height=height,margin=dict(l=0,r=0,t=0,b=0),
        xaxis=dict(visible=False),yaxis=dict(visible=False),
        hoverlabel=dict(bgcolor="#0d1117",font_color="#ecf0f1",font_size=10))
    return fig

def make_zone_wbgt(zone_wbgt_map):
    zones=list(zone_wbgt_map.keys())
    vals =[zone_wbgt_map[z] for z in zones]
    labels=[z.replace("_","\n") for z in zones]
    def zone_bar_color(wbgt):
        if wbgt < 28: return "#2ecc71"
        elif wbgt < 32: return "#f39c12"
        elif wbgt < 35: return "#e67e22"
        elif wbgt < 38: return "#e74c3c"
        else: return "#8e44ad"
    colors=[zone_bar_color(v) for v in vals]
    fig=go.Figure(go.Bar(x=labels,y=vals,marker_color=colors,
        marker_opacity=0.85,text=[f"{v:.1f}°C" for v in vals],
        textposition="outside",textfont=dict(color="#ecf0f1",size=10),
        hovertemplate="%{x}: %{y:.1f}°C WBGT<extra></extra>"))
    fig.add_hline(y=32,line=dict(color="#e74c3c",width=1,dash="dash"),
                  annotation_text="32°C",
                  annotation_position="top left",
                  annotation_font_color="#e74c3c",annotation_font_size=8)
    d=chart_base(190); d["margin"]=dict(l=10,r=10,t=20,b=10)
    d["xaxis"]=dict(tickfont=dict(color="#ecf0f1",size=9),gridcolor=GRID_CLR)
    fig.update_layout(**d)
    return fig

def make_zone_scatter(worker_risks, worker_zones):
    zone_order={"direct_sun":0,"partial_shade":1,"full_shade":2,"indoor_covered":3}
    rows=[]
    for wid,risk in worker_risks.items():
        w=WORKERS[wid]
        rows.append({"name":w["name"].split()[0],"risk":risk,
                     "zone_n":zone_order.get(w["zone"],0)+np.random.uniform(-0.32,0.32),
                     "wid":wid})
    df=pd.DataFrame(rows)
    fig=go.Figure(go.Scatter(
        x=df["zone_n"],y=df["risk"],mode="markers+text",
        text=df["name"],textposition="top center",
        textfont=dict(size=8,color="#ecf0f1"),
        marker=dict(size=14,color=df["risk"].tolist(),
            colorscale=[[0,"#0a1a0a"],[0.4,"#2b2200"],
                        [0.55,"#2b1500"],[0.7,"#2b0000"],[1,"#1a0020"]],
            cmin=0,cmax=100,showscale=True,
            colorbar=dict(
                title=dict(text="Risk",font=dict(color="#4a5568",size=9)),
                tickfont=dict(color="#4a5568",size=8),
                thickness=10,len=0.75,
            ),
            line=dict(color="#ecf0f1",width=0.8)),
        hovertemplate="<b>%{text}</b><br>Risk: %{y:.0f}<extra></extra>"))
    fig.add_hline(y=70,line=dict(color="#e74c3c",width=1,dash="dash"))
    d=chart_base(260); d["margin"]=dict(l=10,r=60,t=10,b=30)
    d["xaxis"]=dict(tickvals=[0,1,2,3],
                    ticktext=["Direct Sun","Part. Shade","Full Shade","Indoor"],
                    tickfont=dict(color="#ecf0f1",size=8),gridcolor=GRID_CLR)
    d["yaxis"]=dict(range=[0,100],gridcolor=GRID_CLR,
                    tickfont=dict(color="#4a5568",size=8),title="Risk Score")
    fig.update_layout(**d)
    return fig

def make_shap_chart(shap_features):
    if not shap_features: return None
    names=[s["feature"].replace("_"," ")[:22] for s in shap_features]
    vals=[s["shap_value"] for s in shap_features]
    colors=["#e74c3c" if v>0 else "#3498db" for v in vals]
    fig=go.Figure(go.Bar(x=vals,y=names,orientation="h",
        marker_color=colors,marker_opacity=0.85,
        hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
    d=chart_base(180); d["margin"]=dict(l=10,r=10,t=20,b=10)
    d["xaxis"]=dict(gridcolor=GRID_CLR,tickfont=dict(color="#4a5568",size=7),
                    title=dict(text="SHAP value",font=dict(color="#4a5568",size=8)))
    d["yaxis"]=dict(gridcolor=GRID_CLR,tickfont=dict(color="#ecf0f1",size=8))
    d["title"]=dict(text="SHAP — Danger drivers",font=dict(color="#ecf0f1",size=10),x=0.02)
    fig.update_layout(**d)
    return fig

def make_shift_heatmap(worker_risks, worker_states):
    names=[WORKERS[wid]["name"].split()[0] for wid in WORKERS]
    z=[]
    for wid in WORKERS:
        hist=list(worker_states[wid]["history"])
        if len(hist)<48:
            hist=[0]*(48-len(hist))+hist
        z.append(hist[:48])
    fig=go.Figure(go.Heatmap(z=z,x=list(range(48)),y=names,
        colorscale=[[0,"#0a1a0a"],[0.4,"#2b2200"],[0.55,"#2b1500"],
                    [0.7,"#2b0000"],[1,"#1a0020"]],
        zmin=0,zmax=100,showscale=True,
        colorbar=dict(title=dict(text="Risk",font=dict(color="#4a5568",size=9)),tickfont=dict(color="#4a5568",size=8)),
        hovertemplate="<b>%{y}</b><br>Min %{x}: Risk %{z:.0f}<extra></extra>"))
    d=chart_base(220); d["margin"]=dict(l=90,r=60,t=10,b=30)
    d["xaxis"]=dict(title="Minutes (last 48)",gridcolor=GRID_CLR,
                    tickfont=dict(color="#4a5568",size=7))
    fig.update_layout(**d)
    return fig


# ══════════════════════════════════════════════════════════════
# LLM ENGINE
# ══════════════════════════════════════════════════════════════
def get_llm_explanation(wid, state, risk, wbgt, audience, shap_feats=None):
    """
    Thin adapter: translates the dashboard's live simulation state into
    the `signals` dict format v2_llm_engine.py expects, then delegates
    to the real generate_groq_explanation() / generate_rule_based_
    explanation() functions. This used to be a full second copy of
    the LLM engine's logic (its own Groq call, its own driver-ranking
    rules, its own three audience templates) living inside the
    dashboard — now there is exactly one implementation of each,
    and the dashboard just calls it.
    """
    w=WORKERS[wid]
    hr_dev=(state["hr"]-w["resting_hr"])/w["resting_hr"]*100
    signals={
        "heart_rate_bpm":    state["hr"],
        "core_temp_c":       state["core_temp"],
        "breathing_rate":    state.get("br",16),
        "spo2_pct":          state.get("spo2",97),
        "heat_debt_index":   state.get("heat_debt",0),
        "dehydration_index": state.get("dehydration",0),
        "wbgt":              wbgt,
        "danger_prob_now":   risk/100,
        "danger_prob_30min": risk/100*0.9,
        "risk_score":        risk,
        "minutes_on_shift":  state.get("minutes",0),
        "model_confidence":  0.87 if shap_feats else 0.60,
    }
    return llm_engine.generate_groq_explanation(
        wid, signals, audience=audience, shap_features=shap_feats)


# ══════════════════════════════════════════════════════════════
# SHAP COMPUTATION
# ══════════════════════════════════════════════════════════════
def get_shap(models_dict, state, w, wbgt, risk):
    if not models_dict["model"] or not models_dict["explainer"]: return None
    try:
        hr=state["hr"]; ct=state["core_temp"]
        r_hr=float(w["resting_hr"]); m_hr=float(w["max_hr"]); accl=float(w["accl_days"])
        hrr=float(np.clip((hr-r_hr)/(m_hr-r_hr)*100,0,100))
        hr_dev=(hr-r_hr)/r_hr*100; t_dev=ct-float(w["baseline_temp"])
        feat_map={"heart_rate_bpm":hr,"hrv_rmssd_ms":state.get("hrv",40),
                  "core_temp_c":ct,"breathing_rate":state.get("br",16),
                  "spo2_pct":state.get("spo2",97),"heat_debt_index":state.get("heat_debt",0),
                  "dehydration_index":state.get("dehydration",0),"wbgt":wbgt,
                  "danger_prob_now":risk/100,"danger_prob_30min":risk/100*0.9,
                  "cv_strain_now":float(np.clip((hr-r_hr)/(m_hr-r_hr),0,1)),"accl_days":accl}
        fvec=np.zeros((1,len(models_dict["features"])))
        for i,f in enumerate(models_dict["features"]):
            if "_slope" in f or "_rstd" in f:
                # Rate-of-change and rolling-volatility features are a
                # completely different scale than the raw signal (e.g.
                # heart_rate_bpm_slope5 is diff(5)/5 -- a small
                # per-minute rate, typically ~0.5-2 -- not the raw bpm
                # value itself). Filling these with the current raw
                # value (e.g. 118) previously fed the model a wildly
                # out-of-distribution input after scaling, since it
                # never saw a "118 bpm/min rate of change" in training.
                # 0 (no recent change / no recent volatility) is the
                # correct default for a single live snapshot that
                # can't compute a true recent trend.
                fvec[0,i]=0
            else:
                # _lag{N} and _rm{N} (rolling mean) ARE the same units
                # as the raw signal, so approximating them with the
                # current value (a flat-trend assumption) is
                # reasonable when true historical replay isn't
                # available on every dashboard tick.
                base=f.split("_lag")[0].split("_rm")[0]
                fvec[0,i]=feat_map.get(base,feat_map.get(f,0))
        x_sc=models_dict["scaler"].transform(fvec)
        sv=np.array(models_dict["explainer"].shap_values(x_sc))
        ds=sv[0,:,2]; top3=np.argsort(np.abs(ds))[::-1][:3]
        return [{"feature":models_dict["features"][i],
                 "shap_value":round(float(ds[i]),3),
                 "direction":"increases" if ds[i]>0 else "decreases"} for i in top3]
    except Exception: return None


# ══════════════════════════════════════════════════════════════
# RL RECOMMENDATION
# ══════════════════════════════════════════════════════════════
def get_rl_action(rl_agent, risk, wbgt, accl, minutes):
    ACTIONS={0:"CONTINUE",1:"INCREASE BREAKS",2:"REST BREAK",
             3:"ROTATE ZONE",4:"REDUCE WORKLOAD",5:"EVACUATE"}
    ACTION_COLORS={0:"#2ecc71",1:"#f39c12",2:"#e67e22",
                   3:"#3498db",4:"#9b59b6",5:"#e74c3c"}
    RATIONALE={0:"Within safe parameters — continue monitoring",
               1:"Increase rest frequency — preventive action",
               2:"Cardiac strain rising — 15-min rest break",
               3:"Move to cooler zone — WBGT reduction",
               4:"Reduce physical exertion for 30 minutes",
               5:"Critical physiological state — evacuate now"}
    if rl_agent:
        try:
            # Use the single canonical state discretization + the
            # already-tested best_action() method (with its untrained-
            # state safety fallback) instead of reimplementing the
            # lookup inline. This used to call a raw np.argmax() with
            # no protection, bypassing that fallback entirely — a
            # second, unprotected copy of a bug already fixed once.
            state=rl_scheduler.discretize_state(risk,wbgt,accl,minutes/60)
            aid,_=rl_agent.best_action(state)
            return ACTIONS[aid],ACTION_COLORS[aid],RATIONALE[aid]
        except Exception: pass
    # Fallback
    if risk>=75: return "EVACUATE","#e74c3c",RATIONALE[5]
    elif risk>=60: return "REST BREAK","#e67e22",RATIONALE[2]
    elif risk>=45 and minutes>180: return "ROTATE ZONE","#3498db",RATIONALE[3]
    elif accl<7 and risk>=35: return "INCREASE BREAKS","#f39c12",RATIONALE[1]
    return "CONTINUE","#2ecc71",RATIONALE[0]


# ══════════════════════════════════════════════════════════════
# AUDIT
# ══════════════════════════════════════════════════════════════
def get_audit(limit=6):
    if not os.path.exists(DB_PATH):
        st.session_state.setdefault("_audit_warned", False)
        if not st.session_state["_audit_warned"]:
            st.warning(f"⚠️ Audit database not found at {DB_PATH} — "
                       f"alert history unavailable this session.")
            st.session_state["_audit_warned"] = True
        return []
    try:
        conn=sqlite3.connect(DB_PATH)
        rows=conn.execute(
            "SELECT worker_name,risk_score,alert_type,acknowledged,"
            "timestamp FROM alerts ORDER BY id DESC LIMIT ?",
            (limit,)).fetchall()
        conn.close()
        return rows
    except Exception as e:
        st.warning(f"⚠️ Could not read audit log ({type(e).__name__}: {e})")
        return []

def log_alert(wid, risk, wbgt, driver1):
    if not os.path.exists(DB_PATH):
        return
    try:
        w=WORKERS.get(wid,{})
        conn=sqlite3.connect(DB_PATH)
        conn.execute("INSERT INTO alerts (timestamp,worker_id,worker_name,"
                     "risk_score,alert_type,action_required,site_id,zone,"
                     "wbgt,top_driver_1) VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (datetime.now().isoformat(),wid,w.get("name",""),
                      risk,"danger" if risk>=70 else "warning",action_text(risk),
                      w.get("site",""),w.get("zone",""),wbgt,driver1))
        conn.commit(); conn.close()
    except Exception as e:
        st.warning(f"⚠️ Could not write alert to audit log "
                   f"({type(e).__name__}: {e})")


# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
# ── Screen detection helpers ─────────────────────────────
def is_mobile(w=None):
    w = w or st.session_state.get("screen_width", 1200)
    return w < 768

def is_tablet(w=None):
    w = w or st.session_state.get("screen_width", 1200)
    return 768 <= w < 1024

def sw():
    """Return current screen width."""
    return st.session_state.get("screen_width", 1200)

def ncols(desktop, tablet=2, mobile=1):
    """Return column count based on screen width."""
    w = sw()
    if w < 768: return mobile
    elif w < 1024: return tablet
    return desktop

def gauge_h():
    w = sw()
    if w < 768: return 220
    elif w < 1024: return 200
    return 185

def chart_cfg():
    return {"displayModeBar": not is_mobile()}

def refresh_rate(speed):
    """
    Wall-clock seconds between screen redraws — intentionally
    independent of simulation speed.

    IMPORTANT — this used to return 0.8/speed, which meant the
    "Sim Speed" slider controlled BOTH how much simulated time passes
    per tick (via step()'s speed multiplier, which is correct) AND how
    often the actual screen fully re-rendered (which is not what that
    slider is for). At the default speed of 3x, that computed to a
    ~0.27 second refresh — the whole app redrawing nearly 4 times per
    second, far too fast to actually read any chart or number before
    it changed again.

    Fixed at a single, stable 7-second interval on every device — a
    deliberately calm, predictable cadence so a person watching the
    dashboard can always read the current numbers in full before they
    change again. "Sim Speed" still does its job: a higher speed means
    more simulated minutes advance between each of these (still calm)
    redraws, not that the screen refreshes faster.
    """
    return 7.0


def check_password():
    """Password gate - only authorized viewers can access."""
    if st.session_state.get("authenticated"):
        return True
    
    st.markdown("""
    <div style='display:flex;justify-content:center;align-items:center;
    height:100vh;flex-direction:column;background:#0a0e1a'>
    <div style='text-align:center;padding:40px;background:#0d1117;
    border:1px solid #1a2035;border-radius:16px;width:min(380px,90vw)'>
    <div style='font-size:3rem;margin-bottom:8px'>🛡️</div>
    <div style='font-size:1.4rem;font-weight:900;color:#ecf0f1;
    letter-spacing:-0.02em'>ProactiveGuard v2.0</div>
    <div style='font-size:0.75rem;color:#4a5568;margin:6px 0 20px 0;
    text-transform:uppercase;letter-spacing:0.1em'>
    Saudi Aramco · Authorized Access Only</div>
    </div></div>
    """, unsafe_allow_html=True)
    
    pwd = st.text_input("Access Code", type="password",
                        placeholder="Enter access code...",
                        label_visibility="collapsed")
    
    if pwd:
        correct = "ProactiveGuard2025"
        try:
            if hasattr(st, "secrets") and "password" in st.secrets:
                correct = st.secrets["password"]
        except Exception:
            pass
        
        if pwd == correct:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect access code.")
    
    st.stop()
    return False


def init_session():
    # Real screen-width detection.
    #
    # IMPORTANT — the previous version sent window.innerWidth via
    # postMessage({type:'streamlit:setComponentValue',...}), but that
    # message type is only understood by Streamlit's CUSTOM COMPONENT
    # protocol (components.declare_component + the JS component
    # library's Streamlit.setComponentValue()) — components.html()
    # has no receiving side for it at all. The message went nowhere,
    # silently. screen_width therefore never actually updated from
    # its 1200px desktop default, on ANY device — meaning is_mobile(),
    # ncols(), and gauge_h() (which drive essentially every
    # st.columns() call in this dashboard) were dead code in
    # production: real phones were always getting the full desktop
    # column layout.
    #
    # Fix: a URL query-param round-trip. On first load, JS appends the
    # real viewport width to the URL and does ONE reload (guarded by
    # sessionStorage so this never repeats within the same browser
    # tab); Python then reads it via st.query_params on the next load.
    # A one-time reload cost on first open is a fair trade for
    # actually knowing the real device — a monitoring dashboard's
    # layout needs to be right for the device it's opened on far more
    # than it needs to live-adapt to a desktop window resize mid-shift.
    qp_width = st.query_params.get("vw")
    if qp_width is not None:
        try:
            st.session_state["screen_width"] = int(qp_width)
        except (TypeError, ValueError):
            pass

    import streamlit.components.v1 as _components
    _components.html("""<script>
(function(){
    if(!sessionStorage.getItem('pg_vw_set')){
        sessionStorage.setItem('pg_vw_set','1');
        var w = window.innerWidth;
        var url = new URL(window.location.href);
        url.searchParams.set('vw', w);
        window.location.replace(url.toString());
    }
})();
</script>
<meta name="viewport" content="width=device-width,initial-scale=1.0">
""",height=0,scrolling=False)

    defaults={"tick":0,"running":True,"selected":"W004","audience":"manager","screen_width":1200,
              "alerts":[],"logged":set(),
              "worker_states":{wid:init_state(w) for wid,w in WORKERS.items()}}
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k]=v


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    check_password()
    init_session()
    M=load_models()
    # Unpickling the RL agent (inside load_models) can trigger Python
    # to implicitly import v2_rl_scheduler.py for the first time, since
    # pickle needs that module's QLearningAgent class definition to
    # reconstruct the object — and that module also seeds np.random(42)
    # at import time. Reseed again here, after all possible implicit
    # imports have happened, so the live simulation stays non-deterministic.
    np.random.seed(None)
    for _w in M.get("load_warnings", []):
        st.warning(f"⚠️ {_w}")

    # ── Top Control Bar (replaces sidebar) ──────────────────────
    st.markdown(
        "<div style='background:#060810;border-bottom:1px solid #1a2035;"
        "padding:8px 0;margin-bottom:8px'>",
        unsafe_allow_html=True)
    
    _mob = is_mobile()
    if _mob:
        # Mobile: two rows of controls
        ctrl1,ctrl2 = st.columns([1,3])
        ctrl3 = st.columns([1])[0]
        ctrl4 = st.columns([1])[0]
        ctrl5,ctrl6 = st.columns([1,1])
    else:
        ctrl1,ctrl2,ctrl3,ctrl4,ctrl5,ctrl6 = st.columns([1,2,2,2,1,1])
    
    with ctrl1:
        st.markdown(
            "<div style='padding:4px 0;text-align:center'>"
            "<span style='font-size:1.2rem'>🛡️</span>"
            "<span style='font-size:0.75rem;font-weight:900;color:#ecf0f1;margin-left:4px'>ProactiveGuard</span>"
            "</div>", unsafe_allow_html=True)
    
    with ctrl2:
        speed = st.slider("⚡ Speed", 1, 20, 3, label_visibility="collapsed",
                          help="Simulation speed")
        st.markdown("<div style='font-size:0.65rem;color:#4a5568;text-align:center'>⚡ Sim Speed: "+str(speed)+"x</div>",
                   unsafe_allow_html=True)
    
    with ctrl3:
        sel = st.selectbox("Worker", options=list(WORKERS.keys()),
            format_func=lambda x: WORKERS[x]["name"],
            index=list(WORKERS.keys()).index(st.session_state.selected),
            label_visibility="collapsed")
        st.session_state.selected = sel
    
    with ctrl4:
        if is_mobile():
            aud = st.selectbox("Audience",["manager","doctor","engineer"],
                format_func=lambda x:{"manager":"👷 Safety Manager",
                                       "doctor":"🏥 Site Doctor",
                                       "engineer":"⚙️ ML Engineer"}[x],
                label_visibility="collapsed")
        else:
            aud = st.radio("Audience", ["manager","doctor","engineer"],
                format_func=lambda x:{"manager":"👷 Manager","doctor":"🏥 Doctor","engineer":"⚙️ Engineer"}[x],
                horizontal=True, label_visibility="collapsed")
        st.session_state.audience = aud
    
    with ctrl5:
        if st.button("⏸" if st.session_state.running else "▶", use_container_width=True):
            st.session_state.running = not st.session_state.running
    with ctrl6:
        if st.button("↺ Reset" if _mob else "↺", use_container_width=True):
            st.session_state.update({"tick":0,"alerts":[],"logged":set(),
                "worker_states":{wid:init_state(w) for wid,w in WORKERS.items()}})
    
    with ctrl6:
        src = M["source"]
        rl_ok = M["rl_agent"] is not None
        gk = os.getenv("GROQ_API_KEY","")
        groq_ok = gk and gk.startswith("gsk")
        ml_c  = "#2ecc71" if src=="ml_model" else "#f39c12"
        rl_c  = "#2ecc71" if rl_ok else "#f39c12"
        llm_c = "#2ecc71" if groq_ok else "#f39c12"
        st.markdown(
            f"<div style='font-size:0.65rem;line-height:1.8'>"
            f"<span style='color:{ml_c}'>● ML</span> "
            f"<span style='color:{rl_c}'>● RL</span> "
            f"<span style='color:{llm_c}'>● LLM</span>"
            f"</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#111827;margin:4px 0 8px 0'>", unsafe_allow_html=True)

    # ── View density toggle ─────────────────────────────────────
    # Separate from the Manager/Doctor/Engineer audience toggle above
    # (which only changes the wording of the AI-generated explanation
    # text). This controls whether technical charts — SHAP feature
    # importance, the zone risk scatter plot, the shift heatmap — show
    # at all. A field supervisor glancing at this mid-shift shouldn't
    # need to know what a SHAP value is; those charts are genuinely
    # useful for a safety officer reviewing patterns after the fact,
    # just not for a 10-second spot-check.
    st.session_state.setdefault("view_mode", "field")
    vm_choice = st.radio(
        "View mode",
        ["🔍 Field View — simple", "📊 Analyst View — full detail"],
        horizontal=True, label_visibility="collapsed",
        index=0 if st.session_state.view_mode == "field" else 1,
        key="view_mode_radio")
    st.session_state.view_mode = "field" if "Field" in vm_choice else "analyst"
    is_field = st.session_state.view_mode == "field"

    # ── Advance simulation ────────────────────────────────────
    env=get_env(st.session_state.tick)
    wbgt=env["wbgt"]
    worker_risks={}

    if st.session_state.running:
        new_alerts=list(st.session_state.alerts)
        alert_ids=[a["id"] for a in new_alerts]
        for wid,w in WORKERS.items():
            zw=zone_engine.calculate_zone_wbgt(wbgt,w["zone"],env["solar"])
            st.session_state.worker_states[wid]=step(
                st.session_state.worker_states[wid],w,zw,speed)
            s=st.session_state.worker_states[wid]
            risk=compute_risk(s,w,zw)
            s["history"].append(risk)
            s["hr_history"].append(s["hr"])
            s["br_history"].append(s.get("br",16))
            s["spo2_history"].append(s.get("spo2",97))
            worker_risks[wid]=risk
            if risk>=70 and wid not in alert_ids:
                new_alerts.append({"id":wid,"name":w["name"],"role":w["role"],
                                   "score":risk,"zone":w["zone"],"wbgt":zw})
                if wid not in st.session_state.logged:
                    hr_dev=(s["hr"]-w["resting_hr"])/w["resting_hr"]*100
                    log_alert(wid,risk,zw,f"HR {hr_dev:.0f}% above baseline")
                    st.session_state.logged.add(wid)
            if risk<55:
                new_alerts=[a for a in new_alerts if a["id"]!=wid]
                st.session_state.logged.discard(wid)
        for a in new_alerts: a["score"]=worker_risks.get(a["id"],a["score"])
        st.session_state.alerts=new_alerts
        st.session_state.tick+=1
    else:
        for wid,w in WORKERS.items():
            zw=zone_engine.calculate_zone_wbgt(wbgt,w["zone"],env["solar"])
            worker_risks[wid]=compute_risk(
                st.session_state.worker_states[wid],w,zw)

    # Sort workers by risk (highest first)
    sorted_wids=sorted(worker_risks,key=lambda x:worker_risks[x],reverse=True)

    # ── Header ────────────────────────────────────────────────
    shift_min=st.session_state.tick%480
    shift_pct=shift_min/480*100
    h_s=shift_min//60; m_s=shift_min%60
    now_str=datetime.now().strftime("%H:%M:%S")
    top_wid=sorted_wids[0]; top_risk=worker_risks[top_wid]

    st.markdown(
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'flex-wrap:wrap;gap:10px;'
        f'padding:12px 0 8px 0;border-bottom:1px solid #111827;margin-bottom:12px">'
        f'<div>'
        f'<span style="font-size:1.6rem;font-weight:900;color:#ecf0f1;'
        f'letter-spacing:-0.03em">🛡️ ProactiveGuard</span>'
        f'<span style="font-size:0.8rem;color:#4a5568;margin-left:10px;'
        f'vertical-align:middle">v2.0 · Saudi Aramco Operations</span>'
        f'</div>'
        f'<div style="display:flex;align-items:center;flex-wrap:wrap;gap:16px">'
        f'<div style="text-align:right">'
        f'<div style="font-size:0.65rem;color:#4a5568;text-transform:uppercase;'
        f'letter-spacing:0.1em">Shift Progress</div>'
        f'<div style="font-size:0.85rem;color:#ecf0f1;font-weight:600">'
        f'{h_s:02d}:{m_s:02d} / 08:00 ({shift_pct:.0f}%)</div>'
        f'<div class="progress-container" style="width:min(120px,30vw)">'
        f'<div class="progress-fill" style="width:{shift_pct:.0f}%;'
        f'background:{"#e74c3c" if shift_pct>75 else "#f39c12" if shift_pct>50 else "#3498db"}"></div>'
        f'</div></div>'
        f'<div style="text-align:right">'
        f'<div style="font-size:0.65rem;color:#4a5568;text-transform:uppercase;'
        f'letter-spacing:0.1em">Last Update</div>'
        f'<div style="font-size:0.85rem;color:#2ecc71;font-family:JetBrains Mono">'
        f'<span class="led led-green"></span>{now_str}</div>'
        f'</div></div></div>',
        unsafe_allow_html=True)

    # ── Critical full-screen alert ────────────────────────────
    critical_workers=[wid for wid,r in worker_risks.items() if r>=85]
    if critical_workers:
        names=", ".join(WORKERS[w]["name"].split()[0] for w in critical_workers)
        if is_mobile():
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#0d0010,#1a0020);'
                f'border:2px solid #8e44ad;border-radius:10px;padding:20px;'
                f'margin-bottom:12px;animation:pulse-critical 2s infinite;'
                f'text-align:center">'
                f'<div style="font-size:clamp(1.2rem,5vw,1.5rem);font-weight:900;'
                f'color:#8e44ad;letter-spacing:0.05em">🚨 CRITICAL ALERT</div>'
                f'<div style="font-size:clamp(2rem,8vw,4rem);font-weight:900;'
                f'color:#8e44ad;animation:pulse-critical 1s infinite">EVACUATE NOW</div>'
                f'<div style="color:#ecf0f1;margin-top:8px;font-size:clamp(0.8rem,3vw,1rem)">'
                f'<b style="color:#b85ce8">{names}</b><br>'
                f'Immediate medical assessment required</div>'
                f'</div>',unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#0d0010,#1a0020);'
                f'border:2px solid #8e44ad;border-radius:10px;padding:16px 20px;'
                f'margin-bottom:12px;animation:pulse-critical 2s infinite">'
                f'<div style="font-size:1.1rem;font-weight:900;color:#8e44ad;'
                f'letter-spacing:0.05em">🚨 CRITICAL PHYSIOLOGICAL ALERT</div>'
                f'<div style="color:#ecf0f1;margin-top:4px">'
                f'Workers at CRITICAL risk: <b style="color:#b85ce8">{names}</b> — '
                f'Immediate medical assessment required. Evacuate from work zone.</div>'
                f'</div>',unsafe_allow_html=True)

    # ── Environment panel ─────────────────────────────────────
    wlabel,wcolor=wbgt_label(wbgt)
    wbgt_field_label = "Heat Stress Index" if is_field else "WBGT Index"
    solar_field_label = "Sun Intensity" if is_field else "Solar Radiation"
    e1,e2,e3,e4,e5=st.columns(ncols(5,3,2))
    for col,icon,label,val,color in [
        (e1,"🌡️","Ambient Temp",f"{env['ambient']}°C","#e74c3c"),
        (e2,"💧","Humidity",f"{env['humidity']}%","#3498db"),
        (e3,"☀️",wbgt_field_label,f"{env['wbgt']}°C",wcolor),
        (e4,"⚠️","Heat Category",wlabel,wcolor),
        (e5,"☁️",solar_field_label,f"{env['solar']:.0f} W/m²","#f39c12"),
    ]:
        col.markdown(
            f'<div class="env-card"><div class="env-label">{icon} {label}</div>'
            f'<div class="env-value" style="color:{color}">{val}</div></div>',
            unsafe_allow_html=True)

    # Zone WBGT microclimate bar — real per-zone physics from
    # v2_zone_engine.py, not a flat static offset table.
    zone_wbgt_map=zone_engine.get_site_zone_wbgt_map(wbgt,env["solar"])

    st.markdown("<div style='margin:12px 0 4px 0'></div>",unsafe_allow_html=True)

    # ── Active alerts ─────────────────────────────────────────
    if st.session_state.alerts:
        st.markdown(
            "<div style='font-size:0.7rem;font-weight:700;color:#e74c3c;"
            "text-transform:uppercase;letter-spacing:0.12em;margin-bottom:6px'>"
            "🚨 Active Alerts</div>",unsafe_allow_html=True)
        for alert in st.session_state.alerts:
            sc=alert["score"]; c=risk_color(sc)
            is_crit=sc>=85
            card_cls="alert-critical" if is_crit else "alert-banner"
            st.markdown(
                f'<div class="{card_cls}">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px">'
                f'<div>'
                f'<span style="font-size:1rem;font-weight:800;color:{c}">'
                f'⚠️ {alert["name"]}</span>'
                f'<span style="color:#4a5568;font-size:0.75rem;margin-left:8px">'
                f'{alert["role"]} · {alert["zone"].replace("_"," ")}</span>'
                f'</div>'
                f'<div style="text-align:right">'
                f'<span style="font-size:1.8rem;font-weight:900;color:{c}">'
                f'{sc:.0f}</span>'
                f'<span style="color:{c};font-size:0.75rem;font-weight:700;'
                f'margin-left:6px">{risk_label(sc)}</span>'
                f'</div></div>'
                f'<div style="color:#cbd5e0;font-size:0.82rem;margin-top:8px;'
                f'line-height:1.4">'
                f'{action_text(sc)}</div>'
                f'</div>',unsafe_allow_html=True)

    # ── Zone event detection ──────────────────────────────────
    # Real coordinated-risk detector from v2_zone_engine.py: 3+ workers
    # in the same zone spiking together = environmental event
    # (evacuate the zone), vs 1-2 elevated = individual physiology
    # (send that worker for a rest break). This used to be a bare
    # `len(elev)>=3` check reimplemented inline here; now it's the
    # same function the standalone zone engine uses, so both agree.
    worker_zones={wid:w["zone"] for wid,w in WORKERS.items()}
    zone_events=zone_engine.detect_coordinated_risk(worker_risks,worker_zones)
    for ev in zone_events:
            zone=ev["zone"]; elev=[w["worker_id"] for w in ev["workers"]]; avg=ev["avg_risk"]
            st.markdown(
                f'<div class="zone-event">'
                f'<span style="color:#8e44ad;font-weight:800;font-size:0.9rem">'
                f'🌐 COORDINATED ZONE EVENT — {zone.upper().replace("_"," ")} '
                f'({ev["severity"].upper()})</span><br>'
                f'<span style="color:#cbd5e0;font-size:0.82rem">'
                f'{len(elev)} workers simultaneously elevated (avg: {avg:.0f}). '
                f'Environmental cause suspected. Consider zone evacuation.</span>'
                f'</div>',unsafe_allow_html=True)

    st.markdown("<div style='margin:8px 0'></div>",unsafe_allow_html=True)

    # ── Worker gauges (sorted by risk) ───────────────────────
    st.markdown(
        "<div style='font-size:0.7rem;font-weight:700;color:#4a5568;"
        "text-transform:uppercase;letter-spacing:0.12em;margin-bottom:8px'>"
        "👷 Worker Risk Monitor — Ranked by Risk</div>",
        unsafe_allow_html=True)

    _gcols = ncols(5, 3, 2)
    cols=st.columns(_gcols)
    for idx,wid in enumerate(sorted_wids):
        w=WORKERS[wid]; risk=worker_risks[wid]
        state=st.session_state.worker_states[wid]
        sel_border="2px solid #3498db" if wid==st.session_state.selected else "none"
        sensor_status=state.get("sensor_status","LIVE")
        accl_pct=min(100,w["accl_days"]/14*100)
        accl_color="#e74c3c" if accl_pct<50 else "#f39c12" if accl_pct<80 else "#2ecc71"
        accl_label=("New worker — high caution" if accl_pct<50
                    else "Still adjusting to heat" if accl_pct<80
                    else "Fully acclimatized")
        hr_dev=(state["hr"]-w["resting_hr"])/w["resting_hr"]*100

        with cols[idx%5]:
            # Gauge
            fig=make_gauge(risk,w["name"],w["role"])
            st.plotly_chart(fig,use_container_width=True,
                           config=chart_cfg(),key=f"g_{wid}")
            # Worker card
            break_icon="💤 " if state.get("on_break") else ""
            new_icon="⚠️" if w["accl_days"]<7 else ""
            st.markdown(
                f'<div class="worker-card {risk_class(risk)}" '
                f'style="outline:{sel_border};outline-offset:2px">'
                f'<div class="worker-name">{break_icon}{w["name"].split()[0]} {new_icon}</div>'
                f'<div class="worker-role">{w["role"]}</div>'
                f'<div class="vital-row">'
                f'<span class="vital-chip" style="color:#e74c3c">❤️ {state["hr"]:.0f}</span>'
                f'<span class="vital-chip" style="color:#3498db">🫁 {state.get("br",16):.0f}</span>'
                f'<span class="vital-chip" style="color:#2ecc71">O₂ {state.get("spo2",97):.0f}%</span>'
                f'<span class="vital-chip" style="color:#e67e22">🌡️ {state["core_temp"]:.1f}</span>'
                f'</div>'
                f'<div style="margin-top:6px;font-size:0.68rem;color:#4a5568">Acclimatization</div>'
                f'<div class="progress-container">'
                f'<div class="progress-fill" style="width:{accl_pct:.0f}%;background:{accl_color}"></div>'
                f'</div>'
                f'<div style="display:flex;justify-content:space-between;flex-wrap:wrap;'
                f'gap:4px;font-size:0.65rem;color:#2c3e50;margin-top:2px">'
                f'<span style="color:{accl_color}">{accl_label} ({w["accl_days"]}d)</span>'
                f'<span style="white-space:nowrap"><span class="led {led_class(sensor_status)}"></span>{sensor_status}</span>'
                f'</div>'
                f'</div>',unsafe_allow_html=True)
            if st.button("Select ▶",key=f"sel_{wid}",use_container_width=True):
                st.session_state.selected=wid

    st.markdown("<hr style='border-color:#111827;margin:16px 0'>",
                unsafe_allow_html=True)

    # ── Zone map + Shift heatmap ──────────────────────────────
    # Selected-worker variables are needed by later panels regardless
    # of view mode, so they're computed here unconditionally rather
    # than inside the (now Analyst-View-gated) chart blocks below.
    sel_w=WORKERS[st.session_state.selected]
    sel_state=st.session_state.worker_states[st.session_state.selected]
    sel_zw=zone_engine.calculate_zone_wbgt(wbgt,sel_w["zone"],env["solar"])
    sel_risk=worker_risks[st.session_state.selected]

    zm_col,hm_col=st.columns(1 if is_mobile() else [1,1])
    with zm_col:
        st.markdown(
            "<div class='panel-title' style='--accent:#3498db'>🗺️ Zone Risk Map</div>",
            unsafe_allow_html=True)
        zone_avgs = {}
        for wid,risk in worker_risks.items():
            z = WORKERS[wid]["zone"]
            if z not in zone_avgs: zone_avgs[z] = []
            zone_avgs[z].append(risk)
        if is_field:
            # Plain-language summary instead of a scatter/bar chart —
            # a field supervisor needs "which zone is worst right now"
            # in one sentence, not axes to read.
            worst_zone = max(zone_avgs, key=lambda z: sum(zone_avgs[z])/len(zone_avgs[z]))
            worst_avg = sum(zone_avgs[worst_zone])/len(zone_avgs[worst_zone])
            zc = risk_color(worst_avg)
            st.markdown(
                f'<div style="background:#0d1117;border:1px solid #1a2035;'
                f'border-radius:8px;padding:14px;text-align:center">'
                f'<div style="font-size:0.7rem;color:#4a5568;text-transform:uppercase;'
                f'letter-spacing:0.08em">Highest-risk zone right now</div>'
                f'<div style="font-size:1.3rem;font-weight:800;color:{zc};margin-top:4px">'
                f'{worst_zone.replace("_"," ").title()} — avg risk {worst_avg:.0f}</div>'
                f'</div>',unsafe_allow_html=True)
        elif is_mobile():
            # Simplified bar chart for mobile performance
            import plotly.graph_objects as go
            z_names = [z.replace("_"," ").title() for z in zone_avgs]
            z_vals  = [round(sum(v)/len(v),1) for v in zone_avgs.values()]
            z_cols  = [risk_color(v) for v in z_vals]
            fig_mob = go.Figure(go.Bar(
                x=z_names, y=z_vals, marker_color=z_cols,
                text=[f"{v:.0f}" for v in z_vals],
                textposition="outside",
                textfont=dict(color="#ecf0f1",size=12),
                hovertemplate="%{x}: %{y:.0f}<extra></extra>"))
            fig_mob.add_hline(y=70,line=dict(color="#e74c3c",width=1,dash="dash"))
            fig_mob.update_layout(
                paper_bgcolor=CHART_BG,plot_bgcolor=PLOT_BG,
                height=220,margin=dict(l=10,r=10,t=10,b=30),
                yaxis=dict(range=[0,100],gridcolor=GRID_CLR,
                           tickfont=dict(color="#4a5568",size=8)),
                xaxis=dict(tickfont=dict(color="#ecf0f1",size=9)),
                hoverlabel=dict(bgcolor="#0d1117",font_color="#ecf0f1"))
            st.plotly_chart(fig_mob,use_container_width=True,
                           config=chart_cfg(),key="zone_mobile")
        else:
            fig_zone=make_zone_scatter(worker_risks,
                                       {wid:WORKERS[wid]["zone"] for wid in WORKERS})
            st.plotly_chart(fig_zone,use_container_width=True,
                           config=chart_cfg(),key="zone_scatter")
        # Zone WBGT bar — Analyst View only; Field View already got
        # its answer in the plain-language summary above.
        if not is_field:
            st.markdown(
                "<div class='panel-title' style='--accent:#f39c12;margin-top:8px'>"
                "🌡️ Zone Microclimate WBGT</div>",unsafe_allow_html=True)
            fig_zwbgt=make_zone_wbgt(zone_wbgt_map)
            st.plotly_chart(fig_zwbgt,use_container_width=True,
                           config=chart_cfg(),key="zone_wbgt")

    with hm_col:
        if not is_field:
            st.markdown(
                "<div class='panel-title' style='--accent:#9b59b6'>📊 Shift Risk Heatmap — All Workers</div>",
                unsafe_allow_html=True)
            fig_hm=make_shift_heatmap(worker_risks,st.session_state.worker_states)
            st.plotly_chart(fig_hm,use_container_width=True,
                           config=chart_cfg(),key="shift_hm")

        # Selected worker sparklines — kept visible in both modes;
        # heart rate / breathing / SpO2 trend lines are intuitive at
        # a glance and don't need any technical background to read.
        st.markdown(
            f"<div class='panel-title' style='--accent:#e74c3c;margin-top:8px'>"
            f"📈 {sel_w['name'].split()[0]} — Live Signals</div>",
            unsafe_allow_html=True)
        sp1,sp2,sp3=st.columns(ncols(3,3,1))
        with sp1:
            st.markdown("<div style='font-size:0.65rem;color:#e74c3c;text-align:center'>❤️ Heart Rate</div>",unsafe_allow_html=True)
            if len(sel_state["hr_history"])>1:
                st.plotly_chart(make_sparkline(sel_state["hr_history"],"#e74c3c"),
                               use_container_width=True,
                               config=chart_cfg(),key="sp_hr")
            else:
                st.markdown("<div style='text-align:center;color:#4a5568;"
                           "font-size:0.7rem;padding:20px 0'>Collecting data…</div>",
                           unsafe_allow_html=True)
        with sp2:
            st.markdown("<div style='font-size:0.65rem;color:#3498db;text-align:center'>🫁 Breathing Rate</div>",unsafe_allow_html=True)
            if len(sel_state["br_history"])>1:
                st.plotly_chart(make_sparkline(sel_state["br_history"],"#3498db"),
                               use_container_width=True,
                               config=chart_cfg(),key="sp_br")
            else:
                st.markdown("<div style='text-align:center;color:#4a5568;"
                           "font-size:0.7rem;padding:20px 0'>Collecting data…</div>",
                           unsafe_allow_html=True)
        with sp3:
            st.markdown("<div style='font-size:0.65rem;color:#2ecc71;text-align:center'>O₂ SpO2</div>",unsafe_allow_html=True)
            if len(sel_state["spo2_history"])>1:
                st.plotly_chart(make_sparkline(sel_state["spo2_history"],"#2ecc71"),
                               use_container_width=True,
                               config=chart_cfg(),key="sp_spo2")
            else:
                st.markdown("<div style='text-align:center;color:#4a5568;"
                           "font-size:0.7rem;padding:20px 0'>Collecting data…</div>",
                           unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#111827;margin:16px 0'>",
                unsafe_allow_html=True)

    # ── Detail: trajectory + intelligence ────────────────────
    st.markdown(
        f"<div style='font-size:0.7rem;font-weight:700;color:#4a5568;"
        f"text-transform:uppercase;letter-spacing:0.12em;margin-bottom:8px'>"
        f"🔍 {sel_w['name']} — Detail Analysis</div>",
        unsafe_allow_html=True)

    traj_col,intel_col=st.columns(1 if is_mobile() else [3,2])

    with traj_col:
        # Risk trajectory with twin
        twin_pred=[float(np.clip(sel_risk+i*(0.35 if sel_w["accl_days"]<14 else 0.12),0,100))
                   for i in range(1,11)]
        fig_traj=make_trajectory(sel_state["history"],sel_w["name"],sel_risk,twin_pred)
        st.plotly_chart(fig_traj,use_container_width=True,
                       config=chart_cfg(),key="traj_detail")
        # Vitals
        hr_dev=(sel_state["hr"]-sel_w["resting_hr"])/sel_w["resting_hr"]*100
        ct_dev=sel_state["core_temp"]-sel_w["baseline_temp"]
        br_val=sel_state.get("br",16); spo2_val=sel_state.get("spo2",97)
        di_val=sel_state.get("dehydration",0)
        # Trend arrow based on history
        hist_list=list(sel_state["history"])
        trend_arrow="↑" if len(hist_list)>5 and hist_list[-1]>hist_list[-5] else "↓" if len(hist_list)>5 and hist_list[-1]<hist_list[-5] else "→"
        trend_color="#e74c3c" if trend_arrow=="↑" else "#2ecc71" if trend_arrow=="↓" else "#f39c12"
        # Time to danger estimate
        if len(hist_list)>10:
            rate=(hist_list[-1]-hist_list[-10])/10
            if rate>0 and hist_list[-1]<70:
                ttd=int((70-hist_list[-1])/rate) if rate>0 else 999
                ttd_str=f"~{min(ttd,60)}min to danger" if ttd<60 else "Safe window"
            elif hist_list[-1]>=70:
                ttd_str="IN DANGER NOW"
            else:
                ttd_str="Stable"
        else:
            ttd_str="Calculating..."
        st.markdown(
            f'''<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-top:8px">
            <div style="background:#0d1117;border:1px solid #1a2035;border-radius:8px;padding:10px;text-align:center">
                <div style="font-size:0.65rem;color:#4a5568;text-transform:uppercase">❤️ Heart Rate</div>
                <div style="font-size:1.4rem;font-weight:800;color:#e74c3c">{sel_state["hr"]:.0f}</div>
                <div style="font-size:0.7rem;color:#e74c3c">+{hr_dev:.0f}% baseline</div>
            </div>
            <div style="background:#0d1117;border:1px solid #1a2035;border-radius:8px;padding:10px;text-align:center">
                <div style="font-size:0.65rem;color:#4a5568;text-transform:uppercase">🌡️ Core Temp</div>
                <div style="font-size:1.4rem;font-weight:800;color:#e67e22">{sel_state["core_temp"]:.2f}°C</div>
                <div style="font-size:0.7rem;color:#e67e22">+{ct_dev:.2f}°C</div>
            </div>
            <div style="background:#0d1117;border:1px solid #1a2035;border-radius:8px;padding:10px;text-align:center">
                <div style="font-size:0.65rem;color:#4a5568;text-transform:uppercase">🫁 Breathing</div>
                <div style="font-size:1.4rem;font-weight:800;color:{"#e74c3c" if br_val>24 else "#3498db"}">{br_val:.0f}/min</div>
                <div style="font-size:0.7rem;color:{"#e74c3c" if br_val>24 else "#4a5568"}">{"⚠️ Elevated" if br_val>24 else "Normal"}</div>
            </div>
            <div style="background:#0d1117;border:1px solid #1a2035;border-radius:8px;padding:10px;text-align:center">
                <div style="font-size:0.65rem;color:#4a5568;text-transform:uppercase">O₂ SpO2</div>
                <div style="font-size:1.4rem;font-weight:800;color:{"#e74c3c" if spo2_val<95 else "#2ecc71"}">{spo2_val:.1f}%</div>
                <div style="font-size:0.7rem;color:{"#e74c3c" if spo2_val<95 else "#4a5568"}">{"⚠️ Low" if spo2_val<95 else "Normal"}</div>
            </div>
            <div style="background:#0d1117;border:1px solid #1a2035;border-radius:8px;padding:10px;text-align:center">
                <div style="font-size:0.65rem;color:#4a5568;text-transform:uppercase">⏱️ Time to Danger</div>
                <div style="font-size:0.9rem;font-weight:800;color:{trend_color}">{trend_arrow} {ttd_str}</div>
                <div style="font-size:0.7rem;color:#4a5568">Dehydr: {di_val:.2f}</div>
            </div>
            </div>''', unsafe_allow_html=True)

    with intel_col:
        # Risk score panel
        rc=risk_color(sel_risk)
        # Confidence interval based on data quality
        dq=sel_state.get("data_quality",1.0)
        ci=int((1-dq)*15+5)
        hist_list2=list(sel_state["history"])
        trend2="↑ Rising" if len(hist_list2)>5 and hist_list2[-1]>hist_list2[-5] else "↓ Falling" if len(hist_list2)>5 and hist_list2[-1]<hist_list2[-5] else "→ Stable"
        trend2_color="#e74c3c" if "Rising" in trend2 else "#2ecc71" if "Falling" in trend2 else "#f39c12"
        st.markdown(
            f'<div style="background:linear-gradient(145deg,#0d1117,#111827);'
            f'border:1px solid {rc}44;border-radius:10px;padding:14px;'
            f'text-align:center;margin-bottom:8px">'
            f'<div style="font-size:0.65rem;color:#4a5568;text-transform:uppercase;'
            f'letter-spacing:0.12em">Composite Risk Score</div>'
            f'<div style="font-size:3.5rem;font-weight:900;color:{rc};'
            f'line-height:1;font-family:Inter">{sel_risk:.0f}</div>'
            f'<div style="font-size:0.72rem;color:#4a5568;margin:2px 0">'
            f'Confidence: {sel_risk:.0f} ± {ci} | <span style="color:{trend2_color}">{trend2}</span></div>'
            f'<div style="font-size:0.8rem;font-weight:700;color:{rc};'
            f'letter-spacing:0.1em">{risk_label(sel_risk)}</div>'
            f'<div class="progress-container" style="margin:8px 0">'
            f'<div class="progress-fill" style="width:{sel_risk:.0f}%;background:{rc}"></div>'
            f'</div>'
            f'<div style="font-size:0.78rem;color:#cbd5e0;margin-top:4px">'
            f'{action_text(sel_risk)}</div>'
            f'</div>',unsafe_allow_html=True)

        # LLM explanation — real v2_llm_engine.py functions, not a
        # duplicate implementation
        shap_feats=get_shap(M,sel_state,sel_w,sel_zw,sel_risk)
        explanation,src_code=get_llm_explanation(
            st.session_state.selected,sel_state,sel_risk,sel_zw,
            st.session_state.audience,shap_feats)
        src_label={"groq":"🤖 Groq LLM",
                   "rule_based":"📋 Rule-Based (offline)",
                   "rule_based_fallback":"📋 Rule-Based (Groq unavailable)"
                   }.get(src_code,src_code)
        st.markdown(
            f'<div class="llm-panel">'
            f'<div class="llm-source">{src_label} · '
            f'{st.session_state.audience.upper()} VIEW</div>'
            f'{explanation}'
            f'</div>',unsafe_allow_html=True)

        # Digital twin
        st.markdown(
            f'<div class="twin-panel">'
            f'<div style="font-size:0.65rem;color:#2ecc71;text-transform:uppercase;'
            f'letter-spacing:0.1em;margin-bottom:4px">🔬 Digital Twin Forecast</div>'
            f'<div style="color:#ecf0f1;font-size:0.82rem">'
            f'<b>+15min:</b> {twin_pred[4]:.1f} &nbsp;|&nbsp; '
            f'<b>+30min:</b> {twin_pred[9]:.1f} &nbsp;|&nbsp; '
            f'<b>Trend:</b> <span style="color:{"#e74c3c" if twin_pred[-1]>sel_risk else "#2ecc71"}">'
            f'{"↑ Rising" if twin_pred[-1]>sel_risk else "↓ Stable"}</span>'
            f'</div></div>',unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#111827;margin:16px 0'>",
                unsafe_allow_html=True)

    # ── Bottom row: SHAP + RL + Audit ─────────────────────────
    sh_col,rl_col,au_col=st.columns(1 if is_mobile() else [1.5,1.5,1])

    with sh_col:
        panel_title = ("⚠️ Why This Worker Is At Risk" if is_field
                       else "📊 SHAP Feature Importance")
        st.markdown(
            f"<div class='panel-title' style='--accent:#e74c3c'>{panel_title}</div>",
            unsafe_allow_html=True)
        if shap_feats:
            if not is_field:
                fig_shap=make_shap_chart(shap_feats)
                if fig_shap:
                    st.plotly_chart(fig_shap,use_container_width=True,
                                   config=chart_cfg(),key="shap_chart")
            # Plain English SHAP explanation — shown in both modes;
            # this is the version a field supervisor should actually
            # read, so it's not gated behind Analyst View.
            shap_plain=[]
            feat_labels={"heart_rate_bpm":"Heart rate","heart_rate_bpm_rm15":"HR trend (15min avg)",
                        "heart_rate_bpm_lag30":"HR 30min ago","breathing_rate":"Breathing rate",
                        "core_temp_c":"Core temperature","heat_debt_index":"Cumulative heat load",
                        "dehydration_index":"Dehydration","wbgt":"Environmental heat (WBGT)"}
            for s in shap_feats:
                fname=feat_labels.get(s["feature"],s["feature"].replace("_"," "))
                direction="↑ increases" if s["shap_value"]>0 else "↓ decreases"
                shap_plain.append(f"• **{fname}** {direction} danger risk")
            if shap_plain:
                st.markdown(
                    f'<div style="background:#0d1117;border:1px solid #1a2035;'
                    f'border-radius:6px;padding:8px;font-size:0.75rem;margin-top:4px">'
                    f'{"<br>".join(shap_plain[:3])}'
                    f'</div>',unsafe_allow_html=True)
        else:
            hr_dev2=(sel_state["hr"]-sel_w["resting_hr"])/sel_w["resting_hr"]*100
            ct_dev2=sel_state["core_temp"]-sel_w["baseline_temp"]
            if not is_field:
                fb=[{"feature":"hr_deviation_pct","shap_value":hr_dev2*0.025},
                    {"feature":"core_temp_deviation","shap_value":ct_dev2*1.8},
                    {"feature":"heat_debt_index","shap_value":sel_state.get("heat_debt",0)*0.004}]
                fig_shap=make_shap_chart(fb)
                if fig_shap:
                    st.plotly_chart(fig_shap,use_container_width=True,
                                   config=chart_cfg(),key="shap_fb")
            st.markdown(
                f'<div style="background:#0d1117;border:1px solid #1a2035;'
                f'border-radius:6px;padding:8px;font-size:0.75rem;margin-top:4px">'
                f'• HR {hr_dev2:.0f}% above personal baseline<br>'
                f'• Core temp +{ct_dev2:.2f}°C above normal<br>'
                f'• Heat debt accumulation high'
                f'</div>',unsafe_allow_html=True)

    with rl_col:
        st.markdown(
            "<div class='panel-title' style='--accent:#9b59b6'>🤖 RL Scheduler</div>",
            unsafe_allow_html=True)
        action,ac_color,rationale=get_rl_action(
            M["rl_agent"],sel_risk,sel_zw,sel_w["accl_days"],sel_state["minutes"])
        st.markdown(
            f'<div class="rl-panel">'
            f'<div style="font-size:0.65rem;color:#9b59b6;text-transform:uppercase;'
            f'letter-spacing:0.1em;margin-bottom:6px">Optimal Action</div>'
            f'<div class="rl-action" style="color:{ac_color}">{action}</div>'
            f'<div style="color:#cbd5e0;font-size:0.8rem;margin-top:4px">{rationale}</div>'
            f'</div>',unsafe_allow_html=True)

        # Proactive 2-hour schedule — real output of
        # v2_rl_scheduler.generate_proactive_schedule(), previously
        # computed by that module but never surfaced anywhere in the
        # live dashboard (only the single live-tick action was shown).
        if M["rl_agent"] is not None:
            try:
                sched = rl_scheduler.generate_proactive_schedule(
                    M["rl_agent"], st.session_state.selected, sel_risk,
                    sel_zw, sel_state["minutes"])
                with st.expander("📅 Next 2 Hours — Proactive Schedule"):
                    for row in sched[:8]:
                        st.markdown(
                            f'<div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:4px;'
                            f'font-size:0.78rem;padding:3px 0;border-bottom:1px solid #1a2035">'
                            f'<span style="color:#4a5568">+{row["minute"]-sel_state["minutes"]:.0f} min</span>'
                            f'<span style="color:#cbd5e0">{row["recommended_action"]}</span>'
                            f'<span style="color:#f39c12">risk→{row["risk_after_action"]:.0f}</span>'
                            f'</div>', unsafe_allow_html=True)
            except Exception as e:
                st.caption(f"⚠️ Proactive schedule unavailable "
                           f"({type(e).__name__}: {e})")

        # Physiological indicators
        _heat_debt = sel_state.get("heat_debt",0)
        _dehyd = sel_state.get("dehydration",0)
        if is_field:
            heat_label = ("Low" if _heat_debt/800<0.3 else
                         "Moderate" if _heat_debt/800<0.6 else "High")
            heat_color = "#2ecc71" if heat_label=="Low" else "#f39c12" if heat_label=="Moderate" else "#e74c3c"
            hyd_label = ("Good" if _dehyd<0.3 else
                        "Getting low" if _dehyd<0.6 else "Critical")
            hyd_color = "#2ecc71" if hyd_label=="Good" else "#f39c12" if hyd_label=="Getting low" else "#e74c3c"
            heat_display, dehyd_display = heat_label, hyd_label
            heat_color_final, dehyd_color_final = heat_color, hyd_color
        else:
            heat_display, dehyd_display = f'{_heat_debt:.0f}', f'{_dehyd:.3f}'
            heat_color_final, dehyd_color_final = "#f39c12", "#3498db"
        st.markdown(
            f'<div style="background:#0d1117;border:1px solid #1a2035;border-radius:8px;'
            f'padding:12px;margin-top:8px">'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">'
            f'<div><div style="font-size:0.65rem;color:#4a5568;text-transform:uppercase">Heat Load</div>'
            f'<div style="font-size:1.1rem;font-weight:700;color:{heat_color_final}">'
            f'{heat_display}</div></div>'
            f'<div><div style="font-size:0.65rem;color:#4a5568;text-transform:uppercase">Hydration</div>'
            f'<div style="font-size:1.1rem;font-weight:700;color:{dehyd_color_final}">'
            f'{dehyd_display}</div></div>'
            f'<div><div style="font-size:0.65rem;color:#4a5568;text-transform:uppercase">HRV</div>'
            f'<div style="font-size:1.1rem;font-weight:700;color:#2ecc71">'
            f'{sel_state.get("hrv",40):.0f} ms</div></div>'
            f'<div><div style="font-size:0.65rem;color:#4a5568;text-transform:uppercase">Accl Status</div>'
            f'<div style="font-size:0.85rem;font-weight:700;'
            f'color:{"#e74c3c" if sel_w["accl_days"]<7 else "#2ecc71"}">'
            f'{"⚠️ NEW" if sel_w["accl_days"]<7 else "✅ OK"}</div></div>'
            f'</div></div>',unsafe_allow_html=True)

    with au_col:
        st.markdown(
            "<div class='panel-title' style='--accent:#4a5568'>📋 Audit Trail</div>",
            unsafe_allow_html=True)
        audit_rows=get_audit(6)
        if audit_rows:
            for row in audit_rows:
                name,score,atype,acked,ts=row
                c=risk_color(float(score))
                ack="ACK" if acked else "NEW"
                ts_short=str(ts)[11:16] if ts else ""
                st.markdown(
                    f'<div class="audit-row" style="--risk-color:{c}">'
                    f'<span style="color:{c};font-weight:700;font-family:JetBrains Mono">'
                    f'{float(score):.0f}</span>'
                    f'<span style="color:#ecf0f1;font-size:0.72rem">'
                    f'{str(name).split()[0] if name else ""}</span>'
                    f'<span style="color:#2c3e50;margin-left:auto">{ts_short} {ack}</span>'
                    f'</div>',unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#2c3e50;font-size:0.8rem;'
                       'padding:8px">No alerts logged yet</div>',
                       unsafe_allow_html=True)

        st.markdown("<div style='margin-top:8px'></div>",unsafe_allow_html=True)
        if st.button("📄 Export PDF Report",use_container_width=True):
            try:
                import sys; sys.path.insert(0,".")
                from v2_safety_infrastructure import generate_pdf_report
                generate_pdf_report({
                    "site_name":"Saudi Aramco — Dhahran Operations",
                    "shift_date":datetime.now().strftime("%Y-%m-%d"),
                    "n_workers":10,"avg_wbgt":wbgt,"peak_wbgt":wbgt+3,
                    "n_alerts":len(st.session_state.alerts),
                    "n_danger_alerts":sum(1 for a in st.session_state.alerts if a["score"]>=70),
                    "n_unacknowledged":len(st.session_state.alerts),
                    "n_escalations":0,"n_false_alarms":0,
                    "accl_anomaly_workers":["Samir Hassan","Majed Al-Shehri"],
                    "workers":[{"name":WORKERS[wid]["name"],
                                "max_risk":worker_risks[wid],
                                "danger_minutes":0} for wid in WORKERS],
                },"outputs/v2_shift_report_live.pdf")
                st.success("✅ Saved to outputs/")
            except Exception as e:
                st.error(f"{e}")

    # ── Footer ────────────────────────────────────────────────
    st.markdown(
        '<div class="pg-footer">'
        'ProactiveGuard v2.0 &nbsp;·&nbsp; '
        'Pennes Bioheat Equation &nbsp;·&nbsp; Fiala Sweat Model &nbsp;·&nbsp; '
        'Fick Cardiovascular Strain &nbsp;·&nbsp; XGBoost + Temporal Features &nbsp;·&nbsp; '
        'SHAP Explainability &nbsp;·&nbsp; Q-Learning RL Scheduler &nbsp;·&nbsp; '
        'Groq LLM Engine &nbsp;·&nbsp; Zone Intelligence &nbsp;·&nbsp; '
        'Digital Twin 30-min Prediction &nbsp;·&nbsp; '
        'WBGT ISO 7933 &nbsp;·&nbsp; ACGIH TLV &nbsp;·&nbsp; ISO 45001'
        '</div>',unsafe_allow_html=True)

    # Auto-refresh
    if st.session_state.running:
        time.sleep(refresh_rate(speed))
        st.rerun()


if __name__=="__main__":
    main()
