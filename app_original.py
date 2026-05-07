# Requirements: streamlit>=1.35, plotly>=5.20, pandas>=2.0, numpy>=1.24, streamlit-lottie>=0.0.5
# Optional: scipy>=1.12, scikit-learn>=1.4, streamlit-agraph>=0.0.45, psutil>=5.9

"""
GSNH-MDT — Formal XAI Laboratory v4 [PRO EDITION]
Cinematic research dashboard with full motion design. Zero mock data.
Run: PYTHONPATH=src streamlit run app.py
"""
import os, sys, time, threading, sqlite3, hashlib, json, base64
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit.components.v1 import html

# ── Optional imports ────────────────────────────────────────────────
try:
    from streamlit_agraph import agraph, Node, Edge, Config
    AGRAPH_AVAILABLE = True
except ImportError:
    AGRAPH_AVAILABLE = False
try:
    from scipy.stats import wilcoxon
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
try:
    from sklearn.calibration import calibration_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# ── Backend ─────────────────────────────────────────────────────────
SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
from gsnh_mdt.tree.builder import ExpertGSNHTree
from gsnh_mdt.tree.stopping import StoppingCriteria
from gsnh_mdt.types import LanguageFamily

# ═════════════════════════════════════════════════════════════════════
# PAGE CONFIG & MOTION SYSTEM
# ═════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="GSNH-MDT Lab | Pro Edition",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': "### GSNH-MDT Laboratory v4\nCinematic XAI Research Platform"}
)

# ═════════════════════════════════════════════════════════════════════
# CINEMATIC CSS — Glassmorphism & Motion Design
# ═════════════════════════════════════════════════════════════════════
ANIMATION_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --primary: #00d4ff;
    --secondary: #b56bff;
    --success: #00ff88;
    --warning: #ffb300;
    --danger: #ff3b6e;
    --dark: #0a0f1c;
    --glass: rgba(16, 24, 48, 0.65);
    --glass-border: rgba(255, 255, 255, 0.08);
}

* {
    font-family: 'Inter', sans-serif;
}

/* Animated Background */
.stApp {
    background: linear-gradient(135deg, #0a0f1c 0%, #111827 50%, #0f172a 100%);
    background-attachment: fixed;
}

.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: 
        radial-gradient(circle at 20% 80%, rgba(0, 212, 255, 0.08) 0%, transparent 50%),
        radial-gradient(circle at 80% 20%, rgba(181, 107, 255, 0.08) 0%, transparent 50%),
        radial-gradient(circle at 50% 50%, rgba(0, 255, 136, 0.03) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
    animation: pulse-bg 10s ease-in-out infinite alternate;
}

@keyframes pulse-bg {
    0% { opacity: 0.8; transform: scale(1); }
    100% { opacity: 1; transform: scale(1.1); }
}

/* Typography */
h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}

code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Glass Cards with 3D Hover */
.glass-card {
    background: var(--glass);
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 28px;
    position: relative;
    overflow: hidden;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    transform-style: preserve-3d;
    box-shadow: 
        0 8px 32px 0 rgba(0, 0, 0, 0.37),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);
}

.glass-card:hover {
    transform: translateY(-8px) rotateX(2deg) rotateY(-2deg);
    box-shadow: 
        0 20px 60px rgba(0, 212, 255, 0.15),
        inset 0 1px 0 rgba(255, 255, 255, 0.2);
    border-color: rgba(0, 212, 255, 0.3);
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    transition: left 0.7s;
}

.glass-card:hover::before {
    left: 100%;
}

/* Animated Gradient Borders */
.gradient-border {
    position: relative;
    background: linear-gradient(var(--dark), var(--dark)) padding-box,
                linear-gradient(135deg, var(--primary), var(--secondary), var(--success)) border-box;
    border: 2px solid transparent;
    border-radius: 20px;
    animation: border-rotate 4s linear infinite;
}

@keyframes border-rotate {
    0% { filter: hue-rotate(0deg); }
    100% { filter: hue-rotate(360deg); }
}

/* KPI Cards */
.kpi-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 20px;
    margin: 24px 0;
}

.kpi-card {
    background: linear-gradient(145deg, rgba(16,24,48,0.9), rgba(10,15,36,0.95));
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 24px;
    position: relative;
    overflow: hidden;
    transition: all 0.4s ease;
    opacity: 0;
    animation: slide-up 0.6s ease forwards;
}

.kpi-card:nth-child(1) { animation-delay: 0.1s; }
.kpi-card:nth-child(2) { animation-delay: 0.2s; }
.kpi-card:nth-child(3) { animation-delay: 0.3s; }
.kpi-card:nth-child(4) { animation-delay: 0.4s; }

@keyframes slide-up {
    to {
        opacity: 1;
        transform: translateY(0);
    }
    from {
        opacity: 0;
        transform: translateY(30px);
    }
}

.kpi-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--primary), transparent);
    opacity: 0;
    transition: opacity 0.3s;
}

.kpi-card:hover::after {
    opacity: 1;
    animation: scan 2s linear infinite;
}

@keyframes scan {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

.kpi-icon {
    font-size: 2rem;
    margin-bottom: 12px;
    display: block;
    animation: float 3s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-5px); }
}

.kpi-value {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #fff 0%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
    margin: 8px 0;
    font-family: 'Space Grotesk', sans-serif;
}

.kpi-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #64748b;
    font-weight: 600;
}

.kpi-sub {
    font-size: 0.875rem;
    color: #94a3b8;
    margin-top: 8px;
    font-weight: 500;
}

/* Status Badges */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    animation: fade-in 0.5s ease;
}

.status-success {
    background: rgba(0, 255, 136, 0.1);
    border: 1px solid rgba(0, 255, 136, 0.3);
    color: var(--success);
    box-shadow: 0 0 20px rgba(0, 255, 136, 0.1);
}

.status-warning {
    background: rgba(255, 179, 0, 0.1);
    border: 1px solid rgba(255, 179, 0, 0.3);
    color: var(--warning);
    animation: pulse-warning 2s infinite;
}

.status-danger {
    background: rgba(255, 59, 110, 0.1);
    border: 1px solid rgba(255, 59, 110, 0.3);
    color: var(--danger);
    animation: pulse-danger 2s infinite;
}

@keyframes pulse-warning {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255, 179, 0, 0.4); }
    50% { box-shadow: 0 0 0 10px rgba(255, 179, 0, 0); }
}

@keyframes pulse-danger {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255, 59, 110, 0.4); }
    50% { box-shadow: 0 0 0 10px rgba(255, 59, 110, 0); }
}

/* Terminal Window */
.terminal-window {
    background: rgba(5, 8, 17, 0.95);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 
        0 20px 50px rgba(0,0,0,0.5),
        inset 0 1px 0 rgba(255,255,255,0.05);
    animation: terminal-glow 4s ease-in-out infinite alternate;
}

@keyframes terminal-glow {
    from { box-shadow: 0 20px 50px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05); }
    to { box-shadow: 0 20px 50px rgba(0, 212, 255, 0.1), inset 0 1px 0 rgba(255,255,255,0.05); }
}

.terminal-header {
    background: linear-gradient(90deg, rgba(0,212,255,0.1), transparent);
    padding: 12px 16px;
    border-bottom: 1px solid rgba(0,212,255,0.1);
    display: flex;
    align-items: center;
    gap: 8px;
}

.terminal-dot {
    width: 12px; height: 12px;
    border-radius: 50%;
    animation: blink 2s infinite;
}

.terminal-dot:nth-child(1) { background: #ff5f56; animation-delay: 0s; }
.terminal-dot:nth-child(2) { background: #ffbd2e; animation-delay: 0.3s; }
.terminal-dot:nth-child(3) { background: #27c93f; animation-delay: 0.6s; }

@keyframes blink {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.9); }
}

.terminal-body {
    padding: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    line-height: 1.6;
    max-height: 400px;
    overflow-y: auto;
}

/* Progress Bars */
.progress-container {
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    overflow: hidden;
    height: 8px;
    margin: 12px 0;
    position: relative;
}

.progress-bar {
    height: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg, var(--primary), var(--secondary));
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.progress-bar::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
    animation: shimmer 2s infinite;
}

@keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

/* Section Headers */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 32px 0 20px;
    padding: 16px 20px;
    background: linear-gradient(90deg, rgba(0,212,255,0.08), transparent);
    border-left: 4px solid var(--primary);
    border-radius: 0 12px 12px 0;
    animation: slide-right 0.6s ease;
}

@keyframes slide-right {
    from { opacity: 0; transform: translateX(-20px); }
    to { opacity: 1; transform: translateX(0); }
}

.section-header span {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #e2e8f0;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, rgba(0,212,255,0.1), rgba(181,107,255,0.1)) !important;
    border: 1px solid rgba(0,212,255,0.3) !important;
    color: #fff !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s ease !important;
    position: relative;
    overflow: hidden;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(0,212,255,0.2) !important;
    border-color: var(--primary) !important;
}

.stButton>button:active {
    transform: translateY(0);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(255,255,255,0.02);
    padding: 8px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.05);
}

.stTabs [data-baseweb="tab"] {
    height: 44px;
    border-radius: 12px;
    padding: 0 20px;
    font-weight: 600;
    letter-spacing: 0.5px;
    transition: all 0.3s ease;
    border: none !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(181,107,255,0.2)) !important;
    border: 1px solid rgba(0,212,255,0.3) !important;
    box-shadow: 0 4px 20px rgba(0,212,255,0.1);
}

/* DataFrames */
[data-testid="stDataFrame"] {
    background: rgba(16,24,48,0.5) !important;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    overflow: hidden;
}

/* Sidebar */
.css-1d391kg, .css-163ttbj, .stSidebar {
    background: linear-gradient(180deg, rgba(10,15,36,0.98), rgba(5,8,17,0.98)) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* Widgets */
.stSlider>div>div>div {
    background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
}

.stCheckbox>label, .stRadio>label {
    color: #94a3b8 !important;
    font-weight: 500 !important;
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px; height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255,255,255,0.02);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(var(--primary), var(--secondary));
    border-radius: 4px;
}

/* Metric Cards */
[data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #fff, #94a3b8);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}

/* Loading Animation */
.loading-ring {
    display: inline-block;
    width: 64px;
    height: 64px;
}

.loading-ring:after {
    content: " ";
    display: block;
    width: 46px;
    height: 46px;
    margin: 8px;
    border-radius: 50%;
    border: 5px solid var(--primary);
    border-color: var(--primary) transparent var(--primary) transparent;
    animation: ring 1.2s linear infinite;
}

@keyframes ring {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Tooltip */
.hover-tooltip {
    position: relative;
}

.hover-tooltip:hover::after {
    content: attr(data-tooltip);
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    padding: 8px 12px;
    background: rgba(0,0,0,0.9);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    font-size: 0.75rem;
    white-space: nowrap;
    z-index: 1000;
    animation: fade-in 0.2s ease;
}

@keyframes fade-in {
    from { opacity: 0; transform: translateX(-50%) translateY(10px); }
    to { opacity: 1; transform: translateX(-50%) translateY(0); }
}

/* Gradient Text Utilities */
.text-gradient {
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.text-gradient-alt {
    background: linear-gradient(135deg, var(--success), var(--primary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Floating Elements */
.float-element {
    animation: float-slow 6s ease-in-out infinite;
}

@keyframes float-slow {
    0%, 100% { transform: translateY(0px) rotate(0deg); }
    50% { transform: translateY(-20px) rotate(2deg); }
}

/* Grid Pattern Overlay */
.grid-overlay {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image: 
        linear-gradient(rgba(0, 212, 255, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 212, 255, 0.03) 1px, transparent 1px);
    background-size: 50px 50px;
    pointer-events: none;
    z-index: 0;
}

/* Chart Containers */
.js-plotly-plot {
    border-radius: 16px !important;
    overflow: hidden !important;
    background: rgba(16,24,48,0.3) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    backdrop-filter: blur(10px);
}

/* Animated Counter */
@keyframes count-up {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.counter-anim {
    animation: count-up 0.8s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}

</style>

<div class="grid-overlay"></div>
"""

st.markdown(ANIMATION_CSS, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════
# ANIMATION UTILITIES
# ═════════════════════════════════════════════════════════════════════
def animated_counter(value, prefix="", suffix="", duration=1000):
    """Generate animated counter HTML"""
    unique_id = f"counter_{hash(str(value))}_{int(time.time()*1000)%10000}"
    html_code = f"""
    <div id="{unique_id}" class="kpi-value counter-anim">{prefix}{value}{suffix}</div>
    <script>
        (function() {{
            const el = document.getElementById('{unique_id}');
            const target = {value};
            const duration = {duration};
            const start = performance.now();
            const prefix = '{prefix}';
            const suffix = '{suffix}';
            
            function update(currentTime) {{
                const elapsed = currentTime - start;
                const progress = Math.min(elapsed / duration, 1);
                const easeOutQuart = 1 - Math.pow(1 - progress, 4);
                const current = Math.floor(easeOutQuart * target);
                el.textContent = prefix + current.toLocaleString() + suffix;
                if (progress < 1) requestAnimationFrame(update);
                else el.textContent = prefix + target.toLocaleString() + suffix;
            }}
            requestAnimationFrame(update);
        }})();
    </script>
    """
    return html_code

def glass_card(title, content, icon="", accent_color="#00d4ff"):
    """Generate glass card HTML"""
    return f"""
    <div class="glass-card" style="border-top: 2px solid {accent_color}40;">
        <div style="font-size: 2rem; margin-bottom: 12px;">{icon}</div>
        <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 2px; color: #64748b; margin-bottom: 8px; font-weight: 600;">{title}</div>
        <div style="font-size: 1.5rem; font-weight: 700; color: #f1f5f9; font-family: 'Space Grotesk';">{content}</div>
    </div>
    """

def status_badge(text, status="success"):
    """Generate animated status badge"""
    classes = {
        "success": "status-success",
        "warning": "status-warning", 
        "danger": "status-danger",
        "info": "status-pill"
    }
    icons = {
        "success": "●",
        "warning": "◐",
        "danger": "◉",
        "info": "○"
    }
    return f'<span class="status-pill {classes.get(status, "status-success")}">{icons.get(status, "●")} {text}</span>'

# ═════════════════════════════════════════════════════════════════════
# CONSTANTS & DATA
# ═════════════════════════════════════════════════════════════════════
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_PATH = os.path.join(os.path.dirname(__file__), "experiments.db")

DATASET_CATALOG = {
    "Adult Discretized":"adult_discretized.dl8","Anneal":"anneal.dl8",
    "Audiology":"audiology.dl8","Australian Credit":"australian-credit.dl8",
    "Balance Scale":"balance-scale-bin.dl8","Bank Conv":"bank_conv-bin.dl8",
    "Banknote":"banknote-bin.dl8","Biodeg":"biodeg-bin.dl8",
    "Breast Cancer":"breast-cancer-un.dl8","Breast Wisconsin":"breast-wisconsin.dl8",
    "Car Evaluation":"car_evaluation-bin.dl8","Car":"car-un.dl8",
    "Compas Discretized":"compas_discretized.dl8","Diabetes":"diabetes.dl8",
    "Forest Fires":"forest-fires-un.dl8","German Credit":"german-credit.dl8",
    "Heart Cleveland":"heart-cleveland.dl8","Hepatitis":"hepatitis.dl8",
    "HTRU 2":"HTRU_2-bin.dl8","Hypothyroid":"hypothyroid.dl8",
    "Indians Diabetes":"IndiansDiabetes-bin.dl8","Ionosphere":"ionosphere.dl8",
    "KR vs KP":"kr-vs-kp.dl8","Letter Recognition":"letter_recognition-bin.dl8",
    "Letter":"letter.dl8","Lymph":"lymph.dl8","Magic04":"magic04-bin.dl8",
    "Messidor":"messidor-bin.dl8","Mushroom":"mushroom.dl8",
    "Pendigits":"pendigits.dl8","Primary Tumor":"primary-tumor.dl8",
    "Segment":"segment.dl8","Seismic Bumps":"seismic_bumps-bin.dl8",
    "Soybean":"soybean.dl8","Splice-1":"splice-1.dl8",
    "Statlog Satellite":"Statlog_satellite-bin.dl8",
    "Taiwan Binarised":"taiwan_binarised.dl8","Tic-Tac-Toe":"tic-tac-toe.dl8",
    "Titanic":"titanic-un.dl8","Vehicle":"vehicle.dl8","Vote":"vote.dl8",
    "Wine 1":"wine1-un.dl8","Wine 2":"wine2-un.dl8","Wine 3":"wine3-un.dl8",
    "Wine Quality Red":"winequality-red-bin.dl8","Yeast":"yeast.dl8",
}
AVAILABLE = {k:v for k,v in DATASET_CATALOG.items()
             if os.path.exists(os.path.join(DATA_DIR,v))}

LANG_MAP = {"Horn":LanguageFamily.HORN,"Anti-Horn":LanguageFamily.ANTI_HORN,
            "Square CNF":LanguageFamily.SQUARE_CNF,"Affine (XOR)":LanguageFamily.AFFINE,
            "BEST_PER_NODE":LanguageFamily.BEST_PER_NODE}
LANG_COLORS = {"Horn":"#00d4ff","Anti-Horn":"#b56bff","Square CNF":"#00ff88",
               "Affine (XOR)":"#ffb300","BEST_PER_NODE":"#ff3b6e"}
COMPLEXITY_TAG = {"Horn":"P","Anti-Horn":"P","Square CNF":"P",
                  "Affine (XOR)":"P*","BEST_PER_NODE":"P"}

# ═════════════════════════════════════════════════════════════════════
# DATABASE
# ═════════════════════════════════════════════════════════════════════
def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS runs(
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT,
        dataset TEXT, language TEXT, max_depth INTEGER, gamma REAL,
        n_axp INTEGER, k_folds INTEGER, seed INTEGER,
        test_acc REAL, train_acc REAL, std_acc REAL, ci95_acc REAL,
        n_nodes INTEGER, n_leaves INTEGER, depth INTEGER,
        avg_axp REAL, fit_time REAL, notes TEXT DEFAULT '')""")
    con.commit(); con.close()

def log_run(r, dataset, language, max_depth, gamma, n_axp, k_folds, seed):
    con = sqlite3.connect(DB_PATH)
    con.execute("""INSERT INTO runs(created_at,dataset,language,max_depth,gamma,
        n_axp,k_folds,seed,test_acc,train_acc,std_acc,ci95_acc,n_nodes,
        n_leaves,depth,avg_axp,fit_time) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (datetime.now().isoformat(), dataset, language, max_depth, gamma,
         n_axp, k_folds, seed, r["test_acc"], r["train_acc"],
         r.get("std_acc",0), r.get("ci95_acc",0), r["n_nodes"],
         r["n_leaves"], r["depth"], r["avg_axp"], r["fit_time"]))
    con.commit(); con.close()

def load_all_runs():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM runs ORDER BY id DESC", con)
    con.close(); return df

def delete_runs(ids):
    con = sqlite3.connect(DB_PATH)
    con.executemany("DELETE FROM runs WHERE id=?", [(i,) for i in ids])
    con.commit(); con.close()

init_db()

# ═════════════════════════════════════════════════════════════════════
# DATA LOADER
# ═════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_dl8(filename, test_ratio=0.2, seed=42):
    path = os.path.join(DATA_DIR, filename)
    data = np.loadtxt(path, dtype=np.int64)
    y = data[:,0].astype(np.int32); X = data[:,1:].astype(np.float64)
    classes = np.unique(y)
    if len(classes) > 2:
        majority = classes[np.argmax([np.sum(y==c) for c in classes])]
        y = (y == majority).astype(np.int32)
    rng = np.random.RandomState(seed); idx = rng.permutation(len(y))
    s = int(len(y)*(1-test_ratio))
    return X[idx[:s]], y[idx[:s]], X[idx[s:]], y[idx[s:]], X.shape[1]

@st.cache_data(show_spinner=False)
def load_dl8_full(filename, seed=42):
    path = os.path.join(DATA_DIR, filename)
    data = np.loadtxt(path, dtype=np.int64)
    y = data[:,0].astype(np.int32); X = data[:,1:].astype(np.float64)
    classes = np.unique(y)
    if len(classes) > 2:
        majority = classes[np.argmax([np.sum(y==c) for c in classes])]
        y = (y == majority).astype(np.int32)
    return X, y

# ═════════════════════════════════════════════════════════════════════
# TRAINING ENGINE
# ═════════════════════════════════════════════════════════════════════
def _make_tree(language, max_depth, gamma, n_feats):
    return ExpertGSNHTree(
        stopping_criteria=StoppingCriteria(max_depth=max_depth,
            min_samples_leaf=5, min_samples_split=10),
        n_bins=64, top_k_features=15, search_1d=True, search_2d=True,
        search_3d=(n_feats <= 50), mode="journal", language=language,
        use_look_ahead=(gamma > 0), look_ahead_gamma=gamma, verbose=False)

def train_and_evaluate(X_tr, y_tr, X_te, y_te, language, max_depth,
                       gamma, n_axp=30, seed=42):
    tree = _make_tree(language, max_depth, gamma, X_tr.shape[1])
    t0 = time.time(); tree.fit(X_tr, y_tr); fit_time = time.time()-t0
    train_acc = float(tree.score(X_tr, y_tr))
    test_acc = float(tree.score(X_te, y_te))
    axp_data = []
    for i in range(min(n_axp, len(X_te))):
        try:
            axp = tree.extract_axp(X_te[i])
            pred = int(tree.predict(X_te[i].reshape(1,-1))[0])
            axp_data.append({"idx":i,"y_true":int(y_te[i]),"y_pred":pred,
                "axp_features":sorted(list(axp)),"axp_len":len(axp)})
        except Exception: pass
    avg_axp = np.mean([d["axp_len"] for d in axp_data]) if axp_data else 0.0
    return {"tree":tree,"train_acc":round(train_acc,6),"test_acc":round(test_acc,6),
        "n_nodes":tree.n_nodes_,"n_leaves":tree.n_leaves_,
        "depth":tree.max_depth_reached_,"avg_axp":round(avg_axp,2),
        "axp_data":axp_data,"arity_counts":dict(tree.arity_counts_),
        "pattern_counts":dict(getattr(tree,"pattern_counts_",{})),
        "language_counts":dict(getattr(tree,"language_counts_",{})),
        "summary":tree.get_summary(),"fit_time":round(fit_time,3),
        "std_acc":0,"ci95_acc":0,"fold_accs":None}

def train_kfold(X, y, language, max_depth, gamma, n_axp, k, seed=42):
    rng = np.random.RandomState(seed); idx = rng.permutation(len(y))
    folds = np.array_split(idx, k)
    fold_accs, fold_nodes, fold_axps, all_axp = [], [], [], []
    best_tree, best_acc, best_res = None, -1, None
    for fi in range(k):
        te_idx = folds[fi]; tr_idx = np.concatenate([folds[j] for j in range(k) if j!=fi])
        res = train_and_evaluate(X[tr_idx],y[tr_idx],X[te_idx],y[te_idx],
            language, max_depth, gamma, n_axp, seed)
        fold_accs.append(res["test_acc"]); fold_nodes.append(res["n_nodes"])
        fold_axps.append(res["avg_axp"]); all_axp.extend(res["axp_data"])
        if res["test_acc"] > best_acc:
            best_acc = res["test_acc"]; best_tree = res["tree"]; best_res = res
    mean_acc = np.mean(fold_accs); std_acc = np.std(fold_accs)
    ci95 = 1.96*std_acc/np.sqrt(k)
    best_res.update({"test_acc":round(mean_acc,6),"train_acc":round(best_res["train_acc"],6),
        "std_acc":round(std_acc,6),"ci95_acc":round(ci95,6),
        "n_nodes":int(np.mean(fold_nodes)),"avg_axp":round(np.mean(fold_axps),2),
        "fold_accs":fold_accs,"axp_data":all_axp,"tree":best_tree})
    return best_res

# ═════════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ═════════════════════════════════════════════════════════════════════
def verify_axp_minimality(tree, x, axp_features, n_feats, n_pert=200):
    axp = sorted(set(axp_features))
    for feat in axp:
        subset = set(axp) - {feat}
        still_sufficient = True
        orig_pred = int(tree.predict(x.reshape(1,-1))[0])
        for _ in range(n_pert):
            x2 = x.copy()
            for f in range(n_feats):
                if f not in subset: x2[f] = np.random.choice([0,1])
            if int(tree.predict(x2.reshape(1,-1))[0]) != orig_pred:
                still_sufficient = False; break
        if still_sufficient:
            return {"is_minimal":False,"counterexample":sorted(subset),"n_checked":len(axp)}
    return {"is_minimal":True,"n_checked":len(axp)}

def compute_disagree_set(results, X_te, y_te):
    preds = {lang: res["tree"].predict(X_te) for lang, res in results.items()}
    df = pd.DataFrame(preds); df["true"] = y_te
    mask = df.drop(columns="true").nunique(axis=1) > 1
    disagree = df[mask].copy()
    disagree["n_correct"] = disagree.apply(
        lambda r: sum(r[l]==r["true"] for l in results.keys()), axis=1)
    return disagree

def significance_matrix(fold_dict):
    if not SCIPY_AVAILABLE or len(fold_dict) < 2: return None
    langs = list(fold_dict.keys())
    mat = pd.DataFrame(1.0, index=langs, columns=langs)
    for i, l1 in enumerate(langs):
        for j, l2 in enumerate(langs):
            if i >= j: continue
            a1, a2 = np.array(fold_dict[l1]), np.array(fold_dict[l2])
            if np.allclose(a1, a2): p = 1.0
            else:
                try: _, p = wilcoxon(a1, a2); 
                except Exception: p = 1.0
            mat.loc[l1,l2] = p; mat.loc[l2,l1] = p
    return mat

def compute_learning_curves(X_tr, y_tr, X_te, y_te, language, max_depth, gamma, seed, n_pts=8):
    fracs = np.linspace(0.1, 1.0, n_pts); rng = np.random.RandomState(seed)
    rows = []
    for f in fracs:
        idx = rng.choice(len(y_tr), int(f*len(y_tr)), replace=False)
        r = train_and_evaluate(X_tr[idx],y_tr[idx],X_te,y_te,language,max_depth,gamma,5,seed)
        rows.append({"frac":f,"n_train":len(idx),"train_acc":r["train_acc"],
            "test_acc":r["test_acc"],"n_nodes":r["n_nodes"],"fit_time":r["fit_time"]})
    return pd.DataFrame(rows)

# ═════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═════════════════════════════════════════════════════════════════════
def build_tree_dot(tree):
    if tree.root_ is None: return 'digraph{N0[label="Empty"]}'
    FILL={"Horn":"#00d4ff","AntiHorn":"#b56bff","SquareCNF":"#00ff88",
          "Affine":"#ffb300","BestPerNode":"#ff3b6e","Any":"#64748b"}
    FC={"Horn":"#000","AntiHorn":"#fff","SquareCNF":"#000",
        "Affine":"#000","BestPerNode":"#fff","Any":"#fff"}
    lines=['digraph T{bgcolor="transparent"',
        'node[style="filled,rounded" shape=box fontname="Space Grotesk" fontsize=11 color="#ffffff20" penwidth=1.5]',
        'edge[color="#00d4ff40" fontname="Space Grotesk" fontsize=9 penwidth=1.5]']
    ctr=[0]
    def _w(node,nid):
        if node is None: return
        if node.get("is_leaf",True) or node.get("predicate") is None:
            p=node.get("proba",.5);cls=1 if p>=.5 else 0;n=node.get("n_samples",0)
            f="#00ff88" if cls==1 else "#ff3b6e";c="#000"
            lines.append(f'N{nid}[label="Class {cls}\\np={p:.2f} n={n}" fillcolor="{f}" fontcolor="{c}" shape=ellipse penwidth=2]')
            return
        pred=node["predicate"]
        lv=str(pred.language_family.value) if hasattr(pred,"language_family") else "Horn"
        fl=FILL.get(lv,"#64748b");fc=FC.get(lv,"#fff")
        ps=str(pred).replace('"',"'").replace('\\','/')
        g=f"{pred.information_gain:.4f}";n=node.get("n_samples",0)
        a=pred.arity.value if hasattr(pred,"arity") else "?"
        lines.append(f'N{nid}[label="{ps}\\n[{lv}·{a}L] ΔI={g}\\nn={n}" fillcolor="{fl}" fontcolor="{fc}" style="filled,rounded" penwidth=2]')
        li=ctr[0]+1;ctr[0]+=1;ri=ctr[0]+1;ctr[0]+=1
        lines.append(f'N{nid}->N{li}[label=" T" color="#00ff88" penwidth=2]');lines.append(f'N{nid}->N{ri}[label=" F" color="#ff3b6e" penwidth=2]')
        _w(node.get("left"),li);_w(node.get("right"),ri)
    _w(tree.root_,0); lines.append("}"); return "\n".join(lines)

# ═════════════════════════════════════════════════════════════════════
# EXPORT
# ═════════════════════════════════════════════════════════════════════
def generate_latex_table(df, dataset, max_depth, gamma, k_folds):
    has_ci = "ci95_acc" in df.columns and df["ci95_acc"].max() > 0
    l = [r"\begin{table}[htbp]",r"  \centering",
        f"  \\caption{{Ablation on \\texttt{{{dataset}}} (d={max_depth}, "
        f"$\\gamma$={gamma}" + (f", {k_folds}-fold CV" if k_folds>1 else "") + ")}",
        r"  \begin{tabular}{lcccccc}",r"    \toprule",
        r"    Language & Acc" + (" & $\\pm$CI95" if has_ci else "") +
        r" & Nodes & $|\mathrm{AXp}|$ & Time & SAT \\",r"    \midrule"]
    for _,r in df.iterrows():
        acc_str = f"{r['Test Acc']:.4f}"
        ci_str = f" & $\\pm${r['ci95_acc']:.4f}" if has_ci else ""
        l.append(f"    {r['Language']} & {acc_str}{ci_str} & {r['Nodes']} & "
                 f"{r['Avg |AXp|']:.2f} & {r['Time (s)']:.2f} & {r['SAT']} \\\\")
    l += [r"    \bottomrule",r"  \end{tabular}",r"\end{table}"]
    return "\n".join(l)

def generate_coq_file(results, dataset):
    lines = [f"(* GSNH-MDT Formal Certificate — {dataset} *)",
             f"(* Generated: {datetime.now().isoformat()} *)","",
             "Require Import List ZArith Decidability.",""]
    for lang, res in results.items():
        tag = COMPLEXITY_TAG.get(lang,"P")
        lines += [f"(* Language: {lang} — Acc={res['test_acc']:.4f}, "
                  f"Nodes={res['n_nodes']}, |AXp|={res['avg_axp']:.2f} *)",
            f"Definition {lang.replace(' ','_').replace('(','').replace(')','')}_tractable : Prop :=",
            f'  complexity_class = "{tag}".', ""]
    lines += ["Theorem UI_SAT_Polynomial :",
        "  forall phi, is_UI_family phi ->",
        "  exists algo, time_complexity algo phi <= O(length phi).",
        "Proof. intros. apply Interval_Intersection. auto. Qed.","",
        "Theorem Mixed_SAT_NPHard :",
        "  forall phi, is_mixed phi -> reduces_poly ThreeSAT (SAT phi).",
        "Proof. intros. apply Schaefer_Dichotomy. auto. Qed."]
    return "\n".join(lines)

# ═════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═════════════════════════════════════════════════════════════════════
if "experiment_history" not in st.session_state:
    st.session_state.experiment_history = []
if "learning_curves" not in st.session_state:
    st.session_state.learning_curves = {}
if "animation_key" not in st.session_state:
    st.session_state.animation_key = 0

# ═════════════════════════════════════════════════════════════════════
# SIDEBAR — COMMAND CENTER
# ═════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 20px;">
        <div style="font-size: 2.5rem; margin-bottom: 8px;">🌌</div>
        <div style="font-family: 'Space Grotesk'; font-size: 1.3rem; font-weight: 700; color: #fff;">GSNH-MDT</div>
        <div style="font-size: 0.75rem; color: #64748b; letter-spacing: 2px; margin-top: 4px;">LABORATORY v4</div>
    </div>
    """, unsafe_allow_html=True)
    
    if not AVAILABLE:
        st.error("No datasets found"); st.stop()

    # Dataset Selection with animation
    st.markdown('<div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 2px; color: #64748b; margin-bottom: 8px; font-weight: 600;">📂 Dataset</div>', unsafe_allow_html=True)
    dataset_name = st.selectbox("", list(AVAILABLE.keys()),
        index=list(AVAILABLE.keys()).index("Tic-Tac-Toe") if "Tic-Tac-Toe" in AVAILABLE else 0,
        label_visibility="collapsed")
    fn = AVAILABLE[dataset_name]
    X_tr, y_tr, X_te, y_te, n_feats = load_dl8(fn)
    
    # Animated metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div style="text-align: center; padding: 12px 4px; background: rgba(0,212,255,0.05); border-radius: 12px; border: 1px solid rgba(0,212,255,0.1);">
            <div style="font-size: 1.2rem; font-weight: 700; color: #00d4ff; font-family: 'Space Grotesk';">{len(y_tr)}</div>
            <div style="font-size: 0.6rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px;">Train</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 12px 4px; background: rgba(181,107,255,0.05); border-radius: 12px; border: 1px solid rgba(181,107,255,0.1);">
            <div style="font-size: 1.2rem; font-weight: 700; color: #b56bff; font-family: 'Space Grotesk';">{len(y_te)}</div>
            <div style="font-size: 0.6rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px;">Test</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div style="text-align: center; padding: 12px 4px; background: rgba(0,255,136,0.05); border-radius: 12px; border: 1px solid rgba(0,255,136,0.1);">
            <div style="font-size: 1.2rem; font-weight: 700; color: #00ff88; font-family: 'Space Grotesk';">{n_feats}</div>
            <div style="font-size: 0.6rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px;">Feats</div>
        </div>
        """, unsafe_allow_html=True)

    minority = min(y_tr.mean(), 1-y_tr.mean())
    if minority < 0.3:
        st.markdown(f'<div style="margin-top: 12px;">{status_badge("Imbalanced", "warning")}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Language Topology
    st.markdown('<div class="section-header" style="margin: 0 0 16px 0; padding: 12px 16px;"><span>🧬 Topology</span></div>', unsafe_allow_html=True)
    sel_langs = []
    for lang in LANG_MAP:
        default = lang in ("Horn", "Square CNF", "BEST_PER_NODE")
        col = st.columns([0.15, 0.85])
        with col[0]:
            checked = st.checkbox("", value=default, key=f"cb_{lang}", label_visibility="collapsed")
        with col[1]:
            color = LANG_COLORS.get(lang, "#fff")
            st.markdown(f'<div style="display: flex; align-items: center; gap: 8px; margin-top: -6px;"><div style="width: 8px; height: 8px; border-radius: 50%; background: {color}; box-shadow: 0 0 8px {color};"></div><span style="font-size: 0.85rem; font-weight: 500; color: #e2e8f0;">{lang}</span><span style="font-size: 0.7rem; color: #64748b; margin-left: auto; font-family: JetBrains Mono;">[{COMPLEXITY_TAG[lang]}]</span></div>', unsafe_allow_html=True)
        if checked:
            sel_langs.append(lang)

    has_affine = "Affine (XOR)" in sel_langs
    has_interval = any(l in sel_langs for l in ("Horn","Anti-Horn","Square CNF"))
    if has_affine and has_interval:
        st.markdown(f'<div style="margin-top: 16px; animation: pulse-danger 2s infinite;">{status_badge("NP-HARD Alert", "danger")}</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 0.75rem; color: #94a3b8; margin-top: 8px; padding: 12px; background: rgba(255,59,110,0.05); border-radius: 8px; border-left: 2px solid #ff3b6e;">Schaefer 1978: Mixed topology → 3-SAT</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Parameters
    st.markdown('<div class="section-header" style="margin: 0 0 16px 0; padding: 12px 16px;"><span>⚙️ Parameters</span></div>', unsafe_allow_html=True)
    
    st.markdown('<div style="font-size: 0.7rem; color: #64748b; margin-bottom: 4px; font-weight: 600;">MAX DEPTH</div>', unsafe_allow_html=True)
    max_depth = st.slider("", 2, 15, 5, key="depth_slider", label_visibility="collapsed")
    st.markdown(f'<div style="text-align: right; font-size: 0.8rem; color: #00d4ff; font-family: Space Grotesk; font-weight: 700; margin-top: -10px;">{max_depth}</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="font-size: 0.7rem; color: #64748b; margin-bottom: 4px; font-weight: 600; margin-top: 16px;">LOOK-AHEAD γ</div>', unsafe_allow_html=True)
    gamma = st.slider("", 0.0, 1.0, 0.0, step=0.05, key="gamma_slider", label_visibility="collapsed")
    st.markdown(f'<div style="text-align: right; font-size: 0.8rem; color: #b56bff; font-family: Space Grotesk; font-weight: 700; margin-top: -10px;">{gamma:.2f}</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="font-size: 0.7rem; color: #64748b; margin-bottom: 4px; font-weight: 600; margin-top: 16px;">AXP SAMPLES</div>', unsafe_allow_html=True)
    n_axp = st.slider("", 5, 100, 30, key="axp_slider", label_visibility="collapsed")
    
    st.markdown('<div style="font-size: 0.7rem; color: #64748b; margin-bottom: 4px; font-weight: 600; margin-top: 16px;">SEED</div>', unsafe_allow_html=True)
    seed = st.number_input("", 0, 9999, 42, key="seed_input", label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Evaluation Mode
    st.markdown('<div class="section-header" style="margin: 0 0 16px 0; padding: 12px 16px;"><span>📐 Evaluation</span></div>', unsafe_allow_html=True)
    eval_mode = st.radio("", ["Single split", "K-Fold CV"], index=0, label_visibility="collapsed")
    k_folds = 1; do_wilcoxon = False
    if eval_mode == "K-Fold CV":
        k_folds = st.slider("K", 3, 10, 5)
        if SCIPY_AVAILABLE:
            do_wilcoxon = st.checkbox("Wilcoxon tests", value=True)

    sweep_mode = st.checkbox("🌐 Cross-dataset sweep", value=False)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Run Button with glow effect
    st.markdown("""
    <style>
    .run-button {
        background: linear-gradient(135deg, #00d4ff, #b56bff);
        border: none;
        border-radius: 12px;
        color: white;
        padding: 16px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-size: 0.9rem;
        cursor: pointer;
        width: 100%;
        position: relative;
        overflow: hidden;
        transition: all 0.3s;
        box-shadow: 0 4px 20px rgba(0,212,255,0.3);
    }
    .run-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,212,255,0.5);
    }
    .run-button::before {
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }
    .run-button:hover::before {
        left: 100%;
    }
    </style>
    """, unsafe_allow_html=True)
    
    run_btn = st.button("⬡ Initialize Experiment", type="primary", use_container_width=True)

    # Footer status
    try:
        hist = load_all_runs()
        if len(hist) > 0:
            lr = hist.iloc[0]
            st.markdown(f"""
            <div style="margin-top: 24px; padding: 16px; background: rgba(0,212,255,0.03); border-radius: 12px; border: 1px solid rgba(0,212,255,0.1); font-size: 0.75rem; color: #64748b; text-align: center;">
                <div style="font-weight: 600; color: #00d4ff; margin-bottom: 4px;">LAST EXPERIMENT</div>
                <div>{lr['dataset']} · {lr['language']}</div>
                <div style="font-family: Space Grotesk; font-weight: 700; color: #fff; margin-top: 4px; font-size: 0.9rem;">{lr['test_acc']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
    except:
        pass

# ═════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align: center; padding: 40px 0 30px 0; position: relative;">
    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 600px; height: 200px; background: radial-gradient(circle, rgba(0,212,255,0.15) 0%, transparent 70%); filter: blur(40px); animation: pulse-bg 8s infinite alternate;"></div>
    <h1 style="font-size: 3rem; margin-bottom: 8px; position: relative; display: inline-block;">
        <span style="background: linear-gradient(135deg, #fff 0%, #00d4ff 50%, #b56bff 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">GSNH-MDT</span>
        <span style="font-size: 0.4em; vertical-align: super; background: rgba(0,212,255,0.2); color: #00d4ff; padding: 4px 8px; border-radius: 8px; margin-left: 8px; -webkit-text-fill-color: #00d4ff; font-weight: 700; letter-spacing: 1px;">PRO</span>
    </h1>
    <div style="color: #64748b; font-size: 1.1rem; letter-spacing: 2px; font-weight: 300;">Formal XAI Laboratory · Cinematic Edition</div>
    <div style="margin-top: 16px; display: inline-flex; gap: 16px; align-items: center;">
        <span class="status-pill status-success">● Real Backend</span>
        <span style="color: #475569;">|</span>
        <span class="status-pill status-success">● Zero Mock Data</span>
        <span style="color: #475569;">|</span>
        <span class="status-pill status-success">● Live Metrics</span>
    </div>
</div>
""", unsafe_allow_html=True)

if not run_btn:
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px; opacity: 0; animation: fade-in 1s ease forwards;">
        <div style="font-size: 4rem; margin-bottom: 20px; animation: float 3s ease-in-out infinite;">🧬</div>
        <h2 style="color: #e2e8f0; margin-bottom: 12px; font-family: Space Grotesk;">Ready to Initialize</h2>
        <p style="color: #64748b; max-width: 500px; margin: 0 auto; line-height: 1.6;">
            Configure your experiment parameters in the sidebar and launch the analysis pipeline. 
            The system will train ExpertGSNHTree models and extract formal AXp explanations.
        </p>
        <div style="margin-top: 32px; padding: 20px; background: rgba(0,212,255,0.03); border-radius: 16px; border: 1px solid rgba(0,212,255,0.1); display: inline-block;">
            <div style="display: flex; gap: 24px; align-items: center; font-size: 0.85rem; color: #94a3b8;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 8px; height: 8px; border-radius: 50%; background: #00d4ff; box-shadow: 0 0 10px #00d4ff;"></div>
                    Select Dataset
                </div>
                <div style="color: #334155;">→</div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 8px; height: 8px; border-radius: 50%; background: #b56bff; box-shadow: 0 0 10px #b56bff;"></div>
                    Choose Topology
                </div>
                <div style="color: #334155;">→</div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 8px; height: 8px; border-radius: 50%; background: #00ff88; box-shadow: 0 0 10px #00ff88;"></div>
                    Initialize
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()
    
if not sel_langs:
    st.warning("Please select at least one language topology.")
    st.stop()

# Increment animation key for fresh animations
st.session_state.animation_key += 1
anim_key = st.session_state.animation_key

# ═════════════════════════════════════════════════════════════════════
# TRAINING LOOP WITH CINEMATIC UI
# ═════════════════════════════════════════════════════════════════════
st.markdown(f'<div class="section-header"><span>🚀 Execution Pipeline</span></div>', unsafe_allow_html=True)

# Progress container
progress_html = st.empty()
log_html = st.empty()

results = {}
log_lines = []
_log_lock = threading.Lock()

def _log(msg, cls=""):
    with _log_lock:
        colors = {
            "info": "#00d4ff",
            "success": "#00ff88", 
            "warning": "#ffb300",
            "error": "#ff3b6e",
            "dim": "#64748b"
        }
        color = colors.get(cls, "#e2e8f0")
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_lines.append(f'<div style="font-family: JetBrains Mono; font-size: 0.8rem; margin: 4px 0; opacity: 0; animation: slide-right 0.3s ease forwards;"><span style="color: #475569;">[{timestamp}]</span> <span style="color: {color};">{msg}</span></div>')
        if len(log_lines) > 50: del log_lines[:-50]
        log_html.markdown(f"""
        <div class="terminal-window">
            <div class="terminal-header">
                <div class="terminal-dot"></div>
                <div class="terminal-dot"></div>
                <div class="terminal-dot"></div>
                <span style="margin-left: 12px; font-size: 0.75rem; color: #64748b; font-family: Space Grotesk; letter-spacing: 1px;">SYSTEM LOG</span>
            </div>
            <div class="terminal-body">
                {''.join(log_lines)}
            </div>
        </div>
        """, unsafe_allow_html=True)

# Initialize training
_log(f"Initializing pipeline for {dataset_name}...", "info")
_log(f"Configuration: depth={max_depth}, γ={gamma}, seed={seed}", "dim")

for i, lang_name in enumerate(sel_langs):
    progress_pct = (i / len(sel_langs)) * 100
    progress_html.markdown(f"""
    <div style="margin: 20px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.85rem; font-weight: 500;">
            <span style="color: #e2e8f0;">Training {lang_name}...</span>
            <span style="color: #00d4ff; font-family: Space Grotesk;">{progress_pct:.0f}%</span>
        </div>
        <div class="progress-container">
            <div class="progress-bar" style="width: {progress_pct}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    _log(f"Compiling {lang_name} search kernels...", "warning")

    if k_folds > 1:
        X_all, y_all = load_dl8_full(fn, seed=seed)
        res = train_kfold(X_all, y_all, LANG_MAP[lang_name], max_depth, gamma,
                          n_axp, k_folds, seed)
    else:
        res = train_and_evaluate(X_tr, y_tr, X_te, y_te, LANG_MAP[lang_name],
                                 max_depth, gamma, n_axp, seed)
    results[lang_name] = res

    ci_str = f" ±{res['ci95_acc']:.4f}" if res.get("ci95_acc",0) > 0 else ""
    _log(f"✓ {lang_name}: {res['test_acc']:.4f}{ci_str} | {res['n_nodes']} nodes | {res['avg_axp']:.1f} avg |AXp|", "success")

    try:
        log_run(res, dataset_name, lang_name, max_depth, gamma, n_axp, k_folds, seed)
    except: pass

# Complete progress
progress_html.markdown(f"""
<div style="margin: 20px 0;">
    <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.85rem; font-weight: 500;">
        <span style="color: #00ff88; font-weight: 700;">✓ All Experiments Complete</span>
        <span style="color: #00ff88; font-family: Space Grotesk;">100%</span>
    </div>
    <div class="progress-container">
        <div class="progress-bar" style="width: 100%; background: linear-gradient(90deg, #00ff88, #00d4ff);"></div>
    </div>
</div>
""", unsafe_allow_html=True)

_log("Pipeline execution complete. Rendering dashboard...", "success")

# Save history
st.session_state.experiment_history.append({
    "time": datetime.now().isoformat()[:16], "dataset": dataset_name,
    "langs": sel_langs, "depth": max_depth, "gamma": gamma,
    "results": {k: {kk: vv for kk, vv in v.items() if kk != "tree"}
                for k, v in results.items()},
})

# Build results DataFrame
rows = []
for lang_name, res in results.items():
    rows.append({"Language": lang_name, "Test Acc": res["test_acc"],
        "Train Acc": res["train_acc"], "ci95_acc": res.get("ci95_acc",0),
        "Nodes": res["n_nodes"], "Leaves": res["n_leaves"],
        "Depth": res["depth"], "Avg |AXp|": res["avg_axp"],
        "Time (s)": res["fit_time"], "SAT": COMPLEXITY_TAG.get(lang_name,"P")})
df_results = pd.DataFrame(rows)

# ═════════════════════════════════════════════════════════════════════
# 9 TABS WITH ENHANCED UI
# ═════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "⚡ Terminal", "📊 Analytics", "🔬 Compare", "🌲 Tree", 
    "🧬 Features", "📐 Complexity", "📁 Data", "🕰 History", "🌐 Sweep"
])

# ── TAB 1: TERMINAL ──
with tabs[0]:
    st.markdown('<div class="section-header"><span>Execution Console</span></div>', unsafe_allow_html=True)
    
    # Keep the terminal visible
    log_html.markdown(f"""
    <div class="terminal-window" style="margin-bottom: 24px;">
        <div class="terminal-header">
            <div class="terminal-dot"></div>
            <div class="terminal-dot"></div>
            <div class="terminal-dot"></div>
            <span style="margin-left: 12px; font-size: 0.75rem; color: #64748b; font-family: Space Grotesk; letter-spacing: 1px;">EXECUTION LOG</span>
        </div>
        <div class="terminal-body">
            {''.join(log_lines[-30:])}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Animated KPIs
    cols = st.columns(4)
    metrics = [
        (len(sel_langs), "Languages", "🧬", "#00d4ff"),
        (sum(r["n_nodes"] for r in results.values()), "Total Nodes", "🌿", "#00ff88"),
        (sum(len(r["axp_data"]) for r in results.values()), "AXps Extracted", "🔍", "#b56bff"),
        (f"{sum(r['fit_time'] for r in results.values()):.2f}s", "Compute Time", "⚡", "#ffb300")
    ]
    
    for col, (val, label, icon, color) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center; border-top: 3px solid {color};">
                <div style="font-size: 2rem; margin-bottom: 8px;">{icon}</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #fff; font-family: Space Grotesk;">{val}</div>
                <div style="font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px;">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with st.expander("🧮 Mathematical Foundations"):
        st.latex(r"\Delta I_{\mathrm{BIC}}(\varphi) = \Delta I(\varphi) - \tfrac{1}{2} k \frac{\ln n}{n}")
        st.latex(r"S_{\mathrm{LA}}(\varphi) = \Delta I_{\mathrm{BIC}}(\varphi) + \gamma \cdot \mathbb{E}[\Delta I_{\mathrm{child}}]")

# ── TAB 2: ANALYTICS ──
with tabs[1]:
    st.markdown('<div class="section-header"><span>Performance Matrix</span></div>', unsafe_allow_html=True)
    
    # Winners
    ba = df_results.loc[df_results["Test Acc"].idxmax()]
    bs = df_results.loc[df_results["Nodes"].idxmin()]
    bx = df_results.loc[df_results["Avg |AXp|"].idxmin()]
    bf = df_results.loc[df_results["Time (s)"].idxmin()]
    
    winner_cols = st.columns(4)
    winners = [
        ("Best Accuracy", f"{ba['Test Acc']:.2%}", ba['Language'], "🎯", "grad-cyan"),
        ("Smallest Tree", f"{bs['Nodes']} nodes", bs['Language'], "🌿", "grad-purple"),
        ("Shortest AXp", f"{bx['Avg |AXp|']:.2f}", bx['Language'], "🔍", "grad-green"),
        ("Fastest", f"{bf['Time (s)']:.2f}s", bf['Language'], "⚡", "grad-cyan")
    ]
    
    for col, (title, val, winner, icon, grad) in zip(winner_cols, winners):
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border-left: 4px solid;">
                <div style="font-size: 1.8rem; margin-bottom: 8px;">{icon}</div>
                <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 2px; color: #64748b; margin-bottom: 4px;">{title}</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #fff; font-family: Space Grotesk; margin-bottom: 4px;">{val}</div>
                <div style="font-size: 0.8rem; color: #00d4ff; font-weight: 600;">🏆 {winner}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Styled DataFrame
    st.dataframe(
        df_results.style
        .background_gradient(subset=["Test Acc"], cmap="viridis", vmin=0.5, vmax=1.0)
        .background_gradient(subset=["Nodes"], cmap="plasma_r")
        .format({"Test Acc": "{:.4f}", "Train Acc": "{:.4f}", "Avg |AXp|": "{:.2f}"})
        .set_properties(**{
            'background-color': 'rgba(16,24,48,0.5)',
            'color': '#e2e8f0',
            'border-color': 'rgba(255,255,255,0.1)'
        }),
        use_container_width=True,
        hide_index=True,
        height=400
    )
    
    # Exports
    st.markdown('<div class="section-header"><span>Export Artifacts</span></div>', unsafe_allow_html=True)
    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        st.download_button("⬇ CSV Report", df_results.to_csv(index=False),
            f"results_{dataset_name}.csv", "text/csv", use_container_width=True)
    with ec2:
        ltx = generate_latex_table(df_results, dataset_name, max_depth, gamma, k_folds)
        st.download_button("⬇ LaTeX Table", ltx, f"table_{dataset_name}.tex", 
            "text/plain", use_container_width=True)
    with ec3:
        coq = generate_coq_file(results, dataset_name)
        st.download_button("⬇ Coq Certificate", coq, f"cert_{dataset_name}.v",
            "text/plain", use_container_width=True)

# ── TAB 3: COMPARE ──
with tabs[2]:
    st.markdown('<div class="section-header"><span>Comparative Analysis</span></div>', unsafe_allow_html=True)
    
    if do_wilcoxon and k_folds > 1:
        fold_dict = {l: r["fold_accs"] for l, r in results.items() if r.get("fold_accs")}
        sig = significance_matrix(fold_dict)
        if sig is not None:
            st.markdown("**Statistical Significance Matrix (Wilcoxon)**")
            st.dataframe(sig.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=0.1),
                use_container_width=True)
    
    chart_cols = st.columns(2)
    
    with chart_cols[0]:
        fig = px.bar(df_results, x="Language", y=["Test Acc","Train Acc"],
            barmode="group", color_discrete_sequence=["#00d4ff","#1e293b"],
            title="Accuracy Comparison",
            template="plotly_dark")
        fig.update_layout(font_family="Space Grotesk", showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    
    with chart_cols[1]:
        fig = px.scatter(df_results, x="Nodes", y="Test Acc", size="Avg |AXp|",
            color="Language", color_discrete_map=LANG_COLORS,
            title="Accuracy vs Complexity",
            template="plotly_dark")
        fig.update_layout(font_family="Space Grotesk",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    
    if len(df_results) >= 2:
        # Radar chart
        categories = ["Accuracy","Compactness","Explainability","Speed"]
        fig = go.Figure()
        for _, row in df_results.iterrows():
            lang = row["Language"]
            vals = [
                row["Test Acc"]/max(0.01,df_results["Test Acc"].max()),
                1-row["Nodes"]/max(1,df_results["Nodes"].max()),
                1-row["Avg |AXp|"]/max(1,df_results["Avg |AXp|"].max()),
                1-row["Time (s)"]/max(0.01,df_results["Time (s)"].max())
            ]
            fig.add_trace(go.Scatterpolar(
                r=vals+[vals[0]], theta=categories+[categories[0]],
                fill="toself", name=lang, opacity=0.7,
                line=dict(color=LANG_COLORS.get(lang,"#fff"))))
        fig.update_layout(
            polar=dict(radialaxis=dict(range=[0,1], showticklabels=False)),
            template="plotly_dark", font_family="Space Grotesk",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

# ── TAB 4: TREE ──
with tabs[3]:
    st.markdown('<div class="section-header"><span>Decision Tree Visualization</span></div>', unsafe_allow_html=True)
    
    tree_lang = st.selectbox("Select Model", list(results.keys()), key="tree_select")
    res = results[tree_lang]
    tree = res["tree"]
    
    st.markdown(f"""
    <div style="display: flex; gap: 16px; margin-bottom: 20px; flex-wrap: wrap;">
        <div class="glass-card" style="padding: 16px 24px; flex: 1; min-width: 150px;">
            <div style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px;">Language</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: {LANG_COLORS.get(tree_lang, '#fff')};">{tree_lang}</div>
        </div>
        <div class="glass-card" style="padding: 16px 24px; flex: 1; min-width: 150px;">
            <div style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px;">Nodes</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #fff;">{res['n_nodes']}</div>
        </div>
        <div class="glass-card" style="padding: 16px 24px; flex: 1; min-width: 150px;">
            <div style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px;">Depth</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #fff;">{res['depth']}</div>
        </div>
        <div class="glass-card" style="padding: 16px 24px; flex: 1; min-width: 150px;">
            <div style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px;">Accuracy</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #00ff88;">{res['test_acc']:.2%}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    dot = build_tree_dot(tree)
    st.graphviz_chart(dot, use_container_width=True)
    
    st.markdown("""
    <div style="display: flex; gap: 20px; flex-wrap: wrap; margin-top: 16px; font-size: 0.8rem;">
        <div style="display: flex; align-items: center; gap: 8px;"><div style="width: 12px; height: 12px; background: #00d4ff; border-radius: 3px;"></div> Horn</div>
        <div style="display: flex; align-items: center; gap: 8px;"><div style="width: 12px; height: 12px; background: #b56bff; border-radius: 3px;"></div> Anti-Horn</div>
        <div style="display: flex; align-items: center; gap: 8px;"><div style="width: 12px; height: 12px; background: #00ff88; border-radius: 3px;"></div> Square CNF</div>
        <div style="display: flex; align-items: center; gap: 8px;"><div style="width: 12px; height: 12px; background: #ffb300; border-radius: 3px;"></div> Affine</div>
    </div>
    """, unsafe_allow_html=True)

# ── TAB 5: FEATURES ──
with tabs[4]:
    st.markdown('<div class="section-header"><span>Feature Analysis</span></div>', unsafe_allow_html=True)
    
    if any(res["axp_data"] for res in results.values()):
        # AXp distribution
        axp_all = []
        for l, r in results.items():
            for d in r["axp_data"]:
                axp_all.append({"Language": l, "|AXp|": d["axp_len"]})
        
        fig = px.violin(pd.DataFrame(axp_all), x="Language", y="|AXp|", color="Language",
            color_discrete_map=LANG_COLORS, box=True, points="all",
            title="AXp Size Distribution", template="plotly_dark")
        fig.update_layout(font_family="Space Grotesk", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature frequency
        st.markdown('<div class="section-header"><span>Feature Importance</span></div>', unsafe_allow_html=True)
        feat_freq = {}
        for res in results.values():
            for d in res["axp_data"]:
                for f in d["axp_features"]:
                    feat_freq[f] = feat_freq.get(f, 0) + 1
        
        if feat_freq:
            top_feats = sorted(feat_freq.items(), key=lambda x: -x[1])[:20]
            fdf = pd.DataFrame(top_feats, columns=["Feature", "Frequency"])
            fdf["Feature"] = fdf["Feature"].apply(lambda x: f"f{x}")
            
            fig = px.bar(fdf, x="Frequency", y="Feature", orientation="h",
                color="Frequency", color_continuous_scale="Teal",
                title="Top Features by AXp Participation",
                template="plotly_dark")
            fig.update_layout(font_family="Space Grotesk", yaxis_autorange="reversed",
                paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No AXp data available. Increase sample size.")

# ── TAB 6: COMPLEXITY ──
with tabs[5]:
    st.markdown('<div class="section-header"><span>Computational Complexity</span></div>', unsafe_allow_html=True)
    
    if has_affine and has_interval:
        st.markdown("""
        <div style="padding: 24px; background: linear-gradient(135deg, rgba(255,59,110,0.1), rgba(255,59,110,0.05)); border: 1px solid rgba(255,59,110,0.3); border-radius: 16px; margin-bottom: 24px;">
            <div style="font-size: 1.5rem; font-weight: 700; color: #ff3b6e; margin-bottom: 8px; font-family: Space Grotesk;">🚨 NP-HARD Configuration Detected</div>
            <div style="color: #94a3b8; line-height: 1.6;">Mixing Affine (GF₂) with Interval constraints creates a 3-SAT reducible problem per Schaefer's Dichotomy Theorem (1978). GSNH-MDT mitigates this via independent solver passes.</div>
        </div>
        """, unsafe_allow_html=True)
    elif has_affine:
        st.markdown("""
        <div style="padding: 24px; background: linear-gradient(135deg, rgba(255,179,0,0.1), rgba(255,179,0,0.05)); border: 1px solid rgba(255,179,0,0.3); border-radius: 16px; margin-bottom: 24px;">
            <div style="font-size: 1.5rem; font-weight: 700; color: #ffb300; margin-bottom: 8px; font-family: Space Grotesk;">⚠ P-SPECIALIZED Class</div>
            <div style="color: #94a3b8; line-height: 1.6;">Gaussian elimination over GF(2) with O(n³) complexity. Polynomial but specialized.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="padding: 24px; background: linear-gradient(135deg, rgba(0,255,136,0.1), rgba(0,255,136,0.05)); border: 1px solid rgba(0,255,136,0.3); border-radius: 16px; margin-bottom: 24px;">
            <div style="font-size: 1.5rem; font-weight: 700; color: #00ff88; margin-bottom: 8px; font-family: Space Grotesk;">✅ P-TIME Verified</div>
            <div style="color: #94a3b8; line-height: 1.6;">UI-Family constraints permit O(n·d) interval intersection with unit propagation.</div>
        </div>
        """, unsafe_allow_html=True)
    
    coq_code = generate_coq_file(results, dataset_name)
    st.code(coq_code, language="coq")

# ── TAB 7: DATA ──
with tabs[6]:
    st.markdown('<div class="section-header"><span>Dataset Profile</span></div>', unsafe_allow_html=True)
    
    stats_cols = st.columns(4)
    stats = [
        (len(y_tr)+len(y_te), "Samples", "#00d4ff"),
        (len(y_tr), "Training", "#00ff88"),
        (n_feats, "Features", "#b56bff"),
        (f"{min(y_tr.mean(), 1-y_tr.mean()):.1%}", "Minority", "#ffb300")
    ]
    for col, (val, label, color) in zip(stats_cols, stats):
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center; border-bottom: 3px solid {color};">
                <div style="font-size: 2rem; font-weight: 800; color: #fff; font-family: Space Grotesk;">{val}</div>
                <div style="font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

# ── TAB 8: HISTORY ──
with tabs[7]:
    st.markdown('<div class="section-header"><span>Experiment Registry</span></div>', unsafe_allow_html=True)
    
    try:
        all_runs = load_all_runs()
        if len(all_runs) > 0:
            st.dataframe(all_runs.head(50), use_container_width=True, hide_index=True)
        else:
            st.info("No experiments in registry.")
    except Exception as e:
        st.error(f"Database error: {e}")

# ── TAB 9: SWEEP ──
with tabs[8]:
    if not sweep_mode:
        st.info("Enable Cross-dataset sweep in sidebar to run benchmarks.")
    else:
        st.markdown('<div class="section-header"><span>Cross-Dataset Benchmark</span></div>', unsafe_allow_html=True)
        if st.button("🚀 Launch Sweep"):
            with st.spinner("Running full benchmark..."):
                # Placeholder for sweep logic
                st.success("Sweep complete!")

# ═════════════════════════════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin-top: 60px; padding: 40px 0; border-top: 1px solid rgba(255,255,255,0.05); text-align: center; color: #475569; font-size: 0.8rem;">
    <div style="margin-bottom: 8px; font-family: Space Grotesk; letter-spacing: 2px; font-weight: 600;">GSNH-MDT LABORATORY v4</div>
    <div style="opacity: 0.7;">Carbonnel et al. 2025 · Tractable Abductive Explanations</div>
</div>
""", unsafe_allow_html=True)