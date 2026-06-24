"""
Dashboard Streamlit — TCC 3W Petrobras
Sistema para Detecção e Classificação de Eventos Indesejáveis em Poços de Petróleo Offshore

Autor  : Alan Pinheiro da Silva
Curso  : Especialização Engenharia de Dados e Big Data — PECE / Escola Politécnica USP
Programa: eEDB-007 | 2026

Execução:
    pip install streamlit plotly pandas numpy joblib
    python download_assets.py   # baixa logos PECE/USP (apenas primeira vez)
    streamlit run app.py
"""

import base64
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
# Helper: carrega imagem local como base64 (evita bloqueio de hotlink externo)
# ─────────────────────────────────────────────────────────────────────────────
def img_b64(path: Path, fallback: str = "") -> str:
    if path.exists():
        ext  = path.suffix.lstrip(".")
        mime = "svg+xml" if ext == "svg" else ext
        data = base64.b64encode(path.read_bytes()).decode()
        return f"data:image/{mime};base64,{data}"
    return fallback

# ─────────────────────────────────────────────────────────────────────────────
# Paleta PECE / USP — Dark Theme
# ─────────────────────────────────────────────────────────────────────────────
PECE_NAVY       = "#003087"
PECE_BLUE       = "#1565C0"
PECE_LIGHT_BLUE = "#2979FF"
USP_GOLD        = "#EDB931"
BG_DARK         = "#0D1B2A"
BG_CARD         = "#142136"
BG_BORDER       = "#1E3A5F"
TEXT_PRIMARY    = "#E2E8F0"
TEXT_MUTED      = "#94A3B8"
SUCCESS         = "#10B981"
DANGER          = "#EF4444"
WARNING         = "#F59E0B"

# ─────────────────────────────────────────────────────────────────────────────
# Caminhos
# ─────────────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent
MODELS = BASE / "models"
DATA   = BASE / "data" / "gold"
ASSETS = BASE / "assets"

# ── Logo PECE: badge institucional SVG ───────────────────────────────────────
_SVG_PECE = (
    "data:image/svg+xml;base64," + base64.b64encode((
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 180 72'>"
        "<defs>"
        "  <linearGradient id='pg' x1='0%' y1='0%' x2='100%' y2='100%'>"
        "    <stop offset='0%' stop-color='#1565C0'/>"
        "    <stop offset='100%' stop-color='#003087'/>"
        "  </linearGradient>"
        "</defs>"
        "<rect width='180' height='72' rx='6' fill='url(#pg)'/>"
        "<rect x='0' y='0' width='180' height='3' rx='1' fill='#EDB931'/>"
        "<rect x='0' y='69' width='180' height='3' rx='1' fill='#EDB931'/>"
        "<text x='90' y='35' font-family='Arial Black,Arial' font-size='28'"
        "  font-weight='900' fill='white' text-anchor='middle' letter-spacing='8'>PECE</text>"
        "<text x='90' y='52' font-family='Arial' font-size='8.5' fill='#EDB931'"
        "  text-anchor='middle' letter-spacing='2'>PROGRAMA DE EDUCAÇÃO CONTINUADA</text>"
        "<text x='90' y='64' font-family='Arial' font-size='7.5'"
        "  fill='rgba(255,255,255,0.75)' text-anchor='middle' letter-spacing='1'>"
        "  ESCOLA POLITÉCNICA · USP</text>"
        "</svg>"
    ).encode()).decode()
)

# ── Logo Minerva / Escola Politécnica USP: selo circular SVG ─────────────────
_SVG_MINERVA = (
    "data:image/svg+xml;base64," + base64.b64encode((
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 200 200'>"
        "<defs>"
        "  <linearGradient id='mg' x1='0%' y1='0%' x2='100%' y2='100%'>"
        "    <stop offset='0%' stop-color='#1976D2'/>"
        "    <stop offset='55%' stop-color='#003087'/>"
        "    <stop offset='100%' stop-color='#0A1628'/>"
        "  </linearGradient>"
        "  <path id='tarc' d='M 22,100 A 78,78 0 0,1 178,100'/>"
        "  <path id='barc' d='M 28,115 A 76,76 0 0,0 172,115'/>"
        "  <clipPath id='cc'><circle cx='100' cy='100' r='96'/></clipPath>"
        "</defs>"
        "<!-- fundo -->"
        "<circle cx='100' cy='100' r='99' fill='url(#mg)' stroke='#EDB931' stroke-width='3'/>"
        "<circle cx='100' cy='100' r='86' fill='none' stroke='rgba(237,185,49,0.35)' stroke-width='1'/>"
        "<!-- silhueta da Minerva (perfil direito simplificado) -->"
        "<g fill='white' opacity='0.92'>"
        "  <!-- elmo/capacete -->"
        "  <path d='M 95,52 C 80,52 68,62 67,76 C 66,88 72,94 78,96"
        "           C 76,100 74,108 74,115 L 88,115 C 88,110 89,106 90,103"
        "           C 95,104 102,103 106,100 C 112,96 116,88 114,76"
        "           C 112,62 110,52 95,52 Z'/>"
        "  <!-- crista do elmo -->"
        "  <path d='M 88,52 C 90,44 92,38 95,34 C 98,38 100,44 102,52 Z'/>"
        "  <!-- rosto / perfil -->"
        "  <path d='M 90,103 C 87,107 85,113 84,120 C 83,126 84,131 87,135"
        "           C 89,138 92,139 95,139 C 98,139 101,137 103,134"
        "           C 105,131 105,127 104,123 C 103,118 101,113 98,108"
        "           C 96,105 93,103 90,103 Z'/>"
        "  <!-- pescoço -->"
        "  <rect x='90' y='138' width='12' height='16' rx='4'/>"
        "  <!-- ombros -->"
        "  <path d='M 75,154 C 78,150 85,148 95,148 C 108,148 118,151 122,156"
        "           L 118,164 C 115,160 108,158 95,158 C 85,158 79,160 76,163 Z'/>"
        "</g>"
        "<!-- texto superior -->"
        "<text font-family='Arial' font-size='12.5' fill='white' letter-spacing='0.5'>"
        "  <textPath href='#tarc' startOffset='4%'>"
        "    UNIVERSIDADE DE SÃO PAULO"
        "  </textPath>"
        "</text>"
        "<!-- separadores -->"
        "<circle cx='19' cy='100' r='3' fill='#EDB931'/>"
        "<circle cx='181' cy='100' r='3' fill='#EDB931'/>"
        "<!-- texto inferior -->"
        "<text font-family='Arial' font-size='12.5' fill='#EDB931' letter-spacing='0.5'>"
        "  <textPath href='#barc' startOffset='8%'>"
        "    ESCOLA POLITÉCNICA"
        "  </textPath>"
        "</text>"
        "</svg>"
    ).encode()).decode()
)

# Logos
LOGO_PECE    = img_b64(ASSETS / "logo-pece-photo.jpg", _SVG_PECE)
MINERVA_ICON = _SVG_MINERVA

# ─────────────────────────────────────────────────────────────────────────────
# Constantes do domínio
# ─────────────────────────────────────────────────────────────────────────────
CLASS_LABELS = {
    0: "Normal",
    1: "BSW Abrupt Increase",
    2: "DHSV Spurious Closure",
    4: "Flow Instability",
    6: "PCK Quick Restriction",
    7: "Scaling in PCK",
}
CLASS_COLORS = {
    0: "#2979FF", 1: "#FFB300", 2: "#EF4444",
    4: "#AB47BC", 6: "#26A69A", 7: "#8D6E63",
}
SENSORS = ["P-PDG","P-TPT","T-TPT","P-MON-CKP","T-JUS-CKP","P-JUS-CKGL","T-JUS-CKGL","QGL"]
WINDOWS = [5, 10, 30]
LAGS    = [1, 3, 5]

PLOTLY_DARK = dict(
    template      = "plotly_dark",
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(20,33,54,0.6)",
    font          = dict(color=TEXT_PRIMARY, family="Inter, sans-serif"),
    margin        = dict(t=30, b=20, l=10, r=10),
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuração da página
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="3W Offshore | PECE USP",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS Global — Dark Theme PECE/USP
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  .stApp, .main, section[data-testid="stMain"] {{
    background-color: {BG_DARK} !important;
  }}
  [data-testid="stSidebar"] {{
    background-color: #0A1220 !important;
    border-right: 1px solid {BG_BORDER};
  }}
  .stTabs [data-baseweb="tab-list"] {{
    background: {BG_CARD};
    border-radius: 10px;
    padding: 4px 6px;
    gap: 4px;
    border: 1px solid {BG_BORDER};
  }}
  .stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: {TEXT_MUTED};
    border-radius: 7px;
    padding: 8px 18px;
    font-weight: 600;
    font-size: 0.87rem;
    transition: all 0.2s;
    border: none;
  }}
  .stTabs [data-baseweb="tab"]:hover {{
    color: {TEXT_PRIMARY};
    background: {BG_BORDER};
  }}
  .stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, {PECE_BLUE}, {PECE_LIGHT_BLUE}) !important;
    color: white !important;
    box-shadow: 0 2px 8px rgba(21,101,192,0.5);
  }}
  .stTabs [data-baseweb="tab-panel"] {{
    background: transparent;
    padding-top: 16px;
  }}
  [data-testid="stMetric"] {{
    background: {BG_CARD};
    border: 1px solid {BG_BORDER};
    border-radius: 10px;
    padding: 16px 20px !important;
    transition: transform 0.15s;
  }}
  [data-testid="stMetric"]:hover {{
    transform: translateY(-2px);
    border-color: {PECE_LIGHT_BLUE};
  }}
  [data-testid="stMetricLabel"] {{ color: {TEXT_MUTED} !important; font-size: 0.82rem; }}
  [data-testid="stMetricValue"] {{ color: {USP_GOLD} !important; font-size: 1.6rem !important; font-weight: 700; }}
  hr {{ border-color: {BG_BORDER} !important; margin: 16px 0; }}
  [data-testid="stExpander"] {{
    background: {BG_CARD};
    border: 1px solid {BG_BORDER};
    border-radius: 8px;
  }}
  h1, h2, h3, h4 {{ color: {TEXT_PRIMARY} !important; }}
  p, li, label   {{ color: {TEXT_MUTED}; }}
  .kpi-card {{
    background: {BG_CARD};
    border: 1px solid {BG_BORDER};
    border-radius: 10px;
    padding: 18px 22px;
    text-align: center;
    transition: transform 0.15s, border-color 0.15s;
    margin-bottom: 8px;
  }}
  .kpi-card:hover {{ transform: translateY(-3px); border-color: {USP_GOLD}; }}
  .kpi-val   {{ font-size: 2rem; font-weight: 800; color: {USP_GOLD}; margin: 4px 0; }}
  .kpi-label {{ font-size: 0.82rem; color: {TEXT_MUTED}; text-transform: uppercase; letter-spacing: 0.05em; }}
  .tech-badge {{
    display: inline-block;
    background: rgba(41,121,255,0.15);
    border: 1px solid {PECE_LIGHT_BLUE};
    color: {PECE_LIGHT_BLUE};
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px;
  }}
  .evt-card {{
    border-left: 4px solid;
    background: {BG_CARD};
    border-radius: 0 8px 8px 0;
    padding: 8px 14px;
    margin: 4px 0;
    font-size: 0.88rem;
  }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Loaders com cache
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_metrics():
    out = {}
    for n in ["lgb", "rf", "xgb"]:
        p = MODELS / f"{n}_baseline_metrics.json"
        if p.exists(): out[n] = json.loads(p.read_text())
    return out

@st.cache_data
def load_thresholds():
    p = MODELS / "threshold_tuning_results.json"
    return json.loads(p.read_text()) if p.exists() else {}

@st.cache_data
def load_shap():
    p = MODELS / "shap_results.json"
    return json.loads(p.read_text()) if p.exists() else {}

@st.cache_data
def load_scaler():
    p = DATA / "scaler_stats.json"
    return json.loads(p.read_text()) if p.exists() else {}

@st.cache_resource
def load_model(name="lgb"):
    p = MODELS / f"{name}_baseline.joblib"
    return joblib.load(p) if p.exists() else None

# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    result = {}
    for sensor in SENSORS:
        col   = df[sensor] if sensor in df.columns else pd.Series(np.zeros(len(df)), index=df.index)
        s_key = sensor.replace("-", "_")
        result[sensor] = col.values
        for w in WINDOWS:
            result[f"{s_key}_roll_mean_{w}"] = col.rolling(w, min_periods=1).mean().values
            result[f"{s_key}_roll_std_{w}"]  = col.rolling(w, min_periods=1).std().fillna(0).values
            result[f"{s_key}_roll_min_{w}"]  = col.rolling(w, min_periods=1).min().values
            result[f"{s_key}_roll_max_{w}"]  = col.rolling(w, min_periods=1).max().values
        for lag in LAGS:
            result[f"{s_key}_lag_{lag}"] = col.shift(lag).bfill().values
        result[f"{s_key}_delta"] = col.diff().fillna(0).values
    feat_df = pd.DataFrame(result, index=df.index)
    for feat in feat_df.columns:
        stats = scaler.get(feat) or scaler.get(feat + "_norm")
        if stats:
            mu, sigma = stats["mean"], stats["std"]
            feat_df[feat] = (feat_df[feat] - mu) / sigma if sigma > 1e-9 else 0.0
    feat_df.rename(columns={c: f"{c}_norm" for c in feat_df.columns}, inplace=True)
    return feat_df

def predict_series(df_raw, model, scaler):
    feat_df  = engineer_features(df_raw, scaler)
    expected = list(model.feature_name_) if hasattr(model, "feature_name_") else feat_df.columns.tolist()
    for c in expected:
        if c not in feat_df.columns: feat_df[c] = 0.0
    feat_df   = feat_df[expected]
    proba     = model.predict_proba(feat_df)
    classes   = list(CLASS_LABELS.keys())
    pred_cls  = [classes[i] for i in proba.argmax(axis=1)]
    confidence= proba.max(axis=1)
    return pred_cls, confidence, proba

def generate_sample_csv(n_rows=80, event_class=2):
    t_jus = np.random.normal(64.4, 0.5, n_rows)
    t_tpt = np.random.normal(116.5, 0.4, n_rows)
    half  = n_rows // 2
    if   event_class == 2: t_jus[half:] -= np.linspace(0, 15, n_rows - half)
    elif event_class == 1: t_tpt[half:] += np.linspace(0, 10, n_rows - half)
    elif event_class == 4: t_jus[half:] += np.sin(np.linspace(0, 4*np.pi, n_rows - half)) * 3
    elif event_class == 7: t_jus[half:] -= np.linspace(0,  5, n_rows - half)
    ts = pd.date_range("2024-01-01", periods=n_rows, freq="1min")
    return pd.DataFrame({"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                         "T-JUS-CKP": np.round(t_jus, 3),
                         "T-TPT":     np.round(t_tpt, 3)})

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(135deg,{PECE_NAVY} 0%,{PECE_BLUE} 60%,#1A237E 100%);
            border-radius:14px;border:1px solid {BG_BORDER};
            box-shadow:0 4px 24px rgba(0,0,0,0.5);overflow:hidden;margin-bottom:18px;">
  <div style="height:4px;background:linear-gradient(90deg,{USP_GOLD},#FFE57F,{USP_GOLD});"></div>
  <div style="display:flex;align-items:center;justify-content:space-between;padding:18px 28px;">
    <div style="display:flex;align-items:center;flex-shrink:0;">
      <img src="{LOGO_PECE}"
           style="height:60px;width:auto;object-fit:contain;
                  filter:drop-shadow(0 2px 6px rgba(0,0,0,0.5));" />
    </div>
    <div style="text-align:center;flex:1;padding:0 24px;">
      <div style="font-size:1.4rem;font-weight:800;color:white;letter-spacing:0.02em;line-height:1.2;">
        🛢️ Detecção e Classificação de Eventos Indesejáveis em Poços Offshore
      </div>
      <div style="font-size:0.82rem;color:rgba(255,255,255,0.7);margin-top:5px;">
        com Machine Learning e Engenharia de Dados · TCC eEDB-007 · Escola Politécnica USP · Alan Pinheiro da Silva · 2026
      </div>
    </div>
    <div style="flex-shrink:0;">
      <img src="{MINERVA_ICON}"
           style="height:72px;width:72px;object-fit:contain;
                  border-radius:50%;
                  box-shadow:0 0 14px rgba(237,185,49,0.5);
                  filter:drop-shadow(0 2px 6px rgba(0,0,0,0.5));" />
    </div>
  </div>
  <div style="height:2px;background:linear-gradient(90deg,transparent,{USP_GOLD},transparent);"></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ABAS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠  Visão Geral",
    "🔄  Pipeline ETL",
    "📊  Exploratória",
    "🤖  Resultados ML",
    "🔍  SHAP",
    "🎯  Demo — Predição",
])

# ── ABA 1: VISÃO GERAL ────────────────────────────────────────────────────────
with tab1:
    kpis = [("7,5 M","Registros"), ("3","Poços"), ("6","Classes"),
            ("136","Features Gold"), ("96,2%","F1-Macro LGB"), ("99,9%","Acurácia LGB")]
    cols = st.columns(len(kpis))
    for col, (val, label) in zip(cols, kpis):
        col.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                     f'<div class="kpi-val">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"<h4 style='color:{USP_GOLD}'>🗂️ Classes — Dataset 3W Petrobras</h4>",
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Classe": list(CLASS_LABELS.keys()),
            "Evento": list(CLASS_LABELS.values()),
            "Arquivos": ["334","3","1","155","6","2"],
        }), hide_index=True, use_container_width=True)

    with col_b:
        st.markdown(f"<h4 style='color:{USP_GOLD}'>⚙️ Stack Tecnológica</h4>",
                    unsafe_allow_html=True)
        for cat, techs in {
            "Linguagem":       ["Python 3.9+","SQL"],
            "ETL":             ["AWS Glue 4.0","PySpark","Apache Airflow 2.9.2"],
            "Armazenamento":   ["MinIO S3","PostgreSQL 15","Parquet"],
            "Machine Learning":["LightGBM","XGBoost","scikit-learn","SHAP","Optuna"],
            "Infraestrutura":  ["Docker","Terraform AWS"],
        }.items():
            badges = " ".join(f'<span class="tech-badge">{t}</span>' for t in techs)
            st.markdown(f"<div style='margin:6px 0'><small style='color:{TEXT_MUTED}'>"
                        f"{cat}</small><br>{badges}</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown(f"<h4 style='color:{USP_GOLD}'>🏆 Destaques</h4>", unsafe_allow_html=True)
    for col, (icon, model, f1, color, note) in zip(st.columns(3), [
        ("🥇","LightGBM","96,2%", SUCCESS, "Acurácia 99,9% · is_unbalance=True"),
        ("🥈","XGBoost","95,7%",  PECE_LIGHT_BLUE, "+5,7pp após threshold tuning"),
        ("🥉","Random Forest","91,1%", WARNING, "150 árvores · class_weight=balanced"),
    ]):
        col.markdown(f"""
        <div style="background:{BG_CARD};border:1px solid {color}44;border-radius:10px;
                    padding:16px;border-left:4px solid {color};">
          <div style="font-size:1.05rem;font-weight:700;color:{color}">{icon} {model}</div>
          <div style="font-size:1.7rem;font-weight:800;color:{USP_GOLD};margin:4px 0">{f1}</div>
          <div style="font-size:0.8rem;color:{TEXT_MUTED}">{note}</div>
        </div>""", unsafe_allow_html=True)

# ── ABA 2: PIPELINE ETL ───────────────────────────────────────────────────────
with tab2:
    st.markdown(f"<h3 style='color:{USP_GOLD}'>🔄 Arquitetura Medallion</h3>", unsafe_allow_html=True)

    p1, a1, p2, a2, p3, a3, p4 = st.columns([2,.3,2,.3,2,.3,2])
    for col, title, desc, color in [
        (p1,"📥 INGESTÃO","GitHub 3W API\nParquet por classe", PECE_NAVY),
        (p2,"🥉 BRONZE",  "Schema padronizado\nbronze_at · dedup", "#4E342E"),
        (p3,"🥈 SILVER",  "Validação física\nFill NaN · dedup",    "#37474F"),
        (p4,"🥇 GOLD",    "136 features\nZ-Score · Split 70/15/15","#1B5E20"),
    ]:
        col.markdown(
            f"<div style='background:{color};border-radius:10px;padding:14px 12px;"
            f"text-align:center;border:1px solid {BG_BORDER};min-height:90px'>"
            f"<div style='font-weight:700;color:white;font-size:0.95rem'>{title}</div>"
            f"<div style='color:rgba(255,255,255,0.75);font-size:0.78rem;margin-top:6px;"
            f"white-space:pre-line'>{desc}</div></div>", unsafe_allow_html=True)
    for col in [a1, a2, a3]:
        col.markdown(f"<div style='text-align:center;font-size:1.8rem;color:{USP_GOLD};"
                     f"padding-top:24px'>▶</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown(f"<h4 style='color:{USP_GOLD}'>🔧 Features Gold</h4>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Tipo":    ["Sensores brutos","Rolling stats","Lag features","Delta","TOTAL"],
            "Qtd":     ["8","96","24","8","136"],
            "Detalhe": ["8 sensores","mean·std·min·max × 3 janelas × 8",
                        "lag t-1·t-3·t-5 × 8","diff(t) × 8","—"],
        }), hide_index=True, use_container_width=True)
    with col_f2:
        st.markdown(f"<h4 style='color:{USP_GOLD}'>📋 DAGs Airflow</h4>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "DAG":    ["dag_etl_3w_pipeline","dag_gold_rebuild","dag_ml_training"],
            "Função": ["Bronze→Silver→Gold","Re-processa Gold","RF+XGB+LGB→Tuning→SHAP"],
        }), hide_index=True, use_container_width=True)

    st.divider()
    c1, c2 = st.columns(2)
    c1.info("🔒 **GITHUB_TOKEN** sempre via variável de ambiente — nunca hardcoded")
    c2.info("📐 **Z-Score** fit exclusivamente no treino — sem data leakage")

# ── ABA 3: EXPLORATÓRIA ───────────────────────────────────────────────────────
with tab3:
    st.markdown(f"<h3 style='color:{USP_GOLD}'>📊 Análise Exploratória de Dados</h3>",
                unsafe_allow_html=True)

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("#### Distribuição de Classes")
        counts = {"Normal":7_200_000,"BSW":9_000,"DHSV":1_200,
                  "Flow":310_000,"PCK":15_000,"Scaling":4_000}
        fig = px.bar(x=list(counts.keys()), y=list(counts.values()),
                     color=list(counts.keys()),
                     color_discrete_sequence=list(CLASS_COLORS.values()), log_y=True)
        fig.update_layout(**PLOTLY_DARK, height=320, showlegend=False)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with col_e2:
        st.markdown("#### Disponibilidade de Sensores (Silver)")
        sens_df = pd.DataFrame({"Sensor": SENSORS,
                                 "% Válidos": [0,0,100,0,100,0,0,0]})
        fig2 = px.bar(sens_df, x="Sensor", y="% Válidos", color="% Válidos",
                      color_continuous_scale=[DANGER, SUCCESS], range_color=[0,100])
        fig2.update_layout(**PLOTLY_DARK, height=320, coloraxis_showscale=False)
        fig2.update_traces(marker_line_width=0)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("⚠️ 6 de 8 sensores com 100% NaN no subconjunto dos 3 poços")

    st.divider()
    st.markdown("#### 📈 Série Temporal — Sensores Ativos")
    ctrl, chart = st.columns([1,3])
    with ctrl:
        n_pts  = st.slider("Pontos", 60, 400, 150, step=50)
        noise  = st.slider("Ruído σ", 0.1, 2.5, 0.5, step=0.1)
        anomaly= st.checkbox("Simular anomalia", value=True)
    with chart:
        t       = np.arange(n_pts)
        t_jus   = 64.4  + noise*np.random.randn(n_pts)
        t_tpt   = 116.5 + noise*np.random.randn(n_pts)
        if anomaly:
            half = n_pts//2
            t_jus[half:] -= np.linspace(0, 12, n_pts-half)
        fts = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=["T-JUS-CKP (°C)","T-TPT (°C)"])
        fts.add_trace(go.Scatter(x=t, y=t_jus, mode="lines",
                      line=dict(color="#EF5350",width=1.8), name="T-JUS-CKP"), row=1, col=1)
        if anomaly:
            fts.add_vrect(x0=n_pts//2, x1=n_pts, fillcolor="rgba(239,68,68,0.08)",
                          line_width=0, row=1, col=1)
        fts.add_trace(go.Scatter(x=t, y=t_tpt, mode="lines",
                      line=dict(color=PECE_LIGHT_BLUE,width=1.8), name="T-TPT"), row=2, col=1)
        fts.update_layout(**PLOTLY_DARK, height=300, showlegend=False)
        st.plotly_chart(fts, use_container_width=True)

# ── ABA 4: RESULTADOS ML ─────────────────────────────────────────────────────
with tab4:
    st.markdown(f"<h3 style='color:{USP_GOLD}'>🤖 Resultados dos Modelos ML</h3>",
                unsafe_allow_html=True)

    metrics = load_metrics()
    thresh  = load_thresholds()
    rows = []
    for key, label in [("lgb","LightGBM ★"),("xgb","XGBoost"),("rf","Random Forest")]:
        m   = metrics.get(key, {})
        t   = thresh.get(key, {})
        b   = m.get("test",{}).get("f1_macro", 0)
        tun = t.get("test",{}).get("tuned",{}).get("f1mac", b)
        rows.append({"Modelo":label,
                     "F1 Val":   f"{m.get('val',{}).get('f1_macro',0)*100:.1f}%",
                     "F1 Teste": f"{b*100:.1f}%",
                     "F1 Tuned": f"{tun*100:.1f}%",
                     "Acurácia": f"{m.get('test',{}).get('accuracy',0)*100:.2f}%",
                     "Ganho":    f"+{(tun-b)*100:.1f}pp" if tun > b else "—"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.divider()
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("#### F1-Macro — Baseline vs Tuning")
        labels_m = ["Random Forest","XGBoost","LightGBM"]
        f1_b  = [0.9113, 0.9002, 0.9619]
        f1_t  = [0.9373, 0.9574, 0.9619]
        fig_f = go.Figure()
        fig_f.add_bar(name="Baseline", x=labels_m, y=f1_b, marker_color=BG_BORDER,
                      text=[f"{v*100:.1f}%" for v in f1_b], textposition="outside")
        fig_f.add_bar(name="Tuned", x=labels_m, y=f1_t, marker_color=PECE_LIGHT_BLUE,
                      text=[f"{v*100:.1f}%" for v in f1_t], textposition="outside")
        fig_f.update_layout(**PLOTLY_DARK, barmode="group", height=360,
                            yaxis=dict(range=[0.85,1.0],tickformat=".0%"),
                            legend=dict(orientation="h",y=-0.18))
        st.plotly_chart(fig_f, use_container_width=True)
    with col_m2:
        st.markdown("#### Matriz de Confusão — LightGBM")
        p = MODELS / "lgb_baseline_confusion.png"
        if p.exists(): st.image(str(p), use_container_width=True)
        else: st.info("lgb_baseline_confusion.png não encontrado em models/")

    st.divider()
    st.markdown(f"<h4 style='color:{USP_GOLD}'>⚠️ Correção: Split Temporal → Estratificado</h4>",
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.dataframe(pd.DataFrame({
        "Abordagem":   ["Temporal (descartado)","Estratificado (adotado)"],
        "RF F1-Macro": ["24,5%","91,1%"],
        "Problema":    ["Classe 7 ausente no treino","—"],
    }), hide_index=True, use_container_width=True)
    c2.error("Divisão **temporal** excluiu a classe 7 do treino. "
             "**StratifiedShuffleSplit** corrigiu: 24,5% → 91,1%.")

# ── ABA 5: SHAP ──────────────────────────────────────────────────────────────
with tab5:
    st.markdown(f"<h3 style='color:{USP_GOLD}'>🔍 Interpretabilidade — SHAP TreeExplainer</h3>",
                unsafe_allow_html=True)

    shap_data = load_shap()
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("#### Importância Global por Sensor")
        gi = shap_data.get("sensor_importance_global", {})
        if gi:
            df_gi = pd.DataFrame([{"Sensor":k,"SHAP (%)":round(v,2)}
                                   for k,v in sorted(gi.items(), key=lambda x:-x[1])])
            fg = px.bar(df_gi, x="SHAP (%)", y="Sensor", orientation="h",
                        color="SHAP (%)", color_continuous_scale=[BG_CARD, USP_GOLD],
                        text="SHAP (%)")
            fg.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fg.update_layout(**PLOTLY_DARK, height=300, coloraxis_showscale=False,
                             yaxis=dict(autorange="reversed"))
            st.plotly_chart(fg, use_container_width=True)
        st.caption("T-JUS-CKP: 69,6% · T-TPT: 30,4% · demais: 0% (100% NaN no Silver)")

    with col_s2:
        st.markdown("#### SHAP por Classe")
        si_rows = []
        for ck, cd in shap_data.get("classes",{}).items():
            si = cd.get("sensor_importance_pct",{})
            si_rows.append({"Evento": CLASS_LABELS.get(int(ck),ck),
                            "T-JUS-CKP": round(si.get("T-JUS-CKP",0),1),
                            "T-TPT":     round(si.get("T-TPT",0),1)})
        if si_rows:
            df_si = pd.DataFrame(si_rows)
            fs = go.Figure()
            fs.add_bar(name="T-JUS-CKP", x=df_si["Evento"], y=df_si["T-JUS-CKP"],
                       marker_color="#EF5350")
            fs.add_bar(name="T-TPT", x=df_si["Evento"], y=df_si["T-TPT"],
                       marker_color=PECE_LIGHT_BLUE)
            fs.update_layout(**PLOTLY_DARK, barmode="stack", height=300,
                             yaxis_title="% SHAP", legend=dict(orientation="h",y=-0.22))
            st.plotly_chart(fs, use_container_width=True)
        st.caption("DHSV Closure: T-TPT domina (inversão única nesta classe)")

    st.divider()
    st.markdown("#### Gráficos gerados pelo TreeExplainer")
    sc = st.columns(3)
    for col, (fn, cap) in zip(sc, [
        ("shap_global_bar.png","Importância Global"),
        ("shap_global_summary.png","Summary Plot"),
        ("shap_sensor_importance.png","Por Sensor"),
    ]):
        p = MODELS/fn
        if p.exists(): col.image(str(p), caption=cap, use_container_width=True)

    st.markdown("#### SHAP por Classe de Evento")
    cc = st.columns(3)
    for i, img in enumerate(sorted(MODELS.glob("shap_class_*.png"))):
        cc[i%3].image(str(img),
                      caption=img.stem.replace("shap_class_","").replace("_"," "),
                      use_container_width=True)

# ── ABA 6: DEMO ──────────────────────────────────────────────────────────────
with tab6:
    st.markdown(f"<h3 style='color:{USP_GOLD}'>🎯 Demo — Predição ao Vivo com LightGBM</h3>",
                unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:{BG_CARD};border:1px solid {BG_BORDER};border-radius:10px;
                padding:14px 18px;margin-bottom:16px">
      <b style="color:{PECE_LIGHT_BLUE}">Como usar:</b>
      <span style="color:{TEXT_MUTED}"> Upload de CSV com colunas
      <code>timestamp</code>, <code>T-JUS-CKP</code> e <code>T-TPT</code>
      (mín. 30 linhas). O sistema calcula 136 features e classifica cada ponto com o LightGBM treinado.
      Gere um CSV de exemplo com o botão ao lado.</span>
    </div>""", unsafe_allow_html=True)

    col_up, col_gen = st.columns([2,1])
    with col_gen:
        st.markdown(f"<div style='color:{USP_GOLD};font-weight:700;margin-bottom:8px'>"
                    f"Gerar dados de exemplo</div>", unsafe_allow_html=True)
        ev = st.selectbox("Evento a simular", [0,1,2,4,7],
                          format_func=lambda x: CLASS_LABELS[x])
        nr = st.slider("Pontos", 60, 200, 100, step=20)
        if st.button("⬇️ Gerar CSV de exemplo", use_container_width=True, type="primary"):
            sample = generate_sample_csv(nr, ev)
            st.download_button("💾 Baixar sample_demo.csv",
                               data=sample.to_csv(index=False).encode(),
                               file_name="sample_demo.csv", mime="text/csv",
                               use_container_width=True)
    with col_up:
        uploaded = st.file_uploader("📂 CSV (timestamp, T-JUS-CKP, T-TPT)", type=["csv"])

    if uploaded:
        try:
            df_raw = pd.read_csv(uploaded)
            missing = {"T-JUS-CKP","T-TPT"} - set(df_raw.columns)
            if missing:
                st.error(f"Colunas ausentes: {missing}")
                st.stop()
            st.success(f"✅ {len(df_raw)} linhas carregadas")
            model  = load_model("lgb")
            scaler = load_scaler()
            if model is None:
                st.error("Modelo não encontrado: models/lgb_baseline.joblib")
                st.stop()
            with st.spinner("🔄 Calculando 136 features e classificando..."):
                pred_cls, confs, _ = predict_series(df_raw, model, scaler)
            df_raw["Classe"]    = pred_cls
            df_raw["Evento"]    = [CLASS_LABELS.get(c, str(c)) for c in pred_cls]
            df_raw["Conf. (%)"] = (confs * 100).round(1)

            st.divider()
            st.markdown(f"<h4 style='color:{USP_GOLD}'>📊 Resultado</h4>", unsafe_allow_html=True)
            vc = pd.Series(pred_cls).value_counts().reset_index()
            vc.columns = ["Classe","Qtd"]
            vc["Evento"]  = vc["Classe"].map(CLASS_LABELS)
            vc["% tempo"] = (vc["Qtd"]/len(df_raw)*100).round(1)

            cr, cp = st.columns([1,2])
            with cr:
                st.dataframe(vc[["Evento","Qtd","% tempo"]], hide_index=True, use_container_width=True)
            with cp:
                fp = px.pie(vc, values="Qtd", names="Evento",
                            color_discrete_sequence=[PECE_LIGHT_BLUE,USP_GOLD,DANGER,
                                                     SUCCESS,WARNING,"#AB47BC"])
                fp.update_layout(**PLOTLY_DARK, height=260, margin=dict(t=10,b=10))
                st.plotly_chart(fp, use_container_width=True)

            st.markdown("#### 📈 Série Temporal com Classes Preditas")
            t_ax = list(range(len(df_raw)))
            fl = make_subplots(rows=3, cols=1, shared_xaxes=True,
                               subplot_titles=["T-JUS-CKP (°C)","T-TPT (°C)","Classe Predita"],
                               row_heights=[0.35,0.35,0.30])
            fl.add_trace(go.Scatter(x=t_ax, y=df_raw["T-JUS-CKP"], mode="lines",
                         line=dict(color="#EF5350",width=1.8), name="T-JUS-CKP"), row=1, col=1)
            fl.add_trace(go.Scatter(x=t_ax, y=df_raw["T-TPT"], mode="lines",
                         line=dict(color=PECE_LIGHT_BLUE,width=1.8), name="T-TPT"), row=2, col=1)
            fl.add_trace(go.Scatter(x=t_ax, y=df_raw["Classe"], mode="markers+lines",
                         marker=dict(size=5, color=df_raw["Classe"], colorscale="Plasma"),
                         line=dict(width=1,color="gray"), name="Classe"), row=3, col=1)
            fl.update_layout(**PLOTLY_DARK, height=500, showlegend=False)
            st.plotly_chart(fl, use_container_width=True)

            with st.expander("📋 Tabela completa"):
                st.dataframe(df_raw[["timestamp","T-JUS-CKP","T-TPT","Evento","Conf. (%)"]],
                             hide_index=True, use_container_width=True)
        except Exception as e:
            st.error(f"Erro: {e}")
            st.exception(e)
    else:
        st.markdown(f"<div style='color:{TEXT_MUTED};margin-bottom:12px'>"
                    f"Aguardando CSV... Classes disponíveis:</div>", unsafe_allow_html=True)
        cc = st.columns(3)
        for i, (cls, label) in enumerate(CLASS_LABELS.items()):
            c = CLASS_COLORS[cls]
            cc[i%3].markdown(
                f'<div class="evt-card" style="border-color:{c}">'
                f'<b style="color:{c}">Classe {cls}</b> — {label}</div>',
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-top:40px;border-top:1px solid {BG_BORDER};padding-top:16px;
            display:flex;align-items:center;justify-content:space-between;">
  <img src="{LOGO_PECE}" style="height:26px;opacity:0.7;" />
  <div style="font-size:0.76rem;color:{TEXT_MUTED};text-align:center;">
    PECE · Escola Politécnica USP · eEDB-007 Engenharia de Dados e Big Data<br>
    Dataset: <a href="https://github.com/petrobras/3W" style="color:{PECE_LIGHT_BLUE}">
    Petrobras 3W v1.70.0</a> &nbsp;|&nbsp;
    <a href="https://github.com/AlanPinheiro19/projeto-3w-aws" style="color:{PECE_LIGHT_BLUE}">
    github.com/AlanPinheiro19/projeto-3w-aws</a>
  </div>
  <img src="{LOGO_USP}" style="height:26px;opacity:0.7;filter:brightness(0) invert(0.6);" />
</div>
""", unsafe_allow_html=True)
