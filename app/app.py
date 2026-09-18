"""Streamlit interface for the 23CSE301 ML Capstone.

Serves the two best models from the notebooks:
  * classification -> tuned SVC, classifies a Kepler Object of Interest
  * regression     -> tuned Random Forest, predicts a stock's next-year price change (%)

Run from the repo root:
    streamlit run app/app.py

Saved models are produced by `python app/export_models.py` (see README).
"""
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.config as stconfig

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

st.set_page_config(page_title="Model Explorer | ML Capstone", page_icon=":material/planet:", layout="wide")

#---------- theme ----------
#streamlit's own widgets follow the theme options, so the toggle rewrites them and reruns once
THEME_CONFIG = {
    "dark": dict(base="dark", primaryColor="#f5b94e", backgroundColor="#0b1020",
                 secondaryBackgroundColor="#121a33", textColor="#e6e9f0"),
    "light": dict(base="light", primaryColor="#2c5bd9", backgroundColor="#f4f5f9",
                  secondaryBackgroundColor="#ffffff", textColor="#1b2233"),
}
PALETTE = {
    "dark": dict(bg="#0b1020", surface="#121a33", surface2="#182244", border="rgba(255,255,255,.10)",
                 text="#eef1f7", muted="#a3adc5", faint="#6f7a96", grid="rgba(255,255,255,.08)",
                 chip="rgba(255,255,255,.06)", amber="#f5b94e", teal="#5eead4", red="#f87171",
                 green="#34d399", blue="#60a5fa", violet="#a78bfa", up="#4ade80", down="#f87171"),
    "light": dict(bg="#f4f5f9", surface="#ffffff", surface2="#eef1f8", border="#e1e5ef",
                  text="#1b2233", muted="#5b6478", faint="#8a93a8", grid="rgba(27,34,51,.09)",
                  chip="#eef1f8", amber="#d18f12", teal="#0e9a86", red="#dc4444",
                  green="#15994f", blue="#2c5bd9", violet="#7c5cd6", up="#15994f", down="#dc4444"),
}
#the theme options are process-wide, so a fresh session starts from whatever is currently applied
CURRENT = "dark" if stconfig.get_option("theme.base") == "dark" else "light"
if "dark" not in st.session_state:
    st.session_state["dark"] = CURRENT == "dark"
MODE = "dark" if st.session_state["dark"] else "light"
T = PALETTE[MODE]
if st.session_state.get("_theme_applied", CURRENT) != MODE:
    for k, v in THEME_CONFIG[MODE].items():
        stconfig.set_option(f"theme.{k}", v)
    st.session_state["_theme_applied"] = MODE
    st.rerun()

MONO = "'JetBrains Mono', ui-monospace, Menlo, monospace"
SANS = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"
FONT_FACES = """
@font-face{font-family:Inter;font-weight:400;src:url(https://cdn.jsdelivr.net/npm/@fontsource/inter@5.2.5/files/inter-latin-400-normal.woff2) format('woff2')}
@font-face{font-family:Inter;font-weight:500;src:url(https://cdn.jsdelivr.net/npm/@fontsource/inter@5.2.5/files/inter-latin-500-normal.woff2) format('woff2')}
@font-face{font-family:Inter;font-weight:600;src:url(https://cdn.jsdelivr.net/npm/@fontsource/inter@5.2.5/files/inter-latin-600-normal.woff2) format('woff2')}
@font-face{font-family:Inter;font-weight:700;src:url(https://cdn.jsdelivr.net/npm/@fontsource/inter@5.2.5/files/inter-latin-700-normal.woff2) format('woff2')}
@font-face{font-family:'JetBrains Mono';font-weight:500;src:url(https://cdn.jsdelivr.net/npm/@fontsource/jetbrains-mono@5.2.5/files/jetbrains-mono-latin-500-normal.woff2) format('woff2')}
@font-face{font-family:'JetBrains Mono';font-weight:600;src:url(https://cdn.jsdelivr.net/npm/@fontsource/jetbrains-mono@5.2.5/files/jetbrains-mono-latin-600-normal.woff2) format('woff2')}
"""

CSS = f"""
<style>
:root {{ --bg:{T['bg']}; --surface:{T['surface']}; --surface2:{T['surface2']}; --border:{T['border']};
        --text:{T['text']}; --muted:{T['muted']}; --faint:{T['faint']}; --chip:{T['chip']};
        --amber:{T['amber']}; --teal:{T['teal']}; --red:{T['red']}; --green:{T['green']}; --blue:{T['blue']}; }}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 1.1rem; padding-bottom: 3rem; max-width: 1240px; }}
section[data-testid="stSidebar"] {{ border-right: 1px solid var(--border); }}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1rem; }}
[data-testid="stPlotlyChart"], [data-testid="stIFrame"], .panel, .hero, .kpi, .empty, .spec-card {{ animation: fadeUp .65s ease both; }}
@keyframes fadeUp {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes pop {{ from {{ opacity: 0; transform: scale(.94); }} to {{ opacity: 1; transform: none; }} }}
@keyframes grow {{ from {{ width: 0; }} }}

.brand {{ display: flex; align-items: center; gap: .75rem; padding: .2rem 0 .9rem; }}
.brand-mark {{ width: 40px; height: 40px; border-radius: 12px; display: grid; place-items: center; color: #fff;
              background: linear-gradient(135deg, #2b3a8f, #0e1530); border: 1px solid rgba(255,255,255,.14); }}
.brand-mark svg {{ width: 24px; height: 24px; }}
.brand-name {{ display: block; font-weight: 700; font-size: 1.05rem; color: var(--text); line-height: 1.15; }}
.brand-sub {{ display: block; font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); margin-top: .15rem; }}
.side-note {{ color: var(--muted); font-size: .82rem; line-height: 1.5; margin-top: .5rem; }}

.hero {{ position: relative; overflow: hidden; border-radius: 18px; padding: 1.6rem 1.8rem 1.5rem; margin-bottom: 1.1rem;
        border: 1px solid var(--border); min-height: 208px; background: var(--surface); }}
.hero.space {{ background: #080d21; border-color: rgba(255,255,255,.10); }}
.hero svg.sky {{ position: absolute; inset: 0; width: 100%; height: 100%; display: block; }}
.hero-body {{ position: relative; z-index: 1; max-width: 660px; }}
.hero.space .hero-body {{ color: #f3f5fa; }}
.eyebrow {{ font-size: .72rem; letter-spacing: .16em; text-transform: uppercase; font-weight: 600; color: var(--muted); }}
.hero.space .eyebrow {{ color: #9fb0e0; }}
.hero-title {{ display: flex; align-items: center; gap: .65rem; font-size: 2rem; font-weight: 700; letter-spacing: -.01em;
              line-height: 1.15; margin: .35rem 0 .45rem; color: var(--text); }}
.hero.space .hero-title {{ color: #f8fafc; }}
.hero-title svg {{ width: 34px; height: 34px; flex: 0 0 auto; }}
.hero-sub {{ font-size: .98rem; line-height: 1.55; color: var(--muted); max-width: 600px; }}
.hero.space .hero-sub {{ color: #c3cce6; }}
.chips {{ display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1rem; }}
.chip {{ display: inline-flex; gap: .4rem; align-items: baseline; padding: .38rem .8rem; border-radius: 999px; font-size: .82rem;
        background: var(--chip); border: 1px solid var(--border); color: var(--muted); animation: pop .55s cubic-bezier(.2,.8,.2,1) both;
        animation-delay: calc(.25s + var(--i) * 90ms); }}
.chip b {{ font-family: {MONO}; font-weight: 600; color: var(--text); font-size: .9rem; }}
.hero.space .chip {{ background: rgba(255,255,255,.08); border-color: rgba(255,255,255,.14); color: #c3cce6; }}
.hero.space .chip b {{ color: #fff; }}

.section {{ font-size: 1rem; font-weight: 600; color: var(--text); margin: .2rem 0 .35rem; }}
.hint {{ color: var(--muted); font-size: .88rem; line-height: 1.55; margin-bottom: .6rem; }}
.caption {{ color: var(--muted); font-size: .84rem; line-height: 1.5; margin: -.2rem 0 .6rem; }}
.panel {{ background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 1.1rem 1.3rem; margin-bottom: 1rem; }}
.empty {{ border: 1px dashed var(--border); border-radius: 16px; padding: 1.3rem 1.4rem; color: var(--muted); font-size: .92rem; line-height: 1.55;
         background: var(--surface); margin-bottom: 1rem; }}
.empty b {{ color: var(--text); }}

.bar-row {{ display: grid; grid-template-columns: minmax(7rem, 11rem) 1fr 3.4rem; align-items: center; gap: .6rem; margin: .42rem 0; animation: fadeUp .5s ease both; }}
.bar-label {{ font-size: .86rem; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.bar-track {{ height: 9px; background: var(--chip); border: 1px solid var(--border); border-radius: 999px; overflow: hidden; }}
.bar-fill {{ height: 100%; border-radius: 999px; width: var(--w); animation: grow 1s cubic-bezier(.2,.8,.2,1) both; }}
.bar-val {{ font-family: {MONO}; font-size: .8rem; color: var(--muted); text-align: right; }}

.spec-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 1.1rem 1.3rem; margin-bottom: 1rem; height: 100%; }}
.spec-head {{ display: flex; align-items: center; gap: .6rem; font-weight: 600; font-size: 1.02rem; color: var(--text); margin-bottom: .7rem; }}
.spec-head svg {{ width: 22px; height: 22px; }}
.spec {{ display: grid; grid-template-columns: 7.5rem 1fr; row-gap: .5rem; column-gap: .8rem; font-size: .88rem; line-height: 1.5; }}
.spec .k {{ color: var(--muted); font-weight: 500; }}
.spec .v {{ color: var(--text); }}
.foot {{ color: var(--faint); font-size: .8rem; margin-top: 1.5rem; line-height: 1.5; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

CLASS_COLORS = {"CONFIRMED": T["teal"], "CANDIDATE": T["amber"], "FALSE POSITIVE": T["red"]}
CLASS_BLURB = {
    "CONFIRMED": "the transit signal looks like a real planet",
    "CANDIDATE": "promising, but not resolved either way yet",
    "FALSE POSITIVE": "the signal is better explained by something other than a planet",
}

CLS_LABELS = {
    "koi_period": "Orbital period (days)", "koi_duration": "Transit duration (hours)",
    "koi_depth": "Transit depth (ppm)", "koi_prad": "Planet radius (Earth radii)",
    "koi_model_snr": "Transit signal-to-noise", "koi_impact": "Impact parameter",
    "koi_steff": "Star temperature (K)", "koi_srad": "Star radius (Solar radii)",
    "koi_teq": "Equilibrium temperature (K)", "koi_slogg": "Star surface gravity (log g)",
    "koi_num_transits": "Transits observed", "koi_kepmag": "Kepler magnitude",
}
CLS_SHORT = {"koi_period": "Period", "koi_duration": "Duration", "koi_depth": "Depth", "koi_prad": "Radius",
             "koi_model_snr": "Signal/noise", "koi_impact": "Impact", "koi_steff": "Star temp", "koi_srad": "Star radius"}
CLS_PRIMARY = ["koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_model_snr", "koi_impact", "koi_steff", "koi_srad"]

REG_LABELS = {
    "EPS": "Earnings per share (EPS)", "PB ratio": "Price-to-book ratio", "returnOnEquity": "Return on equity",
    "Revenue Growth": "Revenue growth", "netProfitMargin": "Net profit margin", "Total assets": "Total assets (USD)",
    "returnOnAssets": "Return on assets", "Net Income Growth": "Net income growth",
    "Total current assets": "Total current assets (USD)", "Total shareholders equity": "Shareholders' equity (USD)",
    "Total liabilities": "Total liabilities (USD)", "Price to Sales Ratio": "Price-to-sales ratio",
    "Asset Growth": "Asset growth", "ROIC": "Return on invested capital", "Growth Adjusted Margin": "Growth-adjusted margin",
    "Revenue": "Revenue (USD)", "Gross Profit": "Gross profit (USD)", "Operating Income": "Operating income (USD)",
    "EBITDA": "EBITDA (USD)", "EBIT": "EBIT (USD)", "Net Income": "Net income (USD)", "Gross Margin": "Gross margin",
    "EBITDA Margin": "EBITDA margin", "operatingProfitMargin": "Operating profit margin",
    "Total current liabilities": "Total current liabilities (USD)", "Total debt": "Total debt (USD)",
    "Cash and cash equivalents": "Cash and equivalents (USD)", "currentRatio": "Current ratio", "quickRatio": "Quick ratio",
    "cashRatio": "Cash ratio", "debtEquityRatio": "Debt-to-equity ratio", "debtRatio": "Debt ratio",
    "interestCoverage": "Interest coverage", "assetTurnover": "Asset turnover", "inventoryTurnover": "Inventory turnover",
    "Operating Cash Flow": "Operating cash flow (USD)", "Free Cash Flow": "Free cash flow (USD)",
    "Capital Expenditure": "Capital expenditure (USD)", "PE ratio": "Price-to-earnings ratio", "EV to Sales": "EV to sales",
    "Dividend Yield": "Dividend yield", "EPS Growth": "EPS growth",
}
REG_PRIMARY = ["EPS", "PB ratio", "returnOnEquity", "Revenue Growth", "netProfitMargin", "Total assets", "returnOnAssets", "Net Income Growth"]

REGRESSION_RESULTS = pd.DataFrame(
    [["Random Forest", 0.1106, 45.53, 31.96], ["Gradient Boosting", 0.0999, 45.80, 31.90],
     ["ElasticNet", 0.0399, 47.30, 33.32], ["Ridge", 0.0390, 47.33, 33.36], ["SVR", 0.0390, 47.33, 31.99],
     ["Lasso", 0.0385, 47.34, 33.36], ["Decision Tree", 0.0367, 47.38, 33.30], ["KNN", 0.0359, 47.40, 32.84],
     ["Linear", 0.0339, 47.45, 33.47], ["Polynomial", 0.0158, 47.89, 33.15]],
    columns=["Model", "R2", "RMSE", "MAE"])
CLASSIFICATION_RESULTS = pd.DataFrame(
    [["SVC", 0.8223, 0.8197, 0.9442], ["Decision Tree (tuned)", 0.8207, 0.8190, 0.9233],
     ["Logistic Regression", 0.8197, 0.8132, 0.9368], ["Decision Tree (baseline)", 0.7930, 0.7938, 0.8383],
     ["KNN (tuned)", 0.7846, 0.7822, 0.9133], ["KNN (baseline)", 0.7737, 0.7705, 0.9006],
     ["Gaussian Naive Bayes", 0.6142, 0.6186, 0.8769]],
    columns=["Model", "Accuracy", "Weighted F1", "ROC-AUC"])


@st.cache_resource(show_spinner=False)
def load_bundles():
    return (joblib.load(MODELS / "classification_svc.pkl"),
            joblib.load(MODELS / "regression_random_forest.pkl"))


#---------- icons and illustrations ----------
ICON_PLANET = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round">'
               '<circle cx="12" cy="12" r="5.4" fill="currentColor" fill-opacity=".18"/>'
               '<ellipse cx="12" cy="12" rx="10.5" ry="3.4" transform="rotate(-24 12 12)"/>'
               '<circle cx="19.2" cy="4.8" r="1" fill="currentColor" stroke="none"/></svg>')
ICON_CHART = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
              '<path d="M3 20h18"/><path d="M4 15.5l5-5.5 4 4 7-8"/><path d="M15.5 6H20v4.5"/>'
              '<path d="M6 20v-2.5M11 20v-4M16 20v-6M21 20v-8" stroke-opacity=".45"/></svg>')
ICON_ORBIT = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round">'
              '<circle cx="12" cy="12" r="3.2" fill="currentColor"/><ellipse cx="12" cy="12" rx="10" ry="4.2" transform="rotate(-30 12 12)"/>'
              '<circle cx="4.6" cy="16.2" r="1.3" fill="currentColor" stroke="none"/></svg>')


def starfield_svg(seed=7, n=175):
    #a contained night sky for the classifier banner, generated once so it does not shimmer between reruns
    rng = random.Random(seed)
    stars = []
    for _ in range(n):
        x, y = rng.uniform(0, 1200), rng.uniform(0, 300)
        r = rng.choice([0.6, 0.7, 0.9, 1.0, 1.2, 1.5, 1.9])
        col = rng.choice(["#ffffff", "#ffffff", "#ffffff", "#dbe6ff", "#ffe7b3"])
        cls = rng.choice(["s0", "s1", "s2", "s3", "s0"])
        op = rng.uniform(.35, .95)
        stars.append(f'<circle class="{cls}" cx="{x:.0f}" cy="{y:.0f}" r="{r}" fill="{col}" opacity="{op:.2f}" '
                     f'style="animation-delay:{-rng.uniform(0, 8):.1f}s"/>')
    bright = ""
    for x, y in [(160, 62), (470, 210), (720, 48), (905, 255)]:
        bright += (f'<g class="s2" style="animation-delay:{-rng.uniform(0, 5):.1f}s">'
                   f'<circle cx="{x}" cy="{y}" r="2.2" fill="#fff"/>'
                   f'<path d="M{x - 9} {y}h18M{x} {y - 9}v18" stroke="#fff" stroke-width=".8" opacity=".7"/></g>')
    planet = ('<g transform="translate(1035 158)">'
              '<ellipse rx="112" ry="26" transform="rotate(-18)" fill="none" stroke="rgba(255,226,170,.55)" stroke-width="7"/>'
              '<ellipse rx="128" ry="31" transform="rotate(-18)" fill="none" stroke="rgba(255,226,170,.22)" stroke-width="3"/>'
              '<circle r="62" fill="url(#planet)"/>'
              '<clipPath id="front"><rect x="-140" y="0" width="280" height="80"/></clipPath>'
              '<ellipse rx="112" ry="26" transform="rotate(-18)" fill="none" stroke="rgba(255,236,190,.75)" stroke-width="7" clip-path="url(#front)"/>'
              '<circle cx="-92" cy="-58" r="7" fill="#d9e2ff" opacity=".9"/></g>')
    defs = ('<defs>'
            '<radialGradient id="planet" cx=".35" cy=".3" r=".8"><stop offset="0" stop-color="#ffd58a"/>'
            '<stop offset=".55" stop-color="#d98f2b"/><stop offset="1" stop-color="#4a2a08"/></radialGradient>'
            '<radialGradient id="nebA"><stop offset="0" stop-color="#5865f2" stop-opacity=".38"/><stop offset="1" stop-color="#5865f2" stop-opacity="0"/></radialGradient>'
            '<radialGradient id="nebB"><stop offset="0" stop-color="#f5b94e" stop-opacity=".16"/><stop offset="1" stop-color="#f5b94e" stop-opacity="0"/></radialGradient>'
            '<linearGradient id="shade" x1="0" x2="1"><stop offset="0" stop-color="#080d21" stop-opacity=".92"/><stop offset="1" stop-color="#080d21" stop-opacity="0"/></linearGradient>'
            '<linearGradient id="tail" x1="0" x2="1"><stop offset="0" stop-color="#fff" stop-opacity="0"/><stop offset="1" stop-color="#fff"/></linearGradient>'
            '</defs>')
    style = ('<style>.s1{animation:tw 3.4s ease-in-out infinite}.s2{animation:tw 5.2s ease-in-out infinite}.s3{animation:tw 7.6s ease-in-out infinite}'
             '@keyframes tw{0%,100%{opacity:.2}50%{opacity:1}}'
             '.shoot{animation:shoot 12s ease-in infinite;animation-delay:1.5s;opacity:0}'
             '@keyframes shoot{0%{transform:translate(180px,20px);opacity:0}3%{opacity:1}16%{transform:translate(760px,190px);opacity:0}100%{transform:translate(760px,190px);opacity:0}}</style>')
    return (f'<svg class="sky" viewBox="0 0 1200 300" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">{defs}{style}'
            f'<rect width="1200" height="300" fill="#080d21"/>'
            f'<ellipse cx="300" cy="70" rx="480" ry="220" fill="url(#nebA)"/><ellipse cx="880" cy="290" rx="420" ry="170" fill="url(#nebB)"/>'
            f'{"".join(stars)}{bright}<line class="shoot" x1="0" y1="0" x2="110" y2="34" stroke="url(#tail)" stroke-width="2" stroke-linecap="round"/>'
            f'{planet}<rect width="760" height="300" fill="url(#shade)"/></svg>')


def market_svg(seed=3):
    #a quiet price line for the stock banner, drawn on when the page loads
    rng = random.Random(seed)
    pts, y = [], 215.0
    for i in range(41):
        y = min(max(y - 2.6 + rng.uniform(-11, 11), 60), 250)
        pts.append((i * 30, y))
    line = "M" + " L".join(f"{x} {y:.0f}" for x, y in pts)
    area = line + " L1200 300 L0 300 Z"
    grid = "".join(f'<line x1="0" x2="1200" y1="{y}" y2="{y}" stroke="{T["text"]}" stroke-opacity=".06"/>' for y in range(60, 300, 60))
    return (f'<svg class="sky" viewBox="0 0 1200 300" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">'
            f'<defs><linearGradient id="area" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="{T["green"]}" stop-opacity=".22"/>'
            f'<stop offset="1" stop-color="{T["green"]}" stop-opacity="0"/></linearGradient>'
            f'<linearGradient id="fade" x1="0" x2="1"><stop offset="0" stop-color="{T["surface"]}"/><stop offset=".42" stop-color="{T["surface"]}" stop-opacity=".92"/>'
            f'<stop offset=".7" stop-color="{T["surface"]}" stop-opacity=".25"/><stop offset="1" stop-color="{T["surface"]}" stop-opacity="0"/></linearGradient></defs>'
            f'<style>.pl{{stroke-dasharray:2600;stroke-dashoffset:2600;animation:draw 2.6s ease-out forwards}}@keyframes draw{{to{{stroke-dashoffset:0}}}}'
            f'.ar{{opacity:0;animation:show 1.4s ease-out 1s forwards}}@keyframes show{{to{{opacity:1}}}}</style>'
            f'{grid}<path class="ar" d="{area}" fill="url(#area)"/>'
            f'<path class="pl" d="{line}" fill="none" stroke="{T["green"]}" stroke-width="2.5" stroke-linejoin="round" vector-effect="non-scaling-stroke"/>'
            f'<rect width="1200" height="300" fill="url(#fade)"/></svg>')


#---------- small ui helpers ----------
def hero(kind, eyebrow, title, icon, subtitle, chips):
    sky = starfield_svg() if kind == "space" else (market_svg() if kind == "finance" else "")
    chip_html = "".join(f'<span class="chip" style="--i:{i}"><b>{v}</b>{k}</span>' for i, (v, k) in enumerate(chips))
    st.markdown(f'<div class="hero {kind}">{sky}<div class="hero-body"><div class="eyebrow">{eyebrow}</div>'
                f'<div class="hero-title">{icon}<span>{title}</span></div><div class="hero-sub">{subtitle}</div>'
                f'<div class="chips">{chip_html}</div></div></div>', unsafe_allow_html=True)


def text(kind, body):
    st.markdown(f'<div class="{kind}">{body}</div>', unsafe_allow_html=True)


def bars(rows):
    #rows: (label, fraction 0..1, display text, colour); bars animate in from zero
    html = ""
    for i, (label, frac, disp, color) in enumerate(rows):
        html += (f'<div class="bar-row" style="animation-delay:{i * 70}ms"><span class="bar-label">{label}</span>'
                 f'<div class="bar-track"><div class="bar-fill" style="--w:{max(frac, 0) * 100:.1f}%;background:{color}"></div></div>'
                 f'<span class="bar-val">{disp}</span></div>')
    return html


def themed(fig, height=340, title=None, legend=True):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=height,
        font=dict(family=SANS, color=T["text"], size=12),
        title=dict(text=title or "", font=dict(size=14, color=T["text"]), x=0, xanchor="left"),
        margin=dict(l=10, r=14, t=46 if title else 14, b=10), showlegend=legend,
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.22, x=0, font=dict(size=11)),
        hoverlabel=dict(bgcolor=T["surface2"], font=dict(color=T["text"], family=SANS), bordercolor=T["border"]),
        xaxis=dict(gridcolor=T["grid"], zerolinecolor=T["grid"], linecolor=T["grid"], title_font=dict(size=12, color=T["muted"])),
        yaxis=dict(gridcolor=T["grid"], zerolinecolor=T["grid"], linecolor=T["grid"], title_font=dict(size=12, color=T["muted"])),
    )
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def fmt_step(v):
    a = abs(v)
    if a >= 1e5:
        return "%.0f", float(10 ** max(int(np.floor(np.log10(a))) - 2, 0))
    if a >= 10:
        return "%.2f", 1.0
    if a >= 1:
        return "%.3f", 0.1
    return "%.4f", 0.01


def number_inputs(features, labels, defaults, prefix, ncols=2):
    #one row of columns per ncols inputs, so the order still reads left to right when rows stack on narrow screens
    values = {}
    for start in range(0, len(features), ncols):
        cols = st.columns(ncols)
        for col, f in zip(cols, features[start:start + ncols]):
            fmt, step = fmt_step(float(defaults[f]))
            values[f] = col.number_input(labels.get(f, f), value=float(defaults[f]), step=step, format=fmt, key=f"{prefix}_{f}")
    return values


def pct_rank(quantiles, v):
    #position of a value inside the 0..100 percentile grid of the training data
    return float(min(np.searchsorted(np.asarray(quantiles), v, side="right"), 100))


#---------- model calls ----------
def classify_koi(cls, values):
    #anything not supplied stays NaN and the pipeline's median imputer fills it
    row = {f: np.nan for f in cls["feature_columns"]}
    row.update(values)
    if values.get("koi_duration"):
        row["transit_depth_per_hour"] = values["koi_depth"] / values["koi_duration"]
    frame = pd.DataFrame([row])[cls["feature_columns"]]
    pipe = cls["pipeline"]
    label = str(pipe.predict(frame)[0])
    probs = dict(zip([str(c) for c in pipe.classes_], pipe.predict_proba(frame)[0]))
    return label, probs


def predict_rows(reg, sector, rows):
    #rows: list of {feature: value}; missing features fall back to training medians
    frame = pd.DataFrame([{**{f: reg["train_medians"][f] for f in reg["raw_features"]}, **r} for r in rows])
    for f, (lo, hi) in reg["clip_bounds"].items():
        frame[f] = frame[f].clip(lo, hi)
    for col in reg["sector_dummy_columns"]:
        frame[col] = 1 if col == f"Sector_{sector}" else 0
    frame["Growth Adjusted Margin"] = frame["Revenue Growth"] * frame["netProfitMargin"]
    frame = frame.reindex(columns=reg["model_columns"], fill_value=0)
    return reg["model"].predict(frame).astype(float)


def predict_return(reg, sector, values):
    return float(predict_rows(reg, sector, [values])[0])


def sensitivity(reg, sector, values, n=7):
    #move one indicator at a time across its training percentiles, keep everything else as entered
    feats = [f for f in reg["feature_importance"] if f in reg["raw_features"]][:n]
    grid = [10, 25, 50, 75, 90]
    rows = [{**values, f: reg["feature_quantiles"][f][p]} for f in feats for p in grid]
    preds = predict_rows(reg, sector, rows).reshape(len(feats), len(grid))
    out = []
    for f, p in zip(feats, preds):
        out.append(dict(feature=f, low=float(p[0]), high=float(p[-1]), lo=float(p.min()), hi=float(p.max()),
                        v_low=reg["feature_quantiles"][f][10], v_high=reg["feature_quantiles"][f][90]))
    return sorted(out, key=lambda d: d["hi"] - d["lo"], reverse=True)


#---------- animated result cards (iframes, so they can count up and sweep in) ----------
def result_card_html(body, extra_css=""):
    return (f"<style>{FONT_FACES}body{{margin:0;font-family:{SANS};color:{T['text']};background:transparent;-webkit-font-smoothing:antialiased}}"
            f".mono{{font-family:{MONO}}}{extra_css}</style>{body}")


def classifier_card(label, probs, per_class):
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    circ = 2 * np.pi * 46
    segs, start = "", 0.0
    for name, p in ranked:
        segs += (f'<circle class="seg" cx="60" cy="60" r="46" stroke="{CLASS_COLORS[name]}" data-off="{circ * (1 - p):.2f}" '
                 f'style="stroke-dasharray:{circ:.2f};stroke-dashoffset:{circ:.2f};transform:rotate({start * 360 - 90:.1f}deg)"/>')
        start += p
    rows = "".join(
        f'<div class="row"><span class="nm">{n}</span><div class="track"><div class="fill" data-w="{p * 100:.1f}" style="background:{CLASS_COLORS[n]}"></div></div>'
        f'<span class="val mono" data-count="{p * 100:.1f}" data-suffix="%">0.0%</span></div>' for n, p in ranked)
    f1 = per_class[label]["f1"]
    css = f"""
.wrap{{display:flex;gap:clamp(14px,4vw,28px);align-items:center;padding:6px 2px}}
.donut{{position:relative;width:clamp(124px,30vw,170px);height:clamp(124px,30vw,170px);flex:0 0 auto}}
.donut svg{{width:100%;height:100%}}
.seg{{fill:none;stroke-width:15;transform-origin:60px 60px;transition:stroke-dashoffset 1.3s cubic-bezier(.2,.8,.2,1)}}
.ring{{fill:none;stroke:{T['chip']};stroke-width:15}}
.center{{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center}}
.pct{{font-size:clamp(22px,6vw,30px);font-weight:600;line-height:1}}
.lbl{{font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:{T['muted']};margin-top:6px}}
.info{{flex:1 1 auto;min-width:0}}
.badge{{display:inline-block;padding:7px 15px;border-radius:999px;font-weight:600;font-size:15px;letter-spacing:.02em;
  background:{CLASS_COLORS[label]}22;color:{CLASS_COLORS[label]};border:1px solid {CLASS_COLORS[label]}66;animation:pop .7s cubic-bezier(.2,.8,.2,1) both}}
@keyframes pop{{from{{opacity:0;transform:scale(.9)}}to{{opacity:1;transform:none}}}}
.note{{color:{T['muted']};font-size:13.5px;line-height:1.5;margin:10px 0 12px}}
.note b{{color:{T['text']}}}
.row{{display:grid;grid-template-columns:minmax(90px,118px) 1fr 52px;align-items:center;gap:10px;margin:7px 0;font-size:13px}}
.nm{{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.track{{height:8px;background:{T['chip']};border:1px solid {T['border']};border-radius:99px;overflow:hidden}}
.fill{{height:100%;width:0;border-radius:99px;transition:width 1.2s cubic-bezier(.2,.8,.2,1)}}
.val{{text-align:right;color:{T['muted']};font-size:12.5px}}
"""
    body = f"""
<div class="wrap">
  <div class="donut"><svg viewBox="0 0 120 120"><circle class="ring" cx="60" cy="60" r="46"/>{segs}</svg>
    <div class="center"><div class="pct mono" data-count="{probs[label] * 100:.0f}" data-suffix="%" data-dec="0">0%</div><div class="lbl">{label}</div></div></div>
  <div class="info"><span class="badge">{label}</span>
    <div class="note">The model reads this object as <b>{label}</b>: {CLASS_BLURB[label]}. On held-out objects it gets this class right
    with an F1 of <b>{f1:.2f}</b>{", the hardest of the three" if label == "CANDIDATE" else ""}.</div>{rows}</div>
</div>
<script>
requestAnimationFrame(()=>setTimeout(()=>{{
  document.querySelectorAll('.seg').forEach(e=>e.style.strokeDashoffset=e.dataset.off);
  document.querySelectorAll('.fill').forEach(e=>e.style.width=e.dataset.w+'%');
  document.querySelectorAll('[data-count]').forEach(e=>{{const t=parseFloat(e.dataset.count),d=e.dataset.dec!==undefined?+e.dataset.dec:1,s=e.dataset.suffix||'',t0=performance.now();
    const step=n=>{{const k=Math.min((n-t0)/1200,1),v=t*(1-Math.pow(1-k,3));e.textContent=v.toFixed(d)+s;if(k<1)requestAnimationFrame(step)}};requestAnimationFrame(step)}});
}},60));
</script>"""
    st.iframe(result_card_html(body, css), height=232)


def gauge_geom(v, lo=-60, hi=60):
    #maps a return to an angle on the half-circle dial, -90 is the left end and +90 the right end
    return -90 + (min(max(v, lo), hi) - lo) / (hi - lo) * 180


def arc_point(deg, r, cx=120, cy=118):
    a = np.radians(deg)
    return cx + r * np.sin(a), cy - r * np.cos(a)


def regression_card(pred, stats, sector, sector_median, r2):
    ticks = ""
    for v in (-60, -30, 0, 30, 60):
        d = gauge_geom(v)
        (x1, y1), (x2, y2), (tx, ty) = arc_point(d, 86), arc_point(d, 96), arc_point(d, 109)
        ticks += (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{T["muted"]}" stroke-width="1.4"/>'
                  f'<text x="{tx:.1f}" y="{ty + 4:.1f}" text-anchor="middle" font-size="9.5" fill="{T["muted"]}">{v:+d}%</text>')
    a0, a1 = gauge_geom(stats["q25"]), gauge_geom(stats["q75"])
    (qx0, qy0), (qx1, qy1) = arc_point(a0, 72), arc_point(a1, 72)
    (mx0, my0), (mx1, my1) = arc_point(gauge_geom(stats["median"]), 66), arc_point(gauge_geom(stats["median"]), 78)
    (lx, ly), (tx_, ty_), (rx, ry) = arc_point(-90, 90), arc_point(0, 90), arc_point(90, 90)
    tone = T["up"] if pred >= 0 else T["down"]
    rel = "above" if pred >= stats["median"] else "below"
    css = f"""
.wrap{{display:flex;gap:clamp(12px,3vw,24px);align-items:center;padding:4px 2px}}
.gauge{{flex:0 0 auto;width:clamp(170px,40vw,240px)}}
.needle{{transform-origin:120px 118px;transform:rotate(-90deg);transition:transform 1.5s cubic-bezier(.3,1.2,.35,1)}}
.info{{flex:1 1 auto;min-width:0}}
.big{{font-size:clamp(30px,8vw,44px);font-weight:600;line-height:1;color:{tone};animation:pop .8s cubic-bezier(.2,.8,.2,1) both}}
@keyframes pop{{from{{opacity:0;transform:scale(.92)}}to{{opacity:1;transform:none}}}}
.lbl{{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:{T['muted']};margin:8px 0 10px}}
.line{{display:flex;justify-content:space-between;gap:12px;font-size:13px;padding:5px 0;border-top:1px solid {T['border']};color:{T['muted']}}}
.line b{{color:{T['text']};font-weight:600;text-align:right;white-space:nowrap}}
.small{{font-size:12px;color:{T['muted']};line-height:1.5;margin-top:8px}}
"""
    body = f"""
<div class="wrap">
  <svg class="gauge" viewBox="0 0 240 132">
    <defs><linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="{T['red']}"/><stop offset=".5" stop-color="{T['faint']}"/><stop offset="1" stop-color="{T['green']}"/></linearGradient></defs>
    <path d="M{lx:.1f} {ly:.1f} A90 90 0 0 1 {tx_:.1f} {ty_:.1f} A90 90 0 0 1 {rx:.1f} {ry:.1f}" fill="none" stroke="url(#g)" stroke-width="12" stroke-linecap="round" opacity=".9"/>
    <path d="M{qx0:.1f} {qy0:.1f} A72 72 0 0 1 {qx1:.1f} {qy1:.1f}" fill="none" stroke="{T['blue']}" stroke-width="5" stroke-linecap="round" opacity=".75"/>
    <line x1="{mx0:.1f}" y1="{my0:.1f}" x2="{mx1:.1f}" y2="{my1:.1f}" stroke="{T['blue']}" stroke-width="2"/>
    {ticks}
    <polygon class="needle" points="120,118 114.5,118 120,42 125.5,118" fill="{T['text']}" data-deg="{gauge_geom(pred):.1f}"/>
    <circle cx="120" cy="118" r="7" fill="{T['surface']}" stroke="{T['text']}" stroke-width="2.5"/>
  </svg>
  <div class="info">
    <div class="big mono" data-count="{pred:.1f}" data-suffix="%" data-signed="1">+0.0%</div>
    <div class="lbl">predicted price change over the next year</div>
    <div class="line"><span>Training median</span><b class="mono">{stats['median']:+.1f}%</b></div>
    <div class="line"><span>{sector} median</span><b class="mono">{sector_median:+.1f}%</b></div>
    <div class="line"><span>Middle half (blue arc)</span><b class="mono">{stats['q25']:+.1f}% to {stats['q75']:+.1f}%</b></div>
    <div class="small">That puts this company <b>{rel}</b> the typical one. Fundamentals explain only about {r2 * 100:.0f}% of the
    variance in next-year returns, so read this as a lean from the numbers, not a forecast.</div>
  </div>
</div>
<script>
requestAnimationFrame(()=>setTimeout(()=>{{
  const n=document.querySelector('.needle');n.style.transform='rotate('+n.dataset.deg+'deg)';
  document.querySelectorAll('[data-count]').forEach(e=>{{const t=parseFloat(e.dataset.count),s=e.dataset.suffix||'',t0=performance.now();
    const step=now=>{{const k=Math.min((now-t0)/1400,1),v=t*(1-Math.pow(1-k,3));e.textContent=(v>=0?'+':'')+v.toFixed(1)+s;if(k<1)requestAnimationFrame(step)}};requestAnimationFrame(step)}});
}},60));
</script>"""
    st.iframe(result_card_html(body, css), height=300)


#---------- charts ----------
def koi_scatter(cls, period=None, depth=None):
    sample = cls["scatter_sample"]
    fig = go.Figure()
    for name, color in CLASS_COLORS.items():
        part = sample[sample["koi_disposition"] == name]
        fig.add_trace(go.Scatter(
            x=part["koi_period"], y=part["koi_depth"], mode="markers", name=name,
            marker=dict(color=color, size=5, opacity=.5),
            hovertemplate="period %{x:.2f} d<br>depth %{y:.0f} ppm<extra>" + name + "</extra>"))
    if period is not None:
        fig.add_trace(go.Scatter(
            x=[max(period, 1e-3)], y=[max(depth, 1e-3)], mode="markers", name="your object",
            marker=dict(symbol="star", size=22, color="#ffffff" if MODE == "dark" else "#1b2233",
                        line=dict(color=T["amber"], width=2.5)),
            hovertemplate="your object<br>period %{x:.2f} d<br>depth %{y:.0f} ppm<extra></extra>"))
    fig.update_xaxes(type="log", title="Orbital period (days, log scale)")
    fig.update_yaxes(type="log", title="Transit depth (ppm, log scale)")
    return themed(fig, 400, f"Your object among {len(sample):,} known Kepler objects" if period is not None
                  else f"{len(sample):,} known Kepler objects by period and depth")


def koi_radar(cls, values):
    feats = [f for f in CLS_PRIMARY if f in cls["feature_quantiles"]]
    labels = [CLS_SHORT.get(f, f) for f in feats] + [CLS_SHORT.get(feats[0], feats[0])]
    fig = go.Figure()
    for name, color in CLASS_COLORS.items():
        r = [pct_rank(cls["feature_quantiles"][f], cls["class_medians"][name][f]) for f in feats]
        fig.add_trace(go.Scatterpolar(r=r + r[:1], theta=labels, name=f"typical {name.lower()}", line=dict(color=color, width=1.5),
                                      opacity=.85, hovertemplate="%{theta}: %{r:.0f}th percentile<extra>" + name + "</extra>"))
    r = [pct_rank(cls["feature_quantiles"][f], values[f]) for f in feats]
    fig.add_trace(go.Scatterpolar(r=r + r[:1], theta=labels, name="your object", fill="toself",
                                  fillcolor="rgba(245,185,78,.18)" if MODE == "dark" else "rgba(209,143,18,.16)",
                                  line=dict(color=T["amber"], width=3), marker=dict(size=7),
                                  hovertemplate="%{theta}: %{r:.0f}th percentile<extra>your object</extra>"))
    fig.update_polars(bgcolor="rgba(0,0,0,0)",
                      radialaxis=dict(range=[0, 100], tickvals=[25, 50, 75, 100], ticksuffix="th", gridcolor=T["grid"],
                                      linecolor=T["grid"], tickfont=dict(size=9, color=T["muted"])),
                      angularaxis=dict(gridcolor=T["grid"], linecolor=T["grid"], tickfont=dict(size=11)))
    return themed(fig, 420, "Each measurement as a percentile of all Kepler objects")


def confusion_heatmap(cls):
    classes, cm = cls["classes"], np.array(cls["confusion"])
    share = cm / cm.sum(axis=1, keepdims=True)
    txt = [[f"{cm[i, j]:,}<br>{share[i, j]:.0%}" for j in range(3)] for i in range(3)]
    fig = go.Figure(go.Heatmap(
        z=share, x=classes, y=classes, text=txt, texttemplate="%{text}", textfont=dict(size=12),
        colorscale=[[0, T["surface2"]], [1, T["teal"]]], showscale=False, xgap=4, ygap=4,
        hovertemplate="actual %{y}<br>predicted %{x}<br>%{text}<extra></extra>"))
    fig.update_xaxes(title="Predicted", side="bottom")
    fig.update_yaxes(title="Actual", autorange="reversed")
    return themed(fig, 360, f"Held-out confusion matrix ({cm.sum():,} objects)", legend=False)


def per_class_bars(cls):
    classes = cls["classes"]
    fig = go.Figure()
    for metric, color in (("precision", T["blue"]), ("recall", T["violet"]), ("f1", T["teal"])):
        vals = [cls["per_class"][c][metric] for c in classes]
        fig.add_trace(go.Bar(name=metric.upper() if metric == "f1" else metric, x=classes, y=vals, marker_color=color,
                             text=[f"{v:.2f}" for v in vals], textposition="outside", textfont=dict(size=11),
                             hovertemplate="%{x}<br>" + metric + " %{y:.3f}<extra></extra>"))
    fig.update_yaxes(range=[0, 1.12], title="score")
    fig.update_layout(bargap=.3, bargroupgap=.08)
    return themed(fig, 360, "Precision, recall and F1 per class")


def class_donut(counts, title):
    names = [n for n in CLASS_COLORS if n in counts]
    fig = go.Figure(go.Pie(labels=names, values=[counts[n] for n in names], hole=.62, sort=False,
                           marker=dict(colors=[CLASS_COLORS[n] for n in names], line=dict(color=T["bg"], width=2)),
                           textinfo="percent", textposition="inside", insidetextorientation="horizontal",
                           textfont=dict(size=11, color="#0b1020" if MODE == "dark" else "#fff"),
                           hovertemplate="%{label}: %{value:,} objects<extra></extra>"))
    fig.add_annotation(text=f"<b>{sum(counts.values()):,}</b><br>objects", showarrow=False, font=dict(size=13, color=T["text"]))
    fig.update_layout(legend=dict(orientation="v", x=1.02, y=.5, xanchor="left"))
    return themed(fig, 232, title)


def return_histogram(reg, pred=None):
    y, s = np.asarray(reg["y_train"], dtype=float), reg["target_stats"]
    fig = go.Figure(go.Histogram(x=y, nbinsx=60, name="training companies",
                                 marker=dict(color=T["blue"], opacity=.55, line=dict(color=T["blue"], width=.5))))
    fig.add_vline(x=s["median"], line_color=T["muted"], line_dash="dot")
    fig.add_annotation(x=s["median"], y=.96, yref="paper", text=f"median {s['median']:+.1f}%", showarrow=False,
                       font=dict(color=T["muted"], size=11), xanchor="left", xshift=6)
    if pred is not None:
        fig.add_vline(x=pred, line_color=T["amber"], line_width=3)
        fig.add_annotation(x=pred, y=1.02, yref="paper", yanchor="bottom", text=f"this company {pred:+.1f}%",
                           showarrow=False, font=dict(color=T["amber"], size=12))
    fig.update_layout(bargap=.06, xaxis_title="Next-year price change (%)", yaxis_title="Companies")
    return themed(fig, 340, f"Where this prediction lands among {len(y):,} training companies" if pred is not None
                  else f"Next-year price change of {len(y):,} training companies", legend=False)


def sensitivity_chart(rows, pred):
    names = [REG_LABELS.get(r["feature"], r["feature"]) for r in rows][::-1]
    rows = rows[::-1]
    fig = go.Figure()
    for name, r in zip(names, rows):
        fig.add_trace(go.Scatter(x=[r["lo"], r["hi"]], y=[name, name], mode="lines", line=dict(color=T["faint"], width=4),
                                 showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[r["low"] for r in rows], y=names, mode="markers", name="input at its 10th percentile",
                             marker=dict(color=T["blue"], size=12, line=dict(color=T["bg"], width=1.5)),
                             customdata=[[r["v_low"]] for r in rows],
                             hovertemplate="%{y} at %{customdata[0]:.3g}<br>prediction %{x:+.1f}%<extra>low input</extra>"))
    fig.add_trace(go.Scatter(x=[r["high"] for r in rows], y=names, mode="markers", name="input at its 90th percentile",
                             marker=dict(color=T["amber"], size=12, line=dict(color=T["bg"], width=1.5)),
                             customdata=[[r["v_high"]] for r in rows],
                             hovertemplate="%{y} at %{customdata[0]:.3g}<br>prediction %{x:+.1f}%<extra>high input</extra>"))
    fig.add_vline(x=pred, line_color=T["muted"], line_dash="dot")
    fig.add_annotation(x=pred, y=1.03, yref="paper", yanchor="bottom", text=f"as entered {pred:+.1f}%", showarrow=False,
                       font=dict(color=T["muted"], size=11))
    fig.update_xaxes(title="Predicted next-year change (%)", ticksuffix="%")
    fig.update_yaxes(automargin=True, tickfont=dict(size=11))
    return themed(fig, 30 * len(rows) + 150, "If one input moved and the rest stayed as entered")


def sector_chart(reg, sector=None):
    stats = reg["sector_stats"]
    order = sorted(stats, key=lambda s: stats[s]["median"])
    colors = [T["amber"] if s == sector else T["teal"] for s in order]
    fig = go.Figure(go.Bar(
        x=[stats[s]["median"] for s in order], y=order, orientation="h", marker=dict(color=colors, opacity=.9),
        text=[f"{stats[s]['median']:+.1f}%" for s in order], textposition="outside", textfont=dict(size=11), cliponaxis=False,
        customdata=[[stats[s]["count"]] for s in order],
        hovertemplate="%{y}<br>median %{x:+.1f}% across %{customdata[0]} training companies<extra></extra>"))
    lo, hi = min(stats[s]["median"] for s in order), max(stats[s]["median"] for s in order)
    fig.update_xaxes(title="Median next-year change (%)", ticksuffix="%", range=[min(lo, 0) - 6, hi + 8])
    fig.update_yaxes(automargin=True, tickfont=dict(size=11))
    return themed(fig, 28 * len(order) + 110, "Typical next-year change by sector (training companies)", legend=False)


def actual_vs_predicted(reg):
    y, p, m = np.asarray(reg["y_test"]), np.asarray(reg["y_pred_test"]), reg["metrics"]
    lo, hi = float(min(y.min(), p.min())), float(max(y.max(), p.max()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="perfect prediction",
                             line=dict(color=T["muted"], dash="dot", width=1.5)))
    fig.add_trace(go.Scatter(x=y, y=p, mode="markers", name="held-out company", marker=dict(color=T["teal"], size=6, opacity=.45),
                             hovertemplate="actual %{x:+.1f}%<br>predicted %{y:+.1f}%<extra></extra>"))
    fig.add_annotation(x=.02, y=.97, xref="paper", yref="paper", xanchor="left", showarrow=False, align="left",
                       text=f"R2 {m['R2']:.3f}<br>RMSE {m['RMSE']:.1f} pts<br>MAE {m['MAE']:.1f} pts",
                       font=dict(size=11, color=T["text"]), bgcolor=T["surface2"], bordercolor=T["border"], borderpad=6)
    fig.update_xaxes(title="Actual next-year change (%)", ticksuffix="%")
    fig.update_yaxes(title="Predicted (%)", ticksuffix="%")
    return themed(fig, 380, f"Actual vs predicted on {len(y):,} held-out companies")


def comparison_bars(df, metric, best_label, title):
    d = df.sort_values(metric, ascending=True)
    fig = go.Figure(go.Bar(
        x=d[metric], y=d["Model"], orientation="h", cliponaxis=False,
        marker=dict(color=[T["amber"] if m == best_label else T["teal"] for m in d["Model"]], opacity=.9),
        text=[f"{v:.3f}" for v in d[metric]], textposition="outside", textfont=dict(size=12),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    fig.update_xaxes(range=[0, float(d[metric].max()) * 1.35], title=metric)
    fig.update_yaxes(automargin=True)
    return themed(fig, 30 * len(d) + 110, title, legend=False)


def grouped_metric_bars(df, metrics, colors, title):
    d = df.iloc[::-1]
    fig = go.Figure()
    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Bar(name=metric, y=d["Model"], x=d[metric], orientation="h", marker_color=color, opacity=.9,
                             hovertemplate="%{y}<br>" + metric + " %{x:.4f}<extra></extra>"))
    fig.update_xaxes(range=[0, 1.02], title="score")
    fig.update_yaxes(automargin=True)
    fig.update_layout(bargap=.25, bargroupgap=.06)
    return themed(fig, 34 * len(d) + 130, title)


def sector_counts_chart(reg):
    counts = reg["sector_counts"]
    order = sorted(counts, key=counts.get)
    fig = go.Figure(go.Bar(x=[counts[s] for s in order], y=order, orientation="h", marker_color=T["blue"], opacity=.85,
                           text=[f"{counts[s]:,}" for s in order], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:,} companies<extra></extra>"))
    fig.update_xaxes(range=[0, max(counts.values()) * 1.25], title="companies")
    fig.update_yaxes(automargin=True, tickfont=dict(size=11))
    return themed(fig, 28 * len(order) + 100, "Companies per sector (2018 data)", legend=False)


#---------- pages ----------
def page_classifier():
    cls, reg = load_bundles()
    m, per_class = cls["metrics"], cls["per_class"]
    total = cls["class_counts_all"]
    hero("space", "Classification · NASA Kepler", "Exoplanet classifier", ICON_PLANET,
         "Enter what Kepler measured for a transit signal and its host star. The tuned Support Vector Classifier, "
         "the best of the five Review 1 classifiers, decides whether it looks like a confirmed planet, a candidate "
         "or a false positive, shows how sure it is, and places it among the objects it learned from.",
         [(f"{sum(total.values()):,}", "objects"), (f"{total.get('CONFIRMED', 0):,}", "confirmed planets"),
          (f"{m['F1_Weighted']:.3f}", "weighted F1"), (f"{m['Accuracy']:.1%}", "accuracy")])

    text("section", "Measurements")
    text("hint", "Pre-filled with typical (median) values. Change the ones you know, then classify.")
    with st.form("cls_form"):
        primary = [f for f in CLS_PRIMARY if f in cls["key_features"]]
        values = number_inputs(primary, CLS_LABELS, cls["feature_medians"], "c", ncols=4)
        more = [f for f in cls["key_features"] if f not in primary]
        a, b = st.columns([2, 1], gap="medium")
        with a:
            with st.expander("More measurements"):
                values.update(number_inputs(more, CLS_LABELS, cls["feature_medians"], "c2", ncols=2))
        with b:
            submitted = st.form_submit_button("Classify object", type="primary", width="stretch")
        if submitted:
            st.session_state["cls"] = (values, classify_koi(cls, values))

    left, right = st.columns([7, 5], gap="large")
    if "cls" not in st.session_state:
        with left:
            text("empty", "Press <b>Classify object</b> to get the predicted class, the probability of each class, "
                          "a comparison with typical objects of each kind, and a report card for the model.")
            show(koi_scatter(cls))
            text("caption", "Confirmed planets and candidates sit in a band of small, shallow transits. False positives "
                            "spread much wider, which is what the classifier picks up on.")
        with right:
            show(class_donut(total, "What the model learned from"))
            text("caption", "All Kepler objects of interest by disposition. Half are false positives, so the classifier "
                            "has to earn its confirmed calls.")
        return

    values, (label, probs) = st.session_state["cls"]
    with left:
        classifier_card(label, probs, per_class)
    with right:
        show(class_donut(total, "What the model learned from"))
    tab1, tab2, tab3 = st.tabs(["Where it sits", "Versus typical objects", "Model report card"])
    with tab1:
        show(koi_scatter(cls, values["koi_period"], values["koi_depth"]))
        text("caption", "The star is your object. Both axes are log-scaled because period and depth span several "
                        "orders of magnitude.")
    with tab2:
        show(koi_radar(cls, values))
        text("caption", "Each axis is the percentile of that measurement among all Kepler objects, so the shapes are "
                        "comparable. Your object's shape follows the class it resembles most.")
    with tab3:
        r1, r2 = st.columns(2, gap="large")
        with r1:
            show(confusion_heatmap(cls))
        with r2:
            show(per_class_bars(cls))
        text("caption", "Rows are the true class, columns the prediction. Candidates are the hardest call: some are "
                        "read as confirmed planets and some as false positives, which is why their F1 is the lowest.")


def page_regression():
    cls, reg = load_bundles()
    m, s = reg["metrics"], reg["target_stats"]
    hero("finance", "Regression · US stock fundamentals", "Stock return predictor", ICON_CHART,
         "Give the tuned Random Forest, the best of the ten Review 1 regressors, a company's 2018 annual fundamentals "
         "and it estimates the share price change over the following year, then shows where that sits among the "
         "companies it was trained on and which inputs move it most.",
         [(f"{reg['n_train'] + reg['n_test']:,}", "companies"), (f"{len(reg['raw_features'])}", "indicators"),
          (f"{m['R2']:.3f}", "test R2"), (f"{m['MAE']:.1f} pts", "test MAE")])

    text("section", "Company fundamentals")
    text("hint", "Pre-filled with training medians. Dollar figures are in USD, ratios as decimals (0.15 = 15%).")
    with st.form("reg_form"):
        sectors = reg["sector_categories"]
        primary = [f for f in REG_PRIMARY if f in reg["raw_features"]]
        row1 = st.columns(3)
        sector = row1[0].selectbox("Sector", sectors, index=sectors.index("Technology") if "Technology" in sectors else 0)
        values = {}
        for col, f in zip(row1[1:], primary[:2]):
            fmt, step = fmt_step(float(reg["train_medians"][f]))
            values[f] = col.number_input(REG_LABELS.get(f, f), value=float(reg["train_medians"][f]), step=step, format=fmt, key=f"r_{f}")
        values.update(number_inputs(primary[2:], REG_LABELS, reg["train_medians"], "r", ncols=3))
        more = [f for f in reg["raw_features"] if f not in primary]
        a, b = st.columns([2, 1], gap="medium")
        with a:
            with st.expander(f"More indicators ({len(more)})"):
                values.update(number_inputs(more, REG_LABELS, reg["train_medians"], "r2", ncols=2))
        with b:
            submitted = st.form_submit_button("Predict next-year return", type="primary", width="stretch")
        if submitted:
            st.session_state["reg"] = (sector, values, predict_return(reg, sector, values))

    imp = list(reg["feature_importance"].items())[:8]
    top = imp[0][1]
    importance_html = ('<div class="panel"><div class="section" style="margin-top:0">What the model weighs most</div>'
                       + bars([(REG_LABELS.get(f, f), v / top, f"{v * 100:.1f}%", T["amber"]) for f, v in imp])
                       + '<div class="caption" style="margin:.6rem 0 0">Random Forest feature importance, share of the total. '
                         'Earnings per share leads, then balance-sheet size and profitability.</div></div>')

    left, right = st.columns([7, 5], gap="large")
    if "reg" not in st.session_state:
        with left:
            text("empty", "Press <b>Predict next-year return</b> to get the predicted change, where it lands among the "
                          "training companies, how each input moves it, and how the model did on held-out companies.")
            show(return_histogram(reg))
            text("caption", "The 2019 price change is centred near zero with a long right tail, so a typical company moved "
                            "only a little and a few moved a lot.")
        with right:
            st.markdown(importance_html, unsafe_allow_html=True)
        return

    sector, values, pred = st.session_state["reg"]
    with left:
        regression_card(pred, s, sector, reg["sector_stats"][sector]["median"], m["R2"])
    with right:
        st.markdown(importance_html, unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["Among all companies", "What moves it", "By sector", "Model report card"])
    with tab1:
        show(return_histogram(reg, pred))
        text("caption", "The amber line is this company. The dotted line is the median training company.")
    with tab2:
        show(sensitivity_chart(sensitivity(reg, sector, values), pred))
        text("caption", "For each input, the model was re-run with that one input at a low (10th percentile) and a "
                        "high (90th percentile) value while everything else stayed as you entered it. Longer bars "
                        "mean the prediction is more sensitive to that input.")
    with tab3:
        show(sector_chart(reg, sector))
        text("caption", f"{sector} is highlighted. Sector is one of the model's inputs, so the same fundamentals "
                        "can lean differently in different sectors.")
    with tab4:
        show(actual_vs_predicted(reg))
        text("caption", "Every model in the capstone lands at a low R2 on this problem: annual fundamentals explain "
                        "only a small share of next-year price moves. The Random Forest still ranks first and its "
                        "predictions lean the right way more often than not.")


def page_about():
    cls, reg = load_bundles()
    hero("plain", "23CSE301 · ML capstone", "About the models", ICON_ORBIT,
         "What the two demos were trained on, how they compare with the other algorithms in the capstone, "
         "and how to run everything yourself.",
         [("10", "regression models"), ("7", "classification runs"), ("2", "datasets"), ("42", "random state")])

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(f'<div class="spec-card"><div class="spec-head">{ICON_PLANET}Classification, Kepler objects of interest</div>'
                    f'<div class="spec"><div class="k">Data</div><div class="v">NASA Kepler Objects of Interest cumulative table, '
                    f'{sum(cls["class_counts_all"].values()):,} objects, three classes.</div>'
                    f'<div class="k">Features</div><div class="v">{len(cls["feature_columns"])} numeric transit and stellar measurements '
                    f'plus an engineered depth-per-hour. The disposition score and false-positive flags are excluded to avoid leakage.</div>'
                    f'<div class="k">Preprocessing</div><div class="v">Median imputation and standard scaling, fitted on training folds only.</div>'
                    f'<div class="k">Split</div><div class="v">Stratified 80/20, {cls["n_train"]:,} train and {cls["n_test"]:,} test objects.</div>'
                    f'<div class="k">Winner</div><div class="v">RBF-kernel SVC (C = 10), tuned with 5-fold GridSearchCV on weighted F1.</div>'
                    f'</div></div>', unsafe_allow_html=True)
        show(grouped_metric_bars(CLASSIFICATION_RESULTS, ["Accuracy", "Weighted F1", "ROC-AUC"], [T["blue"], T["teal"], T["amber"]],
                                 "Part A classifiers on the same held-out split"))
    with c2:
        st.markdown(f'<div class="spec-card"><div class="spec-head">{ICON_CHART}Regression, next-year stock returns</div>'
                    f'<div class="spec"><div class="k">Data</div><div class="v">200+ Financial Indicators of US Stocks (Kaggle), 2018 slice, '
                    f'{reg["n_train"] + reg["n_test"]:,} companies.</div>'
                    f'<div class="k">Target</div><div class="v">Price change over 2019, winsorised at the 1st and 99th percentile.</div>'
                    f'<div class="k">Features</div><div class="v">41 curated fundamentals, one-hot sector and an engineered growth-adjusted margin.</div>'
                    f'<div class="k">Split</div><div class="v">Stratified 80/20 on target quintiles, {reg["n_train"]:,} train and {reg["n_test"]:,} test companies.</div>'
                    f'<div class="k">Winner</div><div class="v">Random Forest (100 trees, depth 8), tuned with 5-fold GridSearchCV. '
                    f'Top-two cross-validated R2: Random Forest 0.061, Gradient Boosting 0.074.</div>'
                    f'</div></div>', unsafe_allow_html=True)
        show(comparison_bars(REGRESSION_RESULTS, "R2", "Random Forest", "All ten regressors by test R2"))

    tab1, tab2 = st.tabs(["Full results tables", "Dataset overview"])
    with tab1:
        text("section", "Classification, Part A (ranked by weighted F1)")
        st.dataframe(CLASSIFICATION_RESULTS, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.4f") for c in ["Accuracy", "Weighted F1", "ROC-AUC"]})
        text("section", "Regression (ranked by test R2)")
        st.dataframe(REGRESSION_RESULTS, hide_index=True, width="stretch",
                     column_config={"R2": st.column_config.NumberColumn(format="%.4f"),
                                    "RMSE": st.column_config.NumberColumn(format="%.2f"),
                                    "MAE": st.column_config.NumberColumn(format="%.2f")})
    with tab2:
        d1, d2 = st.columns(2, gap="large")
        with d1:
            show(class_donut(cls["class_counts_all"], "Kepler objects by disposition"))
        with d2:
            show(sector_counts_chart(reg))

    text("section", "Run it yourself")
    text("hint", "From the repository root, with both datasets in <code>data/</code>:")
    st.code("pip install -r requirements.txt\npython app/export_models.py\nstreamlit run app/app.py", language="bash")
    text("foot", "23CSE301 Machine Learning, Capstone Project. All numbers shown are the ones reported in the project "
                 "notebooks; every model in each track was scored on the same held-out split.")


#---------- app shell ----------
pages = [
    st.Page(page_classifier, title="Exoplanet classifier", icon=":material/planet:", default=True),
    st.Page(page_regression, title="Stock return predictor", icon=":material/finance_mode:", url_path="stocks"),
    st.Page(page_about, title="About the models", icon=":material/info:", url_path="about"),
]
nav = st.navigation(pages, position="hidden")

with st.sidebar:
    st.markdown(f'<div class="brand"><span class="brand-mark">{ICON_ORBIT}</span><span><span class="brand-name">Model Explorer</span>'
                f'<span class="brand-sub">23CSE301 · ML Capstone</span></span></div>', unsafe_allow_html=True)
    for p in pages:
        st.page_link(p, width="stretch")
    st.markdown("---")
    st.toggle(":material/dark_mode: Dark mode", key="dark")
    text("side-note", "Two trained models from the capstone notebooks, served from the saved files in <code>models/</code>. "
                      "Predictions are computed on the spot and nothing you enter is stored.")

try:
    load_bundles()
except FileNotFoundError:
    st.error("Saved models not found. From the repository root run `python app/export_models.py` first.")
    st.stop()

nav.run()
