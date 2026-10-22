"""Streamlit interface for the 23CSE301 ML Capstone.

Serves the notebook models, including the final classifier chosen by cross-validation.

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
                 green="#34d399", blue="#60a5fa", violet="#a78bfa", pink="#f472b6", cyan="#22d3ee", up="#4ade80", down="#f87171"),
    "light": dict(bg="#f4f5f9", surface="#ffffff", surface2="#eef1f8", border="#e1e5ef",
                  text="#1b2233", muted="#5b6478", faint="#8a93a8", grid="rgba(27,34,51,.09)",
                  chip="#eef1f8", amber="#d18f12", teal="#0e9a86", red="#dc4444",
                  green="#15994f", blue="#2c5bd9", violet="#7c5cd6", pink="#db2777", cyan="#0e7490", up="#15994f", down="#dc4444"),
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
.hero.survey {{ background: #090d26; border-color: rgba(34,211,238,.18); }}
.hero svg.sky {{ position: absolute; inset: 0; width: 100%; height: 100%; display: block; }}
.hero-body {{ position: relative; z-index: 1; max-width: 660px; }}
.hero.space .hero-body, .hero.survey .hero-body {{ color: #f3f5fa; }}
.eyebrow {{ font-size: .72rem; letter-spacing: .16em; text-transform: uppercase; font-weight: 600; color: var(--muted); }}
.hero.space .eyebrow, .hero.survey .eyebrow {{ color: #9fb0e0; }}
.hero-title {{ display: flex; align-items: center; gap: .65rem; font-size: 2rem; font-weight: 700; letter-spacing: -.01em;
              line-height: 1.15; margin: .35rem 0 .45rem; color: var(--text); }}
.hero.space .hero-title, .hero.survey .hero-title {{ color: #f8fafc; }}
.hero-title svg {{ width: 34px; height: 34px; flex: 0 0 auto; }}
.hero-sub {{ font-size: .98rem; line-height: 1.55; color: var(--muted); max-width: 600px; }}
.hero.space .hero-sub, .hero.survey .hero-sub {{ color: #c3cce6; }}
.chips {{ display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1rem; }}
.chip {{ display: inline-flex; gap: .4rem; align-items: baseline; padding: .38rem .8rem; border-radius: 999px; font-size: .82rem;
        background: var(--chip); border: 1px solid var(--border); color: var(--muted); animation: pop .55s cubic-bezier(.2,.8,.2,1) both;
        animation-delay: calc(.25s + var(--i) * 90ms); }}
.chip b {{ font-family: {MONO}; font-weight: 600; color: var(--text); font-size: .9rem; }}
.model-chips {{ margin: 1.9rem 0 0; }}
.hero.space .chip, .hero.survey .chip {{ background: rgba(255,255,255,.08); border-color: rgba(255,255,255,.14); color: #c3cce6; }}
.hero.space .chip b, .hero.survey .chip b {{ color: #fff; }}

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

#table header -> metric key inside each saved model
CLS_COLUMNS = {"Accuracy": "Accuracy", "Weighted F1": "F1_Weighted", "ROC-AUC": "ROC_AUC_OvR"}
REG_COLUMNS = {"R2": "R2", "RMSE": "RMSE", "MAE": "MAE"}


@st.cache_resource(show_spinner=False)
def load_bundle(name, stamp):
    #stamp is the file's modification time, so a freshly pulled bundle replaces the cached one without a restart
    return joblib.load(MODELS / name)


def load_bundles():
    return tuple(load_bundle(name, (MODELS / name).stat().st_mtime)
                 for name in ("classification_models.pkl", "regression_models.pkl", "clustering_kmeans.pkl"))


def final_model_text(cls):
    #the notebook's final choice when the bundle carries it, the top of the held-out table otherwise
    final = cls.get("final_model")
    if final:
        return f'{final["name"]}, {final["why"]}'
    return f'{cls["best"]}, the highest weighted F1 on the held-out split.'


def results_table(bundle, columns):
    #one row per saved model in the bundle's ranked order
    return pd.DataFrame([{"Model": n, **{h: bundle["models"][n]["metrics"][k] for h, k in columns.items()}}
                         for n in bundle["model_order"]])


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


ICON_SURVEY = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round">'
               '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 10q9-3 18 0M3 15q9-3 18 0M9 4q-1.5 8 0 16M15 4q1.5 8 0 16" stroke-opacity=".55"/>'
               '<circle cx="8" cy="8.5" r="1.2" fill="currentColor" stroke="none"/><circle cx="14.5" cy="12.5" r="1.4" fill="currentColor" stroke="none"/>'
               '<circle cx="17" cy="8" r="1" fill="currentColor" stroke="none"/></svg>')


def survey_svg(seed=11):
    #a sky-chart banner for the TESS page: curved celestial grid, two coloured swarms of objects, a camera strip scanning past
    rng = random.Random(seed)
    meridians = "".join(f'<path d="M{x} 0 Q{x + 34} 150 {x} 300" fill="none" stroke="#22d3ee" stroke-opacity=".17" stroke-width="1"/>'
                        for x in range(90, 1200, 105))
    parallels = "".join(f'<path d="M0 {y} Q600 {y - 26} 1200 {y}" fill="none" stroke="#22d3ee" stroke-opacity=".17" stroke-width="1"/>'
                        for y in range(42, 300, 52))
    pts = []
    for cx, cy, sx, sy, col, n in [(470, 170, 150, 62, "#a78bfa", 120), (820, 125, 185, 72, "#f472b6", 160)]:
        for _ in range(n):
            x, y = rng.gauss(cx, sx), rng.gauss(cy, sy)
            if 0 < x < 1200 and 0 < y < 300:
                pts.append(f'<circle class="p" cx="{x:.0f}" cy="{y:.0f}" r="{rng.choice([1.2, 1.5, 1.9, 2.3])}" fill="{col}" '
                           f'opacity="{rng.uniform(.35, .9):.2f}" style="animation-delay:{rng.uniform(0, 2.2):.2f}s"/>')
    strip = ('<g class="strip"><g transform="rotate(-14 1010 150)">'
             + "".join(f'<rect x="968" y="{y}" width="84" height="84" fill="#22d3ee" fill-opacity=".05" stroke="#22d3ee" '
                       f'stroke-opacity=".75" stroke-width="1.4" stroke-dasharray="7 5"/>' for y in (-118, -26, 66, 158))
             + '</g></g>')
    style = ('<style>.p{animation:pin 1s ease both;transform-box:fill-box;transform-origin:center}'
             '@keyframes pin{from{opacity:0;transform:scale(.2)}}'
             '.strip{animation:scan 26s ease-in-out infinite alternate}'
             '@keyframes scan{from{transform:translateX(90px)}to{transform:translateX(-330px)}}</style>')
    return (f'<svg class="sky" viewBox="0 0 1200 300" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">{style}'
            f'<defs><linearGradient id="sshade" x1="0" x2="1"><stop offset="0" stop-color="#090d26" stop-opacity=".94"/>'
            f'<stop offset="1" stop-color="#090d26" stop-opacity="0"/></linearGradient></defs>'
            f'<rect width="1200" height="300" fill="#090d26"/>{meridians}{parallels}{"".join(pts)}{strip}'
            f'<rect width="720" height="300" fill="url(#sshade)"/></svg>')


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
    sky = {"space": starfield_svg, "finance": market_svg, "survey": survey_svg}.get(kind, lambda: "")()
    chip_html = "".join(f'<span class="chip" style="--i:{i}"><b>{v}</b>{k}</span>' for i, (v, k) in enumerate(chips))
    st.markdown(f'<div class="hero {kind}">{sky}<div class="hero-body"><div class="eyebrow">{eyebrow}</div>'
                f'<div class="hero-title">{icon}<span>{title}</span></div><div class="hero-sub">{subtitle}</div>'
                f'<div class="chips">{chip_html}</div></div></div>', unsafe_allow_html=True)


def text(kind, body):
    st.markdown(f'<div class="{kind}">{body}</div>', unsafe_allow_html=True)


def model_picker(bundle, key, chips, tag="best", why="The best one on the held-out split is selected by default."):
    #selectbox on the left, the chosen model's held-out scores on the right; the default is the bundle's reported model
    left, right = st.columns([5, 7], gap="large")
    with left:
        name = st.selectbox("Model", bundle["model_order"], key=key, index=bundle["model_order"].index(bundle["best"]),
                            format_func=lambda n: f"{n}  ({tag})" if n == bundle["best"] else n,
                            help=f"Every model from the notebooks is available. {why}")
    m = bundle["models"][name]
    with right:
        st.markdown('<div class="chips model-chips">' + "".join(
            f'<span class="chip" style="--i:{i}"><b>{v}</b>{k}</span>' for i, (v, k) in enumerate(chips(m)))
            + '</div>', unsafe_allow_html=True)
    return name, m


def reg_label(f):
    return f"Sector: {f[7:]}" if f.startswith("Sector_") else REG_LABELS.get(f, f)


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
def koi_frame(cls, values):
    #anything not supplied stays NaN and each pipeline's median imputer fills it
    row = {f: np.nan for f in cls["feature_columns"]}
    row.update(values)
    if values.get("koi_duration"):
        row["transit_depth_per_hour"] = values["koi_depth"] / values["koi_duration"]
    return pd.DataFrame([row])[cls["feature_columns"]]


def classify_with(pipe, frame):
    label = str(pipe.predict(frame)[0])
    probs = dict(zip([str(c) for c in pipe.classes_], pipe.predict_proba(frame)[0]))
    return label, probs


def classify_koi(cls, values, name):
    return classify_with(cls["models"][name]["pipeline"], koi_frame(cls, values))


def classify_all(cls, values):
    frame = koi_frame(cls, values)
    return {n: classify_with(cls["models"][n]["pipeline"], frame) for n in cls["model_order"]}


def build_frame(reg, sector, rows):
    #rows: list of {feature: value}; missing features fall back to training medians
    frame = pd.DataFrame([{**{f: reg["train_medians"][f] for f in reg["raw_features"]}, **r} for r in rows])
    for f, (lo, hi) in reg["clip_bounds"].items():
        frame[f] = frame[f].clip(lo, hi)
    for col in reg["sector_dummy_columns"]:
        frame[col] = 1 if col == f"Sector_{sector}" else 0
    frame["Growth Adjusted Margin"] = frame["Revenue Growth"] * frame["netProfitMargin"]
    return frame.reindex(columns=reg["model_columns"], fill_value=0)


def predict_rows(reg, sector, rows, name):
    return reg["models"][name]["model"].predict(build_frame(reg, sector, rows)).astype(float)


def predict_return(reg, sector, values, name):
    return float(predict_rows(reg, sector, [values], name)[0])


def predict_all(reg, sector, values):
    frame = build_frame(reg, sector, [values])
    return {n: float(reg["models"][n]["model"].predict(frame)[0]) for n in reg["model_order"]}


def sensitivity(reg, sector, values, name, n=7):
    #move one indicator at a time across its training percentiles, keep everything else as entered
    weights = reg["models"][name]["weights"] or reg["models"][reg["best"]]["weights"]
    feats = [f for f in weights if f in reg["raw_features"]][:n]
    grid = [10, 25, 50, 75, 90]
    rows = [{**values, f: reg["feature_quantiles"][f][p]} for f in feats for p in grid]
    preds = predict_rows(reg, sector, rows, name).reshape(len(feats), len(grid))
    out = []
    for f, p in zip(feats, preds):
        out.append(dict(feature=f, low=float(p[0]), high=float(p[-1]), lo=float(p.min()), hi=float(p.max()),
                        v_low=reg["feature_quantiles"][f][10], v_high=reg["feature_quantiles"][f][90]))
    return sorted(out, key=lambda d: d["hi"] - d["lo"], reverse=True)


#---------- animated result cards (iframes, so they can count up and sweep in) ----------
def result_card_html(body, extra_css=""):
    return (f"<style>{FONT_FACES}body{{margin:0;font-family:{SANS};color:{T['text']};background:transparent;-webkit-font-smoothing:antialiased}}"
            f".mono{{font-family:{MONO}}}{extra_css}</style>{body}")


def classifier_card(label, probs, per_class, model_name):
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    circ = 2 * np.pi * 46
    segs, start = "", 0.0
    for cls_name, p in ranked:
        segs += (f'<circle class="seg" cx="60" cy="60" r="46" stroke="{CLASS_COLORS[cls_name]}" data-off="{circ * (1 - p):.2f}" '
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
    <div class="note"><b>{model_name}</b> reads this object as <b>{label}</b>: {CLASS_BLURB[label]}. On held-out objects it gets this class right
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


def regression_card(pred, stats, sector, sector_median, r2, model_name):
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
    <div class="lbl">{model_name}: predicted price change over the next year</div>
    <div class="line"><span>Training median</span><b class="mono">{stats['median']:+.1f}%</b></div>
    <div class="line"><span>{sector} median</span><b class="mono">{sector_median:+.1f}%</b></div>
    <div class="line"><span>Middle half (blue arc)</span><b class="mono">{stats['q25']:+.1f}% to {stats['q75']:+.1f}%</b></div>
    <div class="small">That puts this company <b>{rel}</b> the typical one. {model_name} explains only about {r2 * 100:.0f}% of the
    variance in held-out next-year returns, so read this as a lean from the numbers, not a forecast.</div>
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


def confusion_heatmap(cls, m):
    classes, cm = cls["classes"], np.array(m["confusion"])
    share = cm / cm.sum(axis=1, keepdims=True)
    txt = [[f"{cm[i, j]:,}<br>{share[i, j]:.0%}" for j in range(3)] for i in range(3)]
    fig = go.Figure(go.Heatmap(
        z=share, x=classes, y=classes, text=txt, texttemplate="%{text}", textfont=dict(size=12),
        colorscale=[[0, T["surface2"]], [1, T["teal"]]], showscale=False, xgap=4, ygap=4,
        hovertemplate="actual %{y}<br>predicted %{x}<br>%{text}<extra></extra>"))
    fig.update_xaxes(title="Predicted", side="bottom")
    fig.update_yaxes(title="Actual", autorange="reversed")
    return themed(fig, 360, f"Held-out confusion matrix ({cm.sum():,} objects)", legend=False)


def per_class_bars(cls, m):
    classes = cls["classes"]
    fig = go.Figure()
    for metric, color in (("precision", T["blue"]), ("recall", T["violet"]), ("f1", T["teal"])):
        vals = [m["per_class"][c][metric] for c in classes]
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


def all_models_regression(reg, preds, selected):
    names = reg["model_order"][::-1]
    vals, median = [preds[n] for n in names], reg["target_stats"]["median"]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h", cliponaxis=False,
        marker=dict(color=[T["amber"] if n == selected else T["teal"] for n in names], opacity=.9),
        text=[f"{v:+.1f}%" for v in vals], textposition="outside", textfont=dict(size=11),
        hovertemplate="%{y}: %{x:+.1f}%<extra></extra>"))
    fig.add_vline(x=median, line_color=T["muted"], line_dash="dot")
    fig.add_annotation(x=median, y=1.03, yref="paper", yanchor="bottom", text=f"training median {median:+.1f}%",
                       showarrow=False, font=dict(color=T["muted"], size=11))
    lo, hi = min(vals + [median, 0]), max(vals + [median, 0])
    pad = max((hi - lo) * .25, 5)
    fig.update_xaxes(range=[lo - pad, hi + pad], ticksuffix="%", title="Predicted next-year change (%)")
    fig.update_yaxes(tickvals=names, ticktext=[f"<b>{n}</b>" if n == selected else n for n in names], automargin=True)
    return themed(fig, 30 * len(names) + 130, "What every model predicts for these inputs", legend=False)


def all_models_classification(cls, results, selected):
    names = cls["model_order"][::-1]
    fig = go.Figure()
    for c, color in CLASS_COLORS.items():
        p = [results[n][1].get(c, 0) for n in names]
        fig.add_trace(go.Bar(y=names, x=p, orientation="h", name=c, marker_color=color, opacity=.9,
                             text=[f"{v:.0%}" if v >= .12 else "" for v in p], textposition="inside", insidetextanchor="middle",
                             textfont=dict(size=11, color="#0b1020" if MODE == "dark" else "#fff"),
                             hovertemplate="%{y}<br>" + c + " %{x:.1%}<extra></extra>"))
    for n in names:
        label = results[n][0]
        fig.add_annotation(x=1.02, y=n, xanchor="left", text=label, showarrow=False,
                           font=dict(size=11, color=CLASS_COLORS.get(label, T["text"])))
    fig.update_layout(barmode="stack", bargap=.3)
    fig.update_xaxes(range=[0, 1.34], tickvals=[0, .25, .5, .75, 1], tickformat=".0%", title="Class probability")
    fig.update_yaxes(tickvals=names, ticktext=[f"<b>{n}</b>" if n == selected else n for n in names], automargin=True)
    return themed(fig, 34 * len(names) + 140, "How every model reads this object (label on the right is its call)")


def actual_vs_predicted(reg, m, name):
    y, p, m = np.asarray(reg["y_test"]), np.asarray(m["y_pred_test"]), m["metrics"]
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
    return themed(fig, 380, f"{name}: actual vs predicted on {len(y):,} held-out companies")


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


#---------- clustering helpers ----------
def cluster_palette(k):
    base = [T["violet"], T["pink"], T["cyan"], T["amber"], T["green"], T["blue"], T["red"], T["teal"], "#fb923c", "#a3e635"]
    return base[:k]


def clu_vector(clu, values):
    #same steps as the notebook's section b: log10 the heavy-tailed features, clip, fill with the median, standardise
    row = {}
    for f in clu["features"]:
        v = float(values[f])
        if f in clu["log_features"]:
            v = np.log10(v) if v > 0 else np.nan
        lo, hi = clu["clip_bounds"][f]
        row[f] = float(np.clip(v, lo, hi)) if not np.isnan(v) else np.nan
    frame = pd.DataFrame([row], columns=clu["features"])
    frame = pd.DataFrame(clu["imputer"].transform(frame), columns=clu["features"])
    return clu["scaler"].transform(frame)[0]


def clu_model(clu, algo, k):
    #labels and centres of the chosen model: k-means for any k in the sweep, the notebook's complete-linkage cut for agglomerative
    if algo == "Agglomerative":
        ag = clu["agglomerative"]
        return {"algo": algo, "k": ag["k"], "labels": np.asarray(ag["labels"]), "centers": np.asarray(ag["means"], dtype=float),
                "names": ag["cluster_names"], "title": f"Agglomerative ({ag['linkage']} linkage, k = {ag['k']})", "centre": "cluster mean"}
    km = clu["kmeans"][k]
    return {"algo": algo, "k": k, "labels": km.labels_, "centers": km.cluster_centers_, "names": clu["cluster_names"].get(k, {}),
            "title": f"K-Means (k = {k})", "centre": "centroid"}


def clu_assign(clu, m, vec):
    #nearest centre, inverse-square-distance weights for the other centres, and how central the object is inside its cluster
    d = np.linalg.norm(m["centers"] - vec, axis=1)
    c = int(d.argmin())
    w = 1 / np.maximum(d, 1e-9) ** 2
    w = w / w.sum()
    members = clu["X_scaled"][m["labels"] == c]
    member_d = np.linalg.norm(members - m["centers"][c], axis=1)
    closer_than = float((member_d > d[c]).mean() * 100)
    return c, d, w, closer_than


def clu_profile(clu, m):
    labels = m["labels"]
    z = pd.DataFrame(clu["X_scaled"], columns=clu["features"]).groupby(labels).mean().T
    med = pd.DataFrame(clu["X_raw"], columns=clu["features"]).groupby(labels).median().T
    return z, med


def clu_describe(clu, m, c, z):
    #hand-written names for the reported models, the three biggest deviations otherwise
    names = m["names"]
    if c in names:
        return names[c]
    top = z[c].sort_values(key=np.abs, ascending=False).head(3)
    def short(f):
        name = clu["feature_labels"][f].split(" (")[0]
        return name if name.startswith("TESS") else name.lower()
    return ", ".join(f"{'high' if v > 0 else 'low'} {short(f)}" for f, v in top.items())


def clu_mix(clu, m):
    labels = pd.Series(m["labels"], name="cluster")
    mix = pd.crosstab(labels, pd.Series(clu["disp_group"], name="group"), normalize="index") * 100
    return mix.reindex(columns=[g for g in ["planet", "candidate", "not a planet", "unknown"] if g in mix.columns], fill_value=0)


def cluster_card(clu, m, c, w, closer_than, desc, mix_row):
    k = m["k"]
    pal = cluster_palette(k)
    color = pal[c]
    circ = 2 * np.pi * 46
    rows = "".join(
        f'<div class="row"><span class="nm">cluster {i}</span><div class="track"><div class="fill" data-w="{w[i] * 100:.1f}" style="background:{pal[i]}"></div></div>'
        f'<span class="val mono" data-count="{w[i] * 100:.1f}" data-suffix="%">0.0%</span></div>' for i in np.argsort(-w)[:5])
    chips = "".join(f'<span class="mix"><b class="mono" style="color:{col}">{mix_row.get(g, 0):.0f}%</b> {g}</span>'
                    for g, col in [("planet", T["teal"]), ("candidate", T["amber"]), ("not a planet", T["red"])])
    css = f"""
.wrap{{display:flex;gap:clamp(14px,4vw,28px);align-items:center;padding:6px 2px}}
.ring{{position:relative;width:clamp(124px,30vw,160px);height:clamp(124px,30vw,160px);flex:0 0 auto}}
.ring svg{{width:100%;height:100%}}
.seg{{fill:none;stroke-width:13;transform:rotate(-90deg);transform-origin:60px 60px;transition:stroke-dashoffset 1.3s cubic-bezier(.2,.8,.2,1)}}
.base{{fill:none;stroke:{T['chip']};stroke-width:13}}
.center{{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center}}
.pct{{font-size:clamp(22px,6vw,28px);font-weight:600;line-height:1}}
.lbl{{font-size:9.5px;letter-spacing:.08em;text-transform:uppercase;color:{T['muted']};margin-top:5px;max-width:110px;line-height:1.3}}
.info{{flex:1 1 auto;min-width:0}}
.badge{{display:inline-block;padding:7px 15px;border-radius:999px;font-weight:600;font-size:15px;letter-spacing:.02em;
  background:{color}22;color:{color};border:1px solid {color}66;animation:pop .7s cubic-bezier(.2,.8,.2,1) both}}
@keyframes pop{{from{{opacity:0;transform:scale(.9)}}to{{opacity:1;transform:none}}}}
.desc{{color:{T['text']};font-size:13.5px;line-height:1.5;margin:10px 0 8px}}
.row{{display:grid;grid-template-columns:minmax(70px,90px) 1fr 52px;align-items:center;gap:10px;margin:6px 0;font-size:12.5px}}
.nm{{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:{T['muted']}}}
.track{{height:8px;background:{T['chip']};border:1px solid {T['border']};border-radius:99px;overflow:hidden}}
.fill{{height:100%;width:0;border-radius:99px;transition:width 1.2s cubic-bezier(.2,.8,.2,1)}}
.val{{text-align:right;color:{T['muted']};font-size:12px}}
.mixrow{{display:flex;flex-wrap:wrap;gap:6px 16px;margin-top:10px;font-size:12.5px;color:{T['muted']}}}
.mix b{{font-weight:600}}
.small{{font-size:11.5px;color:{T['muted']};margin-top:4px}}
"""
    body = f"""
<div class="wrap">
  <div class="ring"><svg viewBox="0 0 120 120"><circle class="base" cx="60" cy="60" r="46"/>
    <circle class="seg" cx="60" cy="60" r="46" stroke="{color}" data-off="{circ * (1 - closer_than / 100):.2f}" style="stroke-dasharray:{circ:.2f};stroke-dashoffset:{circ:.2f}"/></svg>
    <div class="center"><div class="pct mono" data-count="{closer_than:.0f}" data-suffix="%" data-dec="0">0%</div><div class="lbl">of the cluster sits farther from the centre</div></div></div>
  <div class="info"><span class="badge">Cluster {c} of {k}</span>
    <div class="desc">{desc[0].upper() + desc[1:]}.</div>
    <div class="small">Nearest {m['centre']} by distance in standardised feature space; the bars share out the closeness to every centre.</div>{rows}
    <div class="mixrow">{chips}</div><div class="small">What this cluster contains, read from the dispositions after fitting.</div></div>
</div>
<script>
requestAnimationFrame(()=>setTimeout(()=>{{
  document.querySelectorAll('.seg').forEach(e=>e.style.strokeDashoffset=e.dataset.off);
  document.querySelectorAll('.fill').forEach(e=>e.style.width=e.dataset.w+'%');
  document.querySelectorAll('[data-count]').forEach(e=>{{const t=parseFloat(e.dataset.count),d=e.dataset.dec!==undefined?+e.dataset.dec:1,s=e.dataset.suffix||'',t0=performance.now();
    const step=n=>{{const k=Math.min((n-t0)/1200,1),v=t*(1-Math.pow(1-k,3));e.textContent=v.toFixed(d)+s;if(k<1)requestAnimationFrame(step)}};requestAnimationFrame(step)}});
}},60));
</script>"""
    st.iframe(result_card_html(body, css), height=250)


def clu_pca_scatter(clu, m, user=None):
    k, pal, P, ev = m["k"], cluster_palette(m["k"]), clu["X_pca"], clu["explained_variance"]
    toi, disp = np.array(clu["toi"]), np.array(clu["disposition"])
    fig = go.Figure()
    for c in range(k):
        sel = m["labels"] == c
        fig.add_trace(go.Scattergl(x=P[sel, 0], y=P[sel, 1], mode="markers", name=f"cluster {c} (n={int(sel.sum()):,})",
                                   marker=dict(color=pal[c], size=4, opacity=.45),
                                   text=[f"TOI {t}<br>{d}" for t, d in zip(toi[sel], disp[sel])],
                                   hovertemplate="%{text}<extra>cluster " + str(c) + "</extra>"))
    cen = clu["pca"].transform(pd.DataFrame(m["centers"], columns=clu["features"]))
    fig.add_trace(go.Scatter(x=cen[:, 0], y=cen[:, 1], mode="markers", name=m["centre"] + "s",
                             marker=dict(symbol="x", size=13, color=T["text"], line=dict(color=T["bg"], width=1.5)),
                             hovertemplate=m["centre"] + " of cluster %{pointNumber}<extra></extra>"))
    if user is not None:
        u = clu["pca"].transform(pd.DataFrame([user], columns=clu["features"]))[0]
        fig.add_trace(go.Scatter(x=[u[0]], y=[u[1]], mode="markers", name="your object",
                                 marker=dict(symbol="star", size=22, color="#ffffff" if MODE == "dark" else "#1b2233",
                                             line=dict(color=T["amber"], width=2.5)),
                                 hovertemplate="your object<extra></extra>"))
    fig.update_xaxes(title=f"PC1 ({ev[0] * 100:.1f}% of variance)")
    fig.update_yaxes(title=f"PC2 ({ev[1] * 100:.1f}% of variance)")
    return themed(fig, 440, f"{clu['n_rows']:,} TESS objects in PCA space, coloured by {m['title']} cluster")


def clu_k_chart(clu, k):
    from plotly.subplots import make_subplots
    sw = clu["sweep"]
    panels = [("inertia", "Inertia (elbow curve)", T["violet"]), ("silhouette", "Silhouette, higher is better", T["pink"]),
              ("davies_bouldin", "Davies-Bouldin, lower is better", T["cyan"]), ("calinski_harabasz", "Calinski-Harabasz, higher is better", T["amber"])]
    fig = make_subplots(rows=1, cols=4, subplot_titles=[t for _, t, _ in panels], horizontal_spacing=.06)
    for i, (col, _, color) in enumerate(panels, start=1):
        fig.add_trace(go.Scatter(x=sw["k"], y=sw[col], mode="lines+markers", line=dict(color=color, width=2.5), marker=dict(size=7),
                                 showlegend=False, hovertemplate="k = %{x}<br>%{y:.4g}<extra></extra>"), row=1, col=i)
        sel = float(sw.loc[sw["k"] == k, col].iloc[0])
        fig.add_trace(go.Scatter(x=[k], y=[sel], mode="markers", marker=dict(size=15, color=color, line=dict(color=T["text"], width=2)),
                                 showlegend=False, hoverinfo="skip"), row=1, col=i)
        fig.update_xaxes(title_text="k", tickvals=list(range(2, 11)), row=1, col=i)
    themed(fig, 330, f"Choosing k: every number of clusters from 2 to 10, with k = {k} highlighted", legend=False)
    fig.update_xaxes(gridcolor=T["grid"], zerolinecolor=T["grid"], linecolor=T["grid"], title_font=dict(size=11, color=T["muted"]))
    fig.update_yaxes(gridcolor=T["grid"], zerolinecolor=T["grid"], linecolor=T["grid"])
    fig.update_annotations(font=dict(size=11, color=T["text"]))
    return fig


def clu_profile_heatmap(clu, k, z):
    labels = [clu["feature_labels"][f] for f in z.index]
    lim = float(np.abs(z.values).max())
    vals = z.values.astype(float)
    fig = go.Figure(go.Heatmap(z=vals, x=[f"cluster {c}" for c in z.columns], y=labels, zmin=-lim, zmax=lim,
                               text=[[f"{v:+.2f}" for v in row] for row in vals], texttemplate="%{text}", textfont=dict(size=11),
                               colorscale=[[0, T["blue"]], [0.5, T["surface2"]], [1, T["red"]]], xgap=3, ygap=3,
                               colorbar=dict(title="std units", thickness=12),
                               hovertemplate="%{y}<br>%{x}: %{z:.2f} standard units from the average<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return themed(fig, 32 * len(labels) + 110, "How each cluster differs from the average TESS object (standard units)", legend=False)


def clu_mix_chart(clu, m):
    mix, k = clu_mix(clu, m), m["k"]
    colors = {"planet": T["teal"], "candidate": T["amber"], "not a planet": T["red"], "unknown": T["faint"]}
    fig = go.Figure()
    for g in mix.columns:
        fig.add_trace(go.Bar(y=[f"cluster {c}" for c in mix.index], x=mix[g], orientation="h", name=g, marker_color=colors[g], opacity=.9,
                             text=[f"{v:.0f}%" if v >= 8 else "" for v in mix[g]], textposition="inside", insidetextanchor="middle",
                             textfont=dict(size=11, color="#0b1020" if MODE == "dark" else "#fff"),
                             hovertemplate="%{y}<br>" + g + " %{x:.1f}%<extra></extra>"))
    fig.update_layout(barmode="stack", bargap=.3)
    fig.update_xaxes(range=[0, 100], ticksuffix="%", title="Share of the cluster")
    fig.update_yaxes(autorange="reversed")
    fig = themed(fig, max(34 * k + 210, 290), "What each cluster contains")
    fig.update_layout(legend=dict(orientation="h", y=-0.42, yanchor="top", x=0), margin=dict(t=46, b=90))
    return fig


def clu_linkage_chart(clu):
    #the agglomerative sweep: three linkages, k from 2 to 6, the reported cut highlighted
    from plotly.subplots import make_subplots
    ag, sw = clu["agglomerative"], clu["agglomerative"]["sweep"]
    panels = [("silhouette", "Silhouette, higher is better"), ("davies_bouldin", "Davies-Bouldin, lower is better"),
              ("calinski_harabasz", "Calinski-Harabasz, higher is better")]
    colors = {"ward": T["violet"], "complete": T["pink"], "average": T["cyan"]}
    fig = make_subplots(rows=1, cols=3, subplot_titles=[t for _, t in panels], horizontal_spacing=.08)
    for i, (col, _) in enumerate(panels, start=1):
        for method, part in sw.groupby("linkage", sort=False):
            fig.add_trace(go.Scatter(x=part["k"], y=part[col], mode="lines+markers", name=f"{method} linkage", legendgroup=method,
                                     showlegend=(i == 1), line=dict(color=colors[method], width=2.5), marker=dict(size=7),
                                     hovertemplate=method + ", k = %{x}<br>%{y:.4g}<extra></extra>"), row=1, col=i)
        sel = sw[(sw["linkage"] == ag["linkage"]) & (sw["k"] == ag["k"])].iloc[0]
        fig.add_trace(go.Scatter(x=[sel["k"]], y=[sel[col]], mode="markers", showlegend=False, hoverinfo="skip",
                                 marker=dict(size=15, color=colors[ag["linkage"]], line=dict(color=T["text"], width=2))), row=1, col=i)
        fig.update_xaxes(title_text="k", tickvals=list(range(2, 7)), row=1, col=i)
    themed(fig, 340, "Cutting the hierarchy: three linkages and every k from 2 to 6, the reported cut highlighted")
    fig.update_layout(legend=dict(orientation="h", y=-0.28, yanchor="top", x=0), margin=dict(t=56, b=70))
    fig.update_xaxes(gridcolor=T["grid"], zerolinecolor=T["grid"], linecolor=T["grid"], title_font=dict(size=11, color=T["muted"]))
    fig.update_yaxes(gridcolor=T["grid"], zerolinecolor=T["grid"], linecolor=T["grid"])
    fig.update_annotations(font=dict(size=11, color=T["text"]))
    return fig


def clu_metrics_compare(clu):
    #the two reported models side by side on the three internal metrics
    from plotly.subplots import make_subplots
    best = clu["sweep"].loc[clu["sweep"]["k"] == clu["best_k"]].iloc[0]
    ag = clu["agglomerative"]["metrics"]
    rows = [("Silhouette, higher is better", best["silhouette"], ag["Silhouette"], ".3f"),
            ("Davies-Bouldin, lower is better", best["davies_bouldin"], ag["Davies-Bouldin"], ".2f"),
            ("Calinski-Harabasz, higher is better", best["calinski_harabasz"], ag["Calinski-Harabasz"], ",.0f")]
    names = [f"K-Means (k = {clu['best_k']})", f"Agglomerative (k = {clu['agglomerative']['k']})"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[r[0] for r in rows], horizontal_spacing=.1)
    for i, (_, km_v, ag_v, fmt) in enumerate(rows, start=1):
        fig.add_trace(go.Bar(x=names, y=[km_v, ag_v], marker_color=[T["violet"], T["pink"]], showlegend=False,
                             text=[format(km_v, fmt), format(ag_v, fmt)], textposition="outside",
                             hovertemplate="%{x}<br>%{y:.4g}<extra></extra>"), row=1, col=i)
    themed(fig, 330, "The two reported models on the same scaled features", legend=False)
    fig.update_xaxes(gridcolor=T["grid"], zerolinecolor=T["grid"], linecolor=T["grid"], tickfont=dict(size=10))
    fig.update_yaxes(gridcolor=T["grid"], zerolinecolor=T["grid"], linecolor=T["grid"], rangemode="tozero")
    fig.update_annotations(font=dict(size=11, color=T["text"]))
    return fig


#---------- pages ----------
def page_classifier():
    cls, reg, clu = load_bundles()
    best, total = cls["models"][cls["best"]]["metrics"], cls["class_counts_all"]
    hero("space", "Classification · NASA Kepler", "Exoplanet classifier", ICON_PLANET,
         f"Pick any of the ten classifiers (the notebook's final model, {cls['best']}, is selected by "
         "default), enter what Kepler measured for a transit signal and its host star, and it decides whether the "
         "object looks like a confirmed planet, a candidate or a false positive, shows how sure it is, and places "
         "it among the objects it learned from.",
         [(f"{sum(total.values()):,}", "objects"), (f"{total.get('CONFIRMED', 0):,}", "confirmed planets"),
          (f"{len(cls['models'])}", "model runs"), (f"{best['F1_Weighted']:.3f}", "final model weighted F1")])

    name, m = model_picker(cls, "cls_model", tag="final model", why="The notebook's final model, chosen by cross-validation, is selected by default.", chips=lambda m: [
        (f"{m['metrics']['F1_Weighted']:.3f}", "weighted F1"), (f"{m['metrics']['Accuracy']:.1%}", "accuracy"),
        (f"{m['metrics']['ROC_AUC_OvR']:.3f}", "ROC-AUC"), (m["params"], "settings")])

    text("section", "Measurements")
    text("hint", "Pre-filled with typical (median) values. Change the ones you know, then classify. "
                 "Switching the model above re-scores the same measurements.")
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
            st.session_state["cls_inputs"] = values

    left, right = st.columns([7, 5], gap="large")
    if "cls_inputs" not in st.session_state:
        with left:
            text("empty", "Press <b>Classify object</b> to get the predicted class, the probability of each class, "
                          "a comparison with typical objects of each kind, what every model says, and a report card "
                          "for the selected model.")
            show(koi_scatter(cls))
            text("caption", "Confirmed planets and candidates sit in a band of small, shallow transits. False positives "
                            "spread much wider, which is what the classifiers pick up on.")
        with right:
            show(class_donut(total, "What the models learned from"))
            text("caption", "All Kepler objects of interest by disposition. Half are false positives, so a classifier "
                            "has to earn its confirmed calls.")
        return

    values = st.session_state["cls_inputs"]
    label, probs = classify_koi(cls, values, name)
    with left:
        classifier_card(label, probs, m["per_class"], name)
    with right:
        show(class_donut(total, "What the models learned from"))
    tab1, tab2, tab3, tab4 = st.tabs(["Where it sits", "Versus typical objects", "All models", "Model report card"])
    with tab1:
        show(koi_scatter(cls, values["koi_period"], values["koi_depth"]))
        text("caption", "The star is your object. Both axes are log-scaled because period and depth span several "
                        "orders of magnitude.")
    with tab2:
        show(koi_radar(cls, values))
        text("caption", "Each axis is the percentile of that measurement among all Kepler objects, so the shapes are "
                        "comparable. Your object's shape follows the class it resembles most.")
    with tab3:
        show(all_models_classification(cls, classify_all(cls, values), name))
        text("caption", "The same measurements scored by every Review 1 classifier, ranked by held-out weighted F1 "
                        "from the top. Gaussian Naive Bayes is usually all-or-nothing because it treats every "
                        "measurement as independent.")
    with tab4:
        r1, r2 = st.columns(2, gap="large")
        with r1:
            show(confusion_heatmap(cls, m))
        with r2:
            show(per_class_bars(cls, m))
        text("caption", f"{name} on the held-out objects. Rows are the true class, columns the prediction. Candidates "
                        "are the hardest call for every model: some are read as confirmed planets and some as false "
                        "positives, which is why their F1 is the lowest.")


def weights_panel(name, m):
    if not m["weights"]:
        return ('<div class="panel"><div class="section" style="margin-top:0">What the model weighs most</div>'
                f'<div class="caption" style="margin:0">{name} has no built-in feature importance: it works from '
                'distances between companies or from polynomial terms rather than per-feature weights. The '
                '<b>What moves it</b> tab shows how it responds to each input instead.</div></div>')
    imp = list(m["weights"].items())[:8]
    top = imp[0][1]
    return ('<div class="panel"><div class="section" style="margin-top:0">What the model weighs most</div>'
            + bars([(reg_label(f), v / top, f"{v * 100:.1f}%", T["amber"]) for f, v in imp])
            + f'<div class="caption" style="margin:.6rem 0 0">{m["weight_label"]}, for {name}.</div></div>')


def page_regression():
    cls, reg, clu = load_bundles()
    best, s = reg["models"][reg["best"]]["metrics"], reg["target_stats"]
    hero("finance", "Regression · US stock fundamentals", "Stock return predictor", ICON_CHART,
         "Pick any of the ten Review 1 regressors (the tuned Random Forest, the best, is selected by default), give "
         "it a company's 2018 annual fundamentals and it estimates the share price change over the following year, "
         "then shows where that sits among the companies it was trained on and which inputs move it most.",
         [(f"{reg['n_train'] + reg['n_test']:,}", "companies"), (f"{len(reg['raw_features'])}", "indicators"),
          (f"{len(reg['models'])}", "models"), (f"{best['R2']:.3f}", "best test R2")])

    name, m = model_picker(reg, "reg_model", lambda m: [
        (f"{m['metrics']['R2']:.3f}", "test R2"), (f"{m['metrics']['RMSE']:.1f} pts", "RMSE"),
        (f"{m['metrics']['MAE']:.1f} pts", "MAE"), (m["params"], "settings")])

    text("section", "Company fundamentals")
    text("hint", "Pre-filled with training medians. Dollar figures are in USD, ratios as decimals (0.15 = 15%). "
                 "Switching the model above re-scores the same fundamentals.")
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
            st.session_state["reg_inputs"] = (sector, values)

    left, right = st.columns([7, 5], gap="large")
    if "reg_inputs" not in st.session_state:
        with left:
            text("empty", "Press <b>Predict next-year return</b> to get the predicted change, where it lands among the "
                          "training companies, how each input moves it, what every model says, and how the selected "
                          "model did on held-out companies.")
            show(return_histogram(reg))
            text("caption", "The 2019 price change is centred near zero with a long right tail, so a typical company moved "
                            "only a little and a few moved a lot.")
        with right:
            st.markdown(weights_panel(name, m), unsafe_allow_html=True)
        return

    sector, values = st.session_state["reg_inputs"]
    pred = predict_return(reg, sector, values, name)
    with left:
        regression_card(pred, s, sector, reg["sector_stats"][sector]["median"], m["metrics"]["R2"], name)
    with right:
        st.markdown(weights_panel(name, m), unsafe_allow_html=True)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Among all companies", "What moves it", "All models", "By sector", "Model report card"])
    with tab1:
        show(return_histogram(reg, pred))
        text("caption", "The amber line is this company. The dotted line is the median training company.")
    with tab2:
        show(sensitivity_chart(sensitivity(reg, sector, values, name), pred))
        text("caption", f"For each input, {name} was re-run with that one input at a low (10th percentile) and a "
                        "high (90th percentile) value while everything else stayed as you entered it. Longer bars "
                        "mean the prediction is more sensitive to that input.")
    with tab3:
        show(all_models_regression(reg, predict_all(reg, sector, values), name))
        text("caption", "The same fundamentals scored by every Review 1 regressor, ranked by held-out R2 from the top. "
                        "The linear models mostly agree with each other; the tree ensembles react more to the inputs.")
    with tab4:
        show(sector_chart(reg, sector))
        text("caption", f"{sector} is highlighted. Sector is one of the model's inputs, so the same fundamentals "
                        "can lean differently in different sectors.")
    with tab5:
        show(actual_vs_predicted(reg, m, name))
        text("caption", "Every model in the capstone lands at a low R2 on this problem: annual fundamentals explain "
                        "only a small share of next-year price moves. The tree ensembles rank first, and their "
                        "predictions lean the right way more often than not.")


def page_clustering():
    cls, reg, clu = load_bundles()
    sw, ag = clu["sweep"], clu["agglomerative"]
    best = sw.loc[sw["k"] == clu["best_k"]].iloc[0]
    hero("survey", "Clustering · NASA TESS objects of interest", "TESS cluster explorer", ICON_SURVEY,
         "Two algorithms group the TESS Objects of Interest by their physical measurements alone, with no labels during "
         "fitting: K-Means, and Agglomerative clustering that merges objects into a hierarchy. Pick an algorithm, enter a "
         "signal's measurements, and see which group it joins, how close to the centre it sits, and what that group "
         "contains once the NASA dispositions are read back in afterwards.",
         [(f"{clu['n_rows']:,}", "TESS objects"), (f"{len(clu['features'])}", "measurements"), ("2", "algorithms"),
          (f"{best['silhouette']:.3f} · {ag['metrics']['Silhouette']:.3f}", "silhouette, K-Means · Agglomerative")])

    left, right = st.columns([5, 7], gap="large")
    with left:
        algo = st.radio("Algorithm", ["K-Means", "Agglomerative"], key="clu_algo", horizontal=True,
                        help="K-Means places k centroids and moves them until every object sits with its nearest one. Agglomerative "
                             "clustering starts with every object alone and merges the closest groups until k remain; with complete "
                             "linkage the distance between two groups is their farthest pair.")
        if algo == "K-Means":
            k = st.selectbox("Number of clusters", list(range(2, 11)), key="clu_k",
                             format_func=lambda v: f"k = {v}  (reported model)" if v == clu["best_k"] else f"k = {v}",
                             help="The notebook swept every k from 2 to 10. Silhouette and Calinski-Harabasz both peak at k = 2.")
        else:
            k = ag["k"]
            st.selectbox("Number of clusters", [k], key="clu_k_agg", disabled=True,
                         format_func=lambda v: f"k = {v}  (reported cut, {ag['linkage']} linkage)",
                         help="The notebook compared Ward, complete and average linkage for k from 2 to 6 and kept the "
                              "complete-linkage cut at k = 2, the best silhouette among cuts with no tiny cluster.")
    m = clu_model(clu, algo, k)
    if algo == "K-Means":
        row = sw.loc[sw["k"] == k].iloc[0]
        chips = [(f"{row['silhouette']:.3f}", "silhouette"), (f"{row['davies_bouldin']:.2f}", "Davies-Bouldin"),
                 (f"{row['calinski_harabasz']:,.0f}", "Calinski-Harabasz"), (f"{row['inertia']:,.0f}", "inertia")]
    else:
        chips = [(f"{ag['metrics']['Silhouette']:.3f}", "silhouette"), (f"{ag['metrics']['Davies-Bouldin']:.2f}", "Davies-Bouldin"),
                 (f"{ag['metrics']['Calinski-Harabasz']:,.0f}", "Calinski-Harabasz"), (" / ".join(f"{n:,}" for n in ag["sizes"]), "cluster sizes")]
    with right:
        st.markdown('<div class="chips model-chips">' + "".join(
            f'<span class="chip" style="--i:{i}"><b>{v}</b>{n}</span>' for i, (v, n) in enumerate(chips)) + '</div>', unsafe_allow_html=True)

    text("section", "Measurements of a TESS object")
    text("hint", "Pre-filled with the median object. Change what you know, then place it. Switching the algorithm or k above re-places the same object.")
    with st.form("clu_form"):
        values = number_inputs(clu["features"], clu["feature_labels"], clu["feature_medians"], "u", ncols=4)
        a, b = st.columns([2, 1], gap="medium")
        with a:
            text("caption", "All eleven measurements go through the notebook's preprocessing: log scale for the heavy-tailed ones, "
                            "clipping, median fill and standardisation.")
        with b:
            submitted = st.form_submit_button("Place this object", type="primary", width="stretch")
        if submitted:
            st.session_state["clu_inputs"] = values

    z, med = clu_profile(clu, m)
    left, right = st.columns([7, 5], gap="large")
    if "clu_inputs" not in st.session_state:
        with left:
            text("empty", "Press <b>Place this object</b> to see which cluster it joins, how central it is, and where it lands "
                          "among all TESS objects. The map below already shows the clusters of the chosen model.")
            show(clu_pca_scatter(clu, m))
        with right:
            show(clu_mix_chart(clu, m))
            text("caption", "Clusters are not dispositions: candidates dominate every cluster. What changes between clusters is "
                            "the share of confirmed planets and of false positives.")
        return

    values = st.session_state["clu_inputs"]
    vec = clu_vector(clu, values)
    c, d, w, closer = clu_assign(clu, m, vec)
    mix = clu_mix(clu, m)
    with left:
        cluster_card(clu, m, c, w, closer, clu_describe(clu, m, c, z), mix.loc[c])
        if algo == "Agglomerative":
            text("caption", "Agglomerative clustering has no predict step: the hierarchy was built once over all objects, so a new "
                            "object is placed with the cluster whose mean is nearest in standardised feature space.")
    with right:
        show(clu_mix_chart(clu, m))
    tab1, tab2, tab3, tab4 = st.tabs(["Where it sits", "Choosing k" if algo == "K-Means" else "Cutting the tree", "Cluster profiles", "Report card"])
    with tab1:
        show(clu_pca_scatter(clu, m, vec))
        text("caption", f"The star is your object, the crosses are the {m['centre']}s. The two principal components keep "
                        f"{sum(clu['explained_variance']) * 100:.0f}% of the variance of the eleven measurements.")
    with tab2:
        if algo == "K-Means":
            show(clu_k_chart(clu, k))
            text("caption", "Inertia always falls as k grows, so the elbow is read from where the drops get small. Silhouette and "
                            "Calinski-Harabasz both peak at k = 2; Davies-Bouldin barely moves across k, so it does not decide.")
        else:
            dendrogram = ROOT / "notebooks" / "figures" / "Agglomerative" / "dendrogram.png"
            if dendrogram.exists():
                st.image(str(dendrogram), width="stretch",
                         caption="Ward and complete-linkage dendrograms from the notebook, truncated to the last 30 merges. "
                                 "The height of a join is the distance at which two groups merged.")
            show(clu_linkage_chart(clu))
            text("caption", "Every cut of the three hierarchies, scored on the same metrics as K-Means. Ward makes the most even "
                            "halves but scores lower; complete linkage at k = 2 is the best silhouette among cuts where no cluster "
                            "is tiny. Average linkage leaves a handful of outliers alone, which inflates its scores without being useful.")
    with tab3:
        show(clu_profile_heatmap(clu, k, z))
        med_table = med.copy()
        med_table.index = [clu["feature_labels"][f] for f in med_table.index]
        med_table.columns = [f"cluster {c}" for c in med_table.columns]
        text("section", "Median measurements per cluster, original units")
        st.dataframe(med_table.round(2), width="stretch")
    with tab4:
        show(clu_metrics_compare(clu))
        text("caption", f"K-Means at k = {clu['best_k']} scores better on all three metrics than the complete-linkage cut at k = 2 "
                        "on the same standardised features. Both split the sky survey the same way at heart: smaller, cooler, "
                        "nearer systems against hot, large-planet signals around distant stars, so the two methods agree on the "
                        "structure even though K-Means draws the tighter boundary.")
        t1, t2 = st.columns(2, gap="large")
        with t1:
            text("section", "K-Means, every k from the notebook's sweep")
            st.dataframe(sw.rename(columns={"k": "k", "inertia": "Inertia", "silhouette": "Silhouette", "davies_bouldin": "Davies-Bouldin",
                                            "calinski_harabasz": "Calinski-Harabasz"}).round(4), hide_index=True, width="stretch")
        with t2:
            text("section", "Agglomerative, every linkage and cut compared")
            st.dataframe(ag["sweep"].rename(columns={"linkage": "Linkage", "k": "k", "smallest_share": "Smallest cluster (%)",
                                                    "silhouette": "Silhouette", "davies_bouldin": "Davies-Bouldin",
                                                    "calinski_harabasz": "Calinski-Harabasz"}).round(4), hide_index=True, width="stretch")
        text("caption", "The NASA dispositions were never used for fitting; they only appear in the composition chart and the mix on the result card.")


def page_about():
    cls, reg, clu = load_bundles()
    hero("plain", "23CSE301 · ML capstone", "About the models", ICON_ORBIT,
         "What the models were trained on, how all of them compare on the same held-out splits, and how to run "
         "everything yourself. Every model listed here can be selected on its page.",
         [(f"{len(reg['models'])}", "regression models"), (f"{len(cls['models'])}", "classification runs"),
          ("2", "clustering models"), ("3", "datasets"), ("42", "random state")])

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(f'<div class="spec-card"><div class="spec-head">{ICON_PLANET}Classification, Kepler objects of interest</div>'
                    f'<div class="spec"><div class="k">Data</div><div class="v">NASA Kepler Objects of Interest cumulative table, '
                    f'{sum(cls["class_counts_all"].values()):,} objects, three classes.</div>'
                    f'<div class="k">Features</div><div class="v">{len(cls["feature_columns"])} numeric transit and stellar measurements '
                    f'plus an engineered depth-per-hour. The disposition score and false-positive flags are excluded to avoid leakage.</div>'
                    f'<div class="k">Preprocessing</div><div class="v">Median imputation and standard scaling, fitted on training folds only.</div>'
                    f'<div class="k">Split</div><div class="v">Stratified 80/20, {cls["n_train"]:,} train and {cls["n_test"]:,} test objects.</div>'
                    f'<div class="k">Final model</div><div class="v">{final_model_text(cls)} '
                    f'All {len(cls["models"])} runs are saved and selectable on the classifier page.</div>'
                    f'</div></div>', unsafe_allow_html=True)
        show(grouped_metric_bars(results_table(cls, CLS_COLUMNS), list(CLS_COLUMNS), [T["blue"], T["teal"], T["amber"]],
                                 "Classifiers on the same held-out split"))
    with c2:
        st.markdown(f'<div class="spec-card"><div class="spec-head">{ICON_CHART}Regression, next-year stock returns</div>'
                    f'<div class="spec"><div class="k">Data</div><div class="v">200+ Financial Indicators of US Stocks (Kaggle), 2018 slice, '
                    f'{reg["n_train"] + reg["n_test"]:,} companies.</div>'
                    f'<div class="k">Target</div><div class="v">Price change over 2019, winsorised at the 1st and 99th percentile.</div>'
                    f'<div class="k">Features</div><div class="v">41 curated fundamentals, one-hot sector and an engineered growth-adjusted margin.</div>'
                    f'<div class="k">Split</div><div class="v">Stratified 80/20 on target quintiles, {reg["n_train"]:,} train and {reg["n_test"]:,} test companies.</div>'
                    f'<div class="k">Winner</div><div class="v">Random Forest (100 trees, depth 8), tuned with 5-fold GridSearchCV. '
                    f'Top-two cross-validated R2: Random Forest 0.061, Gradient Boosting 0.074. '
                    f'All {len(reg["models"])} regressors are saved and selectable on the stock page.</div>'
                    f'</div></div>', unsafe_allow_html=True)
        show(comparison_bars(results_table(reg, REG_COLUMNS), "R2", reg["best"], "All ten regressors by test R2"))

    sw, ag = clu["sweep"], clu["agglomerative"]
    best = sw.loc[sw["k"] == clu["best_k"]].iloc[0]
    st.markdown(f'<div class="spec-card"><div class="spec-head">{ICON_SURVEY}Clustering, TESS objects of interest</div>'
                f'<div class="spec"><div class="k">Data</div><div class="v">NASA TESS Objects of Interest table from the Exoplanet Archive, '
                f'{clu["n_rows"]:,} transit-like signals, downloaded {clu["downloaded"]}.</div>'
                f'<div class="k">Features</div><div class="v">{len(clu["features"])} physical measurements of the signal, the implied planet and the host star. '
                f'The seven heavy-tailed ones are log-scaled, all are clipped at the 1st/99th percentile, median-filled and standardised.</div>'
                f'<div class="k">Labels</div><div class="v">The TFOPWG disposition is never used for fitting; it is read in afterwards to see what each cluster holds.</div>'
                f'<div class="k">Models</div><div class="v">K-Means swept from k = 2 to 10, reported at k = {clu["best_k"]}: silhouette {best["silhouette"]:.3f}, '
                f'Davies-Bouldin {best["davies_bouldin"]:.2f}, Calinski-Harabasz {best["calinski_harabasz"]:,.0f}. Agglomerative clustering with Ward, complete and '
                f'average linkage compared for k from 2 to 6, reported as the {ag["linkage"]}-linkage cut at k = {ag["k"]}: silhouette {ag["metrics"]["Silhouette"]:.3f}, '
                f'Davies-Bouldin {ag["metrics"]["Davies-Bouldin"]:.2f}, Calinski-Harabasz {ag["metrics"]["Calinski-Harabasz"]:,.0f}. '
                f'Both are visualised in PCA space, the hierarchy also through dendrograms and t-SNE.</div>'
                f'</div></div>', unsafe_allow_html=True)
    k1, k2 = st.columns([7, 5], gap="large")
    with k1:
        show(clu_k_chart(clu, clu["best_k"]))
    with k2:
        show(clu_metrics_compare(clu))

    tab1, tab2 = st.tabs(["Full results tables", "Dataset overview"])
    with tab1:
        text("section", "Classification, every run from Parts A and B (ranked by weighted F1)")
        st.dataframe(results_table(cls, CLS_COLUMNS), hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.4f") for c in CLS_COLUMNS})
        text("section", "Regression (ranked by test R2)")
        st.dataframe(results_table(reg, REG_COLUMNS), hide_index=True, width="stretch",
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
    st.Page(page_clustering, title="TESS cluster explorer", icon=":material/satellite_alt:", url_path="clusters"),
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
    text("side-note", "Models from the three capstone notebooks, served from the saved files in <code>models/</code>. "
                      "Predictions are computed on the spot and nothing you enter is stored.")

try:
    load_bundles()
except FileNotFoundError:
    st.error("Saved models not found. From the repository root run `python app/export_models.py` first.")
    st.stop()

nav.run()
