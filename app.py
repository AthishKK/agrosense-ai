# ================================================
# AgroSense AI — Main Streamlit Dashboard
# ================================================

import sys
sys.stdout.reconfigure(encoding='utf-8')

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io

# ── Patch: block numba/llvmlite DLL crash on Windows Python 3.14 ──
class _Stub:
    def __getattr__(self, name): return _Stub()
    def __call__(self, *a, **kw): return _Stub()
    def __repr__(self): return '_Stub()'

_nb = _Stub()
for _mod in [
    'numba','numba.core','numba.core.serialize','numba.core.decorators',
    'numba.core.types','numba.core.typing','numba.core.utils',
    'numba.core.ir','numba.core.config','numba.core.ir_utils',
    'numba.core.registry','numba.core.extending','numba.core.lowering',
    'numba.core.environment','numba.core.pythonapi','numba.core.cgutils',
    'numba.core.imputils','numba.core.datamodel','numba.core.datamodel.models',
    'numba.typed','numba.stencils','numba.stencils.stencil',
    'numba.misc','numba.misc.firstlinefinder','numba.misc.coverage_support',
    'numba._helperlib','numba._dynfunc',
    'llvmlite','llvmlite.ir','llvmlite.binding',
]:
    sys.modules.setdefault(_mod, _nb)

import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
import urllib.parse
import datetime
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from ai_advice import configure_groq, get_farming_advice, get_soil_advice
from rag_chatbot import initialize_rag, answer_question

# ─────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────

st.set_page_config(
    page_title="AgroSense AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #f0fdf4 !important;
    color: #1f2937 !important;
}
.stApp {
    background: linear-gradient(160deg, #f0fdf4 0%, #e8f5e9 50%, #f9fafb 100%) !important;
    min-height: 100vh;
}
.main .block-container {
    padding-top: 1rem; padding-bottom: 2rem;
    max-width: 1400px; background: transparent !important;
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.main-header {
    background: linear-gradient(135deg,#1b5e20,#2e7d32,#1b5e20,#ffd700,#2e7d32);
    background-size: 300% 300%;
    animation: gradientShift 6s ease infinite;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 3.2rem; font-weight: 800;
    text-align: center; padding: 0.5rem 0 0.2rem 0; letter-spacing: -1px;
}
.sub-header { font-size:1rem;color:#6b7280;text-align:center;margin-bottom:0.5rem;font-weight:400; }
.header-divider {
    height:3px;
    background:linear-gradient(90deg,transparent,#1b5e20,#ffd700,#1b5e20,transparent);
    border:none;margin:0.5rem auto 1.5rem auto;width:60%;border-radius:2px;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0a2e0f 0%,#1b5e20 60%,#0d3b14 100%) !important;
    border-right: 1px solid #2e7d32;
}
[data-testid="stSidebar"] * { color: #e8f5e9 !important; }
[data-testid="stSidebar"] hr { border-color: #2e7d32 !important; }
.sidebar-logo-box {
    background:rgba(255,215,0,0.12);border:1px solid rgba(255,215,0,0.3);
    border-radius:16px;padding:1rem;text-align:center;margin-bottom:1rem;
}
.sidebar-logo-box .logo-title { font-size:1.3rem;font-weight:800;color:#ffd700 !important; }
.sidebar-logo-box .logo-sub   { font-size:0.72rem;color:#a5d6a7 !important; }
.section-title {
    font-size:1.35rem;font-weight:700;color:#1b5e20;
    border-left:4px solid #ffd700;padding-left:0.75rem;margin:1.2rem 0 0.8rem 0;
}
.input-panel-header {
    font-size:0.8rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
    color:#1b5e20;background:linear-gradient(90deg,#e8f5e9,transparent);
    border-left:3px solid #ffd700;padding:0.4rem 0.8rem;border-radius:0 8px 8px 0;margin-bottom:0.7rem;
}
.stButton > button {
    background:linear-gradient(135deg,#1b5e20 0%,#2e7d32 50%,#388e3c 100%) !important;
    color:#ffd700 !important;border:2px solid rgba(255,215,0,0.4) !important;
    border-radius:12px !important;padding:0.75rem 2.5rem !important;
    font-size:1.05rem !important;font-weight:700 !important;width:100% !important;
    transition:all 0.25s ease !important;box-shadow:0 4px 15px rgba(27,94,32,0.35) !important;
}
.stButton > button:hover {
    background:linear-gradient(135deg,#ffd700 0%,#ffca28 100%) !important;
    color:#1b5e20 !important;transform:translateY(-2px) !important;
}
@keyframes fadeInUp {
    from { opacity:0;transform:translateY(24px); }
    to   { opacity:1;transform:translateY(0); }
}
.results-wrapper { animation: fadeInUp 0.55s ease both; }
.results-banner {
    background:linear-gradient(135deg,#1b5e20,#2e7d32);
    border-radius:16px;padding:1.2rem 2rem;margin-bottom:1.5rem;
    display:flex;align-items:center;gap:1rem;
    box-shadow:0 4px 20px rgba(27,94,32,0.3);
}
.results-banner-title { font-size:1.6rem;font-weight:800;color:#ffd700;margin:0; }
.results-banner-sub   { font-size:0.88rem;color:#a5d6a7;margin:0; }
.crop-h-card {
    background:rgba(255,255,255,0.88);backdrop-filter:blur(12px);
    border-radius:20px;padding:1.2rem 0.8rem;text-align:center;
    border:2px solid rgba(27,94,32,0.12);
    box-shadow:0 4px 18px rgba(27,94,32,0.09);
    transition:transform 0.2s,box-shadow 0.2s;height:100%;
}
.crop-h-card:hover { transform:translateY(-4px);box-shadow:0 10px 30px rgba(27,94,32,0.18); }
.crop-h-card.top-card {
    border:2.5px solid #ffd700;
    background:linear-gradient(160deg,rgba(255,215,0,0.10),rgba(255,255,255,0.95));
    box-shadow:0 6px 28px rgba(255,215,0,0.25);
}
.crop-h-emoji  { font-size:2.6rem;margin-bottom:0.3rem; }
.crop-h-name   { font-size:1rem;font-weight:800;color:#1b5e20;margin:0.2rem 0; }
.crop-h-rank   { font-size:1.4rem;margin-bottom:0.2rem; }
.crop-h-conf   { font-size:0.78rem;color:#6b7280;margin-bottom:0.5rem; }
.crop-h-bar-bg { background:#e8f5e9;border-radius:8px;height:8px;margin:0.4rem 0.2rem;overflow:hidden; }
.crop-h-bar-fill { height:8px;border-radius:8px;background:linear-gradient(90deg,#1b5e20,#ffd700); }
.crop-h-card.top-card .crop-h-bar-fill { background:linear-gradient(90deg,#ffd700,#ff8f00); }
.soil-grade-badge {
    width:80px;height:80px;border-radius:50%;
    display:flex;flex-direction:column;align-items:center;justify-content:center;
    font-size:2rem;font-weight:900;margin:0 auto 0.5rem auto;
    box-shadow:0 4px 16px rgba(0,0,0,0.15);
}
.grade-A { background:linear-gradient(135deg,#1b5e20,#2e7d32);color:#ffd700; }
.grade-B { background:linear-gradient(135deg,#2e7d32,#66bb6a);color:#fff; }
.grade-C { background:linear-gradient(135deg,#f57c00,#ffa726);color:#fff; }
.grade-D { background:linear-gradient(135deg,#c62828,#ef5350);color:#fff; }
.fert-card {
    background:linear-gradient(135deg,#1b5e20,#2e7d32);
    border-radius:18px;padding:1.5rem;text-align:center;
    box-shadow:0 6px 24px rgba(27,94,32,0.3);margin-bottom:0.8rem;
}
.fert-card .fert-label { font-size:0.75rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#a5d6a7;margin-bottom:0.4rem; }
.fert-card .fert-name  { font-size:1.6rem;font-weight:800;color:#ffd700; }
.fert-tip {
    background:rgba(255,255,255,0.75);backdrop-filter:blur(8px);
    border-radius:10px;padding:0.7rem 1rem;font-size:0.83rem;
    color:#374151;border-left:3px solid #ffd700;
}
.shap-explain-box {
    background:rgba(255,255,255,0.8);border-radius:12px;
    padding:0.9rem 1.1rem;margin:0.4rem 0;border-left:4px solid #1b5e20;
    box-shadow:0 2px 8px rgba(0,0,0,0.06);font-size:0.88rem;color:#374151;
}
.shap-explain-box.positive { border-left-color:#2e7d32;background:linear-gradient(90deg,rgba(46,125,50,0.07),rgba(255,255,255,0.85)); }
.shap-explain-box.negative { border-left-color:#c62828;background:linear-gradient(90deg,rgba(198,40,40,0.07),rgba(255,255,255,0.85)); }
.fancy-divider {
    height:2px;
    background:linear-gradient(90deg,transparent,#1b5e20 30%,#ffd700 50%,#1b5e20 70%,transparent);
    border:none;margin:1.5rem 0;border-radius:2px;
}
.feature-card {
    background:rgba(255,255,255,0.88);border-radius:18px;padding:1.4rem;
    border:1.5px solid rgba(27,94,32,0.15);
    box-shadow:0 6px 24px rgba(27,94,32,0.10);margin-bottom:1rem;
}
.soil-fix-step {
    background:linear-gradient(135deg,rgba(27,94,32,0.06),rgba(255,215,0,0.06));
    border-radius:12px;padding:0.8rem 1rem;margin:0.5rem 0;
    border-left:4px solid #2e7d32;font-size:0.88rem;
}
.rotation-card {
    background:rgba(255,255,255,0.9);border-radius:14px;padding:1rem;text-align:center;
    border:2px solid rgba(27,94,32,0.12);box-shadow:0 4px 14px rgba(27,94,32,0.08);
}
.rotation-season { font-size:0.72rem;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#6b7280; }
.rotation-crop   { font-size:1.1rem;font-weight:800;color:#1b5e20;margin:0.3rem 0; }
.rotation-effect { font-size:0.75rem;padding:0.2rem 0.6rem;border-radius:20px;display:inline-block; }
.effect-good { background:#e8f5e9;color:#1b5e20; }
.effect-bad  { background:#ffebee;color:#c62828; }
/* Download button */
[data-testid="stDownloadButton"] button {
    background:linear-gradient(135deg,#ffd700,#ffca28) !important;
    color:#1b5e20 !important;border:2px solid #1b5e20 !important;
    border-radius:12px !important;font-weight:800 !important;
    font-size:1rem !important;box-shadow:0 4px 16px rgba(255,215,0,0.4) !important;
}
[data-testid="stDownloadButton"] button:hover {
    background:linear-gradient(135deg,#1b5e20,#2e7d32) !important;color:#ffd700 !important;
}
/* Chatbot */
.chat-container { max-height:520px;overflow-y:auto;padding:1rem 0.5rem;display:flex;flex-direction:column;gap:1rem;scroll-behavior:smooth; }
.chat-row { display:flex;align-items:flex-end;gap:0.6rem;animation:fadeInUp 0.3s ease both; }
.chat-row.user-row { flex-direction:row-reverse; }
.chat-avatar { width:38px;height:38px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:1.2rem;flex-shrink:0;box-shadow:0 2px 8px rgba(0,0,0,0.12); }
.avatar-bot  { background:linear-gradient(135deg,#1b5e20,#2e7d32); }
.avatar-user { background:linear-gradient(135deg,#37474f,#546e7a); }
.chat-bubble { max-width:72%;padding:0.85rem 1.1rem;border-radius:18px;font-size:0.92rem;line-height:1.65;box-shadow:0 3px 12px rgba(0,0,0,0.09);word-wrap:break-word; }
.bubble-bot  { background:linear-gradient(135deg,#e8f5e9,#f1f8e9);border:1.5px solid #a5d6a7;border-bottom-left-radius:4px;color:#1a2e1a; }
.bubble-user { background:linear-gradient(135deg,#1b5e20,#2e7d32);border:1.5px solid #388e3c;border-bottom-right-radius:4px;color:#ffffff; }
.chat-ts { font-size:0.68rem;color:#9ca3af;margin-top:0.25rem;text-align:right;padding:0 0.3rem; }
.chat-row.bot-row .chat-ts { text-align:left; }
@keyframes typingBounce {
    0%,80%,100% { transform:translateY(0);opacity:0.4; }
    40%          { transform:translateY(-6px);opacity:1; }
}
.typing-bubble { background:linear-gradient(135deg,#e8f5e9,#f1f8e9);border:1.5px solid #a5d6a7;border-radius:18px;border-bottom-left-radius:4px;padding:0.75rem 1.1rem;display:inline-flex;gap:5px;align-items:center; }
.typing-dot { width:8px;height:8px;background:#2e7d32;border-radius:50%;animation:typingBounce 1.2s infinite; }
.typing-dot:nth-child(2) { animation-delay:0.2s; }
.typing-dot:nth-child(3) { animation-delay:0.4s; }
/* Selectbox */
div[data-baseweb="select"] span { color:#1b5e20 !important; }
div[data-baseweb="select"] div  { color:#1b5e20 !important;background-color:#ffffff !important; }
div[data-baseweb="popover"] li  { color:#1b5e20 !important;background-color:#ffffff !important; }
div[data-baseweb="popover"] li:hover { background-color:#e8f5e9 !important; }
.stSelectbox label { color:#1b5e20 !important; }
/* Expander — force text visible */
[data-testid="stExpander"] { border:1.5px solid rgba(27,94,32,0.18) !important;border-radius:14px !important;margin-bottom:0.5rem !important;background:rgba(255,255,255,0.75) !important; }
[data-testid="stExpander"]:hover { border-color:#ffd700 !important; }
details summary { color:#1b5e20 !important;font-weight:600 !important; }
details summary * { color:#1b5e20 !important; }
details > summary > div { color:#1b5e20 !important; }
details > summary > div > p { color:#1b5e20 !important; }
[data-testid="stExpander"] > details > summary > span { color:#1b5e20 !important; }
[data-testid="stExpander"] > details > summary { background:#f0fdf4 !important; }
details > summary { color:#1b5e20 !important; }
.streamlit-expanderHeader { color:#1b5e20 !important;font-weight:600 !important; }
.streamlit-expanderHeader p { color:#1b5e20 !important; }
button[data-testid="stExpanderToggle"] p { color:#1b5e20 !important; }
/* Number input */
input[type="number"] {
    background-color:#f0fdf4 !important;color:#1b5e20 !important;
    border:1.5px solid #a5d6a7 !important;border-radius:8px !important;font-weight:600 !important;
}
input[type="number"]:focus {
    border-color:#1b5e20 !important;box-shadow:0 0 0 2px rgba(27,94,32,0.2) !important;outline:none !important;
}
.stNumberInput > div > div > input { background-color:#f0fdf4 !important;color:#1b5e20 !important;border:1.5px solid #a5d6a7 !important; }
.stNumberInput button { background-color:#e8f5e9 !important;color:#1b5e20 !important;border:1px solid #a5d6a7 !important; }
[data-testid="stMetric"] { background:rgba(255,255,255,0.8);border-radius:14px;padding:1rem 1.2rem;border:1px solid rgba(27,94,32,0.15); }
[data-testid="stMetricValue"] { color:#1b5e20 !important;font-weight:800 !important; }
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-thumb { background:#2e7d32;border-radius:3px; }
.stMarkdown,.stText,label,p,span { color:#1f2937; }
.stSelectbox > div > div,.stTextInput > div > div > input { background:#ffffff !important;color:#1f2937 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────

@st.cache_resource
def load_models():
    m = {}
    m['crop_model']         = pickle.load(open('models/crop_model.pkl','rb'))
    m['crop_encoder']       = pickle.load(open('models/crop_encoder.pkl','rb'))
    m['fertilizer_model']   = pickle.load(open('models/fertilizer_model.pkl','rb'))
    m['fertilizer_encoder'] = pickle.load(open('models/fertilizer_encoder.pkl','rb'))
    m['soil_encoder']       = pickle.load(open('models/soil_encoder.pkl','rb'))
    m['crop_fert_encoder']  = pickle.load(open('models/crop_fert_encoder.pkl','rb'))
    m['yield_model']        = pickle.load(open('models/yield_model.pkl','rb'))
    m['yield_crop_encoder'] = pickle.load(open('models/yield_crop_encoder.pkl','rb'))
    m['country_encoder']    = pickle.load(open('models/country_encoder.pkl','rb'))
    m['shap_explainer']     = pickle.load(open('models/shap_explainer.pkl','rb'))
    with open('models/crop_model_metadata.json') as f:
        m['crop_metadata'] = json.load(f)
    return m

# ─────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────

def get_season():
    mo = datetime.datetime.now().month
    return "Kharif" if mo in [6,7,8,9,10] else ("Rabi" if mo in [11,12,1,2,3] else "Zaid")

def get_top5_crops(model, encoder, N, P, K, temperature, humidity, ph, rainfall):
    s = pd.DataFrame([[N,P,K,temperature,humidity,ph,rainfall]],
                     columns=['N','P','K','temperature','humidity','ph','rainfall'])
    proba = model.predict_proba(s)[0]
    idx   = np.argsort(proba)[::-1][:5]
    return list(zip(encoder.classes_[idx], proba[idx]*100))

def get_fertilizer_recommendation(models, soil_type, top_crop, temperature, humidity, N, K, P):
    try:
        classes = list(models['crop_fert_encoder'].classes_)
        fert_crop = None
        for c in classes:
            if c.lower() == top_crop.lower(): fert_crop = c; break
        if not fert_crop:
            for c in classes:
                if c.lower() in top_crop.lower() or top_crop.lower() in c.lower():
                    fert_crop = c; break
        if not fert_crop:
            crop_map = {
                'rice':'Paddy','banana':'Sugarcane','mango':'Sugarcane',
                'apple':'Wheat','grapes':'Cotton','orange':'Sugarcane',
                'papaya':'Sugarcane','coconut':'Sugarcane',
                'chickpea':'Pulses','lentil':'Pulses','kidneybeans':'Pulses',
                'blackgram':'Pulses','mungbean':'Pulses','mothbeans':'Pulses',
                'pigeonpeas':'Pulses','pomegranate':'Cotton',
                'watermelon':'Millets','muskmelon':'Millets',
                'jute':'Cotton','coffee':'Sugarcane',
            }
            mapped = crop_map.get(top_crop.lower())
            if mapped and mapped in classes: fert_crop = mapped
        if not fert_crop: fert_crop = classes[0]
        se = models['soil_encoder'].transform([soil_type])[0]
        ce = models['crop_fert_encoder'].transform([fert_crop])[0]
        s = pd.DataFrame([[temperature,humidity,40,se,ce,N,K,P]],
                         columns=['Temparature','Humidity','Moisture',
                                  'soil_encoded','crop_encoded',
                                  'Nitrogen','Potassium','Phosphorous'])
        pred = models['fertilizer_model'].predict(s)[0]
        return models['fertilizer_encoder'].inverse_transform([pred])[0]
    except:
        return "Balanced NPK"

def get_yield_prediction(model, crop_enc, country_enc, rainfall, temp):
    s = pd.DataFrame([[crop_enc,country_enc,2024,rainfall,50,temp]],
                     columns=['crop_encoded','country_encoded','year',
                              'rainfall_mm','pesticides_tonnes','avg_temp'])
    pred = model.predict(s)[0]
    return pred, round(pred*0.90,2), round(pred*1.10,2)

COUNTRY_MAP = {
    'IN':'India','US':'United States','GB':'United Kingdom','AU':'Australia',
    'CN':'China','BR':'Brazil','PK':'Pakistan','BD':'Bangladesh',
    'NP':'Nepal','LK':'Sri Lanka','TH':'Thailand','VN':'Vietnam',
    'ID':'Indonesia','PH':'Philippines','MM':'Myanmar','DE':'Germany',
    'FR':'France','JP':'Japan','KR':'South Korea','ZA':'South Africa',
    'NG':'Nigeria','ET':'Ethiopia','KE':'Kenya','MX':'Mexico','AR':'Argentina',
}

def geocode_location(location):
    api_key = st.secrets.get("WEATHER_API_KEY","")
    if not api_key: return None,"WEATHER_API_KEY not set"
    try:
        r = requests.get("http://api.openweathermap.org/geo/1.0/direct",
                         params={"q":location,"limit":5,"appid":api_key},timeout=5)
        data = r.json()
        if r.status_code!=200 or not data: return None,"No locations found."
        return data, None
    except Exception as e:
        return None, str(e)

def fetch_weather_by_coords(lat, lon):
    import datetime as _dt
    api_key = st.secrets.get("WEATHER_API_KEY","")
    try:
        r = requests.get("https://api.openweathermap.org/data/2.5/weather",
                         params={"lat":lat,"lon":lon,"appid":api_key,"units":"metric"},timeout=5)
        wx = r.json()
        temp = wx["main"]["temp"]; hum = wx["main"]["humidity"]
        today = _dt.date.today(); ago = today - _dt.timedelta(days=30)
        rr = requests.get("https://archive-api.open-meteo.com/v1/archive",
                          params={"latitude":lat,"longitude":lon,
                                  "start_date":str(ago),"end_date":str(today),
                                  "daily":"precipitation_sum","timezone":"auto"},timeout=15)
        rain = round(sum(v for v in rr.json()["daily"]["precipitation_sum"] if v),1)
        return temp, hum, min(rain,300.0), rain, None
    except Exception as e:
        return None,None,None,None,str(e)

def get_soil_scorecard(N, P, K, ph):
    sc = {}
    sc['Nitrogen']   = ('Deficient','🔴','#ffebee') if N<280 else (('Sufficient','🟢','#e8f5e9') if N<=560 else ('Excess','🟠','#fff3e0'))
    sc['Phosphorus'] = ('Deficient','🔴','#ffebee') if P<10  else (('Sufficient','🟢','#e8f5e9') if P<=25  else ('Excess','🟠','#fff3e0'))
    sc['Potassium']  = ('Deficient','🔴','#ffebee') if K<110 else (('Sufficient','🟢','#e8f5e9') if K<=280 else ('Excess','🟠','#fff3e0'))
    if   ph<5.5:  sc['pH']=('Strongly Acidic',   '🔴','#ffebee')
    elif ph<6.0:  sc['pH']=('Moderately Acidic',  '🟠','#fff3e0')
    elif ph<=7.5: sc['pH']=('Optimal',             '🟢','#e8f5e9')
    elif ph<=8.5: sc['pH']=('Moderately Alkaline','🟠','#fff3e0')
    else:         sc['pH']=('Strongly Alkaline',   '🔴','#ffebee')
    return sc

def parse_advice_sections(advice_text):
    section_map = {
        "SEED SELECTION":      "🌾",
        "SEED":                "🌾",
        "FERTILIZER SCHEDULE": "💊",
        "FERTILIZER":          "💊",
        "IRRIGATION PLAN":     "💧",
        "IRRIGATION":          "💧",
        "DISEASE & PEST":      "🦠",
        "DISEASE AND PEST":    "🦠",
        "DISEASE":             "🦠",
        "PEST PREVENTION":     "🦠",
        "PEST":                "🦠",
        "EXPECTED CHALLENGES": "⚠️",
        "CHALLENGES":          "⚠️",
        "HARVEST GUIDANCE":    "🌿",
        "HARVEST":             "🌿",
    }
    display_names = {
        "SEED SELECTION":      "Seed Selection",
        "SEED":                "Seed Selection",
        "FERTILIZER SCHEDULE": "Fertilizer Schedule",
        "FERTILIZER":          "Fertilizer Schedule",
        "IRRIGATION PLAN":     "Irrigation Plan",
        "IRRIGATION":          "Irrigation Plan",
        "DISEASE & PEST":      "Disease & Pest Prevention",
        "DISEASE AND PEST":    "Disease & Pest Prevention",
        "DISEASE":             "Disease & Pest Prevention",
        "PEST PREVENTION":     "Disease & Pest Prevention",
        "PEST":                "Disease & Pest Prevention",
        "EXPECTED CHALLENGES": "Expected Challenges",
        "CHALLENGES":          "Expected Challenges",
        "HARVEST GUIDANCE":    "Harvest Guidance",
        "HARVEST":             "Harvest Guidance",
    }

    result = {}
    current_section = None
    current_lines = []

    for line in advice_text.split('\n'):
        line_clean = line.strip().upper()
        line_clean = line_clean.replace('#','').replace('*','').replace('-','').strip()

        matched = False
        matched_key = None

        for section in section_map:
            if section in line_clean and len(line_clean) < 60:
                matched_key = section
                matched = True
                break

        if matched:
            if current_section and current_lines:
                result[current_section] = '\n'.join(current_lines).strip()
            current_section = matched_key
            current_lines = []
        elif current_section:
            current_lines.append(line)

    if current_section and current_lines:
        result[current_section] = '\n'.join(current_lines).strip()

    # Remove empty sections
    result = {k: v for k, v in result.items() if v.strip()}

    return result, section_map, display_names

# ─────────────────────────────────────────────────
# YIELD GAP — CROP NAME MAPPING
# ─────────────────────────────────────────────────

YIELD_CROP_MAP = {
    'rice':        'Rice, Paddy',
    'wheat':       'Wheat',
    'maize':       'Maize',
    'sugarcane':   'Sugarcane',
    'cotton':      'Soybeans',
    'banana':      'Plantains And Others',
    'cassava':     'Cassava',
    'potatoes':    'Potatoes',
    'sorghum':     'Sorghum',
    'soybeans':    'Soybeans',
    'yams':        'Yams',
    'sweet potatoes':'Sweet Potatoes',
}

# ─────────────────────────────────────────────────
# NEW FEATURE FUNCTIONS
# ─────────────────────────────────────────────────

def show_input_quality_validator(N, P, K, ph, temperature, humidity, rainfall):
    issues=[]; warnings=[]; scores=[]
    checks = [
        ('Nitrogen (N)',   N,           0, 140, 'kg/ha'),
        ('Phosphorus (P)', P,           0, 145, 'kg/ha'),
        ('Potassium (K)',  K,           0, 205, 'kg/ha'),
        ('pH Level',       ph,          4.0, 9.5, ''),
        ('Temperature',    temperature, 10,  45,  '°C'),
        ('Humidity',       humidity,    20,  95,  '%'),
        ('Rainfall',       rainfall,    0,   300, 'mm'),
    ]
    for name, val, mn, mx, unit in checks:
        if mn <= val <= mx:
            scores.append(100)
        elif val < mn or val > mx:
            if name in ['pH Level','Nitrogen (N)','Phosphorus (P)','Potassium (K)']:
                issues.append(f"⚠️ {name} value ({val}{unit}) is outside normal range ({mn}-{mx}{unit})")
                scores.append(50)
            else:
                warnings.append(f"ℹ️ {name} ({val}{unit}) is unusual — verify your data")
                scores.append(70)
        else:
            scores.append(0)

    overall = int(sum(scores)/len(scores))
    grade = "HIGH ✅" if overall>=85 else ("MEDIUM ⚠️" if overall>=65 else "LOW ❌")
    gc    = "#1b5e20" if overall>=85 else ("#f57c00" if overall>=65 else "#c62828")

    st.markdown('<div class="section-title">🔍 Input Quality Report</div>', unsafe_allow_html=True)
    c1,c2 = st.columns([1,2])
    with c1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=overall,
            title={'text':"<b>Input Quality Score</b>",'font':{'size':13,'color':'#1b5e20'}},
            number={'suffix':'/100','font':{'size':28,'color':gc}},
            gauge={'axis':{'range':[0,100]},'bar':{'color':gc,'thickness':0.3},
                   'steps':[{'range':[0,65],'color':'#ffebee'},{'range':[65,85],'color':'#fff3e0'},{'range':[85,100],'color':'#e8f5e9'}]}
        ))
        fig.update_layout(height=200,margin=dict(t=40,b=10,l=10,r=10),paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f'<div style="text-align:center;background:{gc};color:white;border-radius:20px;padding:0.4rem 1rem;font-weight:800">Quality: {grade}</div>', unsafe_allow_html=True)
    with c2:
        for name, val, mn, mx, unit in checks:
            pct   = min(100, int((val-mn)/(mx-mn)*100)) if mx>mn else 0
            inrng = mn<=val<=mx
            color = '#1b5e20' if inrng else '#c62828'
            icon  = '✅' if inrng else '⚠️'
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:0.6rem;margin:0.35rem 0">'
                f'<span style="width:120px;font-size:0.82rem;font-weight:600;color:#374151">{name}</span>'
                f'<div style="flex:1;background:#e8f5e9;border-radius:6px;height:10px;overflow:hidden">'
                f'<div style="width:{pct}%;height:10px;background:{color};border-radius:6px"></div></div>'
                f'<span style="width:80px;font-size:0.82rem;font-weight:700;color:{color}">{val}{unit}</span>'
                f'<span>{icon}</span></div>', unsafe_allow_html=True)
    for i in issues:   st.error(i)
    for w in warnings: st.warning(w)
    if not issues and not warnings:
        st.success("✅ All input values are within normal agricultural ranges. Prediction confidence is HIGH.")
    return overall


def show_soil_improvement_advisor(N, P, K, ph, soil_type, top_crop):
    st.markdown('<div class="section-title">🌱 Soil Health Improvement Advisor</div>', unsafe_allow_html=True)
    steps=[]; urg=0
    if N<280:
        steps.append({'icon':'🟥','priority':'URGENT','title':'Correct Nitrogen Deficiency',
                      'action':'Apply 80-120 kg/ha Urea in split doses (40% at sowing, 30% at tillering, 30% at flowering)',
                      'timeline':'Immediate — before next sowing','expected':'N levels reach sufficient range within 1 season'})
        urg+=1
    elif N>560:
        steps.append({'icon':'🟠','priority':'REDUCE','title':'Reduce Excess Nitrogen',
                      'action':'Reduce nitrogen fertilizer by 40%. Plant a cereal crop to absorb excess N.',
                      'timeline':'Over next 2 seasons','expected':'N levels normalize in 1-2 crop cycles'})
    if P<10:
        steps.append({'icon':'🟥','priority':'URGENT','title':'Correct Phosphorus Deficiency',
                      'action':'Apply 60-80 kg/ha DAP at sowing time, banded near seed zone',
                      'timeline':'Immediate — at next sowing','expected':'P availability improves within 1 season'})
        urg+=1
    elif P>25:
        steps.append({'icon':'🟠','priority':'MONITOR','title':'Manage Excess Phosphorus',
                      'action':'Skip phosphorus fertilizer for 1-2 seasons. Excess P can block zinc absorption.',
                      'timeline':'Next 1-2 seasons','expected':'P levels reduce naturally with crop uptake'})
    if K<110:
        steps.append({'icon':'🟥','priority':'URGENT','title':'Correct Potassium Deficiency',
                      'action':'Apply 80-100 kg/ha Muriate of Potash (MOP) at sowing time',
                      'timeline':'Immediate — before next sowing','expected':'K levels improve within 1 season'})
        urg+=1
    elif K>280:
        steps.append({'icon':'🟠','priority':'REDUCE','title':'Reduce Excess Potassium',
                      'action':'Stop potassium application for 1-2 seasons.',
                      'timeline':'Next 1-2 seasons','expected':'K normalizes as crop uptake reduces levels'})
    if ph<5.5:
        steps.append({'icon':'🟥','priority':'CRITICAL','title':'Correct Strongly Acidic Soil',
                      'action':'Apply 2-3 tonnes/ha agricultural lime. Work into top 15cm.',
                      'timeline':'3-6 months before planting','expected':'pH rises to 6.0-6.5 within 1 season'})
        urg+=1
    elif ph>8.5:
        steps.append({'icon':'🟥','priority':'CRITICAL','title':'Correct Strongly Alkaline Soil',
                      'action':'Apply 1-2 tonnes/ha gypsum or sulphur + 10 tonnes/ha organic matter.',
                      'timeline':'2-4 months before planting','expected':'pH decreases by 0.5-1.0 units per treatment'})
        urg+=1
    steps.append({'icon':'🟢','priority':'RECOMMENDED','title':'Add Organic Matter',
                  'action':'Apply 10-15 tonnes/ha farmyard manure OR compost before planting.',
                  'timeline':'2-4 weeks before planting','expected':'Gradual improvement over 2-3 seasons'})

    if urg==0: st.success("✅ Your soil is in good condition!")
    elif urg==1: st.warning(f"⚠️ 1 urgent correction needed before planting {top_crop.title()}.")
    else: st.error(f"❌ {urg} urgent corrections needed before planting.")

    pc = {'URGENT':'#c62828','CRITICAL':'#c62828','REDUCE':'#f57c00','MONITOR':'#f57c00','RECOMMENDED':'#1b5e20'}
    for i,step in enumerate(steps,1):
        color = pc.get(step['priority'],'#1b5e20')
        with st.expander(f"{step['icon']} Step {i}: {step['title']} — [{step['priority']}]", expanded=(i==1)):
            st.markdown(
                f'<div class="soil-fix-step">'
                f'<div style="font-weight:700;color:{color};margin-bottom:0.4rem">🎯 Action Required:</div>'
                f'<div style="color:#1f2937;margin-bottom:0.6rem">{step["action"]}</div>'
                f'<div style="display:flex;gap:2rem;font-size:0.82rem;color:#6b7280">'
                f'<span>⏱️ <b>Timeline:</b> {step["timeline"]}</span>'
                f'<span>📈 <b>Expected:</b> {step["expected"]}</span>'
                f'</div></div>', unsafe_allow_html=True)


def show_crop_rotation_engine(top_crop, season, N, P, K):
    st.markdown('<div class="section-title">🔄 Crop Rotation Intelligence Engine</div>', unsafe_allow_html=True)
    crop_effects = {
        'rice':{'depletes':'N','effect':'Depletes N','type':'cereal'},
        'wheat':{'depletes':'N','effect':'Depletes N','type':'cereal'},
        'maize':{'depletes':'N,K','effect':'Depletes N,K','type':'cereal'},
        'sugarcane':{'depletes':'N,P,K','effect':'Heavy feeder','type':'cash'},
        'cotton':{'depletes':'N,P','effect':'Depletes N,P','type':'cash'},
        'jute':{'depletes':'N','effect':'Depletes N','type':'fiber'},
        'banana':{'depletes':'N,K','effect':'Heavy feeder','type':'fruit'},
        'chickpea':{'depletes':'','effect':'Fixes N ✅','type':'legume'},
        'lentil':{'depletes':'','effect':'Fixes N ✅','type':'legume'},
        'kidneybeans':{'depletes':'','effect':'Fixes N ✅','type':'legume'},
        'blackgram':{'depletes':'','effect':'Fixes N ✅','type':'legume'},
        'mungbean':{'depletes':'','effect':'Fixes N ✅','type':'legume'},
        'mothbeans':{'depletes':'','effect':'Fixes N ✅','type':'legume'},
        'pigeonpeas':{'depletes':'','effect':'Fixes N ✅','type':'legume'},
        'watermelon':{'depletes':'K','effect':'Depletes K','type':'fruit'},
        'muskmelon':{'depletes':'K','effect':'Depletes K','type':'fruit'},
    }
    ci  = crop_effects.get(top_crop.lower(),{'depletes':'N','effect':'Depletes nutrients','type':'cereal'})
    seasons = ['Kharif','Rabi','Zaid']
    semojis = {'Kharif':'☀️','Rabi':'❄️','Zaid':'🌸'}
    idx = seasons.index(season) if season in seasons else 0
    rotation = [
        {'season':f"{season} (Now)",'crop':top_crop.title(),'emoji':'🌾',
         'effect':ci['effect'],'effect_type':'bad' if ci.get('depletes') else 'good',
         'reason':'AI recommended based on your soil'},
    ]
    s2 = seasons[(idx+1)%3]
    if 'N' in ci.get('depletes','') or ci['type']=='cereal':
        rotation.append({'season':s2,'crop':'Chickpea' if s2=='Rabi' else 'Mungbean','emoji':'🫘',
                         'effect':'Restores N naturally','effect_type':'good',
                         'reason':f'Legume restores nitrogen depleted by {top_crop.title()}'})
    else:
        rotation.append({'season':s2,'crop':'Wheat' if s2=='Rabi' else 'Maize','emoji':'🌾',
                         'effect':'Breaks pest cycle','effect_type':'good',
                         'reason':'Different crop family breaks pest and disease cycle'})
    s3 = seasons[(idx+2)%3]
    rotation.append({'season':s3,'crop':'Maize' if s3=='Kharif' else ('Mustard' if s3=='Rabi' else 'Watermelon'),
                     'emoji':'🌽' if s3=='Kharif' else ('🌻' if s3=='Rabi' else '🍉'),
                     'effect':'Uses restored nutrients','effect_type':'good',
                     'reason':'Benefits from improved soil after legume season'})
    rotation.append({'season':f"{season} (Next Year)",'crop':top_crop.title(),'emoji':'🌾',
                     'effect':'Higher yield expected','effect_type':'good',
                     'reason':'Soil restored — expect 15-25% higher yield'})

    st.markdown('<div style="background:linear-gradient(135deg,rgba(27,94,32,0.06),rgba(255,215,0,0.06));border-radius:14px;padding:1rem;margin-bottom:1rem;border:1px solid rgba(27,94,32,0.15)"><div style="font-size:0.82rem;color:#6b7280;margin-bottom:0.8rem;font-weight:600">🔄 3-Year Rotation Plan</div>', unsafe_allow_html=True)
    rc = st.columns([1, 0.12, 1, 0.12, 1, 0.12, 1])
    rot_cols   = [rc[0], rc[2], rc[4], rc[6]]
    arrow_cols = [rc[1], rc[3], rc[5]]
    for ac in arrow_cols:
        with ac:
            st.markdown(
                '<div style="text-align:center;font-size:1.6rem;'
                'padding-top:3rem;color:#1b5e20">→</div>',
                unsafe_allow_html=True)
    for i,(col,rot) in enumerate(zip(rot_cols,rotation)):
        ec = 'effect-good' if rot['effect_type']=='good' else 'effect-bad'
        bc = '#ffd700' if i==0 else '#a5d6a7'
        with col:
            st.markdown(
                f'<div class="rotation-card" style="border-color:{bc}">'
                f'<div class="rotation-season">{semojis.get(rot["season"].split()[0],"🌱")} {rot["season"]}</div>'
                f'<div style="font-size:2rem;margin:0.4rem 0">{rot["emoji"]}</div>'
                f'<div class="rotation-crop">{rot["crop"]}</div>'
                f'<div class="rotation-effect {ec}">{rot["effect"]}</div>'
                f'<div style="font-size:0.68rem;color:#6b7280;margin-top:0.4rem">{rot["reason"]}</div>'
                f'</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.info("💡 **Why crop rotation?** Breaks pest cycles, restores nitrogen, improves soil. Can increase yield by 15-25% by year 3.")


def show_crop_calendar(top_crop, season):
    st.markdown('<div class="section-title">📅 Seasonal Crop Calendar</div>', unsafe_allow_html=True)

    # Activities in order — calendar starts from TODAY
    activity_sequence = [
        ('Land Prep',         '🚜', 'Plow field, remove weeds, add organic matter'),
        ('Sowing',            '🌱', 'Sow seeds or transplant seedlings'),
        ('First Fertilizer',  '💊', 'Apply basal fertilizer dose'),
        ('Irrigation',        '💧', 'First irrigation, maintain soil moisture'),
        ('Pest Monitoring',   '🔍', 'Monitor for pests and diseases regularly'),
        ('Second Fertilizer', '💊', 'Apply second fertilizer dose'),
        ('Flowering',         '🌸', 'Critical stage — do not miss irrigation'),
        ('Pre-Harvest',       '🌾', 'Stop irrigation, prepare for harvest'),
        ('Harvest',           '✂️', 'Harvest at the right maturity stage'),
    ]

    crop_durations = {
        'rice':5,'wheat':6,'maize':4,'sugarcane':12,'cotton':6,'jute':5,
        'banana':12,'mango':6,'chickpea':4,'lentil':4,'kidneybeans':3,
        'blackgram':3,'mungbean':3,'mothbeans':3,'pigeonpeas':6,
        'watermelon':3,'muskmelon':3,'papaya':9,'coconut':12,
        'coffee':9,'grapes':6,'apple':6,'orange':6,'pomegranate':6,
    }
    duration = crop_durations.get(top_crop.lower(), 5)

    if duration <= 3:   selected = [0,1,3,8]
    elif duration <= 5: selected = [0,1,2,4,6,7,8]
    else:               selected = list(range(min(duration,len(activity_sequence))))

    activities = [activity_sequence[i] for i in selected if i<len(activity_sequence)]

    today         = datetime.datetime.now()
    current_month = today.month
    current_year  = today.year
    month_names   = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                     7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

    # Build months starting from TODAY
    months = [((current_month-1+i)%12)+1 for i in range(len(activities))]

    st.markdown(
        f'<div style="font-size:0.88rem;color:#6b7280;margin-bottom:1rem">'
        f'📅 Your personalized farming plan for <b>{top_crop.title()}</b> '
        f'starting <b>today, {month_names[current_month]} {current_year}</b>. '
        f'Follow this timeline for best results.</div>', unsafe_allow_html=True)

    cal_cols = st.columns(len(months))
    for i,(col,month,(activity,emoji,desc)) in enumerate(zip(cal_cols,months,activities)):
        is_today = (i==0)
        if is_today:
            bg='background:linear-gradient(135deg,#1b5e20,#2e7d32);'
            txt='color:white;'; desc_c='color:#a5d6a7;'
            border='border:2.5px solid #ffd700;'
            badge='<div style="font-size:0.62rem;background:rgba(255,215,0,0.4);border-radius:6px;padding:0.1rem 0.3rem;margin-top:0.3rem;color:#ffd700;font-weight:700">▶ START TODAY</div>'
        else:
            bg='background:rgba(255,255,255,0.9);'
            txt='color:#1b5e20;'; desc_c='color:#6b7280;'
            border='border:1.5px solid rgba(27,94,32,0.15);'; badge=''
        with col:
            st.markdown(
                f'<div style="{bg}{border}border-radius:12px;padding:0.7rem 0.4rem;text-align:center;box-shadow:0 3px 10px rgba(27,94,32,0.1);margin:0.2rem 0">'
                f'<div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;{txt}">{month_names[month]}</div>'
                f'<div style="font-size:1.4rem;margin:0.3rem 0">{emoji}</div>'
                f'<div style="font-size:0.72rem;font-weight:700;{txt}">{activity}</div>'
                f'<div style="font-size:0.62rem;margin-top:0.2rem;{desc_c}">{desc}</div>'
                f'{badge}</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div style="background:#e8f5e9;border-radius:12px;padding:0.85rem 1.2rem;'
        f'margin-top:0.8rem;border-left:4px solid #1b5e20;font-size:0.88rem;color:#1b5e20">'
        f'▶ <b>Start today with Land Preparation!</b> Your {top_crop.title()} farming plan '
        f'runs from <b>{month_names[current_month]}</b> to <b>{month_names[months[-1]]}</b> '
        f'({len(months)} months). Follow each stage for best results.</div>', unsafe_allow_html=True)



def generate_pdf_report(pred, top5, soil_grade, ok_count, quality_score):
    """Generate a visual PDF report using matplotlib"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.gridspec import GridSpec
        import matplotlib
        matplotlib.use('Agg')

        fig = plt.figure(figsize=(11.7, 16.5))  # A4 size
        fig.patch.set_facecolor('#f0fdf4')
        gs = GridSpec(6, 2, figure=fig, hspace=0.5, wspace=0.4)

        # ── HEADER ──
        ax_header = fig.add_subplot(gs[0, :])
        ax_header.set_facecolor('#1b5e20')
        ax_header.text(0.5, 0.7, '🌾 AgroSense AI — Farm Analysis Report',
                       ha='center', va='center', fontsize=18, fontweight='bold',
                       color='#ffd700', transform=ax_header.transAxes)
        ax_header.text(0.5, 0.25, f"Generated: {pred['timestamp']}   |   Location: {pred.get('city','N/A')}, {pred.get('country','N/A')}   |   Season: {pred['season']}",
                       ha='center', va='center', fontsize=9, color='#a5d6a7',
                       transform=ax_header.transAxes)
        ax_header.axis('off')

        # ── SOIL NUTRIENTS BAR CHART ──
        ax1 = fig.add_subplot(gs[1, 0])
        nutrients = ['N', 'P', 'K']
        values    = [pred['N'], pred['P'], pred['K']]
        optimal   = [300, 20, 200]
        colors_n  = ['#c62828' if v < o*0.5 else ('#1b5e20' if v <= o else '#f57c00')
                     for v, o in zip(values, optimal)]
        bars = ax1.bar(nutrients, values, color=colors_n, width=0.5, edgecolor='white')
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                     f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax1.set_title('Soil Nutrients (kg/ha)', fontweight='bold', color='#1b5e20', fontsize=11)
        ax1.set_facecolor('#f9fafb')
        ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
        patches = [mpatches.Patch(color='#1b5e20',label='Sufficient'),
                   mpatches.Patch(color='#c62828',label='Deficient'),
                   mpatches.Patch(color='#f57c00',label='Excess')]
        ax1.legend(handles=patches, fontsize=7, loc='upper right')

        # ── pH GAUGE ──
        ax2 = fig.add_subplot(gs[1, 1])
        ph_val = pred['ph']
        ph_color = '#1b5e20' if 6.0<=ph_val<=7.5 else ('#c62828' if ph_val<5.5 or ph_val>8.5 else '#f57c00')
        ax2.pie([ph_val, 14-ph_val], colors=[ph_color, '#e8f5e9'],
                startangle=90, counterclock=False,
                wedgeprops={'width':0.4, 'edgecolor':'white'})
        ax2.text(0, 0, f'pH\n{ph_val}', ha='center', va='center',
                 fontsize=14, fontweight='bold', color=ph_color)
        ax2.set_title('Soil pH Level', fontweight='bold', color='#1b5e20', fontsize=11)

        # ── TOP 5 CROPS HORIZONTAL BAR ──
        ax3 = fig.add_subplot(gs[2, :])
        crop_names = [c.title() for c,_ in top5]
        crop_confs = [cf for _,cf in top5]
        bar_colors = ['#ffd700'] + ['#2e7d32']*4
        hbars = ax3.barh(crop_names[::-1], crop_confs[::-1], color=bar_colors[::-1],
                         height=0.5, edgecolor='white')
        for bar, val in zip(hbars, crop_confs[::-1]):
            ax3.text(val+0.5, bar.get_y()+bar.get_height()/2,
                     f'{val:.1f}%', va='center', fontsize=9, fontweight='bold', color='#1b5e20')
        ax3.set_xlim(0, 110)
        ax3.set_title('Top 5 Crop Recommendations (Confidence %)', fontweight='bold', color='#1b5e20', fontsize=11)
        ax3.set_facecolor('#f9fafb')
        ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)
        ax3.axvline(x=50, color='#a5d6a7', linestyle='--', alpha=0.5)

        # ── RECOMMENDATION CARDS ──
        ax4 = fig.add_subplot(gs[3, :])
        ax4.set_facecolor('#e8f5e9')
        ax4.set_xlim(0, 3); ax4.set_ylim(0, 1)
        ax4.axis('off')
        cards = [
            (0.0, '🏆 Best Crop',    pred['top_crop'].title(), '#1b5e20'),
            (1.0, '💊 Fertilizer',   pred['fertilizer'],        '#2e7d32'),
            (2.0, '📊 Yield Range',  f"{pred['yield_low']}–{pred['yield_high']} t/ha", '#388e3c'),
        ]
        for x, title, value, color in cards:
            rect = mpatches.FancyBboxPatch((x+0.05, 0.1), 0.88, 0.8,
                                            boxstyle="round,pad=0.02",
                                            facecolor=color, edgecolor='#ffd700', linewidth=2)
            ax4.add_patch(rect)
            ax4.text(x+0.49, 0.72, title,  ha='center', va='center', fontsize=9,  color='#a5d6a7', fontweight='bold')
            ax4.text(x+0.49, 0.38, value, ha='center', va='center', fontsize=12, color='#ffd700', fontweight='bold')
        ax4.set_title('AI Recommendations', fontweight='bold', color='#1b5e20', fontsize=11)

        # ── SOIL SCORECARD ──
        ax5 = fig.add_subplot(gs[4, 0])
        sc_items  = ['Nitrogen', 'Phosphorus', 'Potassium', 'pH']
        sc_status = []
        sc_colors = []
        N,P,K,ph = pred['N'],pred['P'],pred['K'],pred['ph']
        for item in sc_items:
            if   item=='Nitrogen':   s,c = ('Deficient','#c62828') if N<280 else (('Sufficient','#1b5e20') if N<=560 else ('Excess','#f57c00'))
            elif item=='Phosphorus': s,c = ('Deficient','#c62828') if P<10  else (('Sufficient','#1b5e20') if P<=25  else ('Excess','#f57c00'))
            elif item=='Potassium':  s,c = ('Deficient','#c62828') if K<110 else (('Sufficient','#1b5e20') if K<=280 else ('Excess','#f57c00'))
            else:                    s,c = ('Optimal','#1b5e20') if 6.0<=ph<=7.5 else (('Acidic','#c62828') if ph<6.0 else ('Alkaline','#f57c00'))
            sc_status.append(s); sc_colors.append(c)
        y_pos = range(len(sc_items))
        for i,(item,status,color) in enumerate(zip(sc_items,sc_status,sc_colors)):
            ax5.barh(i, 1, color=color, height=0.6, edgecolor='white')
            ax5.text(0.05, i, f'{item}: {status}', va='center', fontsize=9, color='white', fontweight='bold')
        ax5.set_xlim(0, 1.2); ax5.axis('off')
        ax5.set_title(f'Soil Health — Grade {soil_grade} ({ok_count}/4 optimal)',
                      fontweight='bold', color='#1b5e20', fontsize=11)

        # ── INPUT QUALITY ──
        ax6 = fig.add_subplot(gs[4, 1])
        ax6.pie([quality_score, 100-quality_score],
                colors=['#1b5e20','#e8f5e9'],
                startangle=90, counterclock=False,
                wedgeprops={'width':0.4,'edgecolor':'white'})
        ax6.text(0, 0, f'{quality_score}%\nQuality', ha='center', va='center',
                 fontsize=12, fontweight='bold', color='#1b5e20')
        ax6.set_title('Input Quality Score', fontweight='bold', color='#1b5e20', fontsize=11)

        # ── FOOTER ──
        ax_footer = fig.add_subplot(gs[5, :])
        ax_footer.set_facecolor('#1b5e20')
        ax_footer.text(0.5, 0.6, 'AgroSense AI — Powered by Machine Learning, SHAP & Groq LLM',
                       ha='center', va='center', fontsize=9, color='#ffd700',
                       fontweight='bold', transform=ax_footer.transAxes)
        ax_footer.text(0.5, 0.2, '⚠️ These recommendations are based on ICAR and FAO guidelines. Consult your local agricultural officer for field-specific advice.',
                       ha='center', va='center', fontsize=7, color='#a5d6a7',
                       transform=ax_footer.transAxes)
        ax_footer.axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='pdf', bbox_inches='tight', dpi=150)
        buf.seek(0)
        plt.close()
        return buf.getvalue()
    except Exception as e:
        return None

# ─────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────

def main():
    st.markdown('<div class="main-header">🌾 AgroSense AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Intelligent Precision Farming Decision Support System</div>', unsafe_allow_html=True)
    st.markdown('<hr class="header-divider">', unsafe_allow_html=True)

    with st.spinner("⚙️ Initialising AI models..."):
        models = load_models()

    st.sidebar.markdown(
        '<div class="sidebar-logo-box">'
        '<div style="font-size:2.2rem">🌾</div>'
        '<div class="logo-title">AgroSense AI</div>'
        '<div class="logo-sub">Precision Farming Intelligence</div>'
        '</div>', unsafe_allow_html=True)
    st.sidebar.markdown("---")
    groq_key = st.secrets.get("GROQ_API_KEY","")
    st.sidebar.markdown("**📋 Navigation**")
    page = st.sidebar.radio("", ["🏠 Home & Prediction","🤖 AI Farming Chatbot","📋 My Farm Report"])
    st.sidebar.markdown("---")

    auto_season = get_season()

    # ─────────────────────────────────────────────
    # PAGE 1
    # ─────────────────────────────────────────────
    if page == "🏠 Home & Prediction":

        st.markdown('<div class="section-title">🌦️ Live Weather Auto-fill</div>', unsafe_allow_html=True)
        location_input = st.text_input("Enter your city, town or village name",
                                        placeholder="e.g. Chennai, Madurai, Pune, Delhi")
        find_btn = st.button("🔍 Find My Location")

        if find_btn and location_input:
            with st.spinner("🌍 Searching locations..."):
                results, err = geocode_location(location_input)
            if err: st.error(f"⚠️ {err}")
            else:
                options = []
                for r in results:
                    parts = [r.get("name","")]
                    if r.get("state"): parts.append(r["state"])
                    parts.append(COUNTRY_MAP.get(r.get("country",""), r.get("country","")))
                    options.append(", ".join(parts))
                st.session_state["geo_results"]  = results
                st.session_state["geo_options"]  = options
                st.session_state["geo_confirmed"]= False

        if st.session_state.get("geo_options") and not st.session_state.get("geo_confirmed"):
            sel = st.selectbox("Select your exact location:", st.session_state["geo_options"])
            if st.button("✅ Confirm & Fetch Weather"):
                idx    = st.session_state["geo_options"].index(sel)
                chosen = st.session_state["geo_results"][idx]
                lat,lon = chosen["lat"],chosen["lon"]
                cn = COUNTRY_MAP.get(chosen.get("country",""), chosen.get("country",""))
                with st.spinner("🌍 Fetching weather + 30-day rainfall..."):
                    temp,hum,rain,raw,err = fetch_weather_by_coords(lat,lon)
                if err: st.error(f"⚠️ {err}")
                else:
                    import datetime as _dt
                    _m  = _dt.date.today().month
                    _se = "Kharif" if _m in [6,7,8,9,10] else ("Rabi" if _m in [11,12,1,2,3] else "Zaid")
                    st.session_state.update({
                        "wx_temp":float(temp),"wx_humidity":float(hum),
                        "wx_rainfall":float(rain),"wx_country":cn,
                        "wx_state":chosen.get("state",""),"wx_city":chosen.get("name",""),
                        "detected_season":_se,"geo_confirmed":True
                    })
                    loc = ", ".join(p for p in [chosen.get("name",""),chosen.get("state",""),cn] if p)
                    st.session_state["detected_season"] = _se
                    st.success(f"✅ **{loc}** | 🌡️ {temp}°C | 💧 {hum}% | 🌧️ {rain}mm (30-day) | 📅 {_se}")

        # ── SYNCED SLIDER + NUMBER INPUTS ──
        st.markdown('<div class="section-title">📝 Farm Details</div>', unsafe_allow_html=True)
        col1,col2,col3 = st.columns(3)

        def synced_input(label, key, mn, mx, default, step=1, fmt=None):
            if key not in st.session_state:
                st.session_state[key] = default
            val = st.slider(
                label, mn, mx,
                value=float(st.session_state[key]) if isinstance(step, float) else int(st.session_state[key]),
                step=step)
            st.session_state[key] = val
            return val

        with col1:
            st.markdown('<div class="input-panel-header">🌱 Soil Nutrients</div>', unsafe_allow_html=True)
            N  = synced_input("Nitrogen (N) kg/ha",  "N",  0,   140, 90)
            P  = synced_input("Phosphorus (P) kg/ha","P",  0,   145, 42)
            K  = synced_input("Potassium (K) kg/ha", "K",  0,   205, 43)
            ph = synced_input("pH Level",             "ph", 0.0, 14.0, 6.5, step=0.1, fmt="%.1f")

        with col2:
            st.markdown('<div class="input-panel-header">🌤️ Climate Conditions</div>', unsafe_allow_html=True)
            st.session_state["temperature"] = float(st.session_state.get("wx_temp", st.session_state.get("temperature", 25.0)))
            st.session_state["humidity"] = float(st.session_state.get("wx_humidity", st.session_state.get("humidity", 71.0)))
            st.session_state["rainfall"] = float(min(st.session_state.get("wx_rainfall", st.session_state.get("rainfall", 103.0)), 300.0))

            temperature = synced_input("Temperature (°C)",  "temperature", 0.0, 50.0,
                                        st.session_state["temperature"], step=0.1, fmt="%.1f")
            humidity    = synced_input("Humidity (%)",       "humidity",    0.0, 100.0,
                                        st.session_state["humidity"],    step=0.1, fmt="%.1f")
            rainfall    = synced_input("Rainfall (mm/yr)",   "rainfall",    0.0, 300.0,
                                        st.session_state["rainfall"],    step=0.1, fmt="%.1f")

        with col3:
            st.markdown('<div class="input-panel-header">🗺️ Farm Details</div>', unsafe_allow_html=True)
            soil_type = st.selectbox("Soil Type", ['Loamy','Sandy','Clayey','Black','Red'])
            _seasons = ["Kharif","Rabi","Zaid"]
            season   = st.selectbox("Season", _seasons,
                                     index=_seasons.index(st.session_state.get("detected_season",auto_season)))
            st.caption("📅 Auto-detected from current month. You can change this if needed.")
            _cl = sorted(models['country_encoder'].classes_)
            _wc = st.session_state.get("wx_country", "")
            country = st.selectbox("Country/Region", _cl,
                                    index=_cl.index(_wc) if _wc in _cl else 0)

        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
        analyze = st.button("🔍 ✨ Analyse My Farm with AI")

        if analyze:
            with st.spinner("🌿 Running AI analysis..."):

                quality_score = show_input_quality_validator(N,P,K,ph,temperature,humidity,rainfall)
                st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

                top5     = get_top5_crops(models['crop_model'],models['crop_encoder'],N,P,K,temperature,humidity,ph,rainfall)
                top_crop = top5[0][0]

                st.markdown('<div class="results-wrapper">', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="results-banner">'
                    f'<span style="font-size:2.5rem">🌾</span>'
                    f'<div><p class="results-banner-title">AI Analysis Complete</p>'
                    f'<p class="results-banner-sub">Top recommendation: <b style="color:#ffd700">{top_crop.title()}</b> · Season: {season} · {country}</p>'
                    f'</div></div>', unsafe_allow_html=True)

                CROP_EMOJI = {
                    'rice':'🌾','wheat':'🌾','maize':'🌽','cotton':'🧶','sugarcane':'🌿',
                    'jute':'🌿','coffee':'☕','banana':'🍌','mango':'🥭','grapes':'🍇',
                    'apple':'🍎','orange':'🍊','watermelon':'🍉','muskmelon':'🍈',
                    'papaya':'🍍','coconut':'🥥','chickpea':'🫘','lentil':'🫘',
                    'kidneybeans':'🫘','pigeonpeas':'🫘','mothbeans':'🫘',
                    'mungbean':'🫘','blackgram':'🫘','pomegranate':'🍎',
                }
                rank_labels  = ['🥇','🥈','🥉','4️⃣','5️⃣']
                rank_classes = ['top-card','','','','']

                st.markdown('<div class="section-title">🏆 Top 5 Crop Recommendations</div>', unsafe_allow_html=True)
                hcols = st.columns(5)
                for i,(crop,conf) in enumerate(top5):
                    with hcols[i]:
                        st.markdown(
                            f'<div class="crop-h-card {rank_classes[i]}">'
                            f'<div class="crop-h-rank">{rank_labels[i]}</div>'
                            f'<div class="crop-h-emoji">{CROP_EMOJI.get(crop.lower(),"🌱")}</div>'
                            f'<div class="crop-h-name">{crop.title()}</div>'
                            f'<div class="crop-h-conf">{conf:.1f}% confidence</div>'
                            f'<div class="crop-h-bar-bg"><div class="crop-h-bar-fill" style="width:{min(conf,100):.0f}%"></div></div>'
                            f'</div>', unsafe_allow_html=True)

                st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

                # SOIL HEALTH
                st.markdown('<div class="section-title">🧪 Soil Health Dashboard</div>', unsafe_allow_html=True)
                sc = get_soil_scorecard(N,P,K,ph)
                ok_count   = sum(1 for s,_,_ in sc.values() if s in ('Sufficient','Optimal'))
                soil_grade = 'A' if ok_count==4 else ('B' if ok_count==3 else ('C' if ok_count==2 else 'D'))

                g1,g2,g3,g4,g5 = st.columns(5)
                def soil_gauge(label, value, max_val, status, col):
                    color = '#2e7d32' if status in ('Sufficient','Optimal') else ('#c62828' if status=='Deficient' else '#f57c00')
                    fig = go.Figure(go.Indicator(
                        mode='gauge+number', value=value,
                        title={'text':f'<b>{label}</b>','font':{'size':13,'color':'#1b5e20'}},
                        number={'font':{'size':22,'color':color}},
                        gauge={'axis':{'range':[0,max_val]},'bar':{'color':color,'thickness':0.3},
                               'bgcolor':'white','borderwidth':0,'steps':[{'range':[0,max_val],'color':'#f1f5f9'}]}
                    ))
                    fig.update_layout(height=180,margin=dict(t=40,b=5,l=10,r=10),paper_bgcolor='rgba(0,0,0,0)')
                    pc = '#e8f5e9' if status in ('Sufficient','Optimal') else ('#ffebee' if status=='Deficient' else '#fff3e0')
                    tc = '#1b5e20' if status in ('Sufficient','Optimal') else ('#c62828' if status=='Deficient' else '#e65100')
                    with col:
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown(f'<div style="text-align:center;background:{pc};color:{tc};border-radius:20px;padding:0.2rem 0.6rem;font-size:0.75rem;font-weight:700;margin-top:-0.8rem">{status}</div>', unsafe_allow_html=True)

                soil_gauge('Nitrogen (N)',   N,  140, sc['Nitrogen'][0],   g1)
                soil_gauge('Phosphorus (P)', P,  145, sc['Phosphorus'][0], g2)
                soil_gauge('Potassium (K)',  K,  205, sc['Potassium'][0],  g3)
                soil_gauge('pH Level',       ph,  14, sc['pH'][0],         g4)
                with g5:
                    st.markdown(
                        f'<div style="text-align:center;padding-top:1.2rem">'
                        f'<div style="font-size:0.75rem;font-weight:700;color:#6b7280;letter-spacing:1px;text-transform:uppercase;margin-bottom:0.4rem">Overall Grade</div>'
                        f'<div class="soil-grade-badge grade-{soil_grade}">{soil_grade}</div>'
                        f'<div style="font-size:0.78rem;color:#6b7280;margin-top:0.3rem">{ok_count}/4 nutrients optimal</div>'
                        f'</div>', unsafe_allow_html=True)

                st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
                show_soil_improvement_advisor(N,P,K,ph,soil_type,top_crop)
                st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

                # FERTILIZER + YIELD
                col3b, col4b = st.columns(2)
                with col3b:
                    st.markdown('<div class="section-title">🌿 Fertilizer Recommendation</div>', unsafe_allow_html=True)
                    fertilizer = get_fertilizer_recommendation(models,soil_type,top_crop,temperature,humidity,N,K,P)
                    st.markdown(
                        f'<div class="fert-card">'
                        f'<div class="fert-label">✨ AI Recommended Fertilizer</div>'
                        f'<div class="fert-name">🌿 {fertilizer}</div></div>'
                        f'<div class="fert-tip">💡 Apply based on soil test results. Consult your local agricultural officer for exact dosage.</div>',
                        unsafe_allow_html=True)
                    fert_reasons = {
                        'Urea':f'Your Nitrogen ({N} kg/ha) needs boosting. Urea provides 46% N — highest nitrogen content fertilizer.',
                        'DAP':f'Your Phosphorus ({P} kg/ha) is deficient. DAP provides both P and N for strong root development.',
                        'MOP':f'Your Potassium ({K} kg/ha) needs correction. MOP provides 60% K for crop quality and disease resistance.',
                        '20-20':f'Your soil needs balanced nutrition. 20-20 provides equal N, P, K for all-round improvement.',
                        'Balanced NPK':f'Recommended based on your overall soil profile: N={N}, P={P}, K={K}.',
                    }
                    st.markdown(
                        f'<div class="shap-explain-box positive" style="margin-top:0.5rem">'
                        f'🔍 <b>Why {fertilizer}?</b><br>'
                        f'{fert_reasons.get(fertilizer, f"Recommended based on {soil_type} soil and {top_crop.title()} crop combination.")}'
                        f'</div>', unsafe_allow_html=True)

                pred_val=0; low=0; high=0
                with col4b:
                    st.markdown('<div class="section-title">📊 Yield Prediction</div>', unsafe_allow_html=True)
                    try:
                        yc = list(models['yield_crop_encoder'].classes_)
                        yc_sel = yc[0]
                        for c in yc:
                            if c.lower() in top_crop.lower() or top_crop.lower() in c.lower():
                                yc_sel = c; break
                        cey = models['yield_crop_encoder'].transform([yc_sel])[0]
                        cen = models['country_encoder'].transform([country])[0]
                        pred_val, low, high = get_yield_prediction(models['yield_model'],cey,cen,rainfall,temperature)
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta", value=round(pred_val,2),
                            delta={'reference':round(low,2),'increasing':{'color':'#2e7d32'}},
                            title={'text':"<b>Predicted Yield</b><br><span style='font-size:0.85em;color:#6b7280'>tonnes / hectare</span>",'font':{'size':16}},
                            number={'font':{'size':42,'color':'#1b5e20'},'suffix':' t/ha'},
                            gauge={'axis':{'range':[0,max(10,high*1.3)]},'bar':{'color':'#1b5e20','thickness':0.28},
                                   'bgcolor':'white','borderwidth':0,
                                   'steps':[{'range':[0,low],'color':'#ffebee'},{'range':[low,high],'color':'#c8e6c9'},{'range':[high,high*1.3],'color':'#e8f5e9'}],
                                   'threshold':{'line':{'color':'#ffd700','width':3},'thickness':0.85,'value':round(pred_val,2)}}
                        ))
                        fig.update_layout(height=300,margin=dict(t=60,b=10,l=20,r=20),paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown(f'<div style="text-align:center;background:linear-gradient(90deg,#e8f5e9,#f0fdf4);border-radius:10px;padding:0.6rem;border:1px solid #a5d6a7;font-size:0.88rem">📊 Range: <b>{low} — {high}</b> t/ha</div>', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="shap-explain-box" style="margin-top:0.5rem">'
                            f'🔍 <b>Why this yield range?</b><br>'
                            f'Predicted based on {top_crop.title()} historical yield data for {country} with your rainfall ({rainfall}mm) and temperature ({temperature}°C). The ±10% range accounts for seasonal variability and farming practices.'
                            f'</div>', unsafe_allow_html=True)
                    except:
                        st.info("Yield prediction not available for this crop/region.")

                st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
                show_crop_rotation_engine(top_crop, season, N, P, K)
                st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
                show_crop_calendar(top_crop, season)
                st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

                # SHAP
                st.markdown('<div class="section-title">🔍 Why Our AI Recommended This Crop</div>', unsafe_allow_html=True)
                shap_pairs = []
                try:
                    s = pd.DataFrame([[N,P,K,temperature,humidity,ph,rainfall]],
                                     columns=['N','P','K','temperature','humidity','ph','rainfall'])
                    sv  = np.array(models['shap_explainer'].shap_values(s))
                    idx = list(models['crop_encoder'].classes_).index(top_crop)
                    sv  = sv[0,:,idx]
                    fn  = ['Nitrogen','Phosphorus','Potassium','Temperature','Humidity','pH','Rainfall']
                    shap_pairs = sorted(zip(fn,sv), key=lambda x:abs(x[1]), reverse=True)
                    cs1,cs2 = st.columns([3,2])
                    with cs1:
                        colors = ['#1b5e20' if v>0 else '#c62828' for _,v in shap_pairs]
                        fig2 = go.Figure(go.Bar(
                            x=[v for _,v in shap_pairs], y=[f for f,_ in shap_pairs],
                            orientation='h', marker=dict(color=colors,line=dict(width=0)),
                            text=[f"+{v:.4f}" if v>0 else f"{v:.4f}" for _,v in shap_pairs],
                            textposition='outside'))
                        fig2.update_layout(
                            title=dict(text=f"<b>Feature Impact for {top_crop.title()}</b>",font=dict(size=14,color='#1b5e20')),
                            height=320,margin=dict(t=50,b=20,l=10,r=60),
                            paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(248,250,252,0.8)',
                            xaxis=dict(showgrid=True,gridcolor='#e5e7eb',zeroline=True))
                        st.plotly_chart(fig2, use_container_width=True)
                        tf,tv = shap_pairs[0]
                        st.markdown(
                            f'<div class="fert-tip">🧠 <b>In plain English:</b> The most important factor is <b>{tf}</b>, '
                            f'which {"strongly supports" if tv>0 else "works against"} growing <b>{top_crop.title()}</b>. '
                            f'Green bars = factors that helped. Red bars = factors that slightly reduced confidence.</div>',
                            unsafe_allow_html=True)
                    with cs2:
                        st.markdown('<div style="font-size:0.78rem;font-weight:700;color:#6b7280;letter-spacing:1px;text-transform:uppercase;margin-bottom:0.5rem">📊 Feature Breakdown</div>', unsafe_allow_html=True)
                        for feat,val in shap_pairs:
                            cls = 'positive' if val>0 else 'negative'
                            icon= '✅' if val>0 else '⚠️'
                            st.markdown(
                                f'<div class="shap-explain-box {cls}">{icon} <b>{feat}</b> '
                                f'{"boosted" if val>0 else "reduced"} confidence '
                                f'<span style="float:right;font-weight:700">{("+" if val>0 else "")}{val:.4f}</span></div>',
                                unsafe_allow_html=True)
                except:
                    st.info("SHAP explanation not available.")

                st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

                # AI ADVICE
                st.markdown('<div class="section-title">🤖 AI Farming Advice</div>', unsafe_allow_html=True)
                if groq_key:
                    with st.spinner("✨ Generating personalised advice..."):
                        try:
                            client = configure_groq(groq_key)
                            advice = get_farming_advice(
                                client=client, crop=top_crop.title(), fertilizer=fertilizer,
                                yield_range=f"{low}—{high}" if low else "N/A",
                                N=N, P=P, K=K, ph=ph, temperature=temperature,
                                humidity=humidity, rainfall=rainfall,
                                soil_type=soil_type, season=season)
                            parsed, icons, display_names = parse_advice_sections(advice)
                            if parsed:
                                for i,(sec,content) in enumerate(parsed.items()):
                                    with st.expander(f"{icons.get(sec,'📋')} {display_names.get(sec,sec.title())}", expanded=(i==0)):
                                        st.markdown(content)
                            else:
                                st.markdown(advice)
                            st.info("ℹ️ These recommendations are based on verified agronomic data from ICAR and FAO guidelines.")
                        except:
                            st.warning("Could not generate AI advice. Check your API key.")
                else:
                    st.markdown('<div class="fert-tip">💡 Add GROQ_API_KEY to secrets.toml to unlock AI farming advice.</div>', unsafe_allow_html=True)

                st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

                # DOWNLOAD OPTIONS
                st.markdown('<div class="section-title">📄 Download Farm Report</div>', unsafe_allow_html=True)

                report_lines = [
                    "AgroSense AI — Farm Analysis Report","="*45,
                    f"Date    : {datetime.datetime.now().strftime('%d %b %Y %H:%M')}",
                    f"Season  : {season}   |   Country: {country}",
                    f"Input Quality Score: {quality_score}/100","",
                    "── SOIL DATA ──",
                    f"Nitrogen (N): {N} kg/ha   Phosphorus (P): {P} kg/ha",
                    f"Potassium(K): {K} kg/ha   pH: {ph}   Soil Type: {soil_type}","",
                    "── CLIMATE DATA ──",
                    f"Temperature: {temperature}°C   Humidity: {humidity}%   Rainfall: {rainfall}mm","",
                    "── AI RECOMMENDATIONS ──",
                    f"Top Crop    : {top_crop.title()}",
                    f"Fertilizer  : {fertilizer}",
                    f"Yield Range : {low} — {high} t/ha",
                    f"Soil Grade  : {soil_grade}  ({ok_count}/4 nutrients optimal)","",
                    "── TOP 5 CROPS ──",
                ]
                for i,(c,cf) in enumerate(top5):
                    report_lines.append(f"  {i+1}. {c.title():20s} {cf:.1f}% confidence")
                report_lines += ["","="*45,"Powered by AgroSense AI",
                                  "Disclaimer: Consult local agricultural officer for field-specific advice."]
                report_text = "\n".join(report_lines)

                dc1, dc2 = st.columns(2)
                with dc1:
                    st.download_button(
                        label="📥 Download as Text (.txt)",
                        data=report_text,
                        file_name=f"AgroSense_{top_crop.title()}_{datetime.datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain")
                with dc2:
                    pred_data = {
                        'timestamp': datetime.datetime.now().strftime("%d %b %Y, %I:%M %p"),
                        'city':st.session_state.get('wx_city','Unknown'),
                        'state':st.session_state.get('wx_state',''),
                        'country':st.session_state.get('wx_country','Unknown'),
                        'season':season,'N':N,'P':P,'K':K,'ph':ph,
                        'soil_type':soil_type,'temperature':temperature,
                        'humidity':humidity,'rainfall':rainfall,
                        'top_crop':top_crop,
                        'top5':[(str(c),round(float(cf),2)) for c,cf in top5],
                        'fertilizer':fertilizer,
                        'yield_low':float(low),'yield_high':float(high),
                    }
                    pdf_bytes = generate_pdf_report(pred_data, top5, soil_grade, ok_count, quality_score)
                    if pdf_bytes:
                        st.download_button(
                            label="📊 Download Visual PDF Report",
                            data=pdf_bytes,
                            file_name=f"AgroSense_{top_crop.title()}_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf")
                    else:
                        st.info("PDF generation requires matplotlib. Run: pip install matplotlib")

                # Save to session state
                st.session_state["farm_analysis"] = {
                    "crop":top_crop.title(),"soil_type":soil_type,"fertilizer":fertilizer,
                    "yield_low":low,"yield_high":high,"N":N,"P":P,"K":K,"ph":ph,
                    "temperature":temperature,"humidity":humidity,"rainfall":rainfall,
                    "season":season,"country":country,
                }
                st.session_state['last_prediction'] = {**pred_data}
                st.markdown('</div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # PAGE 2 — CHATBOT
    # ─────────────────────────────────────────────
    elif page == "🤖 AI Farming Chatbot":
        st.markdown(
            '<div style="background:linear-gradient(135deg,#1b5e20,#2e7d32);border-radius:18px;'
            'padding:1.4rem 2rem;margin-bottom:1.2rem;box-shadow:0 6px 24px rgba(27,94,32,0.3)">'
            '<div style="font-size:1.7rem;font-weight:800;color:#ffd700">🤖 AgroSense Farming Assistant</div>'
            '<div style="font-size:0.88rem;color:#a5d6a7;margin-top:0.3rem">'
            'Ask any farming question — crops, fertilizers, diseases, government schemes and more</div>'
            '</div>', unsafe_allow_html=True)

        if not groq_key:
            st.warning("⚠️ Groq API key not configured. Please add GROQ_API_KEY to your secrets.toml.")
            st.stop()

        with st.spinner("⚙️ Loading knowledge base..."):
            knowledge_store, embedder, groq_client = initialize_rag(groq_key)

        if 'chat_history' not in st.session_state: st.session_state.chat_history = []
        if 'quick_q'      not in st.session_state: st.session_state.quick_q = None

        farm_ctx = ""
        fa = st.session_state.get("farm_analysis")
        if fa:
            farm_ctx = (f"Farmer analysis: Crop={fa['crop']}, Soil N={fa['N']},P={fa['P']},K={fa['K']},pH={fa['ph']},type={fa['soil_type']}. "
                        f"Climate={fa['temperature']}°C,{fa['humidity']}%,{fa['rainfall']}mm. "
                        f"Fertilizer={fa['fertilizer']}. Yield={fa['yield_low']}-{fa['yield_high']} t/ha. Season={fa['season']}.")
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#e8f5e9,#d0f0d8);border:2px solid #4caf50;'
                f'border-radius:14px;padding:0.85rem 1.2rem;margin-bottom:1rem">'
                f'🌾 <b>Context loaded:</b> <b>{fa["crop"]}</b> recommended for <b>{fa["soil_type"]}</b> soil in <b>{fa["season"]}</b> season'
                f'</div>', unsafe_allow_html=True)

        QUICK = [("🌱","Best crop for my soil?"),("📈","How to increase yield?"),
                 ("🧪","What fertilizer should I use?"),("🐛","Disease prevention tips"),
                 ("🏛️","Government farming schemes")]
        st.markdown('<div style="font-size:0.8rem;font-weight:700;color:#6b7280;letter-spacing:1px;text-transform:uppercase;margin-bottom:0.5rem">⚡ Quick Questions</div>', unsafe_allow_html=True)
        qcols = st.columns(len(QUICK))
        for i,(icon,qt) in enumerate(QUICK):
            with qcols[i]:
                if st.button(f"{icon} {qt}", key=f"qq_{i}", use_container_width=True):
                    st.session_state.quick_q = qt
        st.markdown('<hr class="fancy-divider" style="margin:0.8rem 0">', unsafe_allow_html=True)

        chat_html = '<div class="chat-container">'
        if not st.session_state.chat_history:
            chat_html += ('<div class="chat-row bot-row"><div class="chat-avatar avatar-bot">🤖</div>'
                          '<div><div class="chat-bubble bubble-bot">👋 <b>Hello! I\'m your AgroSense Farming Assistant.</b><br><br>'
                          'Ask me anything about crops, fertilizers, irrigation, pests, government schemes and more!</div></div></div>')
        for msg in st.session_state.chat_history:
            role,content,ts = msg['role'],msg['content'],msg.get('ts','')
            if role=='user':
                chat_html += f'<div class="chat-row user-row"><div class="chat-avatar avatar-user">👨‍🌾</div><div><div class="chat-bubble bubble-user">{content}</div><div class="chat-ts">{ts}</div></div></div>'
            else:
                chat_html += f'<div class="chat-row bot-row"><div class="chat-avatar avatar-bot">🤖</div><div><div class="chat-bubble bubble-bot">{content.replace(chr(10),"<br>")}</div><div class="chat-ts">{ts}</div></div></div>'
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)

        question = st.chat_input("💬 Ask any farming question...")
        if st.session_state.quick_q:
            question = st.session_state.quick_q; st.session_state.quick_q = None

        if question:
            ts = datetime.datetime.now().strftime("%I:%M %p")
            st.session_state.chat_history.append({'role':'user','content':question,'ts':ts})
            ph = st.empty()
            ph.markdown('<div class="chat-row bot-row"><div class="chat-avatar avatar-bot">🤖</div><div class="typing-bubble"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div></div>', unsafe_allow_html=True)
            result = answer_question(question, knowledge_store, embedder, groq_client, farm_context=farm_ctx)
            ph.empty()
            st.session_state.chat_history.append({'role':'assistant','content':result['answer'],'ts':datetime.datetime.now().strftime("%I:%M %p")})
            with st.expander("📚 Knowledge sources used", expanded=False):
                for i,chunk in enumerate(result['sources'],1):
                    st.caption(f"Source {i}: {chunk[:160]}...")
            st.rerun()

        if st.button("🗑️ Clear Chat History", key="clear_chat"):
            st.session_state.chat_history = []; st.rerun()

    # ─────────────────────────────────────────────
    # PAGE 3 — FARM REPORT
    # ─────────────────────────────────────────────
    elif page == "📋 My Farm Report":
        if "last_prediction" not in st.session_state:
            st.info("🌾 No farm analysis found.\n\nPlease go to 🏠 Home & Prediction and click **Analyse My Farm** first.")
            st.stop()

        pred = st.session_state['last_prediction']
        if pred.get('city','Unknown')=='Unknown':
            st.warning("📍 Location not detected. Please use the weather fetch on the Home page.")

        st.markdown(
            '<div style="background:linear-gradient(135deg,#1b5e20,#2e7d32);border-radius:18px;'
            'padding:1.4rem 2rem;margin-bottom:1.2rem;box-shadow:0 6px 24px rgba(27,94,32,0.3)">'
            '<div style="font-size:1.7rem;font-weight:800;color:#ffd700">📋 My Farm Report</div>'
            '<div style="font-size:0.88rem;color:#a5d6a7;margin-top:0.3rem">Summary of your last AI farm analysis</div>'
            '</div>', unsafe_allow_html=True)

        city,state,country = pred.get('city',''),pred.get('state',''),pred.get('country','')
        location = f"{city}, {state}, {country}" if (state and state!=city) else f"{city}, {country}"
        st.markdown(f"**Generated:** {pred['timestamp']}  |  **Location:** {location}  |  **Season:** {pred['season']}")
        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

        r1,r2 = st.columns(2)
        with r1:
            st.markdown("#### 🌱 Soil Profile")
            st.table(pd.DataFrame({
                "Parameter":["Nitrogen (N)","Phosphorus (P)","Potassium (K)","pH","Soil Type"],
                "Value":[f"{pred['N']} kg/ha",f"{pred['P']} kg/ha",f"{pred['K']} kg/ha",pred['ph'],pred['soil_type']]
            }))
            st.markdown("#### 🏆 AI Recommendations")
            st.success(f"**Best Crop:** {pred['top_crop'].title()}")
            st.info(f"**Fertilizer:** {pred['fertilizer']}")
            st.info(f"**Expected Yield:** {pred['yield_low']} — {pred['yield_high']} tonnes/hectare")
        with r2:
            st.markdown("#### 🌤️ Climate Data")
            st.table(pd.DataFrame({
                "Parameter":["Temperature","Humidity","Annual Rainfall"],
                "Value":[f"{pred['temperature']}°C",f"{pred['humidity']}%",f"{pred['rainfall']}mm"],
                "Source":["OpenWeatherMap","OpenWeatherMap","Open-Meteo (30-day)"]
            }))
            st.markdown("#### 🌾 Top 5 Crop Rankings")
            for i,(crop,conf) in enumerate(pred["top5"]):
                st.write(f"{'🥇🥈🥉4️⃣5️⃣'[i*2:i*2+2]} {crop.title()} — {conf}%")
                st.progress(int(min(conf,100)))

        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
        st.warning("⚠️ This report is generated by AgroSense AI using verified ICAR and FAO agronomic data. Monitor your crop and adjust based on actual field conditions.")

        report_text = "\n".join([
            "AGROSENSE AI — FARM ANALYSIS REPORT","="*40,
            f"Generated: {pred['timestamp']}",f"Location: {location}",f"Season: {pred['season']}","",
            "SOIL PROFILE:",f"  N:{pred['N']} P:{pred['P']} K:{pred['K']} pH:{pred['ph']} Type:{pred['soil_type']}","",
            "CLIMATE:",f"  Temp:{pred['temperature']}°C Humidity:{pred['humidity']}% Rain:{pred['rainfall']}mm","",
            "RECOMMENDATIONS:",f"  Crop:{pred['top_crop'].title()} Fertilizer:{pred['fertilizer']}",
            f"  Yield:{pred['yield_low']}-{pred['yield_high']} t/ha","",
            "TOP 5 CROPS:",
            *[f"  {i+1}. {c.title()} — {cf}%" for i,(c,cf) in enumerate(pred['top5'])],
            "","DISCLAIMER: Consult local agricultural officer for field-specific advice."
        ])

        d1,d2 = st.columns(2)
        with d1:
            st.download_button(
                label="📥 Download Text Report",
                data=report_text,
                file_name=f"AgroSense_{city.replace(' ','_')}_{pred['season']}.txt",
                mime="text/plain")
        with d2:
            pdf_bytes = generate_pdf_report(pred, pred['top5'], 'D', 0, 100)
            if pdf_bytes:
                st.download_button(
                    label="📊 Download Visual PDF",
                    data=pdf_bytes,
                    file_name=f"AgroSense_{city.replace(' ','_')}_{pred['season']}.pdf",
                    mime="application/pdf")


if __name__ == "__main__":
    main()