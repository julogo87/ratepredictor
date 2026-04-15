import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from concurrent.futures import ThreadPoolExecutor
import joblib
import hashlib
import os
import tempfile
import glob
import requests
import warnings
import time
from pytrends.request import TrendReq
from g4f.client import Client

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
DEFAULT_ALPHA_API_KEY = "3OWZCTVQY381I6B1"
DEFAULT_EIA_API_KEY   = "GDsNZWWgRGr4axJQrofEreD7epXOfVgUtbWLJ0Pa"
DEFAULT_EX_API_KEY    = "0afae24b7df33abea3d76688"
APP_TITLE      = "Shipping Rate Predictor — Shanghai → LATAM"
HIST_START     = "2022-01-01"
HIST_END       = pd.Timestamp.today().strftime("%Y-%m-%d")
FORECAST_WEEKS = 4
MODEL_CACHE_DIR = os.path.join(tempfile.gettempdir(), "rate_predictor_cache")

ROUTES = {
    "BUN": {"label": "Buenaventura", "color": "#E87722", "emoji": "🟠", "dest_keywords": ["buenaventura", "cobun"]},
    "CTG": {"label": "Cartagena",    "color": "#003DA5", "emoji": "🔵", "dest_keywords": ["cartagena",   "coctg"]},
}

SP_COLORS = {
    "Hapag Lloyd AG (HLCU)":                             "#E87722",
    "CMA / CGM (CMDU)":                                  "#003DA5",
    "MSC Mediterranean Shipping Co. SA (MSCU)":          "#5C2D91",
    "Ocean Network Express (ONEY)":                      "#E4002B",
    "ZIM INTEGRATED SHIPPING SERVICES (ZIMU)":           "#009B77",
    "Hyundai Merchant Marine Co. Ltd. (HMM) (HDMU)":     "#00539C",
}
SP_SHORT = {
    "Hapag Lloyd AG (HLCU)":                             "HLCU",
    "CMA / CGM (CMDU)":                                  "CMDU",
    "MSC Mediterranean Shipping Co. SA (MSCU)":          "MSCU",
    "Ocean Network Express (ONEY)":                      "ONEY",
    "ZIM INTEGRATED SHIPPING SERVICES (ZIMU)":           "ZIMU",
    "Hyundai Merchant Marine Co. Ltd. (HMM) (HDMU)":     "HDMU",
}

CONTAINERS = ["22g0", "45g0", "40rn"]
CONTAINER_LABELS = {"22g0": "22G0 (20ft)", "45g0": "45G0 (40ft HC)", "40rn": "40RN (Reefer)"}
PLOTLY_TMPL = "plotly_white"

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG & SIDEBAR
# ═══════════════════════════════════════════════════════════════
st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="📦")

st.sidebar.header("⚙️ Configuración")
alpha_key_input = st.sidebar.text_input("Alpha Vantage API Key", value=DEFAULT_ALPHA_API_KEY, type="password")
eia_key_input   = st.sidebar.text_input("EIA API Key (Brent)",   value=DEFAULT_EIA_API_KEY,   type="password")
ex_key_input    = st.sidebar.text_input("ExchangeRate API Key",  value=DEFAULT_EX_API_KEY,    type="password")
ALPHA_API_KEY = alpha_key_input.strip() or DEFAULT_ALPHA_API_KEY
EIA_API_KEY   = eia_key_input.strip()   or DEFAULT_EIA_API_KEY
EX_API_KEY    = ex_key_input.strip()    or DEFAULT_EX_API_KEY

st.sidebar.markdown("**Señales exógenas**")
ca, cb, cc = st.sidebar.columns(3)
with ca: use_port_congestion = st.toggle("Trends",  value=True)
with cb: use_brent           = st.toggle("Brent",   value=True)
with cc: use_fx              = st.toggle("USD/CNY", value=True)

st.sidebar.markdown("---")
debug_mode           = st.sidebar.checkbox("Debug mode", value=False)
enable_hparam_tuning = st.sidebar.checkbox("Optimizar hiperparámetros (Optuna)", value=False)

if "refresh_token" not in st.session_state:
    st.session_state.refresh_token = 0
if st.sidebar.button("🔄 Limpiar caché y refrescar"):
    st.cache_data.clear()
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    for f in glob.glob(os.path.join(MODEL_CACHE_DIR, "*.joblib")):
        try: os.remove(f)
        except: pass
    st.session_state.refresh_token += 1
    st.sidebar.success("Caché limpiada.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Horizonte de visualización**")
hist_weeks = st.sidebar.slider(
    "Semanas de historial en gráficos",
    min_value=8, max_value=208, value=52, step=4,
    help="Cuántas semanas hacia atrás mostrar en los gráficos de historial. No afecta el entrenamiento.")
st.sidebar.caption(f"Forecast: {FORECAST_WEEKS} semanas | Desde {HIST_START}")

# ═══════════════════════════════════════════════════════════════
# HELPERS — IA
# ═══════════════════════════════════════════════════════════════
_AI_MODELS   = ("gemini-1.5-flash", "deepseek-v3", "gpt-4o-mini")
_AI_TIMEOUT  = 20   # segundos máximo por intento
_AI_RETRIES  = 2    # reintentos por modelo

def _try_model(client, model: str, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text or "key" in text.lower() or "login" in text.lower():
        raise ValueError(f"Respuesta inválida de {model}")
    return text

def call_ai(prompt: str) -> str:
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
    client = Client()
    last_err = None
    for model in _AI_MODELS:
        for _ in range(_AI_RETRIES):
            try:
                with ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(_try_model, client, model, prompt)
                    return fut.result(timeout=_AI_TIMEOUT)
            except FuturesTimeout:
                last_err = f"{model} no respondió en {_AI_TIMEOUT}s"
            except Exception as e:
                last_err = e
    names = " → ".join(_AI_MODELS)
    return f"❌ Sin respuesta ({names}): {last_err}"

# ═══════════════════════════════════════════════════════════════
# HELPERS — GENERAL
# ═══════════════════════════════════════════════════════════════
def _sp_short(sp: str) -> str:  return SP_SHORT.get(sp, sp[:6])
def _sp_color(sp: str) -> str:  return SP_COLORS.get(sp, "#888888")

def _trend_arrow(delta: float, threshold=50) -> tuple:
    if delta > threshold:  return "↑", "#2ecc71", "Alcista"
    if delta < -threshold: return "↓", "#e74c3c", "Bajista"
    return "→", "#f39c12", "Estable"

def excel_serial_to_date(value):
    if pd.isna(value): return pd.NaT
    if isinstance(value, (int, float)):
        try: return pd.to_datetime("1899-12-30") + pd.to_timedelta(int(value), "D")
        except: pass
    try: return pd.to_datetime(value, errors="coerce")
    except: return pd.NaT

def _get_model_cache_path(df, use_pc, use_b, use_fx_flag, enable_tuning, suffix=""):
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    key = (f"{df.shape}_{df['Week'].max()}_{use_pc}_{use_b}_{use_fx_flag}"
           f"_{enable_tuning}_{suffix}_{df[CONTAINERS].sum().sum():.2f}")
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return os.path.join(MODEL_CACHE_DIR, f"models_{h}.joblib")

# ═══════════════════════════════════════════════════════════════
# DATA SOURCES (cached API calls)
# ═══════════════════════════════════════════════════════════════
def _alpha_get_json(url, tries=5, sleep_seconds=15):
    last_err = None
    for _ in range(tries):
        try:
            r = requests.get(url, timeout=30); r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and ("Note" in data or "Information" in data or "Error Message" in data):
                last_err = data.get("Note") or data.get("Information") or data.get("Error Message")
                time.sleep(sleep_seconds); continue
            return data
        except Exception as e:
            last_err = str(e); time.sleep(3)
    raise ValueError(f"AlphaVantage error: {last_err}")

@st.cache_data(ttl=3600)
def get_brent_data_eia(start_date, end_date, api_key, debug=False, cache_buster=0):
    try:
        url = (f"https://api.eia.gov/v2/petroleum/pri/spt/data/?api_key={api_key}"
               f"&frequency=weekly&data[0]=value&facets[series][]=RBRTE"
               f"&start={pd.to_datetime(start_date).strftime('%Y-%m-%d')}"
               f"&end={pd.to_datetime(end_date).strftime('%Y-%m-%d')}")
        if debug: st.info(f"EIA URL: {url}")
        r = requests.get(url, timeout=30); r.raise_for_status()
        rows = r.json().get("response", {}).get("data", [])
        if not rows: raise ValueError("EIA sin datos")
        df = pd.DataFrame(rows)
        dc = next((c for c in ["period","date","week","periodStart"] if c in df.columns), None)
        if not dc: raise ValueError("Sin columna fecha EIA")
        df["Week"] = pd.to_datetime(df[dc], errors="coerce")
        df["Brent_Price"] = pd.to_numeric(df.get("value"), errors="coerce")
        df = df.dropna(subset=["Week","Brent_Price"])[["Week","Brent_Price"]].sort_values("Week")
        df["Week"] = df["Week"].dt.to_period("W-MON").dt.start_time
        return df.groupby("Week", as_index=False)["Brent_Price"].mean()
    except Exception as e:
        if debug: st.error(f"EIA error: {e}")
        return pd.DataFrame({"Week":[], "Brent_Price":[]})

@st.cache_data(ttl=3600)
def get_brent_data_alpha(start_date, end_date, api_key):
    url  = f"https://www.alphavantage.co/query?function=BRENT&interval=weekly&apikey={api_key}"
    data = _alpha_get_json(url)
    rows = data.get("data", [])
    if not rows: raise ValueError("AlphaVantage BRENT vacío")
    df = pd.DataFrame(rows)
    df["Week"] = pd.to_datetime(df["date"])
    df["Brent_Price"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["Week","Brent_Price"])
    df = df[(df["Week"]>=pd.to_datetime(start_date)) & (df["Week"]<=pd.to_datetime(end_date))]
    df["Week"] = df["Week"].dt.to_period("W-MON").dt.start_time
    return df[["Week","Brent_Price"]].sort_values("Week").reset_index(drop=True)

@st.cache_data(ttl=3600)
def get_brent_data_combined(start_date, end_date, alpha_key, eia_key, debug=False, cache_buster=0):
    df = get_brent_data_eia(start_date, end_date, eia_key, debug=debug, cache_buster=cache_buster)
    if df is not None and not df.empty: return df
    st.warning("EIA sin datos, usando AlphaVantage...")
    try: return get_brent_data_alpha(start_date, end_date, alpha_key)
    except Exception as e:
        st.error(f"AlphaVantage Brent error: {e}")
        return pd.DataFrame({"Week":[], "Brent_Price":[]})

@st.cache_data(ttl=1800)
def get_exchange_data_pair(start_date, end_date, api_key):
    r = requests.get(f"https://v6.exchangerate-api.com/v6/{api_key}/pair/USD/CNY", timeout=30)
    r.raise_for_status(); data = r.json()
    if data.get("result") != "success": raise ValueError(data.get("error-type","error"))
    rate = data.get("conversion_rate")
    if rate is None: raise ValueError("Sin conversion_rate")
    return pd.DataFrame({"Week": pd.date_range(start=start_date, end=end_date, freq="W-MON"),
                          "Exchange_Rate": float(rate)})

@st.cache_data(ttl=3600)
def get_exchange_data_alpha(start_date, end_date, api_key):
    data_w = _alpha_get_json(f"https://www.alphavantage.co/query?function=FX_WEEKLY&from_symbol=USD&to_symbol=CNY&apikey={api_key}")
    weekly = data_w.get("Time Series FX (Weekly)", {})
    if weekly:
        dfw = pd.DataFrame([{"Week": pd.to_datetime(k), "Exchange_Rate": float(v.get("4. close","nan"))} for k,v in weekly.items()])
        dfw = dfw.dropna()
        dfw = dfw[(dfw["Week"]>=pd.to_datetime(start_date)) & (dfw["Week"]<=pd.to_datetime(end_date))]
        dfw["Week"] = dfw["Week"].dt.to_period("W-MON").dt.start_time
        return dfw.sort_values("Week").reset_index(drop=True)
    data_d = _alpha_get_json(f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol=USD&to_symbol=CNY&apikey={api_key}&outputsize=full")
    daily = data_d.get("Time Series FX (Daily)", {})
    if not daily: raise ValueError("FX_DAILY vacío")
    dfd = pd.DataFrame([{"Date": pd.to_datetime(k), "Close": float(v.get("4. close","nan"))} for k,v in daily.items()])
    dfd = dfd.dropna().set_index("Date").sort_index()
    return dfd["Close"].resample("W-MON").mean().reset_index().rename(columns={"Date":"Week","Close":"Exchange_Rate"})

@st.cache_data(ttl=3600)
def get_fx_yahoo(start_date, end_date):
    """Descarga USD/CNY histórico semanal directo desde Yahoo Finance (sin API key)."""
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/USDCNY%3DX"
           "?interval=1wk&range=10y")
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    ts    = data["chart"]["result"][0]
    times = ts["timestamp"]
    close = ts["indicators"]["quote"][0]["close"]
    df = pd.DataFrame({"Week": pd.to_datetime(times, unit="s"), "Exchange_Rate": close})
    df = df.dropna()
    df["Week"] = df["Week"].dt.to_period("W-MON").dt.start_time
    df = df[(df["Week"] >= pd.to_datetime(start_date)) & (df["Week"] <= pd.to_datetime(end_date))]
    return df.sort_values("Week").reset_index(drop=True)

@st.cache_data(ttl=3600)
def get_fx_data_combined(start_date, end_date, alpha_key, ex_key, debug=False, cache_buster=0):
    # 1. Yahoo Finance (histórico real, sin key)
    try:
        df = get_fx_yahoo(start_date, end_date)
        if df is not None and not df.empty: return df
    except Exception as e1:
        if debug: st.warning(f"Yahoo FX error ({e1}).")
    # 2. ExchangeRate-API (solo tasa actual, sin historial)
    try: return get_exchange_data_pair(start_date, end_date, ex_key)
    except Exception as e2:
        if debug: st.warning(f"ExchangeRate API error ({e2}). Usando AlphaVantage...")
    # 3. AlphaVantage
    try: return get_exchange_data_alpha(start_date, end_date, alpha_key)
    except Exception as e3:
        st.error(f"USD/CNY no disponible ({e3}).")
        return pd.DataFrame({"Week":[], "Exchange_Rate":[]})

@st.cache_data(ttl=3600)
def load_google_trends_data_auto():
    try:
        pt = TrendReq(hl="en-US", tz=0)
        pt.build_payload(["port congestion"],
                         timeframe=f"{HIST_START} {pd.Timestamp.today().strftime('%Y-%m-%d')}", geo="")
        df = pt.interest_over_time().reset_index()
        if "isPartial" in df.columns: df = df.drop(columns=["isPartial"])
        df.columns = ["Week","Port_Congestion_Interest"]
        df["Week"] = pd.to_datetime(df["Week"]).dt.to_period("W-MON").dt.start_time
        return df.groupby("Week")["Port_Congestion_Interest"].mean().reset_index()
    except Exception as e:
        st.error(f"Google Trends error: {e}")
        return pd.DataFrame({"Week":[], "Port_Congestion_Interest":[]})

# ═══════════════════════════════════════════════════════════════
# DATA PROCESSING
# ═══════════════════════════════════════════════════════════════
@st.cache_data
def load_and_process_data(uploaded_file, use_pc, use_b, use_fx_flag,
                           brent_df, exchange_df, trends_df, cache_buster=0):
    if uploaded_file is None: return None
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error al leer Excel: {e}"); return None

    required = ["Duration from", "Service provider", "22g0", "45g0", "40rn"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Columnas faltantes: {missing}"); return None

    df["Duration from"] = df["Duration from"].apply(excel_serial_to_date)
    if df["Duration from"].isna().all():
        df["Duration from"] = pd.date_range(start=HIST_START, periods=len(df), freq="D")

    # Outliers imposibles (< $50 USD)
    for col in CONTAINERS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < 50, col] = np.nan

    for col in ["22g0","45g0"]:
        df[col] = df[col].fillna(df[col].median())

    # Imputación 40rn con ratio aprendido por proveedor
    valid_rn = df.dropna(subset=["40rn","45g0"])
    valid_rn = valid_rn[valid_rn["45g0"] > 0]
    ratio_global = (valid_rn["40rn"] / valid_rn["45g0"]).median() if len(valid_rn) >= 10 else 1.1
    sp_ratios = (valid_rn.groupby("Service provider")
                 .apply(lambda g: (g["40rn"] / g["45g0"]).median())
                 .to_dict()) if len(valid_rn) >= 10 else {}
    df["40rn"] = df.apply(
        lambda r: r["45g0"] * sp_ratios.get(r["Service provider"], ratio_global)
        if pd.isna(r["40rn"]) else r["40rn"], axis=1
    )

    df["Week"]   = pd.to_datetime(df["Duration from"]).dt.to_period("W-MON").dt.start_time
    weekly_df    = df.groupby(["Week","Service provider"])[CONTAINERS].mean().reset_index()

    # IQR capping (5–95 percentil por proveedor)
    for _, grp in weekly_df.groupby("Service provider"):
        idx = grp.index
        for col in CONTAINERS:
            q1, q3 = grp[col].quantile(0.05), grp[col].quantile(0.95)
            iqr    = q3 - q1
            weekly_df.loc[idx, col] = weekly_df.loc[idx, col].clip(
                lower=max(q1 - 1.5*iqr, 50), upper=q3 + 1.5*iqr)

    # Merge exógenas
    if use_pc and trends_df is not None and not trends_df.empty:
        weekly_df = pd.merge(weekly_df, trends_df, on="Week", how="left")
        weekly_df["Port_Congestion_Interest"] = weekly_df["Port_Congestion_Interest"].fillna(0)
    else:
        weekly_df["Port_Congestion_Interest"] = 0.0

    if use_b and brent_df is not None and not brent_df.empty:
        weekly_df = pd.merge(weekly_df, brent_df, on="Week", how="left")
    else:
        weekly_df["Brent_Price"] = 0.0

    if use_fx_flag and exchange_df is not None and not exchange_df.empty:
        weekly_df = pd.merge(weekly_df, exchange_df, on="Week", how="left")
    else:
        weekly_df["Exchange_Rate"] = 0.0

    for col, flag in [("Brent_Price", use_b), ("Exchange_Rate", use_fx_flag)]:
        if col in weekly_df.columns and weekly_df[col].isna().any():
            weekly_df[col] = weekly_df[col].fillna(
                0 if not flag else weekly_df[col].median())

    weekly_df["Week_of_Year"] = weekly_df["Week"].dt.isocalendar().week.astype(int)
    weekly_df["Month"]        = weekly_df["Week"].dt.month.astype(int)
    weekly_df["Quarter"]      = weekly_df["Week"].dt.quarter.astype(int)
    weekly_df = weekly_df.sort_values(["Service provider","Week"]).reset_index(drop=True)

    for _, grp in weekly_df.groupby("Service provider"):
        idx = grp.index
        for col in CONTAINERS:
            s = grp[col]; s1 = s.shift(1)
            weekly_df.loc[idx, f"{col}_lag1"]        = s1
            weekly_df.loc[idx, f"{col}_lag2"]        = s.shift(2)
            weekly_df.loc[idx, f"{col}_lag4"]        = s.shift(4)
            weekly_df.loc[idx, f"{col}_pct_chg1"]    = s.pct_change(1)
            weekly_df.loc[idx, f"{col}_roll4_mean"]  = s1.rolling(4,  min_periods=2).mean()
            weekly_df.loc[idx, f"{col}_roll8_mean"]  = s1.rolling(8,  min_periods=4).mean()
            weekly_df.loc[idx, f"{col}_roll4_std"]   = s1.rolling(4,  min_periods=2).std()
            weekly_df.loc[idx, f"{col}_ewm_span4"]   = s1.ewm(span=4, adjust=False).mean()
            ema4 = s1.ewm(span=4, adjust=False).mean()
            ema8 = s1.ewm(span=8, adjust=False).mean()
            weekly_df.loc[idx, f"{col}_macd_signal"] = ema4 - ema8
            # Indicador de régimen: distancia al promedio de 52 semanas (capta picos/valles)
            roll52 = s1.rolling(52, min_periods=12).mean()
            weekly_df.loc[idx, f"{col}_roll52_mean"]   = roll52
            weekly_df.loc[idx, f"{col}_pct_vs_52wk"]  = (s1 - roll52) / (roll52 + 1e-9)
        for col in ["Brent_Price","Exchange_Rate","Port_Congestion_Interest"]:
            weekly_df.loc[idx, f"{col}_lag1"] = grp[col].shift(1)

    lag_sfx = ("lag1","lag2","lag4","pct_chg1","roll4_mean","roll8_mean",
               "roll4_std","ewm_span4","macd_signal","roll52_mean","pct_vs_52wk")
    for col in [c for c in weekly_df.columns if c.endswith(lag_sfx)]:
        med = weekly_df.groupby("Service provider")[col].transform("median")
        weekly_df[col] = weekly_df[col].fillna(med).fillna(0)

    return weekly_df

# ═══════════════════════════════════════════════════════════════
# OPTUNA TUNING
# ═══════════════════════════════════════════════════════════════
def _tune_xgb_params(X, y, n_trials=15):
    try:
        import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
        tscv = TimeSeriesSplit(n_splits=3)
        def objective(trial):
            p = {"max_depth": trial.suggest_int("max_depth",2,5),
                 "learning_rate": trial.suggest_float("learning_rate",0.02,0.12,log=True),
                 "n_estimators": trial.suggest_int("n_estimators",150,500,step=50),
                 "subsample": trial.suggest_float("subsample",0.6,0.9),
                 "colsample_bytree": trial.suggest_float("colsample_bytree",0.6,0.9),
                 "reg_alpha": trial.suggest_float("reg_alpha",0.0,1.0),
                 "reg_lambda": trial.suggest_float("reg_lambda",0.5,3.0),
                 "min_child_weight": trial.suggest_int("min_child_weight",3,10),
                 "random_state": 42}
            rmses = []
            for tr,te in tscv.split(X):
                if len(tr)<26: continue
                m = XGBRegressor(**p)
                m.fit(X.iloc[tr], y.iloc[tr])
                rmses.append(float(np.sqrt(mean_squared_error(y.iloc[te],m.predict(X.iloc[te])))))
            return float(np.mean(rmses)) if rmses else 1e9
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        b = study.best_params
        return {"max_depth":b.get("max_depth",3),"learning_rate":b.get("learning_rate",0.05),
                "n_estimators":b.get("n_estimators",300),"subsample":b.get("subsample",0.8),
                "colsample_bytree":b.get("colsample_bytree",0.8),
                "reg_alpha":b.get("reg_alpha",0.1),"reg_lambda":b.get("reg_lambda",1.5),
                "min_child_weight":b.get("min_child_weight",5)}
    except ImportError: pass
    return {"max_depth":3,"learning_rate":0.05,"n_estimators":300,"subsample":0.8,
            "colsample_bytree":0.8,"reg_alpha":0.1,"reg_lambda":1.5,"min_child_weight":5}

# ═══════════════════════════════════════════════════════════════
# MODEL TRAINING
# ═══════════════════════════════════════════════════════════════
def train_models_xgb(df, use_pc, use_b, use_fx_flag, enable_tuning=False):
    base_feats = [
        "Week_of_Year","Month","Quarter",
        "22g0_lag1","22g0_lag2","22g0_lag4","22g0_pct_chg1",
        "22g0_roll4_mean","22g0_roll8_mean","22g0_roll4_std","22g0_ewm_span4","22g0_macd_signal",
        "22g0_roll52_mean","22g0_pct_vs_52wk",
        "45g0_lag1","45g0_lag2","45g0_lag4","45g0_pct_chg1",
        "45g0_roll4_mean","45g0_roll8_mean","45g0_roll4_std","45g0_ewm_span4","45g0_macd_signal",
        "45g0_roll52_mean","45g0_pct_vs_52wk",
        "40rn_lag1","40rn_lag2","40rn_lag4","40rn_pct_chg1",
        "40rn_roll4_mean","40rn_roll8_mean","40rn_roll4_std","40rn_ewm_span4","40rn_macd_signal",
        "40rn_roll52_mean","40rn_pct_vs_52wk",
    ]
    if use_b:        base_feats += ["Brent_Price","Brent_Price_lag1"]
    if use_fx_flag:  base_feats += ["Exchange_Rate","Exchange_Rate_lag1"]
    if use_pc:       base_feats += ["Port_Congestion_Interest","Port_Congestion_Interest_lag1"]

    features = [f for f in base_feats if f in df.columns]
    # Ventana deslizante: max 78 semanas de train (~1.5 años) + gap de 4 semanas.
    # Evita que datos del régimen 2022 ($1.2k) contaminen predicciones del régimen 2025 ($1.8k).
    tscv = TimeSeriesSplit(n_splits=4, gap=4, max_train_size=78)
    MIN_TRAIN_FOLD = 26   # al menos 6 meses de historia antes de evaluar
    DEFAULT_BP = {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 300,
                  "subsample": 0.8, "colsample_bytree": 0.8,
                  "reg_alpha": 0.1, "reg_lambda": 1.5, "min_child_weight": 5}
    models, metrics = {}, {}

    for sp in df["Service provider"].unique():
        sp_df = df[df["Service provider"]==sp].copy().sort_values("Week").reset_index(drop=True)
        if len(sp_df) < 30: continue
        sp_models, sp_metrics = {}, {}

        # Pesos exponenciales: dato más reciente = más peso (half-life ~26 semanas)
        n = len(sp_df)
        decay = np.exp(np.linspace(-2.0, 0.0, n))   # e^-2 ... e^0
        sample_weights = decay / decay.sum() * n

        for target in CONTAINERS:
            X = sp_df[features]; y = sp_df[target]
            if len(X) < 20 or y.std() < 10: continue

            bp = _tune_xgb_params(X, y, 15) if (enable_tuning and len(X)>=40) else DEFAULT_BP.copy()

            fold_rmses, fold_mapes, fold_r2s, last_te = [], [], [], None
            for tr_i, te_i in tscv.split(X):
                # Ignorar folds con poca historia de entrenamiento
                if len(tr_i) < MIN_TRAIN_FOLD or len(te_i) < 1: continue
                m = XGBRegressor(**bp, random_state=42)
                m.fit(X.iloc[tr_i], y.iloc[tr_i],
                      sample_weight=sample_weights[tr_i])
                p = m.predict(X.iloc[te_i]); y_te = y.iloc[te_i]
                fold_rmses.append(float(np.sqrt(mean_squared_error(y_te, p))))
                fold_mapes.append(float(np.mean(np.abs((y_te - p) / (y_te + 1e-9))) * 100))
                fold_r2s.append(float(r2_score(y_te, p)))
                last_te = te_i

            # Modelo final entrenado en todo el historial con pesos de recencia
            model = XGBRegressor(**bp, random_state=42)
            model.fit(X, y, sample_weight=sample_weights)

            model_lo = model_hi = None
            try:
                model_lo = XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.10, **bp, random_state=42)
                model_hi = XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.90, **bp, random_state=42)
                model_lo.fit(X, y, sample_weight=sample_weights)
                model_hi.fit(X, y, sample_weight=sample_weights)
            except: model_lo = model_hi = None

            sp_models[target] = model
            if model_lo: sp_models[f"{target}_lo"] = model_lo
            if model_hi: sp_models[f"{target}_hi"] = model_hi

            resid_std = float("nan")
            if last_te is not None:
                resid = y.iloc[last_te] - model.predict(X.iloc[last_te])
                resid_std = float(np.std(resid,ddof=1)) if len(resid)>1 else float("nan")

            sp_metrics[target] = {
                "RMSE": float(np.mean(fold_rmses)) if fold_rmses else float("nan"),
                "MAPE": float(np.mean(fold_mapes)) if fold_mapes else float("nan"),
                "R2":   float(np.mean(fold_r2s))   if fold_r2s   else float("nan"),
                "RESID_STD": resid_std, "best_params": bp,
                "feature_importances": dict(zip(features, model.feature_importances_.tolist()))
            }

        if sp_models:
            models[sp] = sp_models; metrics[sp] = sp_metrics

    return models, metrics, features

# ═══════════════════════════════════════════════════════════════
# PREDICTION WITH ROLLFORWARD
# ═══════════════════════════════════════════════════════════════
def _build_fv(row: dict, features: list) -> list:
    vec = []
    for f in features:
        v = row.get(f, 0.0)
        try:
            v = float(v)
            if np.isnan(v) or np.isinf(v): v = 0.0
        except: v = 0.0
        vec.append(v)
    return vec

def generate_predictions_xgb(models, df, features, forecast_weeks=FORECAST_WEEKS):
    results   = {}
    last_date = df["Week"].max()

    for sp in df["Service provider"].unique():
        sp_models = models.get(sp, {})
        if not sp_models: continue
        sp_df = df[df["Service provider"]==sp].copy().sort_values("Week").reset_index(drop=True)
        if sp_df.empty: continue

        last_row = sp_df.iloc[-1]
        history  = {col: list(sp_df[col].values) for col in CONTAINERS}
        preds    = {col: [] for col in CONTAINERS}

        for i in range(1, forecast_weeks+1):
            date = last_date + timedelta(weeks=i)
            nr   = {"Week_of_Year": date.isocalendar().week,
                    "Month": date.month,
                    "Quarter": (date.month-1)//3+1}

            for col in CONTAINERS:
                h = history[col]; hs = pd.Series(h)
                nr[f"{col}_lag1"] = h[-1] if h else 0.0
                nr[f"{col}_lag2"] = h[-2] if len(h)>=2 else (h[-1] if h else 0.0)
                nr[f"{col}_lag4"] = h[-4] if len(h)>=4 else (h[0]  if h else 0.0)
                p1 = h[-1] if h else 1.0
                p2 = h[-2] if len(h)>=2 else p1
                nr[f"{col}_pct_chg1"]   = (p1-p2)/(abs(p2)+1e-9)
                nr[f"{col}_roll4_mean"]  = float(hs.tail(4).mean())  if len(h)>=2 else p1
                nr[f"{col}_roll8_mean"]  = float(hs.tail(8).mean())  if len(h)>=4 else p1
                nr[f"{col}_roll4_std"]   = float(hs.tail(4).std())   if len(h)>=2 else 0.0
                nr[f"{col}_ewm_span4"]   = float(hs.ewm(span=4,adjust=False).mean().iloc[-1])
                ema4v = float(hs.ewm(span=4,adjust=False).mean().iloc[-1])
                ema8v = float(hs.ewm(span=8,adjust=False).mean().iloc[-1])
                nr[f"{col}_macd_signal"] = ema4v - ema8v
                roll52v = float(hs.tail(52).mean()) if len(h)>=12 else p1
                nr[f"{col}_roll52_mean"]  = roll52v
                nr[f"{col}_pct_vs_52wk"]  = (p1 - roll52v) / (roll52v + 1e-9)

            for exo in ["Brent_Price","Exchange_Rate","Port_Congestion_Interest"]:
                v = float(last_row[exo]) if exo in last_row.index and not pd.isna(last_row[exo]) else 0.0
                nr[exo] = v; nr[f"{exo}_lag1"] = v

            for col in CONTAINERS:
                m = sp_models.get(col)
                nr[col] = float(m.predict([_build_fv(nr,features)])[0]) if m else (history[col][-1] if history[col] else 0.0)
                entry = {"Week": date, "Rate": round(nr[col],2)}
                lo, hi = sp_models.get(f"{col}_lo"), sp_models.get(f"{col}_hi")
                if lo and hi:
                    fv2 = _build_fv(nr, features)
                    entry["Rate_lo"] = round(float(lo.predict([fv2])[0]),2)
                    entry["Rate_hi"] = round(float(hi.predict([fv2])[0]),2)
                preds[col].append(entry)
                history[col].append(nr[col])

        results[sp] = preds
    return results

# ═══════════════════════════════════════════════════════════════
# CORRELATIONS
# ═══════════════════════════════════════════════════════════════
def compute_correlations(sp_df):
    rows = []
    for t in CONTAINERS:
        s = sp_df[t]
        rows.append({"Contenedor": t.upper(),
            "vs Brent":    round(s.corr(sp_df["Brent_Price"])             if "Brent_Price" in sp_df.columns else np.nan,2),
            "vs USD/CNY":  round(s.corr(sp_df["Exchange_Rate"])           if "Exchange_Rate" in sp_df.columns else np.nan,2),
            "vs Congest.": round(s.corr(sp_df["Port_Congestion_Interest"]) if "Port_Congestion_Interest" in sp_df.columns else np.nan,2),
        })
    return pd.DataFrame(rows)

def compute_all_correlations(weekly_df):
    return {sp: compute_correlations(g) for sp, g in weekly_df.groupby("Service provider")}

# ═══════════════════════════════════════════════════════════════
# CHART HELPERS
# ═══════════════════════════════════════════════════════════════
def _rate_chart(sp_df, container, sp, preds_xgb, metrics_xgb, hist_start):
    hist = sp_df[sp_df["Week"] >= hist_start][["Week", container]].copy()
    pred_list = preds_xgb.get(sp,{}).get(container, [])
    pred_df   = pd.DataFrame(pred_list)
    color     = _sp_color(sp)
    fig       = go.Figure()

    if not hist.empty:
        fig.add_trace(go.Scatter(x=hist["Week"], y=hist[container],
            mode="lines", line=dict(color="#95a5a6", width=2), name="Histórico"))

    if not pred_df.empty:
        pred_df["Week"] = pd.to_datetime(pred_df["Week"])
        has_q = "Rate_lo" in pred_df.columns and not pred_df["Rate_lo"].isna().all()
        r, g, b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)

        if has_q:
            fig.add_trace(go.Scatter(
                x=pd.concat([pred_df["Week"], pred_df["Week"][::-1]]),
                y=pd.concat([pred_df["Rate_hi"], pred_df["Rate_lo"][::-1]]),
                fill="toself", fillcolor=f"rgba({r},{g},{b},0.15)",
                line=dict(width=0), name="P10–P90"))
        else:
            m = metrics_xgb.get(sp,{}).get(container,{})
            rmse = float(m.get("RMSE",0.0)) if m else 0.0
            if rmse > 0:
                fig.add_trace(go.Scatter(
                    x=pd.concat([pred_df["Week"], pred_df["Week"][::-1]]),
                    y=pd.concat([pred_df["Rate"]+rmse, (pred_df["Rate"]-rmse)[::-1]]),
                    fill="toself", fillcolor="rgba(200,200,200,0.2)",
                    line=dict(width=0), name="±RMSE"))

        fig.add_trace(go.Scatter(
            x=pred_df["Week"], y=pred_df["Rate"],
            mode="lines+markers",
            line=dict(color=color, width=3),
            marker=dict(size=8, color=color),
            name="Pronóstico"))

        last_p = pred_df.iloc[-1]
        fig.add_annotation(
            x=last_p["Week"], y=last_p["Rate"],
            text=f"<b>${last_p['Rate']:,.0f}</b>",
            showarrow=True, arrowhead=2, arrowcolor=color,
            bgcolor="white", bordercolor=color, borderwidth=1,
            font=dict(size=12, color=color))

    m = metrics_xgb.get(sp,{}).get(container,{})
    subtitle = (f"RMSE ${m.get('RMSE',0):,.0f} | MAPE {m.get('MAPE',0):.1f}% | R² {m.get('R2',0):.2f}"
                if m else "Sin métricas")
    fig.update_layout(
        title=dict(text=f"<b>{container.upper()}</b> — {_sp_short(sp)}<br><sup>{subtitle}</sup>",
                   font=dict(size=14)),
        xaxis_title=None, yaxis_title="Tarifa (USD)",
        yaxis_tickprefix="$", yaxis_tickformat=",.0f",
        template=PLOTLY_TMPL, height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified", margin=dict(t=75,b=35,l=10,r=10))
    return fig

# ═══════════════════════════════════════════════════════════════
# PIPELINE RUNNER (por ruta)
# ═══════════════════════════════════════════════════════════════
def run_route_pipeline(uploaded_file, brent_ext, exchange_ext, trends_ext,
                        use_pc, use_b, use_fx_flag, enable_tuning, route_key, cb):
    """Corre todo el pipeline para una ruta. Retorna dict con todos los artefactos."""
    weekly_df = load_and_process_data(
        uploaded_file, use_pc, use_b, use_fx_flag,
        brent_ext, exchange_ext, trends_ext, cache_buster=cb)
    if weekly_df is None:
        return None

    cache_path = _get_model_cache_path(weekly_df, use_pc, use_b, use_fx_flag, enable_tuning, route_key)
    models_xgb = metrics_xgb = trained_features = None

    if os.path.exists(cache_path):
        try:
            models_xgb, metrics_xgb, trained_features = joblib.load(cache_path)
            st.sidebar.success(f"✅ Modelos {route_key} desde caché.")
        except:
            try: os.remove(cache_path)
            except: pass

    if models_xgb is None:
        models_xgb, metrics_xgb, trained_features = train_models_xgb(
            weekly_df, use_pc, use_b, use_fx_flag, enable_tuning)
        try:
            os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
            joblib.dump((models_xgb, metrics_xgb, trained_features), cache_path)
        except: pass

    preds_xgb = generate_predictions_xgb(models_xgb, weekly_df, trained_features)
    corr_all  = compute_all_correlations(weekly_df)
    providers = sorted(weekly_df["Service provider"].unique())

    return {"weekly_df": weekly_df, "models": models_xgb, "metrics": metrics_xgb,
            "features": trained_features, "preds": preds_xgb, "corrs": corr_all,
            "providers": providers, "route_key": route_key}

# ═══════════════════════════════════════════════════════════════
# RENDER: TAB RESUMEN DE RUTA
# ═══════════════════════════════════════════════════════════════
def render_resumen_tab(rd: dict, hist_start, route_label: str, route_color: str, route_key: str = ""):
    weekly_df = rd["weekly_df"]; preds_xgb = rd["preds"]
    metrics_xgb = rd["metrics"]; providers = rd["providers"]

    last_hist = weekly_df["Week"].max()
    next_pred  = last_hist + timedelta(weeks=1)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Navieras", len(providers))
    c2.metric("Semanas de historial", weekly_df.groupby("Service provider")["Week"].nunique().max())
    c3.metric("Última semana", last_hist.strftime("%d %b %Y"))
    c4.metric("Próx. pronóstico", next_pred.strftime("%d %b %Y"))

    st.markdown("---")
    st.markdown("#### Tarifas actuales vs Pronóstico (+1 semana)")

    rows = []
    for sp in providers:
        sp_last = weekly_df[weekly_df["Service provider"]==sp]
        if sp_last.empty: continue
        last = sp_last.iloc[-1]
        row  = {"Naviera": _sp_short(sp)}
        for col in CONTAINERS:
            row[f"{col.upper()} actual"] = f"${last[col]:,.0f}" if not pd.isna(last[col]) else "—"
        for col in CONTAINERS:
            p1 = preds_xgb.get(sp,{}).get(col,[])
            if p1:
                delta = p1[0]["Rate"] - last[col]
                arr, _, _ = _trend_arrow(delta)
                row[f"{col.upper()} próx."] = f"{arr} ${p1[0]['Rate']:,.0f}"
            else:
                row[f"{col.upper()} próx."] = "—"
        rows.append(row)

    if rows:
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    st.markdown("---")
    st.markdown("#### Pronóstico 22G0 — Todas las navieras (próximas 4 semanas)")
    fig_comp = go.Figure()
    for sp in providers:
        pl = preds_xgb.get(sp,{}).get("22g0",[])
        if not pl: continue
        pdf = pd.DataFrame(pl); pdf["Week"] = pd.to_datetime(pdf["Week"])
        fig_comp.add_trace(go.Scatter(x=pdf["Week"], y=pdf["Rate"],
            mode="lines+markers", name=_sp_short(sp),
            line=dict(color=_sp_color(sp), width=2), marker=dict(size=7)))
    fig_comp.update_layout(xaxis_title=None, yaxis_title="USD",
        yaxis_tickprefix="$", yaxis_tickformat=",.0f",
        template=PLOTLY_TMPL, height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified")
    st.plotly_chart(fig_comp, width='stretch', key=f"resumen_comp_{route_key}")

    st.markdown("#### Histórico de tarifas 22G0 — Todas las navieras")
    fig_all = go.Figure()
    for sp in providers:
        sp_data = weekly_df[weekly_df["Service provider"]==sp].sort_values("Week")
        fig_all.add_trace(go.Scatter(x=sp_data["Week"], y=sp_data["22g0"],
            mode="lines", name=_sp_short(sp),
            line=dict(color=_sp_color(sp), width=2), opacity=0.85))
    fig_all.update_layout(xaxis_title=None, yaxis_title="USD",
        yaxis_tickprefix="$", yaxis_tickformat=",.0f",
        template=PLOTLY_TMPL, height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified")
    st.plotly_chart(fig_all, width='stretch', key=f"resumen_all_{route_key}")

    st.markdown("#### MAPE del modelo por naviera y contenedor")
    hm_data = []
    for sp in providers:
        for tgt in CONTAINERS:
            m = metrics_xgb.get(sp,{}).get(tgt,{})
            hm_data.append({"Naviera":_sp_short(sp), "Contenedor":tgt.upper(),
                             "MAPE": round(m.get("MAPE",float("nan")),1) if m else float("nan")})
    hm_df  = pd.DataFrame(hm_data).pivot(index="Naviera", columns="Contenedor", values="MAPE")
    fig_hm = px.imshow(hm_df, text_auto=True, aspect="auto",
        color_continuous_scale="RdYlGn_r",
        labels=dict(color="MAPE %"),
        title="MAPE promedio — menor es mejor")
    fig_hm.update_layout(template=PLOTLY_TMPL, height=280)
    st.plotly_chart(fig_hm, width='stretch', key=f"resumen_hm_{route_key}")

# ═══════════════════════════════════════════════════════════════
# RENDER: TAB PROVEEDOR
# ═══════════════════════════════════════════════════════════════
def render_provider_tab(sp: str, rd: dict, enable_ai: bool, hist_start, route_key: str = ""):
    weekly_df   = rd["weekly_df"]; preds_xgb = rd["preds"]
    metrics_xgb = rd["metrics"];   corr_all  = rd["corrs"]

    sp_df  = weekly_df[weekly_df["Service provider"]==sp].copy()
    color  = _sp_color(sp)
    spm    = metrics_xgb.get(sp, {})

    st.markdown(f"""
    <div style='background:{color}18; border-left:4px solid {color};
         padding:0.6rem 1rem; border-radius:4px; margin-bottom:0.8rem;'>
      <b style='font-size:1.05rem; color:{color};'>{sp}</b>
    </div>""", unsafe_allow_html=True)

    # KPI cards
    k = st.columns(6)
    for ki, (tgt, lbl) in enumerate(zip(CONTAINERS, ["22G0","45G0","40RN"])):
        m_vals = spm.get(tgt,{})
        rmse = m_vals.get("RMSE",float("nan")) if m_vals else float("nan")
        mape = m_vals.get("MAPE",float("nan")) if m_vals else float("nan")
        r2   = m_vals.get("R2",  float("nan")) if m_vals else float("nan")
        k[ki*2].metric(f"RMSE {lbl}",   f"${rmse:,.0f}" if not np.isnan(rmse) else "—")
        k[ki*2+1].metric(f"MAPE {lbl}", f"{mape:.1f}%"  if not np.isnan(mape) else "—",
                          delta=f"R²={r2:.2f}" if not np.isnan(r2) else None,
                          delta_color="off")

    st.markdown("---")

    # Gráficos de predicción
    sp_slug = _sp_short(sp)
    st.markdown("#### Pronóstico por tipo de contenedor")
    gc1, gc2, gc3 = st.columns(3)
    for cont, gcol in zip(CONTAINERS, [gc1, gc2, gc3]):
        gcol.plotly_chart(
            _rate_chart(sp_df, cont, sp, preds_xgb, metrics_xgb, hist_start),
            width='stretch',
            key=f"pred_{route_key}_{sp_slug}_{cont}")

    # Tabla de predicciones
    st.markdown("#### Tabla de predicciones")
    pred_rows = []
    for cont in CONTAINERS:
        for p in preds_xgb.get(sp,{}).get(cont,[]):
            row = {"Contenedor": cont.upper(),
                   "Semana":     p["Week"].strftime("%Y-%m-%d"),
                   "Tarifa":     f"${p['Rate']:,.0f}"}
            if "Rate_lo" in p:
                row["P10"] = f"${p['Rate_lo']:,.0f}"
                row["P90"] = f"${p['Rate_hi']:,.0f}"
            pred_rows.append(row)

    if pred_rows:
        st.dataframe(pd.DataFrame(pred_rows), width='stretch', hide_index=True)
        # CSV directo desde pred_rows (sin re-indexar en preds_xgb)
        csv_df = pd.DataFrame([
            {"Naviera": sp,
             "Contenedor": r["Contenedor"],
             "Semana":     r["Semana"],
             "Tarifa_USD": int(r["Tarifa"].replace("$","").replace(",",""))}
            for r in pred_rows
        ])
        st.download_button("⬇️ Descargar CSV",
                           csv_df.to_csv(index=False).encode("utf-8"),
                           file_name=f"predicciones_{_sp_short(sp)}.csv",
                           mime="text/csv",
                           key=f"dl_{route_key}_{sp_slug}")
    else:
        st.warning("Sin predicciones disponibles.")

    # Macro + correlaciones
    with st.expander("📈 Señales macro e histórico", expanded=False):
        fig_macro = make_subplots(specs=[[{"secondary_y": True}]])
        for col, nm in zip(CONTAINERS, ["22G0","45G0","40RN"]):
            fig_macro.add_trace(go.Scatter(x=sp_df["Week"], y=sp_df[col], mode="lines", name=nm), secondary_y=False)
        if use_brent and "Brent_Price" in sp_df.columns:
            fig_macro.add_trace(go.Scatter(x=sp_df["Week"], y=sp_df["Brent_Price"],
                mode="lines", name="Brent $/bbl", line=dict(dash="dot")), secondary_y=True)
        if use_fx and "Exchange_Rate" in sp_df.columns:
            fig_macro.add_trace(go.Scatter(x=sp_df["Week"], y=sp_df["Exchange_Rate"],
                mode="lines", name="USD/CNY", line=dict(dash="dot")), secondary_y=True)
        if use_port_congestion and "Port_Congestion_Interest" in sp_df.columns:
            fig_macro.add_trace(go.Scatter(x=sp_df["Week"], y=sp_df["Port_Congestion_Interest"],
                mode="lines", name="Congestión (0-100)", line=dict(dash="dot")), secondary_y=True)
        fig_macro.update_layout(
            yaxis=dict(title="Tarifa (USD)", tickprefix="$", tickformat=",.0f"),
            yaxis2=dict(title="Macro"), template=PLOTLY_TMPL, height=340,
            hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_macro, width='stretch', key=f"macro_{route_key}_{sp_slug}")

        corr_df = corr_all.get(sp, pd.DataFrame())
        if not corr_df.empty:
            st.markdown("**Correlaciones Pearson**")
            def _corr_color(val):
                try:
                    v = float(val)
                    if np.isnan(v): return ""
                    # green for positive, red for negative, white at 0
                    intensity = int(abs(v) * 200)
                    if v > 0:
                        return f"background-color: rgb({255-intensity},{255},{255-intensity})"
                    else:
                        return f"background-color: rgb({255},{255-intensity},{255-intensity})"
                except: return ""
            corr_cols = [c for c in ["vs Brent","vs USD/CNY","vs Congest."] if c in corr_df.columns]
            st.dataframe(corr_df.style.map(_corr_color, subset=corr_cols),
                width='stretch', hide_index=True)

    # Feature importance
    with st.expander("🔍 Importancia de características", expanded=False):
        st.caption(
            "**¿Qué mide esto?** El modelo XGBoost asigna un puntaje (F-score) a cada variable "
            "según cuántas veces fue usada para dividir los datos al entrenar. "
            "Mayor F-score = la variable explica más la variación de la tarifa.\n\n"
            "**Grupos de variables:**\n"
            "- `_lag1/2/4` — Tarifa de 1, 2 o 4 semanas atrás (memoria del precio)\n"
            "- `_pct_chg1` — Variación porcentual semanal (velocidad del cambio)\n"
            "- `_roll4/8_mean` — Promedio móvil 4 y 8 semanas (tendencia suavizada)\n"
            "- `_roll4_std` — Volatilidad reciente (dispersión de los últimos 4 datos)\n"
            "- `_ewm_span4` — Media exponencial (da más peso a datos recientes)\n"
            "- `_macd_signal` — Diferencia EMA4–EMA8 (señal de aceleración/desaceleración)\n"
            "- `Brent_Price` — Precio del petróleo (costo de bunker)\n"
            "- `Exchange_Rate` — USD/CNY (competitividad exportadora China)\n"
            "- `Port_Congestion_Interest` — Interés en congestión portuaria (Google Trends)\n"
            "- `Week_of_Year / Month / Quarter` — Estacionalidad"
        )
        fi_c1, fi_c2, fi_c3 = st.columns(3)
        for tgt, gcol in zip(CONTAINERS, [fi_c1, fi_c2, fi_c3]):
            fi = spm.get(tgt,{}).get("feature_importances",{})
            if fi:
                fi_sorted = sorted(fi.items(), key=lambda x:x[1], reverse=True)[:12]
                fi_df = pd.DataFrame(fi_sorted, columns=["Feature","Importancia"])
                fig_fi = go.Figure(go.Bar(x=fi_df["Importancia"], y=fi_df["Feature"],
                    orientation="h", marker_color=color))
                fig_fi.update_layout(title=f"<b>{tgt.upper()}</b>",
                    xaxis_title="F-score", yaxis_autorange="reversed",
                    template=PLOTLY_TMPL, height=370, margin=dict(l=5,r=5,t=35,b=15))
                gcol.plotly_chart(fig_fi, width='stretch', key=f"fi_{route_key}_{sp_slug}_{tgt}")
            else:
                gcol.info(f"Sin importancia para {tgt.upper()}")

    # IA
    if enable_ai:
        with st.expander("🤖 Análisis IA — DeepSeek", expanded=False):
            last_brent = sp_df["Brent_Price"].iloc[-1] if "Brent_Price" in sp_df.columns and not sp_df["Brent_Price"].isna().all() else "N/A"
            last_fx    = sp_df["Exchange_Rate"].iloc[-1] if "Exchange_Rate" in sp_df.columns and not sp_df["Exchange_Rate"].isna().all() else "N/A"
            last_cong  = sp_df["Port_Congestion_Interest"].iloc[-1] if "Port_Congestion_Interest" in sp_df.columns and not sp_df["Port_Congestion_Interest"].isna().all() else "N/A"

            preds_str = "\n".join([
                f"{c.upper()}: " + ", ".join([f"{p['Week'].strftime('%d/%m/%y')}=${p['Rate']:,.0f}"
                                               for p in preds_xgb.get(sp,{}).get(c,[])])
                for c in CONTAINERS if preds_xgb.get(sp,{}).get(c)])

            trend_lines = []
            for tgt in CONTAINERS:
                pl = preds_xgb.get(sp,{}).get(tgt,[])
                if len(pl)>=2:
                    d = pl[-1]["Rate"]-pl[0]["Rate"]
                    trend_lines.append(f"  {tgt.upper()}: {'Alcista' if d>50 else 'Bajista' if d<-50 else 'Estable'} (Δ${d:+,.0f})")

            metrics_lines = [
                f"  {tgt.upper()}: RMSE=${spm.get(tgt,{}).get('RMSE',0):,.0f}, "
                f"MAPE={spm.get(tgt,{}).get('MAPE',0):.1f}%, R²={spm.get(tgt,{}).get('R2',0):.2f}"
                for tgt in CONTAINERS if spm.get(tgt)]

            ai_c1, ai_c2 = st.columns(2)
            with ai_c1:
                st.markdown("**Tendencias y recomendaciones**")
                with st.spinner("Analizando..."):
                    prompt = (
                        f"Freight marítimo Shanghai→{_sp_short(sp)}. {pd.Timestamp.today().strftime('%b %Y')}.\n"
                        f"Métricas: {' | '.join(metrics_lines)}\n"
                        f"Macro: Brent={last_brent} | USD/CNY={last_fx} | Congestión={last_cong}\n"
                        f"Forecast: {preds_str}\n"
                        f"Tendencias: {' | '.join(trend_lines)}\n"
                        f"Responde en español, máx 200 palabras:\n"
                        f"1. Clasificación 22G0/45G0/40RN (Alcista/Estable/Bajista) + driver.\n"
                        f"2. Confianza del modelo (RMSE/MAPE).\n"
                        f"3. Una recomendación de procurement."
                    )
                    st.markdown(call_ai(prompt))

            with ai_c2:
                st.markdown("**Factores de riesgo**")
                rmse_avg = np.mean([v.get("RMSE",0) for v in spm.values() if isinstance(v,dict) and "RMSE" in v]) if spm else 0
                with st.spinner("Analizando..."):
                    prompt2 = (
                        f"Mercado marítimo Asia-LATAM. Naviera {_sp_short(sp)}. {pd.Timestamp.today().strftime('%b %Y')}.\n"
                        f"Forecast próx 4 sem: {preds_str}\n"
                        f"Lista 4 factores clave que impactarán tarifas Shanghai→{_sp_short(sp)}.\n"
                        f"Formato tabla: Factor | Impacto | Dirección. Sin introducción. En español."
                    )
                    st.markdown(call_ai(prompt2))

# ═══════════════════════════════════════════════════════════════
# RENDER: TAB COMPARATIVA BUN vs CTG
# ═══════════════════════════════════════════════════════════════
def render_comparativa_tab(route_data: dict):
    """Muestra análisis cruzado entre las dos rutas."""
    active = {k: v for k, v in route_data.items() if v is not None}
    if len(active) < 2:
        st.info("Sube los archivos de **ambas rutas** para ver la comparativa.")
        return

    bun = active.get("BUN"); ctg = active.get("CTG")
    st.markdown("### Comparativa Shanghai → Buenaventura vs Cartagena")
    st.caption("Análisis cruzado de tarifas, tendencias y diferencial entre rutas.")

    # ── Diferencial de precio por contenedor ──
    st.markdown("#### Diferencial de tarifas actuales (CTG − BUN)")
    diff_rows = []
    all_sp = set(bun["providers"]) | set(ctg["providers"])
    for sp in sorted(all_sp):
        row = {"Naviera": _sp_short(sp)}
        for col in CONTAINERS:
            bun_last = bun["weekly_df"][bun["weekly_df"]["Service provider"]==sp][col].iloc[-1] \
                       if sp in bun["providers"] else None
            ctg_last = ctg["weekly_df"][ctg["weekly_df"]["Service provider"]==sp][col].iloc[-1] \
                       if sp in ctg["providers"] else None
            if bun_last is not None and ctg_last is not None and not pd.isna(bun_last) and not pd.isna(ctg_last):
                diff = ctg_last - bun_last
                arr, _, _ = _trend_arrow(diff, threshold=20)
                row[f"{col.upper()} BUN"]  = f"${bun_last:,.0f}"
                row[f"{col.upper()} CTG"]  = f"${ctg_last:,.0f}"
                row[f"{col.upper()} Δ"]    = f"{arr} ${diff:+,.0f}"
            else:
                row[f"{col.upper()} BUN"] = f"${bun_last:,.0f}" if bun_last is not None else "—"
                row[f"{col.upper()} CTG"] = f"${ctg_last:,.0f}" if ctg_last is not None else "—"
                row[f"{col.upper()} Δ"]   = "—"
        diff_rows.append(row)

    if diff_rows:
        st.dataframe(pd.DataFrame(diff_rows), width='stretch', hide_index=True)

    st.markdown("---")

    # ── Gráfico comparativo de pronóstico por contenedor ──
    st.markdown("#### Pronóstico 4 semanas — BUN vs CTG por contenedor")
    for col in CONTAINERS:
        fig = go.Figure()
        for rkey, rd, style in [("BUN", bun, "solid"), ("CTG", ctg, "dash")]:
            for sp in rd["providers"]:
                pl = rd["preds"].get(sp,{}).get(col,[])
                if not pl: continue
                pdf = pd.DataFrame(pl); pdf["Week"] = pd.to_datetime(pdf["Week"])
                fig.add_trace(go.Scatter(
                    x=pdf["Week"], y=pdf["Rate"],
                    mode="lines+markers",
                    name=f"{_sp_short(sp)} ({rkey})",
                    line=dict(color=_sp_color(sp), width=2,
                              dash=style),
                    marker=dict(size=6,
                                symbol="circle" if rkey=="BUN" else "diamond")))
        fig.update_layout(
            title=f"<b>{col.upper()}</b> — Pronóstico BUN (sólido) vs CTG (punteado)",
            xaxis_title=None, yaxis_title="USD",
            yaxis_tickprefix="$", yaxis_tickformat=",.0f",
            template=PLOTLY_TMPL, height=360,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            hovermode="x unified")
        st.plotly_chart(fig, width='stretch')

    st.markdown("---")

    # ── Tendencia histórica BUN vs CTG (navieras en común) ──
    common_sp = sorted(set(bun["providers"]) & set(ctg["providers"]))
    if common_sp:
        st.markdown("#### Histórico 22G0 — Navieras en común")
        fig_hist = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Buenaventura (BUN)", "Cartagena (CTG)"))
        for sp in common_sp:
            color = _sp_color(sp)
            for ri, rd in enumerate([bun, ctg], start=1):
                sp_data = rd["weekly_df"][rd["weekly_df"]["Service provider"]==sp].sort_values("Week")
                fig_hist.add_trace(go.Scatter(
                    x=sp_data["Week"], y=sp_data["22g0"],
                    mode="lines", name=_sp_short(sp) if ri==1 else None,
                    line=dict(color=color, width=2),
                    showlegend=(ri==1)), row=1, col=ri)
        fig_hist.update_yaxes(tickprefix="$", tickformat=",.0f")
        fig_hist.update_layout(template=PLOTLY_TMPL, height=380, hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_hist, width='stretch')

    # ── Heatmap MAPE comparativo ──
    st.markdown("#### MAPE del modelo — BUN vs CTG")
    hm_cols = st.columns(2)
    for ci, (rkey, rd) in enumerate([("BUN", bun), ("CTG", ctg)]):
        with hm_cols[ci]:
            hm_data = []
            for sp in rd["providers"]:
                for tgt in CONTAINERS:
                    m = rd["metrics"].get(sp,{}).get(tgt,{})
                    hm_data.append({"Naviera":_sp_short(sp),"Contenedor":tgt.upper(),
                                     "MAPE": round(m.get("MAPE",float("nan")),1) if m else float("nan")})
            hm_df = pd.DataFrame(hm_data).pivot(index="Naviera",columns="Contenedor",values="MAPE")
            fig_hm = px.imshow(hm_df, text_auto=True, aspect="auto",
                color_continuous_scale="RdYlGn_r",
                title=f"MAPE {rkey} (%) — menor = mejor")
            fig_hm.update_layout(template=PLOTLY_TMPL, height=280)
            st.plotly_chart(fig_hm, width='stretch')

# ═══════════════════════════════════════════════════════════════
# ── UI PRINCIPAL ─────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════
st.markdown(f"""
<div style='text-align:center; padding:0.4rem 0 1rem 0;'>
  <span style='font-size:1.9rem; font-weight:700; color:#1a1a2e;'>📦 {APP_TITLE}</span><br>
  <span style='color:#555; font-size:0.9rem;'>Predicción con XGBoost + señales macro | BUN &amp; CTG</span>
</div>
""", unsafe_allow_html=True)

# ── File uploaders ─────────────────────────────────────────────
up_col1, up_col2, ai_col = st.columns([2, 2, 1])
with up_col1:
    st.markdown("🟠 **Buenaventura (BUN)**")
    uploaded_bun = st.file_uploader("Archivo BUN", type=["xlsx"], key="f_bun", label_visibility="collapsed")
with up_col2:
    st.markdown("🔵 **Cartagena (CTG)**")
    uploaded_ctg = st.file_uploader("Archivo CTG", type=["xlsx"], key="f_ctg", label_visibility="collapsed")
with ai_col:
    st.markdown(" ")
    enable_ai = st.toggle("Análisis IA (DeepSeek)", value=True)

if not uploaded_bun and not uploaded_ctg:
    st.info("👆 Sube al menos un archivo de tarifas para comenzar.")
    st.stop()

# ── Carga paralela de datos externos ─────────────────────────
with st.spinner("Cargando datos externos..."):
    with ThreadPoolExecutor(max_workers=3) as pool:
        fb = pool.submit(get_brent_data_combined, HIST_START, HIST_END,
                          ALPHA_API_KEY, EIA_API_KEY, debug_mode,
                          st.session_state.refresh_token) if use_brent else None
        ff = pool.submit(get_fx_data_combined, HIST_START, HIST_END,
                          ALPHA_API_KEY, EX_API_KEY, debug_mode,
                          st.session_state.refresh_token) if use_fx else None
        ft = pool.submit(load_google_trends_data_auto) if use_port_congestion else None

    brent_ext    = fb.result() if fb else pd.DataFrame({"Week":[],"Brent_Price":[]})
    exchange_ext = ff.result() if ff else pd.DataFrame({"Week":[],"Exchange_Rate":[]})
    trends_ext   = ft.result() if ft else pd.DataFrame({"Week":[],"Port_Congestion_Interest":[]})

# ── Pipeline por ruta ─────────────────────────────────────────
route_data = {}
cb = st.session_state.refresh_token

if uploaded_bun:
    with st.spinner("Procesando ruta BUN..."):
        route_data["BUN"] = run_route_pipeline(
            uploaded_bun, brent_ext, exchange_ext, trends_ext,
            use_port_congestion, use_brent, use_fx,
            enable_hparam_tuning, "BUN", cb)
else:
    route_data["BUN"] = None

if uploaded_ctg:
    with st.spinner("Procesando ruta CTG..."):
        route_data["CTG"] = run_route_pipeline(
            uploaded_ctg, brent_ext, exchange_ext, trends_ext,
            use_port_congestion, use_brent, use_fx,
            enable_hparam_tuning, "CTG", cb)
else:
    route_data["CTG"] = None

hist_chart_start = pd.to_datetime(HIST_END) - timedelta(weeks=hist_weeks)

# ── Construcción de tabs ──────────────────────────────────────
tab_labels = ["🌐 Comparativa BUN vs CTG"]
for rkey, rinfo in ROUTES.items():
    rd = route_data.get(rkey)
    if rd is None: continue
    tab_labels.append(f"{rinfo['emoji']} {rinfo['label']} — Resumen")
    for sp in rd["providers"]:
        tab_labels.append(f"{rinfo['emoji']} {rinfo['label']} — {_sp_short(sp)}")

main_tabs = st.tabs(tab_labels)
tab_idx   = 0

# Tab Comparativa
with main_tabs[tab_idx]:
    render_comparativa_tab(route_data)
tab_idx += 1

# Tabs por ruta y proveedor
for rkey, rinfo in ROUTES.items():
    rd = route_data.get(rkey)
    if rd is None: continue

    # Resumen de ruta
    with main_tabs[tab_idx]:
        render_resumen_tab(rd, hist_chart_start, rinfo["label"], rinfo["color"], route_key=rkey)
    tab_idx += 1

    # Tab por proveedor
    for sp in rd["providers"]:
        with main_tabs[tab_idx]:
            render_provider_tab(sp, rd, enable_ai, hist_chart_start, route_key=rkey)
        tab_idx += 1
