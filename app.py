import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import requests
import warnings
import time
from pytrends.request import TrendReq
from g4f.client import Client  # gpt4free client

warnings.filterwarnings("ignore")

# ========= CONFIG =========
DEFAULT_ALPHA_API_KEY = "3OWZCTVQY381I6B1"
DEFAULT_EIA_API_KEY = "GDsNZWWgRGr4axJQrofEreD7epXOfVgUtbWLJ0Pa"
DEFAULT_EX_API_KEY = "0afae24b7df33abea3d76688"
APP_TITLE = "Shipping Rate Predictor: Shanghai → Buenaventura"
HIST_START = "2022-01-01"
HIST_END = pd.Timestamp.today().strftime("%Y-%m-%d")
FORECAST_WEEKS = 4
AI_MODELS = ["gpt-4o-mini", "gemini-2.5-flash", "grok-3", "deepseek-v3"]
# ==========================

st.set_page_config(page_title=APP_TITLE, layout="wide")

# ===== Sidebar config =====
st.sidebar.header("Configuración")
alpha_key_input = st.sidebar.text_input("Alpha Vantage API Key", value=DEFAULT_ALPHA_API_KEY, type="password")
eia_key_input = st.sidebar.text_input("EIA API Key (Brent)", value=DEFAULT_EIA_API_KEY, type="password")
ex_key_input = st.sidebar.text_input("ExchangeRate API Key (USD/CNY)", value=DEFAULT_EX_API_KEY, type="password")
ALPHA_API_KEY = alpha_key_input.strip() or DEFAULT_ALPHA_API_KEY
EIA_API_KEY = eia_key_input.strip() or DEFAULT_EIA_API_KEY
EX_API_KEY = ex_key_input.strip() or DEFAULT_EX_API_KEY

# Toggles de señales exógenas
a, b, c = st.sidebar.columns(3)
with a:
    use_port_congestion = st.toggle("Trends", value=True, help="Google Trends: 'port congestion'")
with b:
    use_brent = st.toggle("Brent", value=True)
with c:
    use_fx = st.toggle("USD/CNY", value=True)

# Debug y refresco de caché
debug_mode = st.sidebar.checkbox("Debug mode (mostrar URLs)", value=False)
if "refresh_token" not in st.session_state:
    st.session_state.refresh_token = 0
if st.sidebar.button("🔄 Forzar refresco de datos (limpiar caché)"):
    st.cache_data.clear()
    st.session_state.refresh_token += 1
    st.sidebar.success("Caché limpiada. Vuelve a ejecutar.")

# ===== IA (gpt4free) =====
def call_ai_with_fallback(prompt, model_list=AI_MODELS):
    """
    Intenta IA con fallback entre modelos gpt4free.
    Filtra respuestas que pidan 'key' o 'login'. Devuelve texto con el modelo usado.
    """
    for idx, model in enumerate(model_list):
        try:
            if idx > 0:
                st.warning(f"El modelo anterior falló o no fue válido. Probando con `{model}`…")
            client = Client()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            text = (response.choices[0].message.content or "").strip()
            if not text:
                raise Exception("Respuesta de IA vacía.")
            low = text.lower()
            if "key" in low or "login" in low:
                st.warning(f"`{model}` devolvió un mensaje de credenciales. Probando siguiente modelo…")
                continue
            return f"**Model:** {model}\n\n{text}"
        except Exception as e:
            print(f"Modelo {model} falló con error: {e}")
            continue
    return "❌ Todos los modelos de IA fallaron. Por favor, inténtalo de nuevo más tarde."

# ===== AlphaVantage helper con reintentos =====
def _alpha_get_json(url, tries=5, sleep_seconds=15):
    """Llama AlphaVantage con reintentos cuando devuelve 'Note' (rate limit) o mensajes vacíos."""
    last_err = None
    for _ in range(tries):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and ("Note" in data or "Information" in data or "Error Message" in data):
                last_err = data.get("Note") or data.get("Information") or data.get("Error Message")
                time.sleep(sleep_seconds)
                continue
            return data
        except Exception as e:
            last_err = str(e)
            time.sleep(3)
    raise ValueError(f"AlphaVantage respondió vacío o con límite. Último error: {last_err}")

# ========= Data Sources =========
@st.cache_data(ttl=3600)
def get_brent_data_eia(start_date, end_date, api_key: str, debug: bool = False, cache_buster: int = 0):
    """Brent semanal desde EIA v2 (petroleum/pri/spt, serie semanal RBRTE)."""
    try:
        start_s = pd.to_datetime(start_date).strftime('%Y-%m-%d')
        end_s = pd.to_datetime(end_date).strftime('%Y-%m-%d')
        url = (
            "https://api.eia.gov/v2/petroleum/pri/spt/data/"
            f"?api_key={api_key}&frequency=weekly&data[0]=value&facets[series][]=RBRTE"
            f"&start={start_s}&end={end_s}"
        )
        if debug:
            st.info(f"EIA v2 URL: {url}")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        rows = data.get('response', {}).get('data', [])
        if not rows:
            raise ValueError("EIA v2 sin datos (response.data vacío)")
        df = pd.DataFrame(rows)
        # Detectar columna fecha
        date_col = None
        for c in ['period', 'date', 'week', 'periodStart']:
            if c in df.columns:
                date_col = c
                break
        if date_col is None:
            raise ValueError(f"No se encontró columna de fecha en EIA v2. Columns: {list(df.columns)}")
        df['Week'] = pd.to_datetime(df[date_col], errors='coerce')
        df['Brent_Price'] = pd.to_numeric(df.get('value'), errors='coerce')
        df = df.dropna(subset=['Week', 'Brent_Price'])
        df = df[['Week', 'Brent_Price']].sort_values('Week')
        df['Week'] = df['Week'].dt.to_period('W-MON').dt.start_time
        df = df.groupby('Week', as_index=False)['Brent_Price'].mean()
        return df
    except Exception as e:
        st.error(f"Error fetching Brent from EIA v2: {e}")
        return pd.DataFrame({'Week': [], 'Brent_Price': []})

@st.cache_data(ttl=3600)
def get_brent_data_alpha(start_date, end_date, api_key: str):
    """Brent semanal desde AlphaVantage (función BRENT)."""
    url = f"https://www.alphavantage.co/query?function=BRENT&interval=weekly&apikey={api_key}"
    data = _alpha_get_json(url)
    rows = data.get("data", [])
    if not rows:
        raise ValueError("Respuesta vacía en 'data' de BRENT.")
    df = pd.DataFrame(rows)
    df['Week'] = pd.to_datetime(df['date'])
    df['Brent_Price'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['Week', 'Brent_Price'])
    df = df[(df['Week'] >= pd.to_datetime(start_date)) & (df['Week'] <= pd.to_datetime(end_date))]
    df = df[['Week', 'Brent_Price']].sort_values('Week').reset_index(drop=True)
    df['Week'] = df['Week'].dt.to_period('W-MON').dt.start_time
    return df

@st.cache_data(ttl=3600)
def get_brent_data_combined(start_date, end_date, alpha_key: str, eia_key: str, debug: bool = False, cache_buster: int = 0):
    """Intenta EIA v2 primero; si falla o viene vacío, usa AlphaVantage."""
    df_eia = get_brent_data_eia(start_date, end_date, eia_key, debug=debug, cache_buster=cache_buster)
    if df_eia is not None and not df_eia.empty:
        return df_eia
    st.warning("EIA v2 no devolvió datos válidos. Intentando AlphaVantage…")
    try:
        return get_brent_data_alpha(start_date, end_date, alpha_key)
    except Exception as e:
        st.error(f"AlphaVantage Brent también falló: {e}")
        return pd.DataFrame({'Week': [], 'Brent_Price': []})

# === USD/CNY ===
@st.cache_data(ttl=1800)
def get_exchange_data_exchangerate_pair(start_date, end_date, api_key: str):
    """Obtiene USD/CNY vía ExchangeRate API /pair y lo esparce semanalmente en el rango."""
    url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/USD/CNY"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    if data.get("result") != "success":
        raise ValueError(data.get("error-type", "respuesta no exitosa"))
    rate = data.get("conversion_rate")
    if rate is None:
        raise ValueError("No se encontró conversion_rate en la respuesta")
    weeks = pd.date_range(start=start_date, end=end_date, freq='W-MON')
    return pd.DataFrame({'Week': weeks, 'Exchange_Rate': float(rate)})

@st.cache_data(ttl=3600)
def get_exchange_data_alpha(start_date, end_date, api_key: str):
    """USD/CNY semanal desde AlphaVantage (FX_WEEKLY o FX_DAILY→semanal)."""
    url_w = f"https://www.alphavantage.co/query?function=FX_WEEKLY&from_symbol=USD&to_symbol=CNY&apikey={api_key}"
    data_w = _alpha_get_json(url_w)
    weekly = data_w.get("Time Series FX (Weekly)", {})
    if weekly:
        dfw = pd.DataFrame([
            {"Week": pd.to_datetime(k), "Exchange_Rate": float(v.get("4. close", "nan"))}
            for k, v in weekly.items()
        ])
        dfw = dfw.dropna(subset=['Week', 'Exchange_Rate'])
        dfw = dfw[(dfw['Week'] >= pd.to_datetime(start_date)) & (dfw['Week'] <= pd.to_datetime(end_date))]
        dfw = dfw.sort_values('Week').reset_index(drop=True)
        dfw['Week'] = dfw['Week'].dt.to_period('W-MON').dt.start_time
        return dfw
    # Fallback diario
    url_d = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol=USD&to_symbol=CNY&apikey={api_key}&outputsize=full"
    data_d = _alpha_get_json(url_d)
    daily = data_d.get("Time Series FX (Daily)", {})
    if not daily:
        raise ValueError("Respuesta vacía también en FX_DAILY.")
    dfd = pd.DataFrame([
        {"Date": pd.to_datetime(k), "Close": float(v.get("4. close", "nan"))}
        for k, v in daily.items()
    ])
    dfd = dfd.dropna(subset=['Date', 'Close']).set_index('Date').sort_index()
    dfw2 = dfd['Close'].resample('W-MON').mean().reset_index()
    dfw2 = dfw2.rename(columns={'Date': 'Week', 'Close': 'Exchange_Rate'})
    return dfw2

@st.cache_data(ttl=3600)
def get_fx_data_combined(start_date, end_date, alpha_key: str, ex_key: str, debug: bool = False, cache_buster: int = 0):
    """Obtiene USD/CNY: intenta primero ExchangeRate API /pair, luego AlphaVantage, y como última opción v6.exchangerate-api.com."""
    try:
        # Opción 1: API principal ExchangeRate /pair
        return get_exchange_data_exchangerate_pair(start_date, end_date, ex_key)
    except Exception as e1:
        st.warning(f"ExchangeRate API falló ({e1}). Intentando AlphaVantage…")
        try:
            # Opción 2: AlphaVantage
            return get_exchange_data_alpha(start_date, end_date, alpha_key)
        except Exception as e2:
            st.warning(f"AlphaVantage API falló ({e2}). Intentando v6.exchangerate-api.com…")
            try:
                # Opción 3: endpoint directo v6.exchangerate-api.com
                url = f"https://v6.exchangerate-api.com/v6/{ex_key}/pair/USD/CNY"
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                data = r.json()
                if data.get("result") != "success":
                    raise ValueError(data.get("error-type", "Respuesta no exitosa en v6.exchangerate-api.com"))
                rate = data.get("conversion_rate")
                if rate is None:
                    raise ValueError("No se encontró 'conversion_rate' en la respuesta de v6.exchangerate-api.com")
                weeks = pd.date_range(start=start_date, end=end_date, freq='W-MON')
                return pd.DataFrame({'Week': weeks, 'Exchange_Rate': float(rate)})
            except Exception as e3:
                st.error(f"v6.exchangerate-api.com también falló ({e3}). No se pudo obtener tipo de cambio.")
                return pd.DataFrame({'Week': [], 'Exchange_Rate': []})

@st.cache_data(ttl=3600)
def load_google_trends_data_auto():
    """Google Trends semanal para 'port congestion'."""
    try:
        pytrends = TrendReq(hl='en-US', tz=0)
        pytrends.build_payload(
            kw_list=["port congestion"],
            timeframe=f'{HIST_START} {pd.Timestamp.today().strftime("%Y-%m-%d")}',
            geo=''
        )
        trends_df = pytrends.interest_over_time().reset_index()
        if 'isPartial' in trends_df.columns:
            trends_df = trends_df.drop(columns=['isPartial'])
        trends_df.columns = ['Week', 'Port_Congestion_Interest']
        trends_df['Week'] = pd.to_datetime(trends_df['Week']).dt.to_period('W-MON').dt.start_time
        trends_df = trends_df.groupby('Week')['Port_Congestion_Interest'].mean().reset_index()
        return trends_df
    except Exception as e:
        st.error(f"Error al traer Google Trends automáticamente: {e}")
        return pd.DataFrame({'Week': [], 'Port_Congestion_Interest': []})

# ========= Utility =========
def excel_serial_to_date(value):
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, (int, float)):
        try:
            return pd.to_datetime('1899-12-30') + pd.to_timedelta(int(value), 'D')
        except Exception:
            pass
    try:
        return pd.to_datetime(value, errors='coerce')
    except Exception:
        return pd.NaT

# ========= Data Processing =========
@st.cache_data
def load_and_process_data(
    uploaded_file,
    use_port_congestion,
    use_brent,
    use_fx,
    alpha_key: str,
    eia_key: str,
    ex_key: str,
    debug: bool = False,
    cache_buster: int = 0
):
    if uploaded_file is None:
        st.error("Por favor sube el archivo History_rates.xlsx.")
        return None, None, None, None

    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error al cargar el archivo subido: {e}")
        return None, None, None, None

    required_cols = ['Duration from', 'Service provider', '22g0', '45g0', '40rn']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Faltan columnas en el archivo subido: {missing_cols}")
        return None, None, None, None

    # Fechas
    df['Duration from'] = df['Duration from'].apply(excel_serial_to_date)
    if df['Duration from'].isna().all():
        df['Duration from'] = pd.date_range(start=HIST_START, periods=len(df), freq='D')

    # NA básicos
    for col in ['22g0', '45g0']:
        df[col] = df[col].fillna(df[col].median())
    df['40rn'] = df['40rn'].fillna(df['45g0'] * 1.1)

    # Semana
    df['Week'] = pd.to_datetime(df['Duration from']).dt.to_period('W-MON').dt.start_time

    # Agregación semanal por proveedor
    weekly_df = df.groupby(['Week', 'Service provider'])[['22g0', '45g0', '40rn']].mean().reset_index()

    # Congestión (Google Trends)
    if use_port_congestion:
        trends_data = load_google_trends_data_auto()
        weekly_df = pd.merge(weekly_df, trends_data, on='Week', how='left')
        weekly_df['Port_Congestion_Interest'] = weekly_df['Port_Congestion_Interest'].fillna(0)
    else:
        weekly_df['Port_Congestion_Interest'] = 0
        trends_data = None

    # Brent y FX (con toggles y fallbacks)
    if use_brent:
        brent = get_brent_data_combined(HIST_START, HIST_END, alpha_key, eia_key, debug=debug, cache_buster=cache_buster)
        weekly_df = pd.merge(weekly_df, brent, on='Week', how='left')
    else:
        brent = pd.DataFrame({'Week': weekly_df['Week'].unique(), 'Brent_Price': 0})
        weekly_df = pd.merge(weekly_df, brent, on='Week', how='left')

    if use_fx:
        exchange = get_fx_data_combined(HIST_START, HIST_END, alpha_key, ex_key, debug=debug, cache_buster=cache_buster)
        weekly_df = pd.merge(weekly_df, exchange, on='Week', how='left')
    else:
        exchange = pd.DataFrame({'Week': weekly_df['Week'].unique(), 'Exchange_Rate': 0})
        weekly_df = pd.merge(weekly_df, exchange, on='Week', how='left')

    # Imputación de NA
    if 'Brent_Price' in weekly_df.columns and weekly_df['Brent_Price'].isna().any():
        weekly_df['Brent_Price'] = weekly_df['Brent_Price'].fillna(0 if not use_brent else weekly_df['Brent_Price'].median())
    if 'Exchange_Rate' in weekly_df.columns and weekly_df['Exchange_Rate'].isna().any():
        weekly_df['Exchange_Rate'] = weekly_df['Exchange_Rate'].fillna(0 if not use_fx else weekly_df['Exchange_Rate'].median())

    # Features temporales
    weekly_df['Week_of_Year'] = weekly_df['Week'].dt.isocalendar().week.astype(int)
    weekly_df['Month'] = weekly_df['Week'].dt.month.astype(int)

    # Orden por proveedor y semana
    weekly_df = weekly_df.sort_values(['Service provider', 'Week']).reset_index(drop=True)

    # Lags por proveedor (1, 2, 4) + variaciones % + lags exógenas
    for sp, grp in weekly_df.groupby('Service provider'):
        idx = grp.index
        for col in ['22g0', '45g0', '40rn']:
            weekly_df.loc[idx, f'{col}_lag1'] = grp[col].shift(1)
            weekly_df.loc[idx, f'{col}_lag2'] = grp[col].shift(2)
            weekly_df.loc[idx, f'{col}_lag4'] = grp[col].shift(4)
            weekly_df.loc[idx, f'{col}_pct_chg1'] = grp[col].pct_change(1)
        for col in ['Brent_Price', 'Exchange_Rate', 'Port_Congestion_Interest']:
            weekly_df.loc[idx, f'{col}_lag1'] = grp[col].shift(1)

    # Relleno de lags con mediana por proveedor
    lag_cols = [c for c in weekly_df.columns if c.endswith(('lag1', 'lag2', 'lag4')) or c.endswith('pct_chg1')]
    for col in lag_cols:
        med = weekly_df.groupby('Service provider')[col].transform('median')
        weekly_df[col] = weekly_df[col].fillna(med)
        weekly_df[col] = weekly_df[col].fillna(0)

    return weekly_df, brent, exchange, trends_data

# ========= Model Training (XGB) =========
@st.cache_data
def train_models_xgb(df, use_port_congestion, use_brent, use_fx):
    # features dinámicas según toggles
    features = [
        'Week_of_Year', 'Month',
        '22g0_lag1', '22g0_lag2', '22g0_lag4', '22g0_pct_chg1',
        '45g0_lag1', '45g0_lag2', '45g0_lag4', '45g0_pct_chg1',
        '40rn_lag1', '40rn_lag2', '40rn_lag4', '40rn_pct_chg1'
    ]
    if use_brent:
        features += ['Brent_Price', 'Brent_Price_lag1']
    if use_fx:
        features += ['Exchange_Rate', 'Exchange_Rate_lag1']
    if use_port_congestion:
        features += ['Port_Congestion_Interest', 'Port_Congestion_Interest_lag1']

    targets = ['22g0', '45g0', '40rn']

    service_providers = df['Service provider'].unique()
    models, metrics = {}, {}

    for sp in service_providers:
        sp_df = df[df['Service provider'] == sp].copy()
        if len(sp_df) < 12:
            continue

        sp_models, sp_metrics = {}, {}
        for target in targets:
            X = sp_df[features]
            y = sp_df[target]
            if len(X) < 10:
                continue

            split = int(len(X) * 0.8)
            split = max(5, min(split, len(X) - 1))

            model = XGBRegressor(
                max_depth=4, learning_rate=0.08, n_estimators=260,
                subsample=0.9, colsample_bytree=0.9, random_state=42
            )
            model.fit(X.iloc[:split], y.iloc[:split])
            y_pred = model.predict(X.iloc[split:])

            rmse = float(np.sqrt(mean_squared_error(y.iloc[split:], y_pred)))
            mape = float(np.mean(np.abs((y.iloc[split:] - y_pred) / y.iloc[split:])) * 100)
            r2 = float(r2_score(y.iloc[split:], y_pred))

            sp_models[target] = model
            resid_std = float(np.std(y.iloc[split:] - y_pred, ddof=1))
            sp_metrics[target] = {"RMSE": rmse, "MAPE": mape, "R2": r2, "RESID_STD": resid_std}

        if sp_models:
            models[sp] = sp_models
            metrics[sp] = sp_metrics

    return models, metrics

# ========= Predictions (XGB) =========
def generate_predictions_xgb(models, df, use_port_congestion=True, use_brent=True, use_fx=True, forecast_weeks=FORECAST_WEEKS):
    results = {}
    last_date = df['Week'].max()
    for sp in df['Service provider'].unique():
        sp_models = models.get(sp, {})
        if not sp_models:
            continue
        preds = {}
        sp_df = df[df['Service provider'] == sp].copy()
        if sp_df.empty:
            continue
        last_row = sp_df.iloc[-1]

        for target, model in sp_models.items():
            preds_list = []
            for i in range(1, forecast_weeks + 1):
                date = last_date + timedelta(weeks=i)
                feat = [
                    date.isocalendar().week,
                    date.month,
                    last_row['22g0_lag1'], last_row['22g0_lag2'], last_row['22g0_lag4'], last_row['22g0_pct_chg1'],
                    last_row['45g0_lag1'], last_row['45g0_lag2'], last_row['45g0_lag4'], last_row['45g0_pct_chg1'],
                    last_row['40rn_lag1'], last_row['40rn_lag2'], last_row['40rn_lag4'], last_row['40rn_pct_chg1']
                ]
                if use_brent:
                    feat += [last_row['Brent_Price'], last_row['Brent_Price_lag1']]
                if use_fx:
                    feat += [last_row['Exchange_Rate'], last_row['Exchange_Rate_lag1']]
                if use_port_congestion:
                    feat += [last_row['Port_Congestion_Interest'], last_row.get('Port_Congestion_Interest_lag1', 0.0)]

                pred = float(model.predict([feat])[0])
                preds_list.append({"Week": date, "Rate": round(pred, 2)})
            preds[target] = preds_list

        results[sp] = preds
    return results

# ========= Correlations =========
def compute_correlations(sp_df):
    rows = []
    for target in ['22g0', '45g0', '40rn']:
        series = sp_df[target]
        corr_brent = series.corr(sp_df['Brent_Price']) if 'Brent_Price' in sp_df else np.nan
        corr_fx = series.corr(sp_df['Exchange_Rate']) if 'Exchange_Rate' in sp_df else np.nan
        corr_trends = series.corr(sp_df['Port_Congestion_Interest']) if 'Port_Congestion_Interest' in sp_df else np.nan
        rows.append({
            'Target': target.upper(),
            'Corr vs Brent': corr_brent,
            'Corr vs USD/CNY': corr_fx,
            'Corr vs Port Congestion': corr_trends
        })
    return pd.DataFrame(rows)

# ========= UI =========
st.markdown(f"<h2 style='text-align:center;margin-top:0'>{APP_TITLE}</h2>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Sube el archivo History_rates.xlsx", type=["xlsx"])
enable_ai = st.toggle("Generar análisis con IA (gpt4free)", value=True)

if uploaded_file:
    weekly_df, brent, exchange, trends_data = load_and_process_data(
        uploaded_file, use_port_congestion, use_brent, use_fx,
        ALPHA_API_KEY, EIA_API_KEY, EX_API_KEY,
        debug=debug_mode, cache_buster=st.session_state.refresh_token
    )
    if weekly_df is None:
        st.stop()

    # Entrenamiento (solo XGB)
    models_xgb, metrics_xgb = train_models_xgb(weekly_df, use_port_congestion, use_brent, use_fx)

    # Predicciones (solo XGB)
    preds_xgb = generate_predictions_xgb(models_xgb, weekly_df, use_port_congestion, use_brent, use_fx)

    # Proveedores
    service_providers = sorted(weekly_df['Service provider'].unique())
    if not service_providers:
        st.warning("No hay service providers en el archivo subido.")
        st.stop()

    tabs = st.tabs(service_providers)

    # Rango histórico para gráfico combinado
    forecast_start_date = pd.to_datetime(HIST_END)
    historical_start_date = forecast_start_date - timedelta(weeks=8)

    for idx, sp in enumerate(service_providers):
        with tabs[idx]:
            st.subheader(f"{sp}")

            # 1) Performance (XGB)
            st.markdown("**Desempeño del modelo (Validación)**")
            spm = metrics_xgb.get(sp, {})
            if spm:
                for tname, vals in spm.items():
                    st.write(f"- **{tname.upper()}** → RMSE: {vals['RMSE']:.2f} | MAPE: {vals['MAPE']:.2f}% | R²: {vals['R2']:.3f}")
            else:
                st.info("Sin métricas XGB (pocos datos o no entrenó).")

            # 2) Correlaciones
            st.markdown("**Correlaciones (Pearson)**")
            sp_df = weekly_df[weekly_df['Service provider'] == sp].copy()
            corr_df = compute_correlations(sp_df)
            st.dataframe(
                corr_df.style.format({
                    'Corr vs Brent': "{:.2f}",
                    'Corr vs USD/CNY': "{:.2f}",
                    'Corr vs Port Congestion': "{:.2f}"
                }),
                use_container_width=True,
                hide_index=True
            )

            # 3) Predicciones (tabla + descarga)
            st.markdown("**Predicciones (Próximas semanas)**")
            rows = []
            for container in ['22g0', '45g0', '40rn']:
                for p in preds_xgb.get(sp, {}).get(container, []):
                    rows.append({
                        "Service provider": sp, "Model": "XGB", "Container": container.upper(),
                        "Week": p["Week"].strftime('%Y-%m-%d'),
                        "Predicted Rate (USD)": p["Rate"],
                    })
            if rows:
                pred_df = pd.DataFrame(rows)
                st.dataframe(pred_df, use_container_width=True, hide_index=True)
                csv = pred_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Descargar predicciones CSV",
                    data=csv,
                    file_name=f"predictions_{sp}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No hay predicciones disponibles para este proveedor.")

            # 4) Gráfica histórico + predicción (por contenedor) con banda ±RMSE (XGB)
            for container in ['22g0', '45g0', '40rn']:
                st.markdown(f"**{container.upper()} – Histórico y pronóstico (XGB)**")
                historical = sp_df[
                    (sp_df['Week'] >= historical_start_date) &
                    (sp_df['Week'] < forecast_start_date)
                ][['Week', container]].rename(columns={'Week': 'Week Starting', container: 'Actual Rate ($USD)'})
                historical['Week Starting'] = historical['Week Starting'].dt.strftime('%Y-%m-%d')

                xgb_df = pd.DataFrame(preds_xgb.get(sp, {}).get(container, []))
                if not xgb_df.empty:
                    xgb_df['Week'] = pd.to_datetime(xgb_df['Week'])
                    xgb_df['Week Starting'] = xgb_df['Week'].dt.strftime('%Y-%m-%d')
                    xgb_df = xgb_df.rename(columns={'Rate': 'XGB Pred ($USD)'})

                fig = go.Figure()
                if not historical.empty:
                    fig.add_trace(go.Scatter(
                        x=historical['Week Starting'],
                        y=historical['Actual Rate ($USD)'],
                        mode="lines",
                        name="Actual"
                    ))
                if not xgb_df.empty:
                    # Banda de confianza (XGB) como ±RMSE
                    xgb_m_local = metrics_xgb.get(sp, {}).get(container, {})
                    rmse_xgb = float(xgb_m_local.get('RMSE', 0.0)) if xgb_m_local else 0.0
                    if rmse_xgb > 0:
                        upper = xgb_df['XGB Pred ($USD)'] + rmse_xgb
                        lower = xgb_df['XGB Pred ($USD)'] - rmse_xgb
                        fig.add_trace(go.Scatter(
                            x=xgb_df['Week Starting'], y=upper,
                            mode='lines', line=dict(width=0), showlegend=True,
                            name='XGB ±RMSE'
                        ))
                        fig.add_trace(go.Scatter(
                            x=xgb_df['Week Starting'], y=lower,
                            mode='lines', line=dict(width=0), fill='tonexty',
                            fillcolor='rgba(255,165,0,0.25)', showlegend=False
                        ))
                    fig.add_trace(go.Scatter(
                        x=xgb_df['Week Starting'],
                        y=xgb_df['XGB Pred ($USD)'],
                        mode='lines+markers',
                        name='XGB'
                    ))

                # Anotación con RMSE & MAPE
                xgb_m = metrics_xgb.get(sp, {}).get(container, {})
                note_text = (
                    f"XGB → RMSE {xgb_m.get('RMSE', float('nan')):.1f} | "
                    f"MAPE {xgb_m.get('MAPE', float('nan')):.1f}%"
                    if xgb_m else "Sin métricas"
                )
                fig.add_annotation(
                    text=note_text,
                    xref="paper", yref="paper",
                    x=0.01, y=0.98,
                    showarrow=False,
                    align="left",
                    bordercolor="rgba(0,0,0,0.1)",
                    borderwidth=1,
                    bgcolor="rgba(255,255,255,0.7)"
                )

                fig.update_layout(
                    title=f"{container.upper()} – {sp}",
                    xaxis_title="Week",
                    yaxis_title="Rate ($USD)",
                    template="plotly_white",
                    height=420
                )
                st.plotly_chart(fig, use_container_width=True)

            # 5) Historical Macro (Rates + Brent & USD/CNY)
            st.markdown("**Historical Macro (Rates + Brent & USD/CNY)**")
            fig_macro = go.Figure()

            # Tarifas en eje primario
            fig_macro.add_trace(go.Scatter(x=sp_df['Week'], y=sp_df['22g0'], mode='lines', name='22G0'))
            fig_macro.add_trace(go.Scatter(x=sp_df['Week'], y=sp_df['45g0'], mode='lines', name='45G0'))
            fig_macro.add_trace(go.Scatter(x=sp_df['Week'], y=sp_df['40rn'], mode='lines', name='40RN'))

            # Brent y FX al eje secundario (solo si toggles activos)
            if use_brent:
                fig_macro.add_trace(go.Scatter(x=sp_df['Week'], y=sp_df['Brent_Price'], mode='lines', name='Brent ($)', yaxis='y2'))
            if use_fx:
                fig_macro.add_trace(go.Scatter(x=sp_df['Week'], y=sp_df['Exchange_Rate'], mode='lines', name='USD/CNY', yaxis='y2'))
            if 'Port_Congestion_Interest' in sp_df.columns and use_port_congestion:
                fig_macro.add_trace(go.Scatter(x=sp_df['Week'], y=sp_df['Port_Congestion_Interest'], mode='lines', name='Port Congestion (0-100)', yaxis='y2'))

            fig_macro.update_layout(
                title="Historical Macro (Rates + Brent & USD/CNY)",
                xaxis=dict(title="Week"),
                yaxis=dict(title="Rates ($USD)"),
                yaxis2=dict(title="Brent ($) / USD-CNY / Trends", overlaying='y', side='right'),
                template="plotly_white",
                height=380
            )
            st.plotly_chart(fig_macro, use_container_width=True)

            # 6) AI Insights por Service Provider (solo XGB)
            if enable_ai:
                last_brent = sp_df['Brent_Price'].iloc[-1] if ('Brent_Price' in sp_df and not sp_df['Brent_Price'].isna().all()) else "N/A"
                last_fx = sp_df['Exchange_Rate'].iloc[-1] if ('Exchange_Rate' in sp_df and not sp_df['Exchange_Rate'].isna().all()) else "N/A"
                last_congestion = sp_df['Port_Congestion_Interest'].iloc[-1] if ('Port_Congestion_Interest' in sp_df and not sp_df['Port_Congestion_Interest'].isna().all()) else "N/A"

                # Resumen de predicciones para prompt (solo XGB)
                lines = []
                for container in ['22g0', '45g0', '40rn']:
                    xgb_pl = preds_xgb.get(sp, {}).get(container, [])
                    if xgb_pl:
                        seq = ", ".join([f"{p['Week'].strftime('%Y-%m-%d')}:{p['Rate']}" for p in xgb_pl])
                        lines.append(f"{container.upper()} XGB: {seq}")
                preds_str = "\n".join(lines) if lines else "Sin predicciones."

                # 6a) Análisis de tendencias y drivers (solo XGB)
                prompt_trend = (
                    f"Actúa como analista de mercados marítimos. Resume y explica tendencias para el service provider '{sp}'. "
                    f"Usa el modelo XGBoost entrenado. Relaciona las tarifas con Brent, USD/CNY y Port Congestion. "
                    f"Señala riesgos/model risk (lags, ventanas, cambio de régimen). Da recomendaciones para pricing y procurement.\n\n"
                    f"Contexto macro:\n"
                    f"- Último Brent ($/bbl): {last_brent}\n"
                    f"- Último USD/CNY: {last_fx}\n"
                    f"- Google Trends Port Congestion (0-100): {last_congestion}\n\n"
                    f"Predicciones próximas {FORECAST_WEEKS} semanas:\n{preds_str}\n\n"
                    f"Clasifica por contenedor (Alcista/Estable/Bajista) e indica el driver dominante probable."
                )
                st.markdown("### AI Insights – Tendencias y drivers")
                ai_analysis = call_ai_with_fallback(prompt_trend, AI_MODELS)
                st.write(ai_analysis)

                # 6b) Factores actuales que podrían afectar tarifas
                prompt_factors = (
                    "Considerando el mercado marítimo global y riesgos actuales, "
                    "¿qué otros factores importantes en la actualidad podrían impactar "
                    "las tarifas de flete oceánico en la actualidad (2025) (por ejemplo, disrupciones geopolíticas, "
                    "clima extremo, blank sailings, GRI, capacidad de contenedores, restricciones "
                    "en canales marítimos, normativas IMO, demanda en US/EU, etc.)? "
                    "Clasifica por impacto (Alto/Medio/Bajo) y justifica brevemente cada uno."
                )
                st.markdown("### AI Insights – Factores de riesgo actuales")
                ai_factors = call_ai_with_fallback(prompt_factors, AI_MODELS)
                st.write(ai_factors)
