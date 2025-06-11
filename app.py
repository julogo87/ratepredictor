import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time
import warnings
import os
import pickle
import backoff
import io

warnings.filterwarnings("ignore")

# Streamlit configuration
st.set_page_config(page_title="Shipping Rate Predictor", layout="wide")

# Tailwind CSS and custom styling
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        .header { font-size: 2.5rem; font-weight: bold; color: #1E40AF; text-align: center; margin-bottom: 2rem; }
        .subheader { font-size: 1.5rem; font-weight: 600; color: #1F2937; margin-top: 1.5rem; }
        .table-container { background-color: #F9FAFB; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .custom-table thead th {
            background-color: #1E40AF;
            color: white;
            font-weight: bold;
            padding: 0.75rem;
            text-align: left;
            border-bottom: 2px solid #E5E7EB;
        }
        .custom-table tbody tr {
            border-bottom: 1px solid #E5E7EB;
        }
        .custom-table tbody tr:nth-child(even) {
            background-color: #F3F4F6;
        }
        .custom-table tbody tr:hover {
            background-color: #E5E7EB;
        }
        .custom-table td {
            padding: 0.75rem;
            text-align: left;
        }
    </style>
""", unsafe_allow_html=True)

# Utility function to convert Excel serial date or string to datetime
def excel_serial_to_date(value):
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, (int, float)):
        try:
            return pd.to_datetime('1899-12-30') + pd.to_timedelta(int(value), 'D')
        except:
            pass
    try:
        return pd.to_datetime(value, errors='coerce')
    except:
        return pd.NaT

# Load Google Trends data from uploaded CSV
@st.cache_data
def load_google_trends_data(uploaded_csv):
    if uploaded_csv is None:
        return pd.DataFrame({'Week': [], 'Port_Congestion_Interest': []})
    try:
        trends_df = pd.read_csv(uploaded_csv, skiprows=1)
        trends_df.columns = ['Week', 'Port_Congestion_Interest']
        trends_df['Week'] = pd.to_datetime(trends_df['Week']).dt.to_period('W-MON').dt.start_time
        trends_df = trends_df.groupby('Week')['Port_Congestion_Interest'].mean().reset_index()
        return trends_df
    except Exception as e:
        st.error(f"Error loading Port Congestion CSV: {e}. Using default values.")
        return pd.DataFrame({'Week': [], 'Port_Congestion_Interest': []})

# Cache data loading with enhanced yfinance handling
@st.cache_data
def load_and_process_data(uploaded_file, uploaded_csv, use_port_congestion):
    if uploaded_file is None:
        st.error("Please upload the History_rates.xlsx file.")
        return None, None, None, None

    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading uploaded file: {e}")
        return None, None, None, None

    required_cols = ['Duration from', 'Service provider', '22g0', '45g0', '40rn']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in uploaded file: {missing_cols}")
        return None, None, None, None

    df['Duration from'] = df['Duration from'].apply(excel_serial_to_date)
    if df['Duration from'].isna().all():
        st.warning("All 'Duration from' values are invalid. Imputing with sequential dates starting from 2022-01-01.")
        df['Duration from'] = pd.date_range(start='2022-01-01', periods=len(df), freq='D')
    elif df['Duration from'].isna().any():
        st.warning("Some 'Duration from' values are invalid. Filling with median date.")
        median_date = df['Duration from'].median()
        df['Duration from'] = df['Duration from'].fillna(median_date)

    for col in ['22g0', '45g0']:
        df[col] = df[col].fillna(df[col].median())
    df['40rn'] = df['40rn'].fillna(df['45g0'] * 1.1)

    # Convert dates to weeks and aggregate by week and service provider
    df['Week'] = pd.to_datetime(df['Duration from']).dt.to_period('W-MON').dt.start_time
    weekly_counts = df.groupby(['Week', 'Service provider']).size()
    if (weekly_counts > 1).any():
        st.info("Multiple tariffs found within the same week for some service providers. Averaging tariffs for each container type per week per service provider.")
    weekly_df = df.groupby(['Week', 'Service provider'])[['22g0', '45g0', '40rn']].mean().reset_index()
    if weekly_df.empty:
        st.error("No data after weekly aggregation by service provider.")
        return None, None, None, None

    # Load Google Trends data if toggle is enabled and CSV is uploaded
    trends_data = None
    if use_port_congestion:
        trends_data = load_google_trends_data(uploaded_csv)
        if not trends_data.empty:
            weekly_df = pd.merge(weekly_df, trends_data, on='Week', how='left')
            if weekly_df['Port_Congestion_Interest'].notna().any():
                weekly_df['Port_Congestion_Interest'] = weekly_df['Port_Congestion_Interest'].fillna(weekly_df['Port_Congestion_Interest'].mean())
            else:
                st.warning("No valid Port Congestion Interest data after merge. Setting to 0.")
                weekly_df['Port_Congestion_Interest'] = 0
            weekly_df['Port_Congestion_Interest_lag1'] = weekly_df.groupby('Service provider')['Port_Congestion_Interest'].shift(1).fillna(weekly_df['Port_Congestion_Interest'].mean())
        else:
            st.warning("No Port Congestion data loaded. Setting Port Congestion Interest to 0.")
            weekly_df['Port_Congestion_Interest'] = 0
            weekly_df['Port_Congestion_Interest_lag1'] = 0
    else:
        weekly_df['Port_Congestion_Interest'] = 0
        weekly_df['Port_Congestion_Interest_lag1'] = 0

    # File-based caching for yfinance data
    brent_cache_file = "brent_cache.pkl"
    exchange_cache_file = "exchange_cache.pkl"

    # Function to fetch yfinance data with enhanced retry logic
    @backoff.on_exception(backoff.expo, Exception, max_tries=10, max_time=300)
    def fetch_yfinance_data(ticker, start, end):
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False, timeout=30)
        if data.empty:
            raise ValueError(f"No data returned for ticker {ticker}")
        return data

    # Cache Brent data
    @st.cache_data
    def get_brent_data(_start, _end):
        if os.path.exists(brent_cache_file):
            try:
                with open(brent_cache_file, 'rb') as f:
                    brent = pickle.load(f)
                    if (brent['Week'].min() <= pd.to_datetime(_start) and 
                        brent['Week'].max() >= pd.to_datetime(_end)):
                        st.info("Using cached Brent data from file.")
                        return brent
            except Exception as e:
                st.warning(f"Error loading Brent cache: {e}. Fetching new data.")

        try:
            brent = fetch_yfinance_data('BZ=F', _start, _end)
            brent = brent[['Close']].resample('W-MON').mean().reset_index()
            brent.columns = ['Week', 'Brent_Price']
            brent['Week'] = pd.to_datetime(brent['Week'])
            with open(brent_cache_file, 'wb') as f:
                pickle.dump(brent, f)
            return brent
        except Exception as e:
            st.error(f"Failed to fetch Brent data: {e}. Using simulated Brent data.")
            weeks = pd.date_range(start=_start, end=_end, freq='W-MON')
            brent = pd.DataFrame({
                'Week': weeks,
                'Brent_Price': np.random.normal(loc=90, scale=10, size=len(weeks))
            })
            brent['Brent_Price'] = brent['Brent_Price'].clip(70, 110)
            return brent

    # Cache Exchange Rate data
    @st.cache_data
    def get_exchange_data(_start, _end):
        if os.path.exists(exchange_cache_file):
            try:
                with open(exchange_cache_file, 'rb') as f:
                    exchange = pickle.load(f)
                    if (exchange['Week'].min() <= pd.to_datetime(_start) and 
                        exchange['Week'].max() >= pd.to_datetime(_end)):
                        st.info("Using cached Exchange Rate data from file.")
                        return exchange
            except Exception as e:
                st.warning(f"Error loading Exchange Rate cache: {e}. Fetching new data.")

        try:
            exchange = fetch_yfinance_data('CNY=X', _start, _end)
            exchange = exchange[['Close']].resample('W-MON').mean().reset_index()
            exchange.columns = ['Week', 'Exchange_Rate']
            exchange['Week'] = pd.to_datetime(exchange['Week'])
            with open(exchange_cache_file, 'wb') as f:
                pickle.dump(exchange, f)
            return exchange
        except Exception as e:
            st.error(f"Failed to fetch USD/CNY Exchange Rate data: {e}. Using simulated Exchange Rate data.")
            weeks = pd.date_range(start=_start, end=_end, freq='W-MON')
            exchange = pd.DataFrame({
                'Week': weeks,
                'Exchange_Rate': np.random.normal(loc=7.0, scale=0.3, size=len(weeks))
            })
            exchange['Exchange_Rate'] = exchange['Exchange_Rate'].clip(6.5, 7.5)
            return exchange

    # Fetch Brent and Exchange Rate data
    brent = get_brent_data('2022-01-01', '2025-06-11')
    time.sleep(5)
    exchange = get_exchange_data('2022-01-01', '2025-06-11')

    # Merge Brent and Exchange Rate data with weekly_df
    weekly_df = pd.merge(weekly_df, brent, on='Week', how='left')
    weekly_df = pd.merge(weekly_df, exchange, on='Week', how='left')
    weekly_df['Brent_Price'] = weekly_df['Brent_Price'].fillna(weekly_df['Brent_Price'].median())
    weekly_df['Exchange_Rate'] = weekly_df['Exchange_Rate'].fillna(weekly_df['Exchange_Rate'].median())
    if weekly_df.empty:
        st.error("No data after merging with Brent and Exchange Rate data.")
        return None, None, None, None

    weekly_df['Week_of_Year'] = weekly_df['Week'].dt.isocalendar().week
    weekly_df['Month'] = weekly_df['Week'].dt.month
    for col in ['22g0', '45g0', '40rn', 'Brent_Price', 'Exchange_Rate']:
        weekly_df[f'{col}_lag1'] = weekly_df.groupby('Service provider')[col].shift(1).fillna(weekly_df.groupby('Service provider')[col].transform('median'))
    if weekly_df.empty:
        st.error("No data after feature engineering.")
        return None, None, None, None

    if len(weekly_df) < 2:
        st.error(f"Insufficient data after processing: {len(weekly_df)} rows. Need at least 2 rows for modeling.")
        return None, None, None, None

    return weekly_df, brent, exchange, trends_data

# Cache model training with performance metrics, per service provider
@st.cache_data
def train_models(df, use_port_congestion):
    if df is None:
        st.error("No data provided to train models.")
        return None, None

    features = ['Week_of_Year', 'Month', '22g0_lag1', '45g0_lag1', '40rn_lag1', 'Brent_Price', 'Brent_Price_lag1', 
                'Exchange_Rate', 'Exchange_Rate_lag1']
    if use_port_congestion:
        features.extend(['Port_Congestion_Interest', 'Port_Congestion_Interest_lag1'])
    targets = ['22g0', '45g0', '40rn']
    service_providers = df['Service provider'].unique()
    service_provider_models = {}
    service_provider_performance_metrics = {}

    for sp in service_providers:
        sp_df = df[df['Service provider'] == sp]
        if len(sp_df) < 2:
            st.warning(f"Insufficient data for service provider {sp}: {len(sp_df)} rows. Skipping this service provider.")
            continue

        models = {}
        performance_metrics = {}
        for target in targets:
            X = sp_df[features]
            y = sp_df[target]
            if len(X) < 2:
                st.error(f"Insufficient data for {target} for service provider {sp}: {len(X)} rows. Need at least 2 rows.")
                continue
            train_size = int(0.8 * len(X)) if len(X) > 5 else max(1, len(X) - 1)
            if train_size < 1 or len(X) - train_size < 1:
                st.warning(f"Insufficient data for {target} for service provider {sp}. Training on all {len(X)} rows without validation.")
                model = XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=100, random_state=42)
                model.fit(X, y)
                models[target] = model
                performance_metrics[target] = {'RMSE': 'N/A', 'MAPE': 'N/A', 'R2': 'N/A', 'Certainty (%)': 'N/A'}
                continue

            X_train, X_val = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_val = y.iloc[:train_size], y.iloc[train_size:]

            model = XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100
            r2 = r2_score(y_val, y_pred)
            certainty = max(0, 100 - mape)

            models[target] = model
            performance_metrics[target] = {
                'RMSE': rmse,
                'MAPE': mape,
                'R2': r2,
                'Certainty (%)': certainty
            }

        service_provider_models[sp] = models
        service_provider_performance_metrics[sp] = performance_metrics

    return service_provider_models, service_provider_performance_metrics

# Generate predictions with confidence intervals, per service provider
def generate_predictions(service_provider_models, df, trends_data, forecast_weeks=4, use_port_congestion=True):
    if service_provider_models is None or df is None:
        st.error("No models or data available for prediction.")
        return None

    # Set random seed for reproducibility
    np.random.seed(42)

    service_providers = df['Service provider'].unique()
    service_provider_predictions = {}

    # Extend Brent, Exchange Rate, and Port Congestion data for the forecast period
    last_brent = df['Brent_Price'].iloc[-1]
    last_exchange = df['Exchange_Rate'].iloc[-1]
    current_date = pd.to_datetime('2025-06-11')
    future_weeks = pd.date_range(start=current_date, periods=forecast_weeks, freq='W-MON')
    future_brent = pd.DataFrame({
        'Week': future_weeks,
        'Brent_Price': last_brent
    })
    future_exchange = pd.DataFrame({
        'Week': future_weeks,
        'Exchange_Rate': last_exchange
    })
    future_trends = None
    if use_port_congestion and trends_data is not None and not trends_data.empty:
        last_trends = trends_data['Port_Congestion_Interest'].tail(4).mean()
        future_trends = pd.DataFrame({
            'Week': future_weeks,
            'Port_Congestion_Interest': last_trends
        })

    for sp in service_providers:
        if sp not in service_provider_models:
            st.warning(f"No model available for service provider {sp}. Skipping predictions.")
            continue

        models = service_provider_models[sp]
        sp_df = df[df['Service provider'] == sp]
        if len(sp_df) == 0:
            st.warning(f"No data available for service provider {sp}. Skipping predictions.")
            continue

        last_row = sp_df.iloc[-1]
        predictions = {col: [] for col in ['22g0', '45g0', '40rn']}

        historical_volatility = {}
        for target in ['22g0', '45g0', '40rn']:
            rates = sp_df[target]
            returns = rates.pct_change().dropna()
            historical_volatility[target] = returns.std() * np.sqrt(52) if len(returns) > 0 else 0.05

        trend_factors = {}
        for target in ['22g0', '45g0', '40rn']:
            recent_rates = sp_df[target].tail(4)
            if len(recent_rates) >= 2:
                trend = (recent_rates.iloc[-1] - recent_rates.iloc[0]) / recent_rates.iloc[0] / len(recent_rates)
                trend_factors[target] = trend
            else:
                trend_factors[target] = 0

        for week in range(forecast_weeks):
            forecast_date = current_date + timedelta(days=7 * week)
            week_of_year = forecast_date.isocalendar().week
            month = forecast_date.month

            seasonal_factor = 1.1 if 6 <= month <= 8 else 0.95

            # Get Brent and Exchange Rate for the forecast date
            brent_price = future_brent[future_brent['Week'] == forecast_date]['Brent_Price']
            brent_price = brent_price.iloc[0] if not brent_price.empty else last_brent
            exchange_rate = future_exchange[future_exchange['Week'] == forecast_date]['Exchange_Rate']
            exchange_rate = exchange_rate.iloc[0] if not exchange_rate.empty else last_exchange

            # Get Port Congestion Interest for the forecast date
            port_congestion_interest = 0
            port_congestion_interest_lag1 = last_row['Port_Congestion_Interest'] if week == 0 else predictions[target][-1]['Port_Congestion_Interest']
            if use_port_congestion and future_trends is not None:
                port_congestion = future_trends[future_trends['Week'] == forecast_date]['Port_Congestion_Interest']
                port_congestion_interest = port_congestion.iloc[0] if not port_congestion.empty else last_trends
                port_congestion_interest_lag1 = last_row['Port_Congestion_Interest'] if week == 0 else predictions[target][-1]['Port_Congestion_Interest']

            week_pred = {'Week': forecast_date}

            features = [
                week_of_year,
                month,
                last_row['22g0'],
                last_row['45g0'],
                last_row['40rn'],
                brent_price,
                last_row['Brent_Price'],
                exchange_rate,
                last_row['Exchange_Rate']
            ]
            if use_port_congestion:
                features.extend([port_congestion_interest, port_congestion_interest_lag1])

            for target in ['22g0', '45g0', '40rn']:
                if target not in models:
                    st.warning(f"No model for {target} for service provider {sp}. Skipping.")
                    continue

                pred_samples = []
                for _ in range(10):
                    pred = models[target].predict([features])[0]
                    trend_adj = 1 + trend_factors[target] * (week + 1)
                    pred *= seasonal_factor * trend_adj
                    weekly_variation = np.random.uniform(-0.05, 0.05) * historical_volatility[target]
                    pred *= (1 + weekly_variation)
                    historical_min = sp_df[target].min()
                    historical_max = sp_df[target].max()
                    pred = max(historical_min, min(historical_max, pred))
                    pred_samples.append(pred)

                neutral_pred = np.mean(pred_samples)
                pred_std = np.std(pred_samples)
                ci_lower = neutral_pred - 1.96 * pred_std
                ci_upper = neutral_pred + 1.96 * pred_std

                prev_rate = last_row[target] if week == 0 else predictions[target][-1]['Neutral_Rate']
                change = (neutral_pred - prev_rate) / prev_rate if prev_rate != 0 else 0
                trend = 'Upward' if change > 0.05 else 'Downward' if change < -0.05 else 'Stable'

                week_pred[f'{target}_Neutral_Rate'] = round(neutral_pred, 2)
                week_pred[f'{target}_Neutral_Trend'] = trend
                week_pred[f'{target}_Neutral_CI_Lower'] = round(ci_lower, 2)
                week_pred[f'{target}_Neutral_CI_Upper'] = round(ci_upper, 2)

            for target in ['22g0', '45g0', '40rn']:
                if f'{target}_Neutral_Rate' not in week_pred:
                    continue
                pred_dict = {
                    'Week': forecast_date,
                    'Service provider': sp,
                    'Neutral_Rate': week_pred[f'{target}_Neutral_Rate'],
                    'Neutral_Trend': week_pred[f'{target}_Neutral_Trend'],
                    'Neutral_CI_Lower': week_pred[f'{target}_Neutral_CI_Lower'],
                    'Neutral_CI_Upper': week_pred[f'{target}_Neutral_CI_Upper'],
                    'Port_Congestion_Interest': port_congestion_interest
                }
                predictions[target].append(pred_dict)

        service_provider_predictions[sp] = predictions

    return service_provider_predictions

# Main app
st.markdown('<div class="header">Shipping Rate Predictor: Shanghai to Buenaventura</div>', unsafe_allow_html=True)

# File uploader for History_rates.xlsx
st.markdown('<div class="subheader">Upload Historical Rates</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload History_rates.xlsx", type=["xlsx"])

# File uploader for Port Congestion CSV
st.markdown('<div class="subheader">Upload Port Congestion Data (Optional)</div>', unsafe_allow_html=True)
uploaded_csv = st.file_uploader("Upload Port Congestion CSV", type=["csv"])

# Toggle for port congestion effect
use_port_congestion = st.toggle("Include Port Congestion Effect", value=False, key="port_congestion_toggle", disabled=uploaded_csv is None)

if uploaded_file is not None:
    # Load data
    weekly_df, brent, exchange, trends_data = load_and_process_data(uploaded_file, uploaded_csv, use_port_congestion)

    if weekly_df is not None:
        # Train models
        service_provider_models, service_provider_performance_metrics = train_models(weekly_df, use_port_congestion)
        if service_provider_models is None:
            st.stop()

        # Generate predictions
        service_provider_predictions = generate_predictions(service_provider_models, weekly_df, trends_data, use_port_congestion=use_port_congestion)
        if service_provider_predictions is None:
            st.stop()

        # Calculate average shipping rates for all service providers
        weekly_df['Average_Rate'] = weekly_df[['22g0', '45g0', '40rn']].mean(axis=1)
        avg_rates = weekly_df.groupby(['Week', 'Service provider'])['Average_Rate'].mean().reset_index()

        # Define the forecast start date and calculate the start of the historical period
        forecast_start_date = pd.to_datetime('2025-06-11')
        historical_start_date = forecast_start_date - timedelta(weeks=4)

        # Create tabs for each service provider
        service_providers = sorted(weekly_df['Service provider'].unique())
        sp_tabs = st.tabs(service_providers)

        # Organize information within tabs
        for idx, sp in enumerate(service_providers):
            with sp_tabs[idx]:
                # Section 1: Average Shipping Rates vs Port Congestion Search Interest (if enabled)
                if use_port_congestion and trends_data is not None and not trends_data.empty:
                    st.markdown('<div class="subheader">Average Shipping Rates vs Port Congestion Search Interest</div>', unsafe_allow_html=True)
                    sp_rates = avg_rates[avg_rates['Service provider'] == sp][['Week', 'Average_Rate']]
                    merged_data = pd.merge(sp_rates, weekly_df[['Week', 'Port_Congestion_Interest']].drop_duplicates(), on='Week', how='inner')
                    if merged_data.empty:
                        st.warning(f"No overlapping data for {sp} to plot.")
                    else:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=merged_data['Week'],
                            y=merged_data['Average_Rate'],
                            mode='lines',
                            name='Average Shipping Rate',
                            line=dict(color='#1f77b4'),
                            yaxis='y1'
                        ))
                        fig.add_trace(go.Scatter(
                            x=merged_data['Week'],
                            y=merged_data['Port_Congestion_Interest'],
                            mode='lines',
                            name='Port Congestion Search Interest',
                            line=dict(color='#ff7f0e'),
                            yaxis='y2'
                        ))
                        fig.update_layout(
                            title=f"Average Shipping Rates vs Port Congestion Search Interest for {sp}",
                            xaxis_title="Week",
                            yaxis=dict(
                                title=dict(
                                    text="Average Rate ($USD)",
                                    font=dict(color="#1f77b4")
                                ),
                                tickfont=dict(color="#1f77b4")
                            ),
                            yaxis2=dict(
                                title=dict(
                                    text="Google Trends Interest (0-100)",
                                    font=dict(color="#ff7f0e")
                                ),
                                tickfont=dict(color="#ff7f0e"),
                                overlaying='y',
                                side='right'
                            ),
                            template="plotly_white",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)

                # Section 2: Model Performance
                st.markdown('<div class="subheader">Model Performance</div>', unsafe_allow_html=True)
                if sp in service_provider_performance_metrics:
                    st.markdown(f"**Service Provider: {sp}**")
                    for target in service_provider_performance_metrics[sp]:
                        metrics = service_provider_performance_metrics[sp][target]
                        st.markdown(f"**{target.upper()}**")
                        certainty = metrics['Certainty (%)']
                        formatted_certainty = certainty if certainty == 'N/A' else f"{certainty:.2f}%"
                        st.write(f"- Estimated Certainty: {formatted_certainty}")
                    st.write("")
                else:
                    st.warning(f"No model performance metrics available for {sp}.")

                # Section 3: Predictions
                st.markdown(f'<div class="subheader">Predictions {"(With Port Congestion)" if use_port_congestion else "(Without Port Congestion)"}</div>', unsafe_allow_html=True)
                if sp not in service_provider_predictions:
                    st.warning(f"No predictions available for service provider {sp}.")
                else:
                    predictions = service_provider_predictions[sp]
                    for container in ['22g0', '45g0', '40rn']:
                        st.markdown(f'<div class="subheader">{container.upper()}</div>', unsafe_allow_html=True)
                        
                        df_predictions = pd.DataFrame(predictions[container])
                        df_filtered = df_predictions[df_predictions['Service provider'] == sp]
                        
                        if df_filtered.empty:
                            st.warning(f"No predictions available for {container} for service provider {sp}.")
                            continue
                        
                        df_display = df_filtered[[
                            'Week',
                            'Neutral_Rate',
                            'Neutral_Trend'
                        ]].rename(columns={
                            'Week': 'Week Starting',
                            'Neutral_Rate': 'Neutral Rate ($USD)',
                            'Neutral_Trend': 'Neutral Trend'
                        })
                        df_display['Week Starting'] = df_display['Week Starting'].dt.strftime('%Y-%m-%d')
                        df_display['Neutral Rate ($USD)'] = df_display['Neutral Rate ($USD)'].round(2)
                        
                        st.markdown('<div class="table-container">', unsafe_allow_html=True)
                        st.dataframe(
                            df_display,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Week Starting": st.column_config.TextColumn(width="medium"),
                                "Neutral Rate ($USD)": st.column_config.NumberColumn(width="medium"),
                                "Neutral Trend": st.column_config.TextColumn(width="medium")
                            }
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        sp_df = weekly_df[weekly_df['Service provider'] == sp]
                        historical_data = sp_df[
                            (sp_df['Week'] >= historical_start_date) &
                            (sp_df['Week'] < forecast_start_date)
                        ][['Week', container]].rename(columns={
                            'Week': 'Week Starting',
                            container: 'Actual Rate ($USD)'
                        })
                        historical_data['Week Starting'] = historical_data['Week Starting'].dt.strftime('%Y-%m-%d')
                        
                        plot_data = pd.concat([
                            historical_data,
                            df_display[['Week Starting', 'Neutral Rate ($USD)']]
                        ], ignore_index=True)
                        
                        fig = go.Figure()
                        scenario_colors = {
                            "Actual": {"line": "#1f77b4"},
                            "Neutral": {"line": "#ff7f0e", "ci": "rgba(0, 255, 0, 0.3)"}
                        }
                        
                        if not historical_data.empty:
                            fig.add_trace(go.Scatter(
                                x=historical_data['Week Starting'],
                                y=historical_data['Actual Rate ($USD)'],
                                mode="lines",
                                name="Actual",
                                line=dict(color=scenario_colors["Actual"]["line"]),
                                opacity=1
                            ))
                        else:
                            st.warning(f"No historical data available for {container} for the last 4 weeks for {sp}.")
                        
                        rate_col = 'Neutral Rate ($USD)'
                        if rate_col not in df_display.columns:
                            st.warning(f"Column {rate_col} not found in data for {container} for service provider {sp}.")
                            continue
                        fig.add_trace(go.Scatter(
                            x=df_display['Week Starting'],
                            y=df_display[rate_col],
                            mode="lines+markers",
                            name="Predicted (Neutral)",
                            line=dict(color=scenario_colors["Neutral"]["line"]),
                            opacity=1
                        ))
                        
                        if 'Neutral_CI_Upper' not in df_filtered.columns or 'Neutral_CI_Lower' not in df_filtered.columns:
                            st.warning(f"Confidence interval columns for Neutral not found for {container} for service provider {sp}.")
                        else:
                            fig.add_trace(go.Scatter(
                                x=df_display['Week Starting'],
                                y=df_filtered['Neutral_CI_Upper'],
                                mode='lines',
                                line=dict(width=0),
                                showlegend=True,
                                name="Neutral 95% CI",
                                opacity=0.3,
                                hovertemplate="Upper CI: %{y:.2f} $USD<extra></extra>"
                            ))
                            fig.add_trace(go.Scatter(
                                x=df_display['Week Starting'],
                                y=df_filtered['Neutral_CI_Lower'],
                                mode='lines',
                                line=dict(width=0),
                                fill='tonexty',
                                fillcolor=scenario_colors["Neutral"]["ci"],
                                showlegend=False,
                                opacity=0.3,
                                hovertemplate="Lower CI: %{y:.2f} $USD<extra></extra>"
                            ))
                        
                        fig.update_layout(
                            title=f"{container.upper()} Rate Forecast for {sp} {'(With Port Congestion)' if use_port_congestion else '(Without Port Congestion)'}",
                            xaxis_title="Week Starting",
                            yaxis_title="Rate ($USD)",
                            template="plotly_white",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)

                # Section 4: Historical Trends
                st.markdown('<div class="subheader">Historical Trends</div>', unsafe_allow_html=True)
                sp_df = weekly_df[weekly_df['Service provider'] == sp]
                fig = go.Figure()
                for col in ['22g0', '45g0', '40rn']:
                    fig.add_trace(go.Scatter(
                        x=sp_df['Week'],
                        y=sp_df[col],
                        mode='lines',
                        name=col.upper()
                    ))
                fig.add_trace(go.Scatter(
                    x=sp_df['Week'],
                    y=sp_df['Brent_Price'] * 10,
                    mode='lines',
                    name='Brent Price (Scaled)',
                    yaxis='y2'
                ))
                fig.add_trace(go.Scatter(
                    x=sp_df['Week'],
                    y=sp_df['Exchange_Rate'] * 1000,
                    mode='lines',
                    name='USD/CNY Exchange Rate (Scaled)',
                    yaxis='y3'
                ))
                fig.update_layout(
                    title=f"Historical Shipping Rates, Brent Prices, and USD/CNY Exchange Rate for {sp}",
                    xaxis_title="Week",
                    yaxis=dict(
                        title=dict(
                            text="Rate ($USD)",
                            font=dict(color="#1f77b4")
                        ),
                        tickfont=dict(color="#1f77b4")
                    ),
                    yaxis2=dict(
                        title=dict(
                            text="Brent Price ($)",
                            font=dict(color="#ff7f0e")
                        ),
                        tickfont=dict(color="#ff7f0e"),
                        overlaying='y',
                        side='right'
                    ),
                    yaxis3=dict(
                        title=dict(
                            text="USD/CNY Exchange Rate",
                            font=dict(color="#2ca02c")
                        ),
                        tickfont=dict(color="#2ca02c"),
                        overlaying='y',
                        side='right',
                        position=0.85
                    ),
                    template="plotly_white",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

# Global Sections
st.markdown('<div class="subheader">Methodology & Insights</div>', unsafe_allow_html=True)
st.markdown("""
- **Data**: Historical rates uploaded via `History_rates.xlsx` (2022-2025), Brent oil prices, and USD/CNY exchange rates from `yfinance`. Tariffs are averaged per week per service provider for each container type. Google Trends data for "port congestion" from an optional uploaded CSV is included as an optional feature via toggle.
- **Model**: XGBoost trained on weekly rates per service provider, using lagged rates, Brent prices, exchange rates, time features, and optionally Google Trends "port congestion" search interest.
- **Scenarios**:
  - **Neutral**: Baseline scenario influenced by Brent prices, USD/CNY exchange rates, and optionally port congestion search interest.
- **Correlation Visualization**: When port congestion is enabled, graphs plot average shipping rates against Google Trends search interest for "port congestion" to visualize relationships.
- **Certainty**: Estimated as 100% - MAPE, with confidence intervals shown in the forecast chart (95% confidence level).
- **Insight**: Brent prices, USD/CNY exchange rates, and port congestion search interest (when enabled) influence predictions. `40rn` rates are ~10-20% higher than `45g0` due to reefer requirements.
""")

st.markdown('<div class="subheader">Interesting Fact</div>', unsafe_allow_html=True)
st.markdown("""
In August 2024, `40rn` rates peaked at ~$6500 for some service providers, reflecting high demand for refrigerated cargo, elevated Brent prices (~$95/barrel), and a stronger USD against CNY (~7.2).
""")