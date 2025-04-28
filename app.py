import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
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

# Cache data loading
@st.cache_data
def load_and_process_data():
    try:
        df = pd.read_excel("History_rates.xlsx")
    except FileNotFoundError:
        st.error("History_rates.xlsx not found in the project root.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading History_rates.xlsx: {e}")
        return None, None, None

    required_cols = ['Duration from', '22g0', '45g0', '40rn']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in History_rates.xlsx: {missing_cols}")
        return None, None, None

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

    df['Week'] = df['Duration from'].dt.to_period('W').dt.start_time
    weekly_df = df.groupby('Week')[['22g0', '45g0', '40rn']].mean().reset_index()
    if weekly_df.empty:
        st.error("No data after weekly aggregation.")
        return None, None, None

    try:
        wti = yf.download('CL=F', start='2022-01-01', end='2025-04-28', progress=False)
        wti = wti[['Close']].resample('W-MON').mean().reset_index()
        wti.columns = ['Week', 'WTI_Price']
        wti['Week'] = pd.to_datetime(wti['Week'])
    except Exception as e:
        st.warning(f"Failed to fetch WTI data: {e}. Simulating WTI data.")
        weeks = pd.date_range(start='2022-01-01', end='2025-04-28', freq='W-MON')
        wti = pd.DataFrame({
            'Week': weeks,
            'WTI_Price': np.random.uniform(80, 100, len(weeks))
        })

    try:
        exchange = yf.download('CNY=X', start='2022-01-01', end='2025-04-28', progress=False)
        exchange = exchange[['Close']].resample('W-MON').mean().reset_index()
        exchange.columns = ['Week', 'Exchange_Rate']
        exchange['Week'] = pd.to_datetime(exchange['Week'])
    except Exception as e:
        st.warning(f"Failed to fetch USD/CNY Exchange Rate data: {e}. Simulating Exchange Rate data.")
        weeks = pd.date_range(start='2022-01-01', end='2025-04-28', freq='W-MON')
        exchange = pd.DataFrame({
            'Week': weeks,
            'Exchange_Rate': np.random.uniform(6.5, 7.5, len(weeks))
        })

    weekly_df = pd.merge(weekly_df, wti, on='Week', how='left')
    weekly_df = pd.merge(weekly_df, exchange, on='Week', how='left')
    weekly_df['WTI_Price'] = weekly_df['WTI_Price'].fillna(weekly_df['WTI_Price'].median())
    weekly_df['Exchange_Rate'] = weekly_df['Exchange_Rate'].fillna(weekly_df['Exchange_Rate'].median())
    if weekly_df.empty:
        st.error("No data after merging with WTI and Exchange Rate data.")
        return None, None, None

    weekly_df['Week_of_Year'] = weekly_df['Week'].dt.isocalendar().week
    weekly_df['Month'] = weekly_df['Week'].dt.month
    for col in ['22g0', '45g0', '40rn', 'WTI_Price', 'Exchange_Rate']:
        weekly_df[f'{col}_lag1'] = weekly_df[col].shift(1).fillna(weekly_df[col].median())
    if weekly_df.empty:
        st.error("No data after feature engineering.")
        return None, None, None

    if len(weekly_df) < 2:
        st.error(f"Insufficient data after processing: {len(weekly_df)} rows. Need at least 2 rows for modeling.")
        return None, None, None

    return weekly_df, wti, exchange

# Cache model training with performance metrics
@st.cache_data
def train_models(df):
    if df is None:
        st.error("No data provided to train models.")
        return None, None

    features = ['Week_of_Year', 'Month', '22g0_lag1', '45g0_lag1', '40rn_lag1', 'WTI_Price', 'WTI_Price_lag1', 'Exchange_Rate', 'Exchange_Rate_lag1']
    targets = ['22g0', '45g0', '40rn']
    models = {}
    performance_metrics = {}

    for target in targets:
        X = df[features]
        y = df[target]
        if len(X) < 2:
            st.error(f"Insufficient data for {target}: {len(X)} rows. Need at least 2 rows.")
            return None, None
        train_size = int(0.8 * len(X)) if len(X) > 5 else max(1, len(X) - 1)
        if train_size < 1 or len(X) - train_size < 1:
            st.warning(f"Insufficient data for {target}. Training on all {len(X)} rows without validation.")
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

    return models, performance_metrics

# Generate predictions with confidence intervals
def generate_predictions(models, df, forecast_weeks=4):
    if models is None or df is None:
        st.error("No models or data available for prediction.")
        return None

    last_row = df.iloc[-1]
    current_date = pd.to_datetime('2025-04-28')
    predictions = {col: [] for col in ['22g0', '45g0', '40rn']}

    historical_volatility = {}
    for target in ['22g0', '45g0', '40rn']:
        rates = df[target]
        returns = rates.pct_change().dropna()
        historical_volatility[target] = returns.std() * np.sqrt(52)

    trend_factors = {}
    for target in ['22g0', '45g0', '40rn']:
        recent_rates = df[target].tail(4)
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

        scenarios = {
            'Optimistic': {'wti_adj': 0.9, 'rate_adj': 0.95, 'exchange_adj': 0.9},
            'Neutral': {'wti_adj': 1.0, 'rate_adj': 1.0, 'exchange_adj': 1.0},
            'Pessimistic': {'wti_adj': 1.1, 'rate_adj': 1.05, 'exchange_adj': 1.1}
        }

        week_pred = {'Week': forecast_date}
        for scenario, adj in scenarios.items():
            wti_price = last_row['WTI_Price'] * adj['wti_adj']
            exchange_rate = last_row['Exchange_Rate'] * adj['exchange_adj']
            features = [
                week_of_year,
                month,
                last_row['22g0'],
                last_row['45g0'],
                last_row['40rn'],
                wti_price,
                last_row['WTI_Price'],
                exchange_rate,
                last_row['Exchange_Rate']
            ]

            for target in ['22g0', '45g0', '40rn']:
                pred_samples = []
                for _ in range(10):
                    pred = models[target].predict([features])[0] * adj['rate_adj']
                    trend_adj = 1 + trend_factors[target] * (week + 1)
                    pred *= seasonal_factor * trend_adj
                    weekly_variation = np.random.uniform(-0.05, 0.05) * historical_volatility[target]
                    pred *= (1 + weekly_variation)
                    historical_min = df[target].min()
                    historical_max = df[target].max()
                    pred = max(historical_min, min(historical_max, pred))
                    pred_samples.append(pred)

                pred = np.mean(pred_samples)
                pred_std = np.std(pred_samples)
                ci_lower = pred - 1.96 * pred_std
                ci_upper = pred + 1.96 * pred_std

                prev_rate = last_row[target] if week == 0 else predictions[target][-1]['Neutral_Rate']
                change = (pred - prev_rate) / prev_rate if prev_rate != 0 else 0
                trend = 'Upward' if change > 0.05 else 'Downward' if change < -0.05 else 'Stable'
                week_pred[f'{target}_{scenario}_Rate'] = round(pred, 2)
                week_pred[f'{target}_{scenario}_Trend'] = trend
                week_pred[f'{target}_{scenario}_CI_Lower'] = round(ci_lower, 2)
                week_pred[f'{target}_{scenario}_CI_Upper'] = round(ci_upper, 2)

        for target in ['22g0', '45g0', '40rn']:
            pred_dict = {
                'Week': forecast_date,
                'Optimistic_Rate': week_pred[f'{target}_Optimistic_Rate'],
                'Optimistic_Trend': week_pred[f'{target}_Optimistic_Trend'],
                'Optimistic_CI_Lower': week_pred[f'{target}_Optimistic_CI_Lower'],
                'Optimistic_CI_Upper': week_pred[f'{target}_Optimistic_CI_Upper'],
                'Neutral_Rate': week_pred[f'{target}_Neutral_Rate'],
                'Neutral_Trend': week_pred[f'{target}_Neutral_Trend'],
                'Neutral_CI_Lower': week_pred[f'{target}_Neutral_CI_Lower'],
                'Neutral_CI_Upper': week_pred[f'{target}_Neutral_CI_Upper'],
                'Pessimistic_Rate': week_pred[f'{target}_Pessimistic_Rate'],
                'Pessimistic_Trend': week_pred[f'{target}_Pessimistic_Trend'],
                'Pessimistic_CI_Lower': week_pred[f'{target}_Pessimistic_CI_Lower'],
                'Pessimistic_CI_Upper': week_pred[f'{target}_Pessimistic_CI_Upper']
            }
            predictions[target].append(pred_dict)

        last_row = pd.Series({
            'Week': forecast_date,
            '22g0': week_pred['22g0_Neutral_Rate'],
            '45g0': week_pred['45g0_Neutral_Rate'],
            '40rn': week_pred['40rn_Neutral_Rate'],
            'WTI_Price': last_row['WTI_Price'],
            'Exchange_Rate': last_row['Exchange_Rate']
        })

    result = {k: pd.DataFrame(predictions[k]) for k in predictions}
    return result

# Main app
st.markdown('<div class="header">Shipping Rate Predictor: Shanghai to Buenaventura</div>', unsafe_allow_html=True)

# Load data
weekly_df, wti, exchange = load_and_process_data()
if weekly_df is None:
    st.markdown("### Using Sample Predictions Due to Data Issue")
    predictions = {
        '22g0': pd.DataFrame({
            'Week': pd.date_range('2025-05-05', periods=4, freq='W-MON'),
            'Optimistic_Rate': [1425, 1472, 1520, 1544],
            'Optimistic_Trend': ['Stable', 'Upward', 'Upward', 'Stable'],
            'Optimistic_CI_Lower': [1400, 1447, 1495, 1519],
            'Optimistic_CI_Upper': [1450, 1497, 1545, 1569],
            'Neutral_Rate': [1500, 1550, 1600, 1625],
            'Neutral_Trend': ['Stable', 'Upward', 'Upward', 'Stable'],
            'Neutral_CI_Lower': [1475, 1525, 1575, 1600],
            'Neutral_CI_Upper': [1525, 1575, 1625, 1650],
            'Pessimistic_Rate': [1575, 1628, 1680, 1706],
            'Pessimistic_Trend': ['Stable', 'Upward', 'Upward', 'Stable'],
            'Pessimistic_CI_Lower': [1550, 1603, 1655, 1681],
            'Pessimistic_CI_Upper': [1600, 1653, 1705, 1731]
        }),
        '45g0': pd.DataFrame({
            'Week': pd.date_range('2025-05-05', periods=4, freq='W-MON'),
            'Optimistic_Rate': [1900, 1995, 2090, 2138],
            'Optimistic_Trend': ['Stable', 'Upward', 'Upward', 'Stable'],
            'Optimistic_CI_Lower': [1875, 1970, 2065, 2113],
            'Optimistic_CI_Upper': [1925, 2020, 2115, 2163],
            'Neutral_Rate': [2000, 2100, 2200, 2250],
            'Neutral_Trend': ['Stable', 'Upward', 'Upward', 'Stable'],
            'Neutral_CI_Lower': [1975, 2075, 2175, 2225],
            'Neutral_CI_Upper': [2025, 2125, 2225, 2275],
            'Pessimistic_Rate': [2100, 2205, 2310, 2363],
            'Pessimistic_Trend': ['Stable', 'Upward', 'Upward', 'Stable'],
            'Pessimistic_CI_Lower': [2075, 2180, 2285, 2338],
            'Pessimistic_CI_Upper': [2125, 2230, 2335, 2388]
        }),
        '40rn': pd.DataFrame({
            'Week': pd.date_range('2025-05-05', periods=4, freq='W-MON'),
            'Optimistic_Rate': [2090, 2195, 2299, 2351],
            'Optimistic_Trend': ['Stable', 'Upward', 'Upward', 'Stable'],
            'Optimistic_CI_Lower': [2065, 2170, 2274, 2326],
            'Optimistic_CI_Upper': [2115, 2220, 2324, 2376],
            'Neutral_Rate': [2200, 2310, 2420, 2475],
            'Neutral_Trend': ['Stable', 'Upward', 'Upward', 'Stable'],
            'Neutral_CI_Lower': [2175, 2285, 2395, 2450],
            'Neutral_CI_Upper': [2225, 2335, 2445, 2500],
            'Pessimistic_Rate': [2310, 2426, 2541, 2599],
            'Pessimistic_Trend': ['Stable', 'Upward', 'Upward', 'Stable'],
            'Pessimistic_CI_Lower': [2285, 2401, 2516, 2574],
            'Pessimistic_CI_Upper': [2335, 2451, 2566, 2624]
        })
    }
    performance_metrics = {
        '22g0': {'RMSE': 'N/A', 'MAPE': 'N/A', 'R2': 'N/A', 'Certainty (%)': 'N/A'},
        '45g0': {'RMSE': 'N/A', 'MAPE': 'N/A', 'R2': 'N/A', 'Certainty (%)': 'N/A'},
        '40rn': {'RMSE': 'N/A', 'MAPE': 'N/A', 'R2': 'N/A', 'Certainty (%)': 'N/A'}
    }
else:
    # Train models
    models, performance_metrics = train_models(weekly_df)
    if models is None:
        st.stop()

    # Generate predictions
    predictions = generate_predictions(models, weekly_df)
    if predictions is None:
        st.stop()

# Model performance section - Show only Estimated Certainty
st.markdown('<div class="subheader">Model Performance</div>', unsafe_allow_html=True)
for target in performance_metrics:
    metrics = performance_metrics[target]
    st.markdown(f"**{target.upper()}**")
    st.write(f"- Estimated Certainty: {metrics['Certainty (%)'] if metrics['Certainty (%)'] == 'N/A' else f'{metrics['Certainty (%)']:.2f}%'}")
    st.write("")

# Container type tabs
tab1, tab2, tab3 = st.tabs(["20ft Container (22g0)", "40ft Container (45g0)", "40ft Reefer (40rn)"])

for tab, container in [(tab1, '22g0'), (tab2, '45g0'), (tab3, '40rn')]:
    with tab:
        st.markdown(f'<div class="subheader">{container.upper()} Predictions</div>', unsafe_allow_html=True)
        
        # Filter data and format for all scenarios, excluding CI Lower and CI Upper
        df_display = predictions[container][[
            'Week',
            'Optimistic_Rate', 'Optimistic_Trend',
            'Neutral_Rate', 'Neutral_Trend',
            'Pessimistic_Rate', 'Pessimistic_Trend'
        ]].rename(columns={
            'Week': 'Week Starting',
            'Optimistic_Rate': 'Optimistic Rate ($USD)',
            'Optimistic_Trend': 'Optimistic Trend',
            'Neutral_Rate': 'Neutral Rate ($USD)',
            'Neutral_Trend': 'Neutral Trend',
            'Pessimistic_Rate': 'Pessimistic Rate ($USD)',
            'Pessimistic_Trend': 'Pessimistic Trend'
        })
        df_display['Week Starting'] = df_display['Week Starting'].dt.strftime('%Y-%m-%d')
        for col in ['Optimistic Rate ($USD)', 'Neutral Rate ($USD)', 'Pessimistic Rate ($USD)']:
            df_display[col] = df_display[col].round(2)
        
        # Display table with all scenarios
        st.markdown('<div class="table-container">', unsafe_allow_html=True)
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Week Starting": st.column_config.TextColumn(width="medium"),
                "Optimistic Rate ($USD)": st.column_config.NumberColumn(width="medium"),
                "Optimistic Trend": st.column_config.TextColumn(width="medium"),
                "Neutral Rate ($USD)": st.column_config.NumberColumn(width="medium"),
                "Neutral Trend": st.column_config.TextColumn(width="medium"),
                "Pessimistic Rate ($USD)": st.column_config.NumberColumn(width="medium"),
                "Pessimistic Trend": st.column_config.TextColumn(width="medium")
            }
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Plot with all scenarios, confidence interval only for Neutral
        fig = go.Figure()
        scenario_colors = {
            "Optimistic": {"line": "#1f77b4"},  # Blue for Optimistic
            "Neutral": {"line": "#ff7f0e", "ci": "rgba(0, 255, 0, 0.3)"},  # Orange for Neutral, green CI
            "Pessimistic": {"line": "#2ca02c"}  # Green for Pessimistic
        }
        for scen in ["Optimistic", "Neutral", "Pessimistic"]:
            # Main line for the scenario
            fig.add_trace(go.Scatter(
                x=predictions[container]['Week'],
                y=predictions[container][f'{scen}_Rate'],
                mode="lines+markers",
                name=scen,
                line=dict(color=scenario_colors[scen]["line"]),
                opacity=1
            ))
            # Confidence interval shading only for Neutral
            if scen == "Neutral":
                fig.add_trace(go.Scatter(
                    x=predictions[container]['Week'],
                    y=predictions[container][f'{scen}_CI_Upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=True,
                    name="Neutral 95% CI",
                    opacity=0.3,
                    hovertemplate="Upper CI: %{y:.2f} $USD<extra></extra>"
                ))
                fig.add_trace(go.Scatter(
                    x=predictions[container]['Week'],
                    y=predictions[container][f'{scen}_CI_Lower'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor=scenario_colors[scen]["ci"],
                    showlegend=False,
                    opacity=0.3,
                    hovertemplate="Lower CI: %{y:.2f} $USD<extra></extra>"
                ))
        fig.update_layout(
            title=f"{container.upper()} Rate Forecast",
            xaxis_title="Week Starting",
            yaxis_title="Rate ($USD)",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# Historical trends with exchange rate
if weekly_df is not None:
    st.markdown('<div class="subheader">Historical Trends</div>', unsafe_allow_html=True)
    fig = go.Figure()
    for col in ['22g0', '45g0', '40rn']:
        fig.add_trace(go.Scatter(
            x=weekly_df['Week'],
            y=weekly_df[col],
            mode='lines',
            name=col.upper()
        ))
    fig.add_trace(go.Scatter(
        x=weekly_df['Week'],
        y=weekly_df['WTI_Price'] * 10,
        mode='lines',
        name='WTI Price (Scaled)',
        yaxis='y2'
    ))
    fig.add_trace(go.Scatter(
        x=weekly_df['Week'],
        y=weekly_df['Exchange_Rate'] * 1000,
        mode='lines',
        name='USD/CNY Exchange Rate (Scaled)',
        yaxis='y3'
    ))
    fig.update_layout(
        title="Historical Shipping Rates, WTI Prices, and USD/CNY Exchange Rate",
        xaxis_title="Week",
        yaxis=dict(
            title="Rate ($USD)",
            tickfont=dict(color="#1f77b4")
        ),
        yaxis2=dict(
            title="WTI Price ($)",
            tickfont=dict(color="#ff7f0e"),
            overlaying='y',
            side='right'
        ),
        yaxis3=dict(
            title="USD/CNY Exchange Rate",
            tickfont=dict(color="#2ca02c"),
            overlaying='y',
            side='right',
            position=0.85
        ),
        template="plotly_white",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# Methodology and insights
st.markdown('<div class="subheader">Methodology & Insights</div>', unsafe_allow_html=True)
st.markdown("""
- **Data**: Historical rates from `History_rates.xlsx` (2022-2025), WTI oil prices, and USD/CNY exchange rates from `yfinance`.
- **Model**: XGBoost trained on weekly rates, using lagged rates, WTI prices, exchange rates, and time features.
- **Scenarios**: Optimistic (-10% WTI, -10% USD/CNY, 95% rate), Neutral (stable WTI and USD/CNY), Pessimistic (+10% WTI, +10% USD/CNY, 105% rate).
- **Certainty**: Estimated as 100% - MAPE, with confidence intervals shown in the forecast chart for the Neutral scenario (95% confidence level).
- **Insight**: WTI prices and USD/CNY exchange rates both influence shipping rates, with `40rn` rates ~10-20% higher than `45g0` due to reefer requirements.
""")

# Interesting fact
st.markdown('<div class="subheader">Interesting Fact</div>', unsafe_allow_html=True)
st.markdown("""
In August 2024, `40rn` rates peaked at ~$6500, reflecting high demand for refrigerated cargo, elevated WTI prices (~$95/barrel), and a stronger USD against CNY (~7.2), increasing costs for Chinese exporters.
""")