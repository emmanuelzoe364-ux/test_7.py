import os
import time
import datetime 
from datetime import timedelta, timezone 
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging 

# Configure basic logging for simulated alerts
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Configuration & Constants ---
st.set_page_config(page_title="BTC/ETH Combined Index Signal Tracker", layout="wide")

# Initialize session state for persistent signal tracking (for simulating alerts)
if 'last_signal_time' not in st.session_state:
    st.session_state.last_signal_time = pd.Timestamp.min.tz_localize(None)

# -----------------------
# Helper function for external alert (SIMULATION)
# -----------------------

def send_external_alert(signal_type: str, message: str, email: str, phone: str, conviction_score: int, lead_asset: str):
    """
    Simulates sending an external alert via Email/SMS API.
    Includes the Conviction Score and Lead Asset for immediate quality assessment.
    """
    if email or phone:
        logging.info(f"*** EXTERNAL ALERT SENT (Simulated) ***")
        if email:
            logging.info(f"EMAIL To: {email}")
        if phone:
            logging.info(f"SMS To: {phone}")
        logging.info(f"CONTENT (Score {conviction_score}, Lead: {lead_asset}): {message.replace('\n', ' | ')}")
    else:
        logging.info("External alert skipped: No email or phone recipient configured.")


# -----------------------
# Helper functions for calculations
# -----------------------

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Calculates the EMA-based Relative Strength Index (EMA-RSI)."""
    delta = series.diff().dropna()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(span=length, adjust=False).mean()
    ma_down = down.ewm(span=length, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, 1e-10) 
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def get_asset_stacking_score(ema_f: float, ema_m: float, ema_s: float) -> tuple[int, str]:
    """Calculates the EMA stacking score for a single asset."""
    score = 0
    desc = "Mixed/Consolidation"
    
    if ema_s > ema_m and ema_m > ema_f:
        score = 2; desc = "Perfect Bullish Stack"
    elif ema_s < ema_m and ema_m < ema_f:
        score = -2; desc = "Perfect Bearish Stack"
    elif ema_s > ema_m or ema_m > ema_f:
        score = 1; desc = "Developing Bullish Bias (Partial Alignment)"
    elif ema_s < ema_m or ema_m < ema_f:
        score = -1; desc = "Developing Bearish Bias (Partial Alignment)"
        
    return score, desc

# --- ADX Calculation Function (Corrected and Refined) ---
def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.DataFrame:
    """Calculates ADX, +DI, and -DI using EMA smoothing approximation."""
    
    df_adx = pd.DataFrame(index=close.index)
    
    # --- 1. True Range (TR) ---
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    df_adx['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # --- 2. Directional Movement (DM) ---
    df_adx['UpMove'] = high - high.shift(1)
    df_adx['DownMove'] = low.shift(1) - low
    
    # +DM
    df_adx['+DM'] = np.where(
        (df_adx['UpMove'] > df_adx['DownMove']) & (df_adx['UpMove'] > 0),
        df_adx['UpMove'], 0.0
    )
    # -DM
    df_adx['-DM'] = np.where(
        (df_adx['DownMove'] > df_adx['UpMove']) & (df_adx['DownMove'] > 0),
        df_adx['DownMove'], 0.0
    )
    
    # --- 3. Exponentially Smoothed TR and DM (Using EMA for smoothing, similar to Wilder's) ---
    smoother = lambda series: series.ewm(span=n, adjust=False).mean()
    
    df_adx['ATR'] = smoother(df_adx['TR'])
    df_adx['+DM_Smoothed'] = smoother(df_adx['+DM'])
    df_adx['-DM_Smoothed'] = smoother(df_adx['-DM'])
    
    # --- 4. Directional Indicators (DI) ---
    # Handle division by zero/NaN for DI
    df_adx['+DI'] = 100 * (df_adx['+DM_Smoothed'] / df_adx['ATR']).replace(np.inf, 0).fillna(0)
    df_adx['-DI'] = 100 * (df_adx['-DM_Smoothed'] / df_adx['ATR']).replace(np.inf, 0).fillna(0)
    
    # --- 5. Directional Movement Index (DX) ---
    sum_di = df_adx['+DI'] + df_adx['-DI']
    # Only calculate DX where sum_di is not zero
    df_adx['DX'] = (abs(df_adx['+DI'] - df_adx['-DI']) / sum_di).replace(np.inf, 0).fillna(0) * 100
    
    # --- 6. Average Directional Index (ADX) ---
    # ADX is the smoothed average of DX
    df_adx['ADX'] = smoother(df_adx['DX'])
    
    return df_adx[['+DI', '-DI', 'ADX']]
# -----------------------


# -----------------------
# Sidebar / user inputs
# -----------------------
st.sidebar.header("Index & Indicator Settings")
TICKERS = ["BTC-USD", "ETH-USD"]

MAX_INTRADAY_DAYS = 60
period_days = st.sidebar.number_input("Fetch period (days)", min_value=7, max_value=365, value=30) 
interval = st.sidebar.selectbox("Interval", options=["15m","30m","1h","1d"], index=2)

if interval in ["15m", "30m", "1h"] and period_days > MAX_INTRADAY_DAYS:
    period_days = MAX_INTRADAY_DAYS
    st.sidebar.warning(f"Intraday period capped at {MAX_INTRADAY_DAYS} days.")

st.session_state.interval = interval 

st.sidebar.markdown("---")
rsi_length = st.sidebar.number_input("RSI length", min_value=7, max_value=30, value=14)
index_ema_span = st.sidebar.number_input("Cumulative Index EMA Span (Smoother)", min_value=1, max_value=10, value=5)

# ADX Settings 
st.sidebar.header("ADX/Directional Movement Settings")
# Input for ADX length
adx_length = st.sidebar.number_input("ADX Length", min_value=7, max_value=50, value=14)
st.sidebar.markdown(f"_ADX is calculated using BTC's High/Low/Close data._")

st.sidebar.markdown("---")

ema_short = st.sidebar.number_input("EMA Short (14)", min_value=5, max_value=25, value=14)
ema_long = st.sidebar.number_input("EMA Medium (30)", min_value=26, max_value=50, value=30)
ema_very_long = st.sidebar.number_input("EMA Very Long (72)", min_value=51, max_value=365, value=72)

st.sidebar.markdown("---")
min_bars_after_cycle = st.sidebar.number_input("Max bars to look for re-alignment (0 = unlimited)", min_value=0, max_value=9999, value=0)

volume_length = st.sidebar.number_input("Volume MA Length", min_value=1, max_value=50, value=14)
enable_volume_filter = st.sidebar.checkbox("Require Volume Confirmation", value=False)


st.sidebar.header("External Notification Settings")
recipient_email = st.sidebar.text_input("Recipient Email (for simulation)", value="")
recipient_phone = st.sidebar.text_input("Recipient Phone (for simulation, e.g., +15551234)", value="")
st.markdown("---")

# --- Main Title & Time Display ---
st.title(f"🔥 BTC/ETH Combined Index (50/50) Tracker")
st.subheader(f"Triple-EMA Confirmation: {ema_short} > {ema_long} > {ema_very_long}")

current_gmt_time = datetime.datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S GMT")
st.markdown(f"**Current GMT Time:** {current_gmt_time}")


# -----------------------
# 🚀 Data Fetching and Processing
# -----------------------

@st.cache_data(ttl=timedelta(seconds=300))
def fetch_and_process_data(tickers: list, period_days: int, interval: str, index_ema_span: int, ema_short: int, ema_long: int, ema_very_long: int, adx_length: int):
    """
    Fetches data, creates the normalized combined index, and calculates all required indicators,
    including ADX.
    """
    
    status_text = st.empty()
    status_text.info(f"Fetching {', '.join(tickers)} {interval} data for {period_days} days. Cache TTL: 300 seconds.")
    
    end_date = pd.Timestamp.now()
    start_date = end_date - timedelta(days=period_days)
    
    try:
        # Note: We fetch all columns for ADX calculation
        raw = yf.download(tickers, start=start_date, end=end_date, interval=interval, progress=False)
    except Exception as e:
        status_text.error(f"Failed during data fetch: {e}. Check ticker symbols or try reducing period.")
        return pd.DataFrame()

    if raw.empty:
        status_text.error("Fetched data is empty. Try a different interval or period.")
        return pd.DataFrame()
        
    # --- 1. CLEANING AND NORMALIZATION ---
    df = pd.DataFrame()
    total_volume = 0
    btc_raw_data = None
    
    for ticker in tickers:
        ticker_base = ticker.split("-")[0].lower()
        
        # Determine if data is multi-column (multiple tickers) or single-column (one ticker)
        if isinstance(raw['Close'], pd.DataFrame):
            close_raw = raw['Close'][ticker]
            volume_raw = raw['Volume'][ticker]
            high_raw = raw['High'][ticker]
            low_raw = raw['Low'][ticker]
            
            if ticker == "BTC-USD":
                btc_raw_data = {'High': high_raw, 'Low': low_raw, 'Close': close_raw}

        else: # Single Ticker Case (unlikely with ["BTC-USD", "ETH-USD"] but safe)
            close_raw = raw['Close']
            volume_raw = raw['Volume']
            high_raw = raw['High']
            low_raw = raw['Low']
            
            if ticker == "BTC-USD":
                btc_raw_data = {'High': high_raw, 'Low': low_raw, 'Close': close_raw}

        # Handle timezones and fill NA values
        close_raw = close_raw.ffill().bfill()
        volume_raw = volume_raw.ffill().bfill()
        
        if close_raw.index.tz is not None:
            close_raw.index = close_raw.index.tz_localize(None)

        base_price = close_raw.iloc[0]
        if base_price == 0:
            status_text.error(f"Initial price for {ticker} is zero, cannot normalize.")
            return pd.DataFrame()
        
        # Normalized Cumulative Price
        df[f'{ticker_base}_cum'] = close_raw / base_price
        
        # Calculate individual asset EMAs
        df[f'{ticker_base}_ema_short'] = df[f'{ticker_base}_cum'].ewm(span=ema_short, adjust=False).mean()
        df[f'{ticker_base}_ema_long'] = df[f'{ticker_base}_cum'].ewm(span=ema_long, adjust=False).mean()
        df[f'{ticker_base}_ema_very_long'] = df[f'{ticker_base}_cum'].ewm(span=ema_very_long, adjust=False).mean()
        
        total_volume += volume_raw

    # --- 2. COMBINED INDEX ---
    df['index_cum'] = df[[f'{t.split("-")[0].lower()}_cum' for t in tickers]].mean(axis=1)
    df['index_cum_smooth'] = df['index_cum'].ewm(span=index_ema_span, adjust=False).mean()
    
    # --- ADX CALCULATION (Using BTC as proxy for Index Directional Movement) ---
    if btc_raw_data:
        # Align index for BTC raw data and remove timezone
        btc_high = btc_raw_data['High'].ffill().bfill()
        btc_low = btc_raw_data['Low'].ffill().bfill()
        btc_close = btc_raw_data['Close'].ffill().bfill()
        
        if btc_high.index.tz is not None: btc_high.index = btc_high.index.tz_localize(None)
        if btc_low.index.tz is not None: btc_low.index = btc_low.index.tz_localize(None)
        if btc_close.index.tz is not None: btc_close.index = btc_close.index.tz_localize(None)
             
        # Recalculate ADX data and join
        adx_data = calculate_adx(btc_high, btc_low, btc_close, adx_length)
        df = df.join(adx_data) 

    # --- 3. OTHER INDICATORS ---
    df['ema_14_divergence'] = df['btc_ema_short'] - df['eth_ema_short']
    df['volume'] = total_volume
    df['Volume_MA'] = df['volume'].rolling(volume_length, min_periods=1).mean()
    
    status_text.empty()
    return df.dropna(subset=['ADX', '+DI', '-DI']) # Drop rows until ADX/DI has enough data points

# --- Execution ---
df = fetch_and_process_data(TICKERS, period_days, interval, index_ema_span, ema_short, ema_long, ema_very_long, adx_length)

if df.empty:
    st.stop()

# -----------------------
# Indicators & Cycles (Calculated on the SMOOTHED Combined Index)
# -----------------------
df['EMA_short'] = df['index_cum_smooth'].ewm(span=ema_short, adjust=False).mean()
df['EMA_long'] = df['index_cum_smooth'].ewm(span=ema_long, adjust=False).mean() 
df['EMA_very_long'] = df['index_cum_smooth'].ewm(span=ema_very_long, adjust=False).mean()
df['RSI'] = rsi(df['index_cum_smooth'], length=rsi_length)

# --- RSI Cycle Detection (Unchanged Logic) ---
cycle_id = 0
in_cycle = False
cycle_type = None
cycle_start_idx = None
cycles = [] 
rsi_series = df['RSI']

df_index_list = df.index.to_list() 

if len(df_index_list) > 1:
    prev_rsi = rsi_series.iloc[0]
    for idx in df_index_list[1:]:
        cur_rsi = rsi_series.loc[idx]
        
        # Cycle start detection
        if not in_cycle:
            if (prev_rsi <= 29) and (cur_rsi > 29):
                in_cycle = True; cycle_type = 'rising'; cycle_start_idx = idx; cycle_id += 1
            elif (prev_rsi >= 71) and (cur_rsi < 71):
                in_cycle = True; cycle_type = 'falling'; cycle_start_idx = idx; cycle_id += 1
        # Cycle end detection
        else:
            if cycle_type == 'rising' and (prev_rsi < 71) and (cur_rsi >= 71):
                cycles.append({'id': cycle_id, 'type': 'rising', 'start': cycle_start_idx, 'end': idx})
                in_cycle = False; cycle_type = None; cycle_start_idx = None
            elif cycle_type == 'falling' and (prev_rsi > 29) and (cur_rsi <= 29):
                cycles.append({'id': cycle_id, 'type': 'falling', 'start': cycle_start_idx, 'end': idx})
                in_cycle = False; cycle_type = None; cycle_start_idx = None
        prev_rsi = cur_rsi

# -----------------------
# Realignment detection and signal setting 
# -----------------------
df['signal'] = 0 # 1 buy, -1 sell
df['signal_reason'] = None
df['conviction_score'] = 0
df['btc_stack_desc'] = None
df['eth_stack_desc'] = None
df['lead_contributor'] = None

def check_volume_confirmation(idx):
    if not enable_volume_filter:
        return True
    return df.at[idx, 'volume'] > df.at[idx, 'Volume_MA']

# Function to determine which asset crossed its EMA 14/30 first in the search window
def find_lead_contributor(df_slice, direction):
    btc_cross_idx = None; eth_cross_idx = None
    
    for t in df_slice.index.to_list():
        if direction == 'rising':
            if btc_cross_idx is None and df.at[t, 'btc_ema_short'] > df.at[t, 'btc_ema_long']:
                if t != df_slice.index[0]: btc_cross_idx = t
            if eth_cross_idx is None and df.at[t, 'eth_ema_short'] > df.at[t, 'eth_ema_long']:
                if t != df_slice.index[0]: eth_cross_idx = t
        elif direction == 'falling':
            if btc_cross_idx is None and df.at[t, 'btc_ema_short'] < df.at[t, 'btc_ema_long']:
                if t != df_slice.index[0]: btc_cross_idx = t
            if eth_cross_idx is None and df.at[t, 'eth_ema_short'] < df.at[t, 'eth_ema_long']:
                if t != df_slice.index[0]: eth_cross_idx = t

    if btc_cross_idx is None and eth_cross_idx is None: return 'Index Alignment Only'
    if btc_cross_idx is not None and eth_cross_idx is not None:
        if btc_cross_idx <= eth_cross_idx: return 'BTC Lead'
        else: return 'ETH Lead'
    elif btc_cross_idx is not None: return 'BTC Lead (ETH lagged)'
    elif eth_cross_idx is not None: return 'ETH Lead (BTC lagged)'
    else: return 'Unclear'


for c in cycles:
    end_idx = c['end']
    search_idx_list = df.loc[end_idx:].index.to_list()
    if len(search_idx_list) <= 1: continue
    
    lookback_window = min(len(df_index_list) - df_index_list.index(end_idx) - 1, 10) 
    
    if min_bars_after_cycle > 0:
        search_idx_list = search_idx_list[1:min_bars_after_cycle+2] 
    else:
        search_idx_list = search_idx_list[1:]

    dipped = False; spiked = False
    
    if c['type'] == 'rising':
        dip_idx = None; reclaim_idx = None
        for t in search_idx_list:
            if (not dipped) and (df.at[t, 'index_cum_smooth'] < df.at[t, 'EMA_long']):
                dipped = True; dip_idx = t
            if dipped and (df.at[t, 'index_cum_smooth'] > df.at[t, 'EMA_long']):
                reclaim_idx = t
                is_stacked = (df.at[reclaim_idx, 'EMA_short'] > df.at[reclaim_idx, 'EMA_long']) and \
                             (df.at[reclaim_idx, 'EMA_long'] > df.at[reclaim_idx, 'EMA_very_long'])
                             
                if is_stacked and check_volume_confirmation(reclaim_idx):
                    btc_score, btc_desc = get_asset_stacking_score(df.at[reclaim_idx, 'btc_ema_very_long'], df.at[reclaim_idx, 'btc_ema_long'], df.at[reclaim_idx, 'btc_ema_short'])
                    eth_score, eth_desc = get_asset_stacking_score(df.at[reclaim_idx, 'eth_ema_very_long'], df.at[reclaim_idx, 'eth_ema_long'], df.at[reclaim_idx, 'eth_ema_short'])
                    total_conviction = btc_score + eth_score
                    
                    lookback_slice = df.loc[dip_idx:reclaim_idx]
                    lead_contributor = find_lead_contributor(lookback_slice, 'rising')

                    df.at[reclaim_idx, 'signal'] = 1
                    df.at[reclaim_idx, 'conviction_score'] = total_conviction
                    df.at[reclaim_idx, 'btc_stack_desc'] = btc_desc
                    df.at[reclaim_idx, 'eth_stack_desc'] = eth_desc
                    df.at[reclaim_idx, 'lead_contributor'] = lead_contributor
                    
                    vol_note = " (Vol Confirmed)" if enable_volume_filter else ""
                    df.at[reclaim_idx, 'signal_reason'] = f"BUY | Lead: {lead_contributor} | Conviction: {total_conviction}/4{vol_note}"
                    break
                else: break
                    
    elif c['type'] == 'falling':
        spike_idx = None; drop_idx = None
        for t in search_idx_list:
            if (not spiked) and (df.at[t, 'index_cum_smooth'] > df.at[t, 'EMA_long']):
                spiked = True; spike_idx = t
            if spiked and (df.at[t, 'index_cum_smooth'] < df.at[t, 'EMA_long']):
                drop_idx = t
                is_stacked = (df.at[drop_idx, 'EMA_short'] < df.at[drop_idx, 'EMA_long']) and \
                             (df.at[drop_idx, 'EMA_long'] < df.at[drop_idx, 'EMA_very_long'])
                             
                if is_stacked and check_volume_confirmation(drop_idx):
                    btc_score, btc_desc = get_asset_stacking_score(df.at[drop_idx, 'btc_ema_very_long'], df.at[drop_idx, 'btc_ema_long'], df.at[drop_idx, 'btc_ema_short'])
                    eth_score, eth_desc = get_asset_stacking_score(df.at[drop_idx, 'eth_ema_very_long'], df.at[drop_idx, 'eth_ema_long'], df.at[drop_idx, 'eth_ema_short'])
                    total_conviction = btc_score + eth_score
                    
                    lookback_slice = df.loc[spike_idx:drop_idx]
                    lead_contributor = find_lead_contributor(lookback_slice, 'falling')
                    
                    df.at[drop_idx, 'signal'] = -1
                    df.at[drop_idx, 'conviction_score'] = total_conviction
                    df.at[drop_idx, 'btc_stack_desc'] = btc_desc
                    df.at[drop_idx, 'eth_stack_desc'] = eth_desc
                    df.at[drop_idx, 'lead_contributor'] = lead_contributor
                    
                    vol_note = " (Vol Confirmed)" if enable_volume_filter else ""
                    df.at[drop_idx, 'signal_reason'] = f"SELL | Lead: {lead_contributor} | Conviction: {total_conviction}/4{vol_note}"
                    break
                else: break

# -----------------------
# Real-time Alerting (External + Internal)
# -----------------------
latest_signal = df[df['signal'] != 0].tail(1)

if not latest_signal.empty:
    latest_time = latest_signal.index[0] 
    signal_value = latest_signal['signal'].iloc[0]
    signal_type = "BUY" if signal_value == 1 else "SELL"
    conviction_score = latest_signal['conviction_score'].iloc[0]
    lead_contributor = latest_signal['lead_contributor'].iloc[0]
    
    if latest_time > st.session_state.last_signal_time:
        st.session_state.last_signal_time = latest_time
        
        # --- 1. Internal Alert Message (in the app) ---
        alert_message = (
            f"🔔 **NEW ALERT ({signal_type})**: Cycle Realignment Signal Fired for BTC/ETH Index!\n\n"
            f"**Time**: {latest_time.strftime('%Y-%m-%d %H:%M:%S GMT')} ({interval})\n"
            f"**Action**: {signal_type}\n"
            f"**Lead Contributor**: **{lead_contributor}** (First to cross 14/30 EMA)\n"
            f"**Conviction Score**: {conviction_score} / 4 (Cross-Asset Confirmation)\n"
            f"**BTC Stacking**: {latest_signal['btc_stack_desc'].iloc[0]}\n"
            f"**ETH Stacking**: {latest_signal['eth_stack_desc'].iloc[0]}"
        )
        st.error(alert_message, icon="🚨") 

        # --- 2. External Alert Generation (Simulated) ---
        external_message = f"BTC/ETH Index ALERT ({interval}): {signal_type} @ {latest_time.strftime('%H:%M GMT')}. Lead: {lead_contributor}. Score: {conviction_score}/4."
        send_external_alert(signal_type, external_message, recipient_email, recipient_phone, conviction_score, lead_contributor)


# -----------------------
# Plotting: main chart (Price/EMAs) + RSI subplot + ADX Subplot 
# -----------------------
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.05,
                    specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]])

# 1. Price Normalized Tracks (Faded)
fig.add_trace(go.Scatter(x=df.index, y=df['btc_cum'], mode='lines', name='BTC Normalized', line=dict(color='rgba(247, 147, 26, 0.5)', dash='dash'), opacity=0.8), row=1, col=1) 
fig.add_trace(go.Scatter(x=df.index, y=df['eth_cum'], mode='lines', name='ETH Normalized', line=dict(color='rgba(130, 130, 130, 0.5)', dash='dash'), opacity=0.8), row=1, col=1) 

# 2. Combined Index (The primary line) & EMAs
fig.add_trace(go.Scatter(x=df.index, y=df['index_cum_smooth'], mode='lines', name=f'Combined Index EMA {index_ema_span}', line=dict(color='#0077c9', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_short'], mode='lines', name=f'Index EMA {ema_short}', line=dict(color='orange')), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_long'], mode='lines', name=f'Index EMA {ema_long}', line=dict(color='red')), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_very_long'], mode='lines', name=f'Index EMA {ema_very_long}', line=dict(color='purple', dash='dot')), row=1, col=1) 

fig.add_hline(y=1.0, line=dict(color='gray', dash='dash'), row=1, col=1) 
fig.update_yaxes(title="Normalized Index (Base 1.0)", row=1, col=1)


# 3. Signals Markers
buys = df[df['signal'] == 1]
sells = df[df['signal'] == -1]
if not buys.empty:
    fig.add_trace(go.Scatter(x=buys.index, y=buys['index_cum_smooth'], mode='markers', marker_symbol='triangle-up',
                             marker_color='green', marker_size=12, name='BUY', marker_line_width=1), row=1, col=1)
if not sells.empty:
    fig.add_trace(go.Scatter(x=sells.index, y=sells['index_cum_smooth'], mode='markers', marker_symbol='triangle-down',
                             marker_color='red', marker_size=12, name='SELL', marker_line_width=1), row=1, col=1)

# 4. RSI Subplot
fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name=f'RSI({rsi_length})', line=dict(color='black')), row=2, col=1)
fig.add_hrect(y0=71, y1=100, fillcolor="red", opacity=0.1, line_width=0, row=2, col=1) 
fig.add_hrect(y0=0, y1=29, fillcolor="green", opacity=0.1, line_width=0, row=2, col=1)
fig.add_hline(y=50, line=dict(color='grey', dash='dot'), row=2, col=1)
fig.update_yaxes(range=[0, 100], title="RSI", row=2, col=1) 

# 5. ADX Subplot 
fig.add_trace(go.Scatter(x=df.index, y=df['+DI'], mode='lines', name=f'+DI({adx_length})', line=dict(color='lime', width=2)), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['-DI'], mode='lines', name=f'-DI({adx_length})', line=dict(color='pink', width=2)), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], mode='lines', name=f'ADX({adx_length})', line=dict(color='blue', width=2, dash='solid')), row=3, col=1)
# ADX Thresholds
fig.add_hline(y=20, line=dict(color='grey', dash='dot', width=1), row=3, col=1) # Trend Threshold
fig.add_hline(y=40, line=dict(color='red', dash='dot', width=1), row=3, col=1) # Strong Trend Threshold

fig.update_yaxes(title=f"ADX/DI ({adx_length})", row=3, col=1) 

fig.update_layout(title="BTC/ETH Combined Index Momentum Dashboard",
                  xaxis=dict(rangeslider=dict(visible=False)), height=900, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Show table of signals and diagnostics 
# -----------------------
st.markdown("### Signals and Diagnostics")
if df['signal'].abs().sum() == 0:
    st.info("No signals found in the selected period with current parameters. Try adjusting settings.")
else:
    # Included ADX, +DI, -DI in the output table
    sig_df = df[df['signal'] != 0][['index_cum_smooth','EMA_short','EMA_long','EMA_very_long','conviction_score', 'lead_contributor', 'ema_14_divergence', 'ADX', '+DI', '-DI', 'btc_stack_desc', 'eth_stack_desc', 'RSI','volume', 'Volume_MA','signal_reason','signal']].copy()
    sig_df.index = sig_df.index.strftime('%Y-%m-%d %H:%M:%S GMT') 
    
    # Rename columns for clarity in the output table
    sig_df.rename(columns={
        'index_cum_smooth': 'Index Price',
        'conviction_score': 'Total Conviction (Max 4)',
        'ema_14_divergence': 'BTC - ETH EMA 14 Divergence',
        'btc_stack_desc': 'BTC Stacking Status',
        'eth_stack_desc': 'ETH Stacking Status',
        'lead_contributor': 'Lead Contributor (14/30 Cross)',
    }, inplace=True)
    
    st.dataframe(sig_df.tail(50))

# small metrics
st.markdown("### Summary")
st.write(f"Total cycles detected: **{len(cycles)}**")
st.write(f"Total signals detected: **{int(df['signal'].abs().sum())}** (Filtered by Combined Index Triple EMA Stack)")
st.write(f"Last signal timestamp recorded: **{st.session_state.last_signal_time.strftime('%Y-%m-%d %H:%M:%S GMT')}** (Used to prevent duplicate alerts.)")

# -----------------------
# Auto-Refresh / Manual Refresh
# -----------------------
st.markdown("---")
col_button, col_timer = st.columns([1, 4])

# Refresh button
if col_button.button(f"🔄 Refresh / Re-fetch Index Data"):
    fetch_and_process_data.clear()
    st.experimental_rerun()

# Auto-refresh timer logic
placeholder = col_timer.empty()
refresh = 300
if refresh > 0:
    for i in range(refresh, 0, -1):
        with placeholder.container():
            st.markdown(f"Next auto-refresh in **{i}** seconds...")
        time.sleep(1)
    
    fetch_and_process_data.clear()
    st.experimental_rerun()
else:
    placeholder.markdown("Auto refresh is **disabled**.")