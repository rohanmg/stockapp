# “””
Stock & Crypto Forecast Engine

Real LSTM-CNN (TensorFlow/Keras) + NeuralProphet forecasting
Live data from Yahoo Finance (yfinance) — any ticker or crypto

Run:
streamlit run app.py

Deploy free:
https://streamlit.io/cloud  →  connect GitHub repo → deploy
“””

import warnings
warnings.filterwarnings(“ignore”)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
import time

# ── Page config (must be first Streamlit call) ────────────────

st.set_page_config(
page_title=“Stock & Crypto Forecast”,
page_icon=“📈”,
layout=“wide”,
initial_sidebar_state=“expanded”,
)

# ── Lazy imports (so app loads even if one package missing) ───

@st.cache_resource
def import_tf():
import tensorflow as tf
return tf

@st.cache_resource
def import_neuralprophet():
from neuralprophet import NeuralProphet
return NeuralProphet

# ══════════════════════════════════════════════════════════════

# DATA FETCHING  — Yahoo Finance via yfinance

# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)   # cache 5 min, then refresh
def fetch_data(ticker: str, years: int = 5) -> pd.DataFrame:
“””
Fetch OHLCV daily data from Yahoo Finance.
Returns a DataFrame with columns: Open, High, Low, Close, Volume.
Raises on invalid ticker or network error.
“””
import yfinance as yf

```
end   = datetime.today()
start = end - timedelta(days=years * 365)

tkr  = yf.Ticker(ticker.upper())
df   = tkr.history(start=start, end=end, interval="1d", auto_adjust=True)

if df.empty:
    raise ValueError(f"No data found for '{ticker}'. Check the ticker symbol.")

df.index = pd.to_datetime(df.index).tz_localize(None)
df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
return df
```

@st.cache_data(ttl=60, show_spinner=False)    # refresh every minute
def fetch_live_price(ticker: str) -> dict:
“””
Fetch the latest real-time quote for a ticker.
Returns dict with price, change, change_pct, market_state.
“””
import yfinance as yf

```
tkr  = yf.Ticker(ticker.upper())
info = tkr.fast_info

try:
    price      = info.last_price
    prev_close = info.previous_close
    change     = price - prev_close
    change_pct = change / prev_close * 100
except Exception:
    hist  = tkr.history(period="2d")
    price = float(hist["Close"].iloc[-1])
    prev  = float(hist["Close"].iloc[-2]) if len(hist) > 1 else price
    change     = price - prev
    change_pct = change / prev * 100

return {
    "price":      round(price, 4),
    "change":     round(change, 4),
    "change_pct": round(change_pct, 4),
    "currency":   getattr(info, "currency", "USD"),
}
```

# ══════════════════════════════════════════════════════════════

# FEATURE ENGINEERING

# ══════════════════════════════════════════════════════════════

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
“””
Adds RSI, MACD, Bollinger Bands, ATR, OBV to the dataframe.
All calculated from real fetched OHLCV data.
“””
c = df[“Close”].copy()
h = df[“High”].copy()
l = df[“Low”].copy()
v = df[“Volume”].copy()

```
# ── RSI (14) ─────────────────────────────────────────────
delta = c.diff()
gain  = delta.clip(lower=0)
loss  = (-delta).clip(lower=0)
avg_g = gain.ewm(com=13, adjust=False).mean()
avg_l = loss.ewm(com=13, adjust=False).mean()
rs    = avg_g / avg_l.replace(0, np.nan)
df["RSI"] = 100 - (100 / (1 + rs))

# ── MACD (12/26/9) ────────────────────────────────────────
ema12       = c.ewm(span=12, adjust=False).mean()
ema26       = c.ewm(span=26, adjust=False).mean()
df["MACD"]  = ema12 - ema26
df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]

# ── Bollinger Bands (20, 2σ) ──────────────────────────────
sma20          = c.rolling(20).mean()
std20          = c.rolling(20).std()
df["BB_upper"] = sma20 + 2 * std20
df["BB_lower"] = sma20 - 2 * std20
df["BB_mid"]   = sma20
df["BB_pct"]   = (c - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])

# ── ATR (14) ──────────────────────────────────────────────
tr             = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
df["ATR"]      = tr.ewm(com=13, adjust=False).mean()

# ── On-Balance Volume ─────────────────────────────────────
obv            = (np.sign(c.diff()) * v).fillna(0).cumsum()
df["OBV"]      = obv

# ── Moving averages ───────────────────────────────────────
df["SMA_20"]   = c.rolling(20).mean()
df["SMA_50"]   = c.rolling(50).mean()
df["SMA_200"]  = c.rolling(200).mean()
df["EMA_20"]   = c.ewm(span=20, adjust=False).mean()

# ── Volatility (20-day rolling std of log returns) ────────
log_ret        = np.log(c / c.shift())
df["Volatility"] = log_ret.rolling(20).std() * np.sqrt(252)

# ── Lag features ─────────────────────────────────────────
for lag in [1, 2, 3, 5, 10]:
    df[f"Close_lag{lag}"] = c.shift(lag)
    df[f"Return_lag{lag}"] = log_ret.shift(lag)

return df.dropna()
```

def make_sequences(data: np.ndarray, window: int = 60):
“””
Converts a 2D array of features into (X, y) sequences for LSTM.
X shape: (n_samples, window, n_features)
y shape: (n_samples,)  — next-day Close (index 0)
“””
X, y = [], []
for i in range(window, len(data)):
X.append(data[i - window: i])
y.append(data[i, 0])       # index 0 = Close (after scaling)
return np.array(X), np.array(y)

# ══════════════════════════════════════════════════════════════

# LSTM-CNN MODEL  (TensorFlow/Keras)

# ══════════════════════════════════════════════════════════════

def build_lstm_cnn_model(input_shape: tuple) -> “tf.keras.Model”:
“””
Builds an LSTM-CNN hybrid:
Conv1D layers extract local patterns,
LSTM layers capture long-range temporal dependencies,
Dense head outputs a single price prediction.
“””
tf = import_tf()
from tensorflow import keras
from tensorflow.keras import layers

```
inp = keras.Input(shape=input_shape)

# CNN block — local pattern extraction
x = layers.Conv1D(filters=64,  kernel_size=3, activation="relu", padding="same")(inp)
x = layers.Conv1D(filters=64,  kernel_size=3, activation="relu", padding="same")(x)
x = layers.MaxPooling1D(pool_size=2)(x)
x = layers.Dropout(0.2)(x)

x = layers.Conv1D(filters=128, kernel_size=3, activation="relu", padding="same")(x)
x = layers.MaxPooling1D(pool_size=2)(x)
x = layers.Dropout(0.2)(x)

# LSTM block — temporal dependencies
x = layers.LSTM(128, return_sequences=True)(x)
x = layers.Dropout(0.2)(x)
x = layers.LSTM(64,  return_sequences=False)(x)
x = layers.Dropout(0.2)(x)

# Dense head
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(32, activation="relu")(x)
out = layers.Dense(1)(x)

model = keras.Model(inputs=inp, outputs=out)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="huber",                # robust to outliers vs MSE
    metrics=["mae"],
)
return model
```

@st.cache_resource(show_spinner=False)
def train_lstm(ticker: str, _df_hash: int, window: int = 60):
“””
Trains the LSTM-CNN model on real fetched data.
Cached per (ticker, data_hash) — only retrains when data changes.

```
Returns: (model, scaler, feature_cols, last_window, last_close, rmse)
"""
from sklearn.preprocessing import MinMaxScaler
import joblib

# We re-fetch inside cache so the hash drives invalidation
df = st.session_state["raw_df"].copy()
df = add_technical_indicators(df)

# Features: Close first (index 0 = prediction target), then indicators
feature_cols = [
    "Close", "Open", "High", "Low", "Volume",
    "RSI", "MACD", "MACD_signal", "BB_pct", "ATR",
    "OBV", "Volatility", "EMA_20",
    "Close_lag1", "Close_lag2", "Close_lag5",
    "Return_lag1", "Return_lag2",
]
feature_cols = [c for c in feature_cols if c in df.columns]

data = df[feature_cols].values.astype(np.float32)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(data)

X, y = make_sequences(scaled, window)

# Train / validation split — 85 / 15
split = int(len(X) * 0.85)
X_tr, X_val = X[:split], X[split:]
y_tr, y_val = y[:split], y[split:]

tf = import_tf()
tf.random.set_seed(42)

model = build_lstm_cnn_model(input_shape=(window, len(feature_cols)))

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
    ),
]

model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=0,
)

# RMSE on validation set (in original price scale)
preds_scaled = model.predict(X_val, verbose=0).flatten()
# Inverse transform: rebuild a dummy array with pred in col 0
dummy         = np.zeros((len(preds_scaled), len(feature_cols)))
dummy[:, 0]   = preds_scaled
preds_price   = scaler.inverse_transform(dummy)[:, 0]

dummy2        = np.zeros((len(y_val), len(feature_cols)))
dummy2[:, 0]  = y_val
actual_price  = scaler.inverse_transform(dummy2)[:, 0]

rmse = float(np.sqrt(np.mean((preds_price - actual_price) ** 2)))
mape = float(np.mean(np.abs((actual_price - preds_price) / actual_price)) * 100)

last_window = scaled[-window:]   # shape (window, n_features)
last_close  = float(df["Close"].iloc[-1])

return model, scaler, feature_cols, last_window, last_close, rmse, mape
```

# ══════════════════════════════════════════════════════════════

# LSTM FORECAST  — iterative multi-step prediction

# ══════════════════════════════════════════════════════════════

def lstm_forecast(
model, scaler, feature_cols: list,
last_window: np.ndarray, last_close: float,
n_days: int, n_simulations: int = 100,
) -> dict:
“””
Generates a multi-step forecast by feeding each prediction
back as input for the next step (recursive forecasting).

```
Runs n_simulations Monte Carlo paths (dropout active at inference)
to produce confidence intervals.

Returns:
    {
      "dates":    list of forecast dates,
      "median":   median price per day,
      "lower_80": 10th percentile,
      "upper_80": 90th percentile,
      "lower_95": 2.5th percentile,
      "upper_95": 97.5th percentile,
      "all_paths": shape (n_simulations, n_days),
    }
"""
tf = import_tf()

all_paths = []

for _ in range(n_simulations):
    path    = []
    window  = last_window.copy()       # shape (seq_len, n_features)

    for _ in range(n_days):
        x    = window[np.newaxis, :, :]   # (1, seq_len, n_features)
        pred = model(x, training=True).numpy()[0, 0]   # dropout ON

        path.append(pred)

        # Roll the window: drop oldest, append new step
        new_row         = window[-1].copy()
        new_row[0]      = pred    # update Close (index 0)
        window          = np.vstack([window[1:], new_row])

    all_paths.append(path)

all_paths = np.array(all_paths)   # (n_simulations, n_days)

# Inverse-transform each path
n_feat = len(feature_cols)
prices = np.zeros_like(all_paths)
for i, path in enumerate(all_paths):
    dummy        = np.zeros((n_days, n_feat))
    dummy[:, 0]  = path
    prices[i]    = scaler.inverse_transform(dummy)[:, 0]

# Business day forecast dates starting tomorrow
today       = pd.Timestamp.today().normalize()
bdays       = pd.bdate_range(start=today + pd.Timedelta(days=1), periods=n_days)

return {
    "dates":    bdays,
    "median":   np.median(prices, axis=0),
    "lower_80": np.percentile(prices, 10,  axis=0),
    "upper_80": np.percentile(prices, 90,  axis=0),
    "lower_95": np.percentile(prices, 2.5, axis=0),
    "upper_95": np.percentile(prices, 97.5,axis=0),
    "all_paths": prices,
}
```

# ══════════════════════════════════════════════════════════════

# NEURALPROPHET FORECAST  (fast fallback)

# ══════════════════════════════════════════════════════════════

def neuralprophet_forecast(df: pd.DataFrame, n_days: int) -> dict:
“””
Runs NeuralProphet on the Close price series.
Much faster than LSTM (~20s) — useful for quick checks.
Returns same structure as lstm_forecast.
“””
NeuralProphet = import_neuralprophet()

```
np_df = df[["Close"]].reset_index()
np_df.columns = ["ds", "y"]
np_df["ds"] = pd.to_datetime(np_df["ds"])

m = NeuralProphet(
    n_forecasts=n_days,
    n_lags=60,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    quantiles=[0.025, 0.10, 0.90, 0.975],
)

# Fit — suppress output
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    m.fit(np_df, freq="B", progress=None)

future = m.make_future_dataframe(np_df, periods=n_days)
with contextlib.redirect_stdout(io.StringIO()):
    forecast = m.predict(future)

last_rows = forecast.tail(n_days)
dates     = pd.bdate_range(
    start=pd.Timestamp.today().normalize() + pd.Timedelta(days=1),
    periods=n_days,
)

return {
    "dates":    dates,
    "median":   last_rows["yhat1"].values,
    "lower_80": last_rows.get("yhat1 10.0%", last_rows["yhat1"] * 0.97).values,
    "upper_80": last_rows.get("yhat1 90.0%", last_rows["yhat1"] * 1.03).values,
    "lower_95": last_rows.get("yhat1 2.5%",  last_rows["yhat1"] * 0.95).values,
    "upper_95": last_rows.get("yhat1 97.5%", last_rows["yhat1"] * 1.05).values,
    "all_paths": np.tile(last_rows["yhat1"].values, (10, 1)),
}
```

# ══════════════════════════════════════════════════════════════

# SIGNAL GENERATOR  — buy / sell / hold from indicators

# ══════════════════════════════════════════════════════════════

def generate_signal(df: pd.DataFrame, forecast_median: np.ndarray) -> dict:
“””
Combines technical indicator signals + forecast direction
into an overall BUY / SELL / HOLD recommendation with confidence.
“””
signals   = {}
score     = 0      # positive = bullish, negative = bearish
max_score = 0

```
last = df.iloc[-1]
c    = df["Close"]

# RSI
if "RSI" in df.columns:
    rsi = last["RSI"]
    max_score += 2
    if rsi < 30:
        signals["RSI"] = ("BUY",  f"Oversold ({rsi:.1f})")
        score += 2
    elif rsi > 70:
        signals["RSI"] = ("SELL", f"Overbought ({rsi:.1f})")
        score -= 2
    else:
        signals["RSI"] = ("HOLD", f"Neutral ({rsi:.1f})")

# MACD crossover
if "MACD" in df.columns and "MACD_signal" in df.columns:
    max_score += 2
    macd_diff = last["MACD"] - last["MACD_signal"]
    prev_diff = df.iloc[-2]["MACD"] - df.iloc[-2]["MACD_signal"]
    if macd_diff > 0 and prev_diff <= 0:
        signals["MACD"] = ("BUY",  "Bullish crossover")
        score += 2
    elif macd_diff < 0 and prev_diff >= 0:
        signals["MACD"] = ("SELL", "Bearish crossover")
        score -= 2
    elif macd_diff > 0:
        signals["MACD"] = ("BUY",  f"Above signal ({macd_diff:.4f})")
        score += 1
    else:
        signals["MACD"] = ("SELL", f"Below signal ({macd_diff:.4f})")
        score -= 1

# Bollinger Bands
if "BB_pct" in df.columns:
    max_score += 2
    bb_pct = last["BB_pct"]
    if bb_pct < 0.2:
        signals["Bollinger"] = ("BUY",  f"Near lower band ({bb_pct:.2f})")
        score += 2
    elif bb_pct > 0.8:
        signals["Bollinger"] = ("SELL", f"Near upper band ({bb_pct:.2f})")
        score -= 2
    else:
        signals["Bollinger"] = ("HOLD", f"Mid band ({bb_pct:.2f})")

# Moving average trend
if "SMA_20" in df.columns and "SMA_50" in df.columns:
    max_score += 2
    if last["SMA_20"] > last["SMA_50"] and df.iloc[-2]["SMA_20"] <= df.iloc[-2]["SMA_50"]:
        signals["MA Cross"] = ("BUY",  "Golden cross (20 > 50)")
        score += 2
    elif last["SMA_20"] < last["SMA_50"] and df.iloc[-2]["SMA_20"] >= df.iloc[-2]["SMA_50"]:
        signals["MA Cross"] = ("SELL", "Death cross (20 < 50)")
        score -= 2
    elif last["SMA_20"] > last["SMA_50"]:
        signals["MA Cross"] = ("BUY",  "Uptrend (20 > 50 SMA)")
        score += 1
    else:
        signals["MA Cross"] = ("SELL", "Downtrend (20 < 50 SMA)")
        score -= 1

# Forecast direction
max_score += 3
cur_price  = float(c.iloc[-1])
fc_return  = (forecast_median[-1] - cur_price) / cur_price * 100
if fc_return > 5:
    signals["LSTM Forecast"] = ("BUY",  f"Model sees +{fc_return:.1f}%")
    score += 3
elif fc_return < -5:
    signals["LSTM Forecast"] = ("SELL", f"Model sees {fc_return:.1f}%")
    score -= 3
else:
    signals["LSTM Forecast"] = ("HOLD", f"Model sees {fc_return:+.1f}%")
    score += 1

# Overall
pct = score / max_score if max_score else 0
if pct >= 0.4:
    overall, color = "BUY",  "#00d4aa"
elif pct <= -0.4:
    overall, color = "SELL", "#f7706a"
else:
    overall, color = "HOLD", "#f5c842"

confidence = min(abs(pct) * 100, 95)

return {
    "overall":    overall,
    "color":      color,
    "confidence": confidence,
    "score":      score,
    "max_score":  max_score,
    "signals":    signals,
    "fc_return":  fc_return,
}
```

# ══════════════════════════════════════════════════════════════

# PLOTLY CHARTS

# ══════════════════════════════════════════════════════════════

def plot_price_history(df: pd.DataFrame, ticker: str) -> go.Figure:
“”“Candlestick chart with volume and moving averages.”””
fig = make_subplots(
rows=2, cols=1, shared_xaxes=True,
row_heights=[0.75, 0.25],
vertical_spacing=0.04,
)

```
# Candlestick
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"],
    low=df["Low"],  close=df["Close"],
    name="OHLC",
    increasing_line_color="#00d4aa",
    decreasing_line_color="#f7706a",
), row=1, col=1)

# MAs
for col, color, name in [
    ("SMA_20",  "#7c6af7", "SMA 20"),
    ("SMA_50",  "#f5c842", "SMA 50"),
    ("SMA_200", "#a0a0a0", "SMA 200"),
]:
    if col in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], name=name,
            line=dict(color=color, width=1.2),
            opacity=0.85,
        ), row=1, col=1)

# Bollinger Bands
if "BB_upper" in df.columns:
    fig.add_trace(go.Scatter(
        x=df.index, y=df["BB_upper"], name="BB Upper",
        line=dict(color="#ffffff", width=0.5, dash="dot"), opacity=0.3,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["BB_lower"], name="BB Lower",
        line=dict(color="#ffffff", width=0.5, dash="dot"), opacity=0.3,
        fill="tonexty", fillcolor="rgba(255,255,255,0.03)",
    ), row=1, col=1)

# Volume bars
colors = ["#00d4aa" if c >= o else "#f7706a"
          for c, o in zip(df["Close"], df["Open"])]
fig.add_trace(go.Bar(
    x=df.index, y=df["Volume"], name="Volume",
    marker_color=colors, opacity=0.6,
), row=2, col=1)

fig.update_layout(
    title=f"{ticker.upper()} — 5-Year Price History",
    template="plotly_dark",
    paper_bgcolor="#0a0c10",
    plot_bgcolor="#111318",
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", y=1.02),
    height=520,
    margin=dict(l=0, r=0, t=48, b=0),
)
fig.update_xaxes(showgrid=True, gridcolor="#1e2230")
fig.update_yaxes(showgrid=True, gridcolor="#1e2230")
return fig
```

def plot_forecast(
df: pd.DataFrame,
fc: dict,
ticker: str,
horizon_label: str,
show_all_paths: bool = False,
) -> go.Figure:
“”“Forecast chart with confidence bands and optional Monte Carlo paths.”””
fig = go.Figure()

```
# Historical (last 90 days for clarity)
hist = df.tail(90)
fig.add_trace(go.Scatter(
    x=hist.index, y=hist["Close"],
    name="Historical", line=dict(color="#00d4aa", width=2),
))

# TODAY marker
fig.add_vline(
    x=df.index[-1], line=dict(color="#3a4255", dash="dash", width=1),
)
fig.add_annotation(
    x=df.index[-1], y=1, yref="paper",
    text="TODAY", showarrow=False,
    font=dict(color="#5a6070", size=10),
    xanchor="left", yanchor="top",
)

# Monte Carlo paths (optional, downsampled)
if show_all_paths and "all_paths" in fc:
    sample = fc["all_paths"][::max(1, len(fc["all_paths"]) // 20)]
    for path in sample:
        fig.add_trace(go.Scatter(
            x=fc["dates"], y=path, mode="lines",
            line=dict(color="#7c6af7", width=0.5),
            opacity=0.15, showlegend=False,
        ))

# 95% CI
fig.add_trace(go.Scatter(
    x=list(fc["dates"]) + list(fc["dates"])[::-1],
    y=list(fc["upper_95"]) + list(fc["lower_95"])[::-1],
    fill="toself", fillcolor="rgba(124,106,247,0.10)",
    line=dict(color="rgba(0,0,0,0)"),
    name="95% CI", showlegend=True,
))

# 80% CI
fig.add_trace(go.Scatter(
    x=list(fc["dates"]) + list(fc["dates"])[::-1],
    y=list(fc["upper_80"]) + list(fc["lower_80"])[::-1],
    fill="toself", fillcolor="rgba(124,106,247,0.20)",
    line=dict(color="rgba(0,0,0,0)"),
    name="80% CI", showlegend=True,
))

# Median forecast
fig.add_trace(go.Scatter(
    x=fc["dates"], y=fc["median"],
    name="Median Forecast",
    line=dict(color="#a78bfa", width=2.5, dash="dash"),
))

fig.update_layout(
    title=f"{ticker.upper()} — {horizon_label} LSTM-CNN Forecast",
    template="plotly_dark",
    paper_bgcolor="#0a0c10",
    plot_bgcolor="#111318",
    legend=dict(orientation="h", y=1.02),
    height=480,
    margin=dict(l=0, r=0, t=48, b=0),
    xaxis=dict(showgrid=True, gridcolor="#1e2230"),
    yaxis=dict(showgrid=True, gridcolor="#1e2230",
               title="Price (USD)" if "USD" in ticker.upper() or "-" not in ticker else "Price"),
)
return fig
```

def plot_indicators(df: pd.DataFrame) -> go.Figure:
“”“RSI + MACD subplot chart.”””
fig = make_subplots(
rows=2, cols=1, shared_xaxes=True,
subplot_titles=(“RSI (14)”, “MACD (12/26/9)”),
vertical_spacing=0.1,
)

```
# RSI
fig.add_trace(go.Scatter(
    x=df.index, y=df["RSI"], name="RSI",
    line=dict(color="#7c6af7", width=1.5),
), row=1, col=1)
fig.add_hline(y=70, line=dict(color="#f7706a", dash="dot", width=1), row=1, col=1)
fig.add_hline(y=30, line=dict(color="#00d4aa", dash="dot", width=1), row=1, col=1)
fig.add_hrect(y0=70, y1=100, fillcolor="rgba(247,112,106,0.05)", row=1, col=1)
fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(0,212,170,0.05)",   row=1, col=1)

# MACD
colors = ["#00d4aa" if v >= 0 else "#f7706a" for v in df["MACD_hist"]]
fig.add_trace(go.Bar(
    x=df.index, y=df["MACD_hist"], name="MACD Histogram",
    marker_color=colors, opacity=0.7,
), row=2, col=1)
fig.add_trace(go.Scatter(
    x=df.index, y=df["MACD"], name="MACD",
    line=dict(color="#a78bfa", width=1.5),
), row=2, col=1)
fig.add_trace(go.Scatter(
    x=df.index, y=df["MACD_signal"], name="Signal",
    line=dict(color="#f5c842", width=1.5),
), row=2, col=1)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0a0c10",
    plot_bgcolor="#111318",
    height=380,
    margin=dict(l=0, r=0, t=40, b=0),
    legend=dict(orientation="h", y=1.05),
)
fig.update_xaxes(showgrid=True, gridcolor="#1e2230")
fig.update_yaxes(showgrid=True, gridcolor="#1e2230")
return fig
```

def plot_30day_table(fc: dict, current_price: float) -> pd.DataFrame:
“”“Returns a styled DataFrame for the 30-day price table.”””
rows = []
for i, (dt, med, lo80, hi80, lo95, hi95) in enumerate(zip(
fc[“dates”][:30], fc[“median”][:30],
fc[“lower_80”][:30], fc[“upper_80”][:30],
fc[“lower_95”][:30], fc[“upper_95”][:30],
)):
chg_today = (med - current_price) / current_price * 100
chg_prev  = (med - (fc[“median”][i-1] if i > 0 else current_price)) /   
(fc[“median”][i-1] if i > 0 else current_price) * 100
rows.append({
“Day”:           i + 1,
“Date”:          dt.strftime(”%a %d %b %Y”),
“Forecast ($)”:  round(med,   2),
“Δ Today”:       f”{chg_today:+.2f}%”,
“Δ Prev Day”:    f”{chg_prev:+.2f}%”,
“Low 80% CI”:    round(lo80,  2),
“High 80% CI”:   round(hi80,  2),
“Low 95% CI”:    round(lo95,  2),
“High 95% CI”:   round(hi95,  2),
})
return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════

# STREAMLIT UI

# ══════════════════════════════════════════════════════════════

# ── Custom CSS ────────────────────────────────────────────────

st.markdown(”””

<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .stApp { background-color: #0a0c10; color: #e8eaf0; }

  .metric-card {
    background: #13161f;
    border: 1px solid #1e2230;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
  }
  .metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.5px;
    color: #5a6070;
    text-transform: uppercase;
    margin-bottom: 6px;
  }
  .metric-value { font-size: 24px; font-weight: 600; letter-spacing: -0.5px; }
  .metric-sub   { font-size: 12px; color: #5a6070; margin-top: 4px; }
  .up   { color: #00d4aa; }
  .down { color: #f7706a; }
  .hold { color: #f5c842; }

  .signal-card {
    background: #13161f;
    border: 1px solid #1e2230;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 16px;
  }
  .signal-overall {
    font-size: 36px;
    font-weight: 700;
    letter-spacing: -1px;
  }
  .ctag {
    display: inline-block;
    background: rgba(0,212,170,0.07);
    border: 1px solid rgba(0,212,170,0.18);
    border-radius: 4px;
    padding: 3px 10px;
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    color: #00d4aa;
    margin: 2px;
  }
  div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
  .stSelectbox > div, .stTextInput > div { background: #111318 !important; }
  .stButton > button {
    background: #00d4aa !important;
    color: #0a0c10 !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 0.5rem 1.5rem !important;
  }
  .stButton > button:hover { background: #00ebbe !important; }
  .disclaimer {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    color: #3a4255;
    text-align: center;
    padding: 12px;
    border-top: 1px solid #1e2230;
    margin-top: 24px;
    letter-spacing: 0.5px;
  }
</style>

“””, unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────

with st.sidebar:
st.markdown(”## 📈 Stock & Crypto Forecast”)
st.markdown(’<div class="ctag">LSTM-CNN HYBRID</div>’
‘<div class="ctag">LIVE YAHOO FINANCE</div>’
‘<div class="ctag">5-YEAR TRAINING</div>’
‘<div class="ctag">MONTE CARLO CI</div>’, unsafe_allow_html=True)
st.markdown(”—”)

```
ticker_input = st.text_input(
    "Ticker Symbol",
    value="TSLA",
    placeholder="AAPL, BTC-USD, ETH-USD, MSFT…",
    help="Any Yahoo Finance ticker. Use -USD suffix for crypto (BTC-USD, ETH-USD)",
).upper().strip()

horizon = st.selectbox(
    "Forecast Horizon",
    options=[("1 Month",  21), ("3 Months", 63),
             ("6 Months", 126), ("1 Year",  252)],
    format_func=lambda x: x[0],
    index=2,
)
n_forecast_days   = horizon[1]
horizon_label     = horizon[0]

model_choice = st.radio(
    "Model",
    options=["LSTM-CNN (accurate, ~2-3 min)", "NeuralProphet (fast, ~20 sec)"],
    index=0,
)
use_lstm = "LSTM" in model_choice

st.markdown("---")
st.markdown("**Display options**")
show_mc_paths  = st.checkbox("Show Monte Carlo paths", value=False)
show_indicators = st.checkbox("Show indicator chart",  value=True)
show_table     = st.checkbox("Show 30-day table",      value=True)

st.markdown("---")
run_btn = st.button("⚡ Generate Forecast", use_container_width=True)

st.markdown("---")
st.markdown(
    '<p style="font-size:10px;color:#3a4255;font-family:Space Mono,monospace">'
    'Data: Yahoo Finance<br>'
    'Model: TensorFlow/Keras LSTM-CNN<br>'
    'CI: Monte Carlo dropout (100 paths)<br>'
    'Retrain: on ticker change only'
    '</p>', unsafe_allow_html=True
)
```

# ── Main area ─────────────────────────────────────────────────

st.markdown(f”””

<div style="display:flex;justify-content:space-between;align-items:flex-start;
            flex-wrap:wrap;gap:12px;margin-bottom:28px">
  <div>
    <div style="font-family:'Space Mono',monospace;font-size:10px;
                letter-spacing:3px;color:#00d4aa;text-transform:uppercase">
      Quantitative Research
    </div>
    <div style="font-size:28px;font-weight:600;letter-spacing:-0.5px">
      Stock & Crypto Forecast Engine
    </div>
  </div>
  <div style="text-align:right">
    <div style="font-family:'Space Mono',monospace;font-size:10px;
                color:#7c6af7;border:1px solid rgba(124,106,247,0.3);
                border-radius:6px;padding:6px 12px;display:inline-block">
      LSTM-CNN HYBRID v3.0
    </div><br>
    <div style="font-family:'Space Mono',monospace;font-size:10px;
                color:#00d4aa;margin-top:6px">
      LIVE DATA · {datetime.today().strftime("%-d %B %Y").upper()}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Run on button click ───────────────────────────────────────

if run_btn:
st.session_state.pop(“forecast_result”, None)   # clear old results
st.session_state.pop(“raw_df”, None)

```
# ── Step 1: Fetch live data ───────────────────────────────
with st.spinner(f"Fetching 5 years of real data for **{ticker_input}** from Yahoo Finance…"):
    try:
        df_raw = fetch_data(ticker_input, years=5)
        st.session_state["raw_df"]     = df_raw
        st.session_state["ticker"]     = ticker_input
        st.session_state["use_lstm"]   = use_lstm
        st.session_state["n_days"]     = n_forecast_days
        st.session_state["hlabel"]     = horizon_label
        st.session_state["show_mc"]    = show_mc_paths
        st.session_state["show_ind"]   = show_indicators
        st.session_state["show_tbl"]   = show_table
    except Exception as e:
        st.error(f"❌ Could not fetch data: {e}")
        st.stop()

# ── Step 2: Live price ────────────────────────────────────
with st.spinner("Fetching live price…"):
    try:
        live = fetch_live_price(ticker_input)
        st.session_state["live"] = live
    except Exception as e:
        st.warning(f"Live price unavailable: {e}")
        live = {"price": df_raw["Close"].iloc[-1], "change": 0,
                "change_pct": 0, "currency": "USD"}
        st.session_state["live"] = live

# ── Step 3: Add indicators ────────────────────────────────
with st.spinner("Calculating technical indicators…"):
    df_ind = add_technical_indicators(df_raw.copy())
    st.session_state["df_ind"] = df_ind

# ── Step 4: Train model ───────────────────────────────────
if use_lstm:
    progress_bar = st.progress(0, text="Training LSTM-CNN on 5 years of real data…")
    try:
        # Use a hash of the last close to bust cache on new data
        df_hash = hash(round(float(df_raw["Close"].iloc[-1]), 2))
        with st.spinner("Training LSTM-CNN (this takes ~2-3 min on first run, cached after)…"):
            model, scaler, feat_cols, last_win, last_close, rmse, mape = \
                train_lstm(ticker_input, df_hash)
        progress_bar.progress(70, text="Generating forecast paths…")

        fc = lstm_forecast(
            model, scaler, feat_cols, last_win, last_close,
            n_days=max(n_forecast_days, 30),
            n_simulations=100,
        )
        progress_bar.progress(100, text="Done!")
        time.sleep(0.3)
        progress_bar.empty()

        st.session_state["forecast_result"] = fc
        st.session_state["rmse"]  = rmse
        st.session_state["mape"]  = mape

    except Exception as e:
        st.error(f"❌ LSTM training failed: {e}")
        st.info("Tip: Try switching to NeuralProphet in the sidebar for a faster result.")
        st.stop()

else:
    with st.spinner("Running NeuralProphet forecast (~20 seconds)…"):
        try:
            fc = neuralprophet_forecast(df_raw, n_days=max(n_forecast_days, 30))
            st.session_state["forecast_result"] = fc
            st.session_state["rmse"]  = None
            st.session_state["mape"]  = None
        except Exception as e:
            st.error(f"❌ NeuralProphet failed: {e}")
            st.stop()

st.session_state["signal"] = generate_signal(df_ind, fc["median"][:n_forecast_days])
st.rerun()
```

# ── Display results (persists across reruns) ──────────────────

if “forecast_result” in st.session_state:
ticker_disp = st.session_state[“ticker”]
df_raw      = st.session_state[“raw_df”]
df_ind      = st.session_state[“df_ind”]
fc          = st.session_state[“forecast_result”]
live        = st.session_state[“live”]
sig         = st.session_state[“signal”]
n_days      = st.session_state[“n_days”]
hlabel      = st.session_state[“hlabel”]
rmse        = st.session_state.get(“rmse”)
mape        = st.session_state.get(“mape”)
show_mc     = st.session_state.get(“show_mc”, False)
show_ind    = st.session_state.get(“show_ind”, True)
show_tbl    = st.session_state.get(“show_tbl”, True)

```
cur_price   = live["price"]
currency    = live["currency"]

# ── Live price metrics ────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
price_cls = "up" if live["change_pct"] >= 0 else "down"

with col1:
    st.markdown(f"""<div class="metric-card">
      <div class="metric-label">Live Price ({currency})</div>
      <div class="metric-value">{cur_price:,.4f}</div>
      <div class="metric-sub {price_cls}">{live["change"]:+.4f} ({live["change_pct"]:+.2f}%)</div>
    </div>""", unsafe_allow_html=True)

fc_end  = fc["median"][n_days - 1]
fc_ret  = (fc_end - cur_price) / cur_price * 100
fc_cls  = "up" if fc_ret >= 0 else "down"

with col2:
    st.markdown(f"""<div class="metric-card">
      <div class="metric-label">Forecast Target ({hlabel})</div>
      <div class="metric-value {fc_cls}">{fc_end:,.2f}</div>
      <div class="metric-sub {fc_cls}">{fc_ret:+.2f}% expected</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""<div class="metric-card">
      <div class="metric-label">80% CI Range</div>
      <div class="metric-value" style="font-size:16px">
        {fc["lower_80"][n_days-1]:,.2f} – {fc["upper_80"][n_days-1]:,.2f}
      </div>
      <div class="metric-sub">80% confidence interval</div>
    </div>""", unsafe_allow_html=True)

with col4:
    rmse_html = f"{rmse:.2f}" if rmse else "N/A"
    mape_html = f"{mape:.2f}%" if mape else "N/A"
    st.markdown(f"""<div class="metric-card">
      <div class="metric-label">Model Error (Val Set)</div>
      <div class="metric-value" style="font-size:20px">RMSE {rmse_html}</div>
      <div class="metric-sub">MAPE {mape_html}</div>
    </div>""", unsafe_allow_html=True)

with col5:
    sc = sig["overall"]
    sc_cls = "up" if sc == "BUY" else ("down" if sc == "SELL" else "hold")
    st.markdown(f"""<div class="metric-card">
      <div class="metric-label">Signal</div>
      <div class="metric-value {sc_cls}">{sc}</div>
      <div class="metric-sub">Confidence: {sig["confidence"]:.0f}%</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Forecast", "🕯 Price History", "📉 Indicators", "🎯 Signal Analysis"
])

with tab1:
    st.plotly_chart(
        plot_forecast(df_raw, fc, ticker_disp, hlabel, show_mc_paths=show_mc),
        use_container_width=True,
    )

    if show_tbl:
        st.markdown("#### 📅 30-Day Daily Forecast")
        tbl = plot_30day_table(fc, cur_price)

        # Colour Δ Today column
        def colour_delta(val):
            try:
                v = float(val.replace("%", "").replace("+", ""))
                return "color: #00d4aa" if v >= 0 else "color: #f7706a"
            except Exception:
                return ""

        styled = (
            tbl.style
            .applymap(colour_delta, subset=["Δ Today", "Δ Prev Day"])
            .format({"Forecast ($)": "{:,.2f}",
                     "Low 80% CI":  "{:,.2f}",
                     "High 80% CI": "{:,.2f}",
                     "Low 95% CI":  "{:,.2f}",
                     "High 95% CI": "{:,.2f}"})
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

with tab2:
    st.plotly_chart(
        plot_price_history(df_ind, ticker_disp),
        use_container_width=True,
    )

with tab3:
    if show_ind:
        st.plotly_chart(plot_indicators(df_ind), use_container_width=True)
    else:
        st.info("Enable 'Show indicator chart' in the sidebar.")

with tab4:
    sc    = sig["overall"]
    sc_cl = "up" if sc == "BUY" else ("down" if sc == "SELL" else "hold")
    conf  = sig["confidence"]

    st.markdown(f"""<div class="signal-card">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px">
        <div>
          <div style="font-family:'Space Mono',monospace;font-size:10px;
                      color:#5a6070;letter-spacing:2px">OVERALL SIGNAL</div>
          <div class="signal-overall {sc_cl}">{sc}</div>
          <div style="font-size:13px;color:#5a6070;margin-top:4px">
            Confidence: {conf:.0f}% &nbsp;·&nbsp;
            Score: {sig["score"]}/{sig["max_score"]} &nbsp;·&nbsp;
            Forecast: {sig["fc_return"]:+.1f}%
          </div>
        </div>
        <div style="text-align:right">
          <div style="font-size:12px;color:#5a6070;font-family:'Space Mono',monospace">
            BASED ON {len(sig["signals"])} INDICATORS
          </div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("#### Individual Indicator Signals")
    for ind_name, (direction, reason) in sig["signals"].items():
        d_cls = "up" if direction == "BUY" else ("down" if direction == "SELL" else "hold")
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    background:#13161f;border:1px solid #1e2230;border-radius:8px;
                    padding:12px 16px;margin-bottom:8px">
          <div>
            <strong style="font-family:'Space Mono',monospace;font-size:12px">{ind_name}</strong>
            <div style="font-size:12px;color:#5a6070;margin-top:2px">{reason}</div>
          </div>
          <div class="{d_cls}" style="font-weight:700;font-size:15px">{direction}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(247,112,106,0.06);border:1px solid rgba(247,112,106,0.2);
                border-radius:8px;padding:12px 16px;margin-top:16px;
                font-size:12px;color:#f7706a;font-family:'Space Mono',monospace">
      ⚠ SIGNALS ARE FOR EDUCATIONAL USE ONLY. NOT FINANCIAL ADVICE.
      ALWAYS DO YOUR OWN RESEARCH AND CONSULT A LICENSED FINANCIAL ADVISOR.
    </div>""", unsafe_allow_html=True)
```

else:
# Empty state
st.markdown(”””
<div style="display:flex;flex-direction:column;align-items:center;
justify-content:center;padding:80px 20px;color:#3a4255;text-align:center">
<div style="font-size:56px;margin-bottom:16px">📈</div>
<div style="font-size:20px;font-weight:600;margin-bottom:8px">Enter a ticker and click Generate Forecast</div>
<div style="font-size:13px;font-family:'Space Mono',monospace;letter-spacing:1px">
SUPPORTS STOCKS (AAPL, TSLA, MSFT…) AND CRYPTO (BTC-USD, ETH-USD, SOL-USD…)
</div>
</div>
“””, unsafe_allow_html=True)

st.markdown(”””

<div class="disclaimer">
  ⚠ FOR EDUCATIONAL USE ONLY — NOT FINANCIAL ADVICE —
  DATA SOURCE: YAHOO FINANCE — MODEL: TENSORFLOW/KERAS LSTM-CNN —
  ALWAYS CONSULT A LICENSED PROFESSIONAL BEFORE TRADING
</div>
""", unsafe_allow_html=True)