# Stock & Crypto Forecast Engine — Setup & Deployment Guide

============================================================

## What this app does

- Fetches REAL live prices and 5-year historical OHLCV data from Yahoo Finance
- Trains a real LSTM-CNN model (TensorFlow/Keras) on that fetched data
- Generates forecasts with 80% and 95% confidence intervals via Monte Carlo dropout
- Supports ANY ticker: AAPL, TSLA, BTC-USD, ETH-USD, SOL-USD, NFLX, NVDA, etc.
- Outputs: price forecast + confidence bands + buy/sell/hold signal + 30-day table
- NeuralProphet as a fast fallback (~20 seconds vs ~2-3 minutes for LSTM)

-----

## File structure

```
stockapp/
├── app.py                  ← Main Streamlit app (everything in one file)
├── requirements.txt        ← Python dependencies
├── .streamlit/
│   └── config.toml         ← Dark theme + server config
└── DEPLOY.md               ← This file
```

-----

## Run locally

### 1. Create a virtual environment (recommended)

```
python3 -m venv venv
source venv/bin/activate          # Mac/Linux
venv\Scripts\activate             # Windows
```

### 2. Install dependencies

```
pip install -r requirements.txt

# TensorFlow on Apple Silicon Mac:
pip install tensorflow-macos tensorflow-metal

# If NeuralProphet install fails, run this first:
pip install pystan==2.19.1.1
pip install neuralprophet
```

### 3. Run

```
streamlit run app.py
# Opens at http://localhost:8501
```

-----

## Deploy FREE on Streamlit Community Cloud

This is the easiest free deployment — permanent public URL, no credit card.

### Steps

1. Push the stockapp/ folder to a GitHub repo:
   
   ```
   git init
   git add .
   git commit -m "initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/stock-forecast.git
   git push -u origin main
   ```
1. Go to https://share.streamlit.io
- Sign in with GitHub
- Click “New app”
- Repository: YOUR_USERNAME/stock-forecast
- Branch: main
- Main file path: app.py
- Click “Deploy”
1. Your app is live at:
   https://YOUR_USERNAME-stock-forecast-app-XXXX.streamlit.app

### Important: Memory limit on free tier

Streamlit Cloud free tier has 1GB RAM.

- LSTM training on 5 years of data uses ~400-600MB — should fit.
- If you hit memory errors, switch to NeuralProphet in the sidebar.
- Consider adding @st.cache_resource to the model so it only trains once per session.

-----

## Deploy on Render (free tier)

1. Push to GitHub (same as above)
1. Go to https://render.com → New → Web Service
1. Connect your GitHub repo
1. Settings:
   Runtime:        Python 3
   Build Command:  pip install -r requirements.txt
   Start Command:  streamlit run app.py –server.port $PORT –server.address 0.0.0.0
   Instance Type:  Free (512MB RAM — use NeuralProphet on this tier)
1. Done — live at https://your-app.onrender.com

Note: Free Render instances spin down after 15 minutes of inactivity.
First request after spin-down takes ~30 seconds to wake up.

-----

## Deploy on Railway (free $5/month credit)

1. Go to https://railway.app → New Project → Deploy from GitHub repo
1. Add environment variable: PORT=8501
1. Add a Procfile:
   web: streamlit run app.py –server.port $PORT –server.address 0.0.0.0
1. Deploy

-----

## Optional: Add a live price API key (Alpha Vantage)

yfinance is free and no key needed, but can be rate-limited.
To add Alpha Vantage as a fallback, update fetch_live_price() in app.py:

```
import requests

AV_KEY = "YOUR_FREE_KEY"   # get at https://www.alphavantage.co/support/#api-key

def fetch_live_price_av(ticker):
    url = (f"https://www.alphavantage.co/query"
           f"?function=GLOBAL_QUOTE&symbol={ticker}&apikey={AV_KEY}")
    data = requests.get(url).json()["Global Quote"]
    return {
        "price":      float(data["05. price"]),
        "change":     float(data["09. change"]),
        "change_pct": float(data["10. change percent"].replace("%","")),
        "currency":   "USD",
    }
```

Store the key safely using Streamlit secrets:

- Create .streamlit/secrets.toml locally: AV_KEY = “your_key”
- On Streamlit Cloud: Settings → Secrets → paste the key
- Access in code: st.secrets[“AV_KEY”]

-----

## Ticker examples

Stocks:        AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, NFLX, AMD
Crypto:        BTC-USD, ETH-USD, SOL-USD, BNB-USD, ADA-USD, DOGE-USD
Indices:       ^GSPC (S&P 500), ^IXIC (Nasdaq), ^DJI (Dow Jones)
ETFs:          SPY, QQQ, ARKK, GLD
International: RELIANCE.NS (NSE India), 0700.HK (Tencent), BABA

-----

## Model details

Architecture:  Conv1D → Conv1D → MaxPool → Dropout
→ Conv1D → MaxPool → Dropout
→ LSTM(128) → Dropout
→ LSTM(64)  → Dropout
→ Dense(64) → Dense(32) → Dense(1)

Loss:          Huber (robust to outliers)
Optimizer:     Adam (lr=0.001) with ReduceLROnPlateau
Regularisation: Dropout(0.2) + EarlyStopping(patience=10)
Split:         85% train / 15% validation
Window:        60 trading days look-back
Confidence:    Monte Carlo dropout, 100 paths, 80% + 95% percentiles

Features used per timestep:
Close, Open, High, Low, Volume,
RSI(14), MACD, MACD Signal, BB%, ATR(14),
OBV, Volatility(20d), EMA(20),
Close lags (1, 2, 5 days), Return lags (1, 2 days)

NeuralProphet (fallback):
n_lags=60, yearly + weekly seasonality,
quantile regression for CI (2.5%, 10%, 90%, 97.5%)