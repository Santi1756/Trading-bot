import os
import math
import datetime as dt
import pytz
import time

import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from alpaca_trade_api.rest import REST

# === CONFIG ===
ALPACA_API_KEY    = 'PKJ2KEGNB3NRQB38A93Y'
ALPACA_SECRET_KEY = 'kJsHKF7n3cys0yCbvtL3YaQPCSLc1MTmctayguHx'
ALPACA_BASE_URL   = 'https://paper-api.alpaca.markets'  # paper trading

WATCHLIST   = ['AAPL', 'MSFT', 'GOOG', 'TSLA']
CONF_THRESH = 0.6    # buy if P(up)>0.6, sell if P(up)<0.4

EAST         = pytz.timezone('US/Eastern')
MARKET_OPEN  = dt.time(9, 30)
MARKET_CLOSE = dt.time(16, 0)

# === REST client & smoke-test ===
alpaca_api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)
try:
    acct = alpaca_api.get_account()
    if acct.trading_blocked:
        raise SystemExit("🚫 Trading is blocked—check your account status.")
    print(f"✅ Connected! Account status: {acct.status}")
except Exception as e:
    raise SystemExit(f"❌ Alpaca auth error: {e}")


def in_market_hours(now_utc):
    now_e = now_utc.astimezone(EAST).time()
    return MARKET_OPEN <= now_e <= MARKET_CLOSE


def get_data(symbol):
    """
    Download the last 60 calendar days of daily bars.
    If yf.download() fails or returns empty, fall back to Ticker.history().
    """
    end   = dt.datetime.now().date()
    start = end - dt.timedelta(days=60)
    try:
        df = yf.download(
            symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval='1d',
            progress=False,
            threads=False
        )
    except Exception as e:
        print(f"❌ {symbol} download() error: {e}. Trying history() fallback.")
        df = yf.Ticker(symbol).history(period='60d', interval='1d')

    # If still empty, return right away
    if df.empty:
        return df

    # Compute features & label
    df['Return'] = df['Close'].pct_change().shift(-1)
    df['SMA_5']  = df['Close'].rolling(5).mean()
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['Target'] = (df['Return'] > 0).astype(int)
    return df.dropna()


def train_model(df):
    X = df[['SMA_5', 'SMA_10']]
    y = df['Target']
    m = RandomForestClassifier(n_estimators=100)
    m.fit(X, y)
    return m


def get_signal(model, df):
    # Predict probability of an up-day, return as float
    return float(model.predict_proba(df[['SMA_5','SMA_10']].tail(1))[0,1])


def gather_signals():
    cash = float(alpaca_api.get_account().cash)
    signals = []
    for sym in WATCHLIST:
        df = get_data(sym)
        if df.empty:
            print(f"⚠️ {sym}: no data returned, skipping.")
            continue

        try:
            m = train_model(df)
            p = get_signal(m, df)
            signals.append((sym, p))
        except Exception as e:
            print(f"⚠️ {sym} model error: {e}")
    return cash, signals


def flatten_weak(signals):
    positions = {p.symbol: p for p in alpaca_api.list_positions()}
    for sym, p in signals:
        if sym in positions and p < (1 - CONF_THRESH):
            qty = int(float(positions[sym].qty))
            alpaca_api.submit_order(
                symbol=sym,
                qty=qty,
                side='sell',
                type='market',
                time_in_force='gtc'
            )
            print(f"🔽 Sold {sym} @ P(up)={p:.2f}")


def trade_best(signals, cash):
    if not signals:
        return
    sym, p = max(signals, key=lambda x: x[1])
    p = float(p)
    if p <= CONF_THRESH:
        print("No buy signals above threshold.")
        return

    # Fetch the very latest close price
    df1 = yf.download(sym, period='1d', interval='1d', progress=False, threads=False)
    if df1.empty:
        print(f"⚠️ No price data for {sym}, skipping buy.")
        return
    price = float(df1['Close'].iloc[-1])

    # Determine how many shares we can buy
    qty = math.floor((cash * p) / price)
    if qty < 1:
        print(f"Insufficient funds to buy a share of {sym}.")
        return

    alpaca_api.submit_order(
        symbol=sym,
        qty=qty,
        side='buy',
        type='market',
        time_in_force='gtc'
    )
    print(f"🔼 Bought {qty} {sym} @ ${price:.2f} (P(up)={p:.2f})")


if __name__ == "__main__":
    print("🚀 Starting continuous trading loop…")
    while True:
        now = dt.datetime.now(tz=pytz.UTC)
        if in_market_hours(now):
            cash, signals = gather_signals()
            flatten_weak(signals)
            trade_best(signals, cash)
        else:
            print("Market closed—sleeping…")
        time.sleep(1)
