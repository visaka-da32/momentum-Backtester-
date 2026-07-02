# run_demo.py
import os
from src.data_io import fetch_daily, fetch_intraday, get_shares_info, minute_fallback_from_daily
from src.features import compute_monthly_momentum_from_daily, compute_monthly_turnover, compute_intraday_features_minute
from src.models import train_ridge_time_series
from src.backtester import SimpleEventBacktester
from src.utils import ensure_dir, sharpe, save_plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS = "results"
ensure_dir(RESULTS)

DEFAULT_TICKERS = ["AAPL","MSFT","AMZN","GOOGL","NVDA","TSLA","META","JPM","BAC","WMT",
                   "PG","KO","DIS","CSCO","ORCL","INTC","AMD","NFLX","C","GS"]

def assign_deciles_per_date(series, n=10):
    s = series.dropna()
    if s.empty:
        return pd.Series(index=series.index, data=np.nan)
    try:
        labels = pd.qcut(s, q=n, labels=False, duplicates='drop')
        return labels.reindex(series.index)
    except Exception:
        ranks = series.rank(method='first', pct=True)
        bins = np.floor(ranks * n).astype(int)
        bins[bins == n] = n - 1
        return pd.Series(bins, index=series.index)

def monthly_replication(daily_df, shares_info):
    monthly = compute_monthly_momentum_from_daily(daily_df, lookback_months=12, skip_months=1)
    monthly = compute_monthly_turnover(monthly, shares_info_map=shares_info, lookback_months=3)

    if 'mom_J' not in monthly.columns:
        for alt in ['momentum','mom','mom_J']:
            if alt in monthly.columns:
                monthly = monthly.rename(columns={alt: 'mom_J'})
                break

    df = monthly.dropna(subset=['mom_J']).copy()
    if df.empty:
        print("No monthly momentum data available after cleaning.")
        return

    df['decile'] = df.groupby('date')['mom_J'].transform(lambda s: assign_deciles_per_date(s, n=10))
    # compute next month return using adj_close
    df['next_ret'] = df.groupby('ticker')['adj_close'].pct_change().shift(-1)
    df = df.dropna(subset=['next_ret','decile'])

    if df.empty:
        print("No rows after next_ret/decile filtering.")
        return

    ew = df.groupby(['date','decile'])['next_ret'].mean().unstack(level='decile')
    if ew.empty:
        print("No decile returns calculated.")
        return

    if 9 in ew.columns and 0 in ew.columns:
        mom_ret = ew[9] - ew[0]
    else:
        top = ew.max(axis=1)
        bot = ew.min(axis=1)
        mom_ret = top - bot

    mom_ret = mom_ret.dropna()
    if mom_ret.empty:
        print("No momentum returns computed.")
        return

    print("Monthly replication: mean monthly mom_ret:", mom_ret.mean())
    print("Sharpe (annualized, 12 periods):", sharpe(mom_ret.values, freq_per_year=12))

    cum = (1 + mom_ret).cumprod()
    fig = plt.figure(figsize=(8,3))
    plt.plot(cum.index, cum.values)
    plt.title("Cumulative Top-minus-Bottom Momentum (yfinance)")
    save_plot(fig, os.path.join(RESULTS, "monthly_mom_cum.png"))

def intraday_pipeline(minute_df, daily_df):
    if minute_df.empty:
        print("No intraday 1m data found via yfinance; using synthetic fallback from daily data.")
        minute_df = minute_fallback_from_daily(daily_df, tickers=daily_df['ticker'].unique())

    feats = compute_intraday_features_minute(minute_df, window_minutes=30)

    # Defensive coercion: ensure numeric volume in daily_df before grouping
    daily_df = daily_df.copy()
    if 'volume' in daily_df.columns:
        daily_df['volume'] = pd.to_numeric(daily_df['volume'], errors='coerce')
    else:
        daily_df['volume'] = np.nan

    # adv_map: average daily volume (fill missing with a sensible default)
    adv_map_series = daily_df.groupby('ticker')['volume'].mean()
    adv_map = {}
    for t, v in adv_map_series.items():
        if pd.isna(v) or v <= 0:
            adv_map[t] = 100_000  # fallback adv
        else:
            adv_map[t] = float(v)

    # shares_map: last known adj_close per ticker (robust)
    if 'adj_close' in daily_df.columns:
        shares_map_series = daily_df.groupby('ticker')['adj_close'].apply(
            lambda s: pd.to_numeric(s, errors='coerce').dropna().iloc[-1] if s.notna().any() else np.nan
        )
    else:
        # fallback to 'close' if adj_close missing
        shares_map_series = daily_df.groupby('ticker')['close'].apply(
            lambda s: pd.to_numeric(s, errors='coerce').dropna().iloc[-1] if s.notna().any() else np.nan
        )
    shares_map = shares_map_series.to_dict()

    # compute per-ticker volatility map (coerce returns to numeric)
    if 'adj_close' in daily_df.columns:
        daily_df['ret'] = daily_df.groupby('ticker')['adj_close'].pct_change()
    else:
        daily_df['ret'] = daily_df.groupby('ticker')['close'].pct_change()
    daily_df['ret'] = pd.to_numeric(daily_df['ret'], errors='coerce')
    vol_map_series = daily_df.groupby('ticker')['ret'].std()
    vol_map = {}
    for t, v in vol_map_series.items():
        vol_map[t] = float(v) if (not pd.isna(v) and v > 0) else 0.02

    f = feats.dropna().copy()
    # target next-minute return
    f['next_price'] = f.groupby('ticker')['price'].shift(-1)
    f['next_ret'] = f['next_price'] / f['price'] - 1
    f = f.dropna(subset=['next_ret'])
    feature_cols = ['ret_1m','ret_5m','vol_roll_sum','vol_zscore','signed_vol_roll']
    available_feats = [c for c in feature_cols if c in f.columns]
    if len(available_feats) == 0:
        raise RuntimeError("No intraday features available for training.")

    X = f[available_feats].values
    y = f['next_ret'].values
    split = int(len(X) * 0.7) if len(X) > 100 else int(len(X) * 0.6)
    model_dict, mses = train_ridge_time_series(X[:split], y[:split], n_splits=3, alpha=1.0)
    scaler = model_dict['scaler']
    model = model_dict['model']

    Xs = scaler.transform(X)
    scores = model.predict(Xs)
    f = f.iloc[:len(scores)].copy()
    f['score'] = scores
    print("Intraday model CV MSEs (training folds):", mses)
    return f, shares_map, adv_map, vol_map

def backtest_run(feature_df, shares_map, adv_map, vol_map):
    """
    Run the SimpleEventBacktester. Ensure 'score' exists in feature_df and is passed
    into the backtester so predict_fn can access it.
    """
    # Ensure 'score' exists
    if 'score' not in feature_df.columns:
        print("[backtest_run] warning: 'score' column missing — filling with zeros so backtest can run.")
        feature_df = feature_df.copy()
        feature_df['score'] = 0.0

    # canonical columns we pass into the backtester (must include 'score')
    columns_for_bt = ['datetime','ticker','price','ret_1m','ret_5m','vol_roll_sum','vol_zscore','signed_vol_roll','score']
    # if any of these columns are missing, create them (NaN) to avoid KeyError in the backtester
    for c in columns_for_bt:
        if c not in feature_df.columns:
            feature_df[c] = np.nan

    # pass the dataframe with 'score' column included
    bt = SimpleEventBacktester(feature_df[columns_for_bt].copy(), shares_map, adv_map, vol_map, cash=1_000_000, latency_ms=5)

    def predict_fn(row):
        # row is a pandas Series representing a single event; 'score' should exist
        # return 0 if missing as a safety fallback
        try:
            return float(row.get('score', 0.0))
        except Exception:
            return 0.0

    bt.run_strategy(predict_fn, size_shares=50, threshold=1e-5)
    pnl_df, trades_df, positions, cash, pv_series = bt.results()
    if len(pnl_df) > 0:
        print("Backtest total PnL:", pnl_df['pnl'].sum())
        print("Trades made:", len(trades_df))
        fig = plt.figure(figsize=(8,3))
        plt.plot(pnl_df['datetime'], pnl_df['pnl'].cumsum())
        plt.title("Cumulative PnL (simple event backtest)")
        save_plot(fig, os.path.join(RESULTS, "intraday_cum_pnl.png"))
        trades_df.to_csv(os.path.join(RESULTS, "trades.csv"), index=False)
    else:
        print("No PnL data from backtest.")

def main():
    tickers = DEFAULT_TICKERS
    print("Fetching daily data via yfinance...")
    daily_df = fetch_daily(tickers, start="2018-01-01", end="2024-12-31", verbose=True)
    print("Fetching shares info...")
    shares_info = get_shares_info(tickers)
    print("Running monthly replication...")
    monthly_replication(daily_df, shares_info)
    print("Attempting to fetch intraday minute bars (recent ~7d)...")
    minute_df = fetch_intraday(tickers, period='7d', interval='1m', verbose=True)
    print("Preparing intraday features & model...")
    features_df, shares_map, adv_map, vol_map = intraday_pipeline(minute_df, daily_df)
    print("Running backtest...")
    backtest_run(features_df, shares_map, adv_map, vol_map)
    print("Done. Results in", RESULTS)

if __name__ == "__main__":
    main()
