"""Download and clean daily price data for the VN30 basket from Yahoo Finance.

Run: python data.py  -> creates data/panel.pkl (a dict of date x ticker DataFrames).
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

# VN30 constituents (2024 review). The current list is used for the whole 2019-2026 period,
# so there is survivorship bias -- stated clearly in the report.
VN30 = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
]
START, END = "2019-01-01", "2026-10-31"
OUT = Path(__file__).parent / "data" / "panel.pkl"


def download() -> dict:
    raw = yf.download([f"{t}.VN" for t in VN30], start=START, end=END,
                      auto_adjust=True, progress=False, group_by="column")
    panel = {}
    for field in ["Open", "High", "Low", "Close", "Volume"]:
        df = raw[field].copy()
        df.columns = [c.replace(".VN", "") for c in df.columns]
        panel[field.lower()] = df[VN30]
    return panel


def clean(panel: dict) -> dict:
    vol = panel["volume"]
    # Yahoo inserts holidays with volume = 0 for every ticker -> drop days where >80% of tickers did not trade
    traded = (vol > 0).sum(axis=1) / vol.notna().sum(axis=1).clip(lower=1)
    keep = traded > 0.2
    out = {k: v.loc[keep] for k, v in panel.items()}
    # A day on which a ticker did not trade (volume 0) -> treat its prices as missing, no interpolation to avoid fake signals
    no_trade = out["volume"] <= 0
    for k in ["open", "high", "low", "close"]:
        out[k] = out[k].mask(no_trade)
    out["volume"] = out["volume"].mask(no_trade)
    # HOSE price limit is ±7% (HNX ±10%): a daily move > 10.5% is almost always a stock split/bonus issue
    # that Yahoo has not adjusted (e.g. VCB -33% on 2025-03-03). Set that day's return to 0 and
    # rebuild the price series from the corrected returns; OHLC are multiplied by the same adjustment factor.
    close = out["close"]
    r = close / close.ffill().shift(1) - 1
    bad = r.abs() > 0.105
    r = r.mask(bad, 0.0)
    rebuilt = (1 + r.fillna(0)).cumprod().where(close.notna())
    factor = (rebuilt / close)
    # normalise so the last day's price = the real price (easier to read); the factor is constant between two fixes
    factor = factor / factor.ffill().iloc[-1]
    for k in ["open", "high", "low", "close"]:
        out[k] = out[k] * factor
    out["volume"] = out["volume"] / factor  # keep traded value (price x volume) unchanged
    out["n_fixed"] = int(bad.sum().sum())
    # drop sessions with too few tickers that have data
    enough = out["close"].count(axis=1) >= 10
    for k in ["open", "high", "low", "close", "volume"]:
        out[k] = out[k].loc[enough]
    return out


if __name__ == "__main__":
    panel = clean(download())
    OUT.parent.mkdir(exist_ok=True)
    pd.to_pickle(panel, OUT)
    close = panel["close"]
    print(f"{close.shape[0]} sessions, {close.index[0].date()} -> {close.index[-1].date()}")
    first = close.apply(lambda s: s.first_valid_index())
    print("First date with data for each ticker:")
    print(first.dt.date.sort_values().to_string())
    print("Share of missing cells:", round(close.isna().mean().mean(), 3))
    print("Abnormal moves fixed:", panel["n_fixed"])
    r = close.pct_change(fill_method=None).stack()
    print("Largest daily return after fixing:", round(r.max(), 3), "smallest:", round(r.min(), 3))
