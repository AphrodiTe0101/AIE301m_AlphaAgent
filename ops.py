"""The alpha language (DSL): the operators the agent is allowed to use.

Every variable is a DataFrame (date x ticker). ts_* operators only look into the past (rolling),
cs_* / rank operators compare tickers within the same day -> no look-ahead.
"""
import numpy as np
import pandas as pd


# ---------- cross-sectional (within one day) ----------
def rank(x):
    return x.rank(axis=1, pct=True)


def zscore(x):
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)


def demean(x):
    return x.sub(x.mean(axis=1), axis=0)


# ---------- time series (per ticker) ----------
def delay(x, d):
    return x.shift(int(d))


def delta(x, d):
    return x - x.shift(int(d))


def pct(x, d):
    return x / x.shift(int(d)) - 1


def ts_mean(x, d):
    return x.rolling(int(d), min_periods=max(2, int(d) // 2)).mean()


def ts_std(x, d):
    return x.rolling(int(d), min_periods=max(2, int(d) // 2)).std()


def ts_sum(x, d):
    return x.rolling(int(d), min_periods=max(2, int(d) // 2)).sum()


def ts_max(x, d):
    return x.rolling(int(d), min_periods=max(2, int(d) // 2)).max()


def ts_min(x, d):
    return x.rolling(int(d), min_periods=max(2, int(d) // 2)).min()


def ts_zscore(x, d):
    return (x - ts_mean(x, d)) / ts_std(x, d)


def ts_rank(x, d):
    """Percentile position of today's value within the last d sessions."""
    d = int(d)
    return x.rolling(d, min_periods=max(2, d // 2)).apply(
        lambda a: (a[-1] > a[:-1]).mean() if len(a) > 1 else np.nan, raw=True)


def ts_corr(x, y, d):
    return x.rolling(int(d), min_periods=max(3, int(d) // 2)).corr(y)


def ts_skew(x, d):
    return x.rolling(int(d), min_periods=max(3, int(d) // 2)).skew()


def decay_linear(x, d):
    d = int(d)
    w = np.arange(1, d + 1, dtype=float)
    w /= w.sum()
    return x.rolling(d, min_periods=d).apply(lambda a: np.dot(a, w), raw=True)


# ---------- element-wise ----------
def sign(x):
    return np.sign(x)


def log(x):
    return np.log(x.where(x > 0))


def abs_(x):
    return x.abs()


def clip(x, lo, hi):
    return x.clip(lo, hi)


def where(cond, a, b):
    return a.where(cond, b) if isinstance(a, pd.DataFrame) else b.mask(cond, a)


OPS = {f.__name__.rstrip("_"): f for f in [
    rank, zscore, demean, delay, delta, pct, ts_mean, ts_std, ts_sum, ts_max, ts_min,
    ts_zscore, ts_rank, ts_corr, ts_skew, decay_linear, sign, log, abs_, clip, where,
]}
