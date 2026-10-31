"""Ngôn ngữ alpha (DSL): các toán tử mà agent được phép dùng.

Mọi biến là DataFrame (ngày x mã). Toán tử ts_* chỉ nhìn về quá khứ (rolling),
toán tử cs_* / rank so sánh giữa các mã trong cùng một ngày -> không có look-ahead.
"""
import numpy as np
import pandas as pd


# ---------- cắt ngang (cross-sectional, trong một ngày) ----------
def rank(x):
    return x.rank(axis=1, pct=True)


def zscore(x):
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)


def demean(x):
    return x.sub(x.mean(axis=1), axis=0)


# ---------- chuỗi thời gian (theo từng mã) ----------
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
    """Vị trí phần trăm của giá trị hôm nay trong cửa sổ d phiên gần nhất."""
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


# ---------- phần tử ----------
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
