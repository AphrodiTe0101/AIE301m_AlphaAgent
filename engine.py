"""Alpha evaluation engine: computes the signal from a DSL expression, measures IC and backtests a portfolio.

Timing conventions (no look-ahead):
  - alpha[t] only uses data up to the end of session t;
  - entry at the close of session t+1 (no assumption of filling at the close of session t);
  - target return = close[t+1+H] / close[t+1] - 1.
"""
import ast
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ops import OPS, ts_mean

ROOT = Path(__file__).parent
PERIODS = {
    "train":   ("2019-01-01", "2022-12-31"),   # the ONLY period the agent sees while searching
    "test":    ("2023-01-01", "2024-12-31"),   # main out-of-sample
    "holdout": ("2025-01-01", "2026-10-30"),   # additional holdout
    "blind":   ("2026-07-01", "2026-10-30"),   # after the LLM's knowledge cutoff (06/2026)
}
HORIZON = 5        # hold for 5 sessions
TOP_K = 6          # long-only portfolio: the 6 highest-scoring tickers (~20% of the basket)
FEE = 0.002        # cost per unit of one-way turnover (~0.15% fee + 0.1% sell tax split evenly)
MIN_NAMES = 10     # need >= 10 tickers with data to compute a day's IC


@lru_cache(maxsize=1)
def load_env():
    p = pd.read_pickle(ROOT / "data" / "panel.pkl")
    o, h, l, c, v = (p[k] for k in ["open", "high", "low", "close", "volume"])
    env = {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "vwap": (h + l + c) / 3,
        "returns": c.pct_change(fill_method=None),
        "value": c * v,                       # traded value (VND)
    }
    env["adv20"] = ts_mean(env["value"], 20)
    fwd = c.shift(-(1 + HORIZON)) / c.shift(-1) - 1
    daily_ret = c.pct_change(fill_method=None).fillna(0.0)
    return env, fwd, daily_ret


_ALLOWED_NODES = (ast.Expression, ast.Call, ast.Name, ast.Load, ast.Constant, ast.BinOp,
                  ast.UnaryOp, ast.Compare, ast.operator, ast.unaryop, ast.cmpop)


def check_expr(expr: str, names) -> ast.Expression:
    """Only allows DSL variable/operator names, numbers, arithmetic and comparisons. No attributes, no indexing."""
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"syntax not allowed: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in names:
            raise ValueError(f"name does not exist in the DSL: {node.id}")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in OPS):
            raise ValueError("only DSL operators may be called")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError("constants must be numbers")
    return tree


def compute_alpha(expr: str) -> pd.DataFrame:
    env, _, _ = load_env()
    scope = {**OPS, **env}
    tree = check_expr(expr, scope)
    out = eval(compile(tree, "<alpha>", "eval"), {"__builtins__": {}}, scope)
    if not isinstance(out, pd.DataFrame) or out.shape != env["close"].shape:
        raise ValueError(f"alpha must return a DataFrame {env['close'].shape}, got {type(out).__name__}")
    out = out.astype(float).replace([np.inf, -np.inf], np.nan)
    return out.where(env["close"].notna())


def _slice(df, period):
    lo, hi = PERIODS[period]
    df = df.loc[lo:hi]
    # drop the last H+1 sessions so forward returns do not spill into the next period
    return df.iloc[: -(HORIZON + 1)] if period != "blind" else df


def daily_ic(alpha: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    m = alpha.notna() & fwd.notna()
    ar = alpha.where(m).rank(axis=1)
    fr = fwd.where(m).rank(axis=1)
    ar = ar.sub(ar.mean(axis=1), axis=0)
    fr = fr.sub(fr.mean(axis=1), axis=0)
    ic = (ar * fr).sum(axis=1) / np.sqrt((ar ** 2).sum(axis=1) * (fr ** 2).sum(axis=1))
    return ic.where(m.sum(axis=1) >= MIN_NAMES)


def ic_stats(alpha, period):
    _, fwd, _ = load_env()
    ic = _slice(daily_ic(alpha, fwd), period).dropna()
    n = len(ic)
    mean, std = ic.mean(), ic.std()
    return {
        "ic": mean,
        "ic_ir": mean / std if std > 0 else np.nan,
        # overlapping 5-session returns -> only ~n/H independent observations
        "t_stat": mean / std * np.sqrt(n / HORIZON) if std > 0 else np.nan,
        "ic_pos": (ic > 0).mean(),
        "n_days": n,
    }, ic


def backtest(alpha, period, top_k=TOP_K, fee=FEE):
    """Long-only top-k, rebalanced every H sessions, compared with the equal-weight VN30 basket."""
    env, _, ret = load_env()
    lo, hi = PERIODS[period]
    a = alpha.loc[lo:hi]
    valid = a.notna()
    rebal_dates = a.index[::HORIZON]
    pick = a.loc[rebal_dates].rank(axis=1, ascending=False, method="first") <= top_k
    target = pick.astype(float).div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    turnover = target.diff().abs().sum(axis=1)
    turnover.iloc[0] = target.iloc[0].abs().sum()
    w = target.reindex(a.index).ffill().fillna(0.0)
    # signal on day s -> filled at the close of s+1 -> earns returns from session s+2
    pos = w.shift(2).fillna(0.0)
    cost = (turnover * fee).reindex(a.index).fillna(0.0).shift(2).fillna(0.0)
    r = ret.loc[lo:hi]
    strat = (pos * r).sum(axis=1) - cost
    avail = env["close"].loc[lo:hi].notna() & env["close"].loc[lo:hi].shift(1).notna()
    bench = r.where(avail).mean(axis=1).fillna(0.0)
    strat, bench = strat.iloc[2:], bench.iloc[2:]

    def stats(x):
        eq = (1 + x).cumprod()
        yrs = len(x) / 252
        return {
            "ann_ret": eq.iloc[-1] ** (1 / yrs) - 1,
            "sharpe": x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan,
            "max_dd": (eq / eq.cummax() - 1).min(),
        }

    s, b = stats(strat), stats(bench)
    ex = strat - bench
    return {
        "ann_ret": s["ann_ret"], "sharpe": s["sharpe"], "max_dd": s["max_dd"],
        "bench_ann_ret": b["ann_ret"], "bench_sharpe": b["sharpe"], "bench_max_dd": b["max_dd"],
        "excess_ann": s["ann_ret"] - b["ann_ret"],
        "info_ratio": ex.mean() / ex.std() * np.sqrt(252) if ex.std() > 0 else np.nan,
        "turnover_ann": turnover.mean() * 252 / HORIZON,
    }, strat, bench


def evaluate(expr, period="train"):
    alpha = compute_alpha(expr)
    s, ic = ic_stats(alpha, period)
    bt, _, _ = backtest(alpha, period)
    s.update({"excess_ann": bt["excess_ann"], "sharpe": bt["sharpe"], "turnover_ann": bt["turnover_ann"],
              "coverage": _slice(alpha, period).notna().mean().mean()})
    return s, alpha, ic
