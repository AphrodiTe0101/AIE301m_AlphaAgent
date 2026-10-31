"""Engine đánh giá alpha: tính tín hiệu từ biểu thức DSL, đo IC và backtest danh mục.

Quy ước thời gian (chống look-ahead):
  - alpha[t] chỉ dùng dữ liệu đến hết phiên t;
  - vào lệnh ở giá đóng cửa phiên t+1 (không giả định khớp được ngay giá đóng cửa phiên t);
  - lợi nhuận mục tiêu = close[t+1+H] / close[t+1] - 1.
"""
import ast
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ops import OPS, ts_mean

ROOT = Path(__file__).parent
PERIODS = {
    "train":   ("2019-01-01", "2022-12-31"),   # agent CHỈ được thấy giai đoạn này khi tìm kiếm
    "test":    ("2023-01-01", "2024-12-31"),   # out-of-sample chính
    "holdout": ("2025-01-01", "2026-10-30"),   # holdout bổ sung
    "blind":   ("2026-07-01", "2026-10-30"),   # sau knowledge cutoff của LLM (06/2026)
}
HORIZON = 5        # nắm giữ 5 phiên
TOP_K = 6          # danh mục long-only: 6 mã điểm cao nhất (~20% rổ)
FEE = 0.002        # chi phí mỗi đơn vị turnover một chiều (~0.15% phí + thuế bán 0.1% chia đều)
MIN_NAMES = 10     # cần >= 10 mã có dữ liệu để tính IC trong ngày


@lru_cache(maxsize=1)
def load_env():
    p = pd.read_pickle(ROOT / "data" / "panel.pkl")
    o, h, l, c, v = (p[k] for k in ["open", "high", "low", "close", "volume"])
    env = {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "vwap": (h + l + c) / 3,
        "returns": c.pct_change(fill_method=None),
        "value": c * v,                       # giá trị giao dịch (VND)
    }
    env["adv20"] = ts_mean(env["value"], 20)
    fwd = c.shift(-(1 + HORIZON)) / c.shift(-1) - 1
    daily_ret = c.pct_change(fill_method=None).fillna(0.0)
    return env, fwd, daily_ret


_ALLOWED_NODES = (ast.Expression, ast.Call, ast.Name, ast.Load, ast.Constant, ast.BinOp,
                  ast.UnaryOp, ast.Compare, ast.operator, ast.unaryop, ast.cmpop)


def check_expr(expr: str, names) -> ast.Expression:
    """Chỉ cho phép: tên biến/toán tử DSL, số, phép toán và so sánh. Không thuộc tính, không index."""
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"cú pháp không được phép: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in names:
            raise ValueError(f"tên không tồn tại trong DSL: {node.id}")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in OPS):
            raise ValueError("chỉ được gọi các toán tử của DSL")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError("hằng số phải là số")
    return tree


def compute_alpha(expr: str) -> pd.DataFrame:
    env, _, _ = load_env()
    scope = {**OPS, **env}
    tree = check_expr(expr, scope)
    out = eval(compile(tree, "<alpha>", "eval"), {"__builtins__": {}}, scope)
    if not isinstance(out, pd.DataFrame) or out.shape != env["close"].shape:
        raise ValueError(f"alpha phải trả về DataFrame {env['close'].shape}, nhận {type(out).__name__}")
    out = out.astype(float).replace([np.inf, -np.inf], np.nan)
    return out.where(env["close"].notna())


def _slice(df, period):
    lo, hi = PERIODS[period]
    df = df.loc[lo:hi]
    # bỏ H+1 phiên cuối để lợi nhuận tương lai không tràn sang giai đoạn sau
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
        # lợi nhuận 5 phiên chồng lấn -> chỉ ~n/H quan sát độc lập
        "t_stat": mean / std * np.sqrt(n / HORIZON) if std > 0 else np.nan,
        "ic_pos": (ic > 0).mean(),
        "n_days": n,
    }, ic


def backtest(alpha, period, top_k=TOP_K, fee=FEE):
    """Long-only top-k, tái cơ cấu mỗi H phiên, so với rổ VN30 chia đều tỷ trọng."""
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
    # tín hiệu ngày s -> khớp lệnh đóng cửa s+1 -> hưởng lợi nhuận từ phiên s+2
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
