"""Đánh giá cuối: chạy MỘT lần sau khi agent dừng tìm kiếm.

Áp dụng quy tắc chọn đã định trước (t-stat IC train cao nhất; combo 5 alpha corr < 0.7),
rồi mới mở dữ liệu test / holdout / blind. Kết quả ghi vào results/.
"""
import json

import numpy as np
import pandas as pd

import engine
from engine import PERIODS, ROOT, backtest, compute_alpha, ic_stats, load_env, daily_ic
from run_generation import BOARD, CACHE, rank_corr

OUT = ROOT / "results"
PERIODS["oos_all"] = ("2023-01-01", "2026-10-30")   # test + holdout liền mạch cho đường vốn
EVAL = ["train", "test", "holdout", "blind"]


def pick_combo(ok, k=5, max_corr=0.7):
    chosen = []
    for _, r in ok.sort_values("t_stat", ascending=False).iterrows():
        a = pd.read_pickle(CACHE / f"{r['name']}.pkl")
        if all(abs(rank_corr(a, pd.read_pickle(CACHE / f"{c}.pkl"))) < max_corr for c in chosen):
            chosen.append(r["name"])
        if len(chosen) == k:
            break
    return chosen


def full_report(alpha):
    rep = {}
    for p in EVAL:
        s, _ = ic_stats(alpha, p)
        bt, _, _ = backtest(alpha, p)
        rep[p] = {**s, **bt}
    return rep


def main():
    board = pd.read_csv(BOARD)
    agent = board[board.gen >= 1]
    ok = board[board.status == "ok"]

    # ---- quy tắc chọn định trước (chỉ dựa trên train) ----
    best = ok.loc[ok.t_stat.idxmax()]
    combo_names = pick_combo(ok)
    best_alpha = pd.read_pickle(CACHE / f"{best['name']}.pkl")
    combo_alpha = sum(pd.read_pickle(CACHE / f"{n}.pkl").rank(axis=1, pct=True) for n in combo_names) / len(combo_names)

    results = {
        "n_alpha_total": int(len(board)),
        "n_agent": int(len(agent)),
        "n_agent_ok": int((agent.status == "ok").sum()),
        "runnable_rate": float((agent.status == "ok").mean()),
        "generations": int(board.gen.max()),
        "best": {"name": best["name"], "expr": best["expr"], "report": full_report(best_alpha)},
        "combo": {"names": combo_names, "report": full_report(combo_alpha)},
    }
    b = results["best"]["report"]
    results["best"]["overfit_ratio"] = b["test"]["ic"] / b["train"]["ic"]
    c = results["combo"]["report"]
    results["combo"]["overfit_ratio"] = c["test"]["ic"] / c["train"]["ic"]

    # ---- IC train vs test cho mọi alpha (chỉ để phân tích sau, không dùng để chọn) ----
    rows = []
    for _, r in ok.iterrows():
        a = pd.read_pickle(CACHE / f"{r['name']}.pkl")
        rows.append({"gen": int(r.gen), "name": r["name"], "ic_train": r.ic, "t_train": r.t_stat,
                     "ic_test": ic_stats(a, "test")[0]["ic"], "ic_holdout": ic_stats(a, "holdout")[0]["ic"],
                     "ic_blind": ic_stats(a, "blind")[0]["ic"]})
    allic = pd.DataFrame(rows)
    allic.to_csv(OUT / "ic_train_vs_oos.csv", index=False)
    results["spearman_train_vs_test_ic"] = float(allic[["ic_train", "ic_test"]].corr("spearman").iloc[0, 1])
    results["share_positive_test_ic"] = float((allic.ic_test > 0).mean())
    base = allic[allic.gen == 0]
    results["baselines_test_ic"] = dict(zip(base.name, base.ic_test.round(4)))

    # ---- tiến hoá: alpha tốt nhất (theo train) tại mỗi thế hệ ----
    evo = []
    for g in sorted(allic.gen.unique()):
        upto = allic[allic.gen <= g]
        top = upto.loc[upto.t_train.idxmax()]
        gen_rows = allic[allic.gen == g]
        evo.append({"gen": int(g), "best_name": top["name"], "best_ic_train": top.ic_train,
                    "best_ic_test": top.ic_test, "gen_median_ic_train": gen_rows.ic_train.median(),
                    "n": int((board.gen == g).sum())})
    pd.DataFrame(evo).to_csv(OUT / "evolution.csv", index=False)

    # ---- IC theo năm của alpha tốt nhất ----
    _, fwd, _ = load_env()
    ic = daily_ic(best_alpha, fwd)
    ic_c = daily_ic(combo_alpha, fwd)
    yearly = pd.DataFrame({"best": ic.groupby(ic.index.year).mean(), "combo": ic_c.groupby(ic_c.index.year).mean()})
    yearly.index.name = "year"
    yearly.round(4).to_csv(OUT / "ic_by_year.csv")

    # ---- đường vốn OOS liền mạch 2023-01 -> 2026-10 ----
    _, s_best, bench = backtest(best_alpha, "oos_all")
    _, s_combo, _ = backtest(combo_alpha, "oos_all")
    eq = pd.DataFrame({"best": (1 + s_best).cumprod(), "combo": (1 + s_combo).cumprod(), "vn30_ew": (1 + bench).cumprod()})
    eq.resample("W-FRI").last().round(4).to_csv(OUT / "equity_oos_weekly.csv")
    results["oos_all"] = {
        "best": backtest(best_alpha, "oos_all")[0], "combo": backtest(combo_alpha, "oos_all")[0]}
    # beta thực tế của danh mục so với rổ (đo rủi ro thị trường ngầm)
    results["oos_all"]["best_beta_vs_ew"] = float(np.cov(s_best, bench)[0, 1] / bench.var())
    results["oos_all"]["combo_beta_vs_ew"] = float(np.cov(s_combo, bench)[0, 1] / bench.var())

    (OUT / "final.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    # ---- in tóm tắt ----
    print(f"Agent: {results['n_agent_ok']}/{results['n_agent']} alpha chạy được ({results['runnable_rate']:.1%}), {results['generations']} thế hệ")
    for key in ["best", "combo"]:
        r = results[key]
        print(f"\n=== {key.upper()}: {r.get('name', r.get('names'))}  overfit={r['overfit_ratio']:.2f}")
        print(pd.DataFrame(r["report"]).T[["ic", "t_stat", "ic_pos", "ann_ret", "sharpe", "max_dd",
                                            "bench_ann_ret", "bench_sharpe", "excess_ann", "turnover_ann"]].round(3).to_string())
    print("\nOOS 2023-01 -> 2026-10:", json.dumps(results["oos_all"], default=lambda x: round(float(x), 3), indent=1))
    print("\nSpearman(IC train, IC test) trên mọi alpha:", round(results["spearman_train_vs_test_ic"], 3),
          "| tỷ lệ alpha có IC test > 0:", round(results["share_positive_test_ic"], 3))
    print("Baseline IC test:", results["baselines_test_ic"])
    print("\nIC theo năm:\n", yearly.round(3).to_string())
    print("\nTiến hoá:\n", pd.DataFrame(evo).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
