"""Công cụ của agent: chấm một thế hệ alpha trên giai đoạn TRAIN và in phản hồi.

Chạy: python run_generation.py alphas/gen_03.json
File JSON: [{"name": ..., "expr": ..., "hypothesis": ..., "parent": ...}, ...]

Chỉ in số liệu train -- agent không bao giờ nhìn thấy test/holdout trong lúc tìm kiếm.
"""
import json
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

from engine import ROOT, evaluate

BOARD = ROOT / "results" / "leaderboard.csv"
CACHE = ROOT / "results" / "cache"
SHOW = ["gen", "name", "ic", "ic_ir", "t_stat", "ic_pos", "excess_ann", "sharpe", "turnover_ann", "max_corr"]


def rank_corr(a, b):
    """Tương quan hạng trung bình giữa hai alpha (mỗi 5 phiên), đo độ trùng lặp ý tưởng."""
    ra, rb = a.rank(axis=1).iloc[::5], b.rank(axis=1).iloc[::5]
    return ra.corrwith(rb, axis=1).loc["2019":"2022"].mean()


def main(path, results_dir=None):
    """Chấm file thế hệ `path`; `results_dir` cho phép chạy demo ở thư mục riêng."""
    board_path = Path(results_dir) / "leaderboard.csv" if results_dir else BOARD
    cache = Path(results_dir) / "cache" if results_dir else CACHE
    gen = int(Path(path).stem.split("_")[1])
    cands = json.loads(Path(path).read_text(encoding="utf-8"))
    board = pd.read_csv(board_path) if board_path.exists() else pd.DataFrame()
    if len(board):
        board = board[board["gen"] != gen]  # chạy lại cùng thế hệ thì ghi đè
    cache.mkdir(parents=True, exist_ok=True)
    elite = board[board["status"] == "ok"].nlargest(5, "t_stat") if len(board) else board
    elite_alphas = {n: pd.read_pickle(cache / f"{n}.pkl") for n in elite.get("name", [])}

    rows = []
    for c in cands:
        row = {"gen": gen, **c}
        t0 = time.time()
        try:
            stats, alpha, _ = evaluate(c["expr"], "train")
            if stats["coverage"] < 0.5:
                raise ValueError(f"coverage chỉ {stats['coverage']:.0%} -> quá nhiều NaN")
            alpha.to_pickle(cache / f"{c['name']}.pkl")
            corrs = {n: rank_corr(alpha, a) for n, a in elite_alphas.items()}
            row.update(stats, status="ok", error="",
                       max_corr=max(corrs.values(), key=abs) if corrs else float("nan"),
                       corr_with=max(corrs, key=lambda n: abs(corrs[n])) if corrs else "")
        except Exception as e:  # code do agent sinh ra có thể lỗi -> ghi nhận, không dừng vòng lặp
            row.update(status="error", error=f"{type(e).__name__}: {e}"[:200])
            if "--debug" in sys.argv:
                traceback.print_exc()
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)

    new = pd.DataFrame(rows)
    board = pd.concat([board, new], ignore_index=True)
    board.to_csv(board_path, index=False)

    pd.set_option("display.width", 200)
    fmt = lambda df: df.reindex(columns=SHOW).round(3).to_string(index=False)
    ok = new[new.status == "ok"]
    out = [f"=== THẾ HỆ {gen}: {len(ok)}/{len(new)} alpha chạy được (TRAIN 2019-2022) ==="]
    if len(ok):
        out.append(fmt(ok.sort_values("t_stat", ascending=False)))
    out += [f"[LỖI] {r['name']}: {r['error']}" for _, r in new[new.status == "error"].iterrows()]
    allok = board[board.status == "ok"]
    out.append(f"\n=== TOP 8 TOÀN CỤC (theo t-stat IC train), {len(allok)}/{len(board)} alpha chạy được ===")
    out.append(fmt(allok.nlargest(8, "t_stat")))
    if len(ok) and ok["corr_with"].ne("").any():
        out.append("\nTrùng lặp lớn nhất với elite: " + ", ".join(
            f"{r['name']}~{r['corr_with']}={r['max_corr']:.2f}" for _, r in ok.iterrows() if r["corr_with"]))
    report = "\n".join(out)
    print(report)
    return report


if __name__ == "__main__":
    main(sys.argv[1])
