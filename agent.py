"""AI Quant Researcher chạy tự động bằng Claude API.

Mỗi thế hệ: Claude đọc bảng xếp hạng (chỉ số liệu TRAIN) -> đề xuất alpha mới (JSON có schema)
-> run_generation chấm điểm -> phản hồi quay lại cho Claude ở thế hệ sau.

Chạy:  pip install anthropic
       set ANTHROPIC_API_KEY=...        (hoặc đăng nhập bằng `ant auth login`)
       python agent.py --generations 8 --per-gen 8          # ghi vào demo_run/
       streamlit run demo_app.py                            # hoặc chạy từ giao diện demo

Kết quả của báo cáo nằm ở alphas/ và results/; mặc định agent ghi vào demo_run/ để không đè lên.
"""
import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import anthropic
import pandas as pd

from engine import ROOT
from ops import OPS
from run_generation import main as score_generation

MODEL = "claude-opus-5-5"
PRICE_IN, PRICE_OUT = 4.00, 20.00  # USD / 1M token (Claude Opus 5.5)

SYSTEM = f"""Bạn là một nhà nghiên cứu định lượng (quant researcher) tìm alpha cho rổ VN30 (30 cổ phiếu vốn hoá lớn nhất HOSE).

Mỗi alpha là MỘT biểu thức Python trả về DataFrame (ngày x mã); giá trị cao = kỳ vọng lợi nhuận 5 phiên tới cao hơn.
Biến có sẵn: open, high, low, close, volume, vwap, returns (lợi nhuận ngày), value (giá trị giao dịch VND), adv20.
Toán tử được phép: {", ".join(sorted(OPS))}.
- rank/zscore/demean: so sánh giữa các mã trong cùng ngày. ts_*(x, d), delay, delta, pct, decay_linear: theo thời gian, cửa sổ d phiên.
- Phép toán + - * / và so sánh (>, <) được phép; KHÔNG có builtins, import, hay tên mã cổ phiếu.
- decay_linear cần đủ d giá trị không NaN; ưu tiên ts_mean khi dữ liệu có thể thiếu.

Đánh giá (chỉ trên giai đoạn train 2019-2022): IC = tương quan hạng với lợi nhuận 5 phiên (vào lệnh ở giá đóng cửa phiên sau),
t_stat của IC (mục tiêu chính), excess_ann = lợi nhuận vượt rổ chia đều của danh mục long-only 6 mã sau phí 0,2%/chiều,
turnover_ann = số lần quay vòng danh mục mỗi năm (cao thì phí ăn lợi nhuận), max_corr = độ trùng lặp với các alpha dẫn đầu.

Cách làm việc: mỗi alpha phải có một giả thuyết kinh tế/hành vi rõ ràng (dòng tiền, tâm lý nhà đầu tư cá nhân, thanh khoản...).
Kết hợp đột biến tham số, lai ghép các alpha tốt, đảo dấu alpha có IC âm ổn định, và khám phá ý tưởng mới ít tương quan.
Kiểm tra độ bền (đổi tham số) trước khi tin một alpha. Tránh overfit: ưu tiên công thức đơn giản."""

SCHEMA = {
    "type": "object",
    "properties": {
        "reflection": {"type": "string", "description": "Rút ra gì từ phản hồi thế hệ trước (2-4 câu)"},
        "alphas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "snake_case ngắn, không có tiền tố thế hệ"},
                    "expr": {"type": "string"},
                    "hypothesis": {"type": "string"},
                    "parent": {"type": "string", "description": "tên alpha cha, hoặc chuỗi rỗng"},
                },
                "required": ["name", "expr", "hypothesis", "parent"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["reflection", "alphas"],
    "additionalProperties": False,
}


def ask_claude(client, feedback, gen, per_gen):
    prompt = (f"Thế hệ {gen}. Phản hồi hiện tại:\n\n{feedback}\n\n"
              f"Đề xuất đúng {per_gen} alpha mới cho thế hệ {gen}.")
    resp = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # nếu bị từ chối do bộ lọc an toàn, API tự chạy lại trên model dự phòng
        system=SYSTEM,
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"Claude từ chối: {resp.stop_details}")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("Hết max_tokens trước khi trả đủ JSON")
    data = json.loads(next(b.text for b in resp.content if b.type == "text"))
    return data, resp.usage


def log_usage(path, gen, usage):
    cost = usage.input_tokens / 1e6 * PRICE_IN + usage.output_tokens / 1e6 * PRICE_OUT
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["gen", "input_tokens", "output_tokens", "usd"])
        w.writerow([gen, usage.input_tokens, usage.output_tokens, round(cost, 4)])
    return cost


def leaderboard_text(board):
    top = board[board.status == "ok"].nlargest(10, "t_stat")
    cols = ["gen", "name", "expr", "ic", "t_stat", "excess_ann", "turnover_ann"]
    return f"Bảng xếp hạng hiện tại ({len(board)} alpha đã thử), top 10:\n" + top[cols].round(3).to_string(index=False)


def evolve(workdir=ROOT, generations=8, per_gen=8, patience=2, api_key=None):
    """Generator: mỗi thế hệ yield một dict để CLI hoặc giao diện demo hiển thị.

    workdir chứa alphas/ và results/ riêng -> chạy demo không đè lên kết quả của báo cáo.
    """
    workdir = Path(workdir)
    alphas_dir, results_dir = workdir / "alphas", workdir / "results"
    alphas_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    board_path = results_dir / "leaderboard.csv"
    notes_path = results_dir / "notes.json"
    notes = json.loads(notes_path.read_text(encoding="utf-8")) if notes_path.exists() else {}
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    if not board_path.exists():  # chưa có baseline -> chấm thế hệ 0 do người viết
        base = alphas_dir / "gen_00.json"
        if not base.exists():
            shutil.copy(ROOT / "alphas" / "gen_00.json", base)
        report = score_generation(str(base), results_dir)
        yield {"gen": 0, "reflection": "Baseline do người viết", "report": report,
               "alphas": json.loads(base.read_text(encoding="utf-8")), "usd": 0.0, "stop": False}
        feedback = report
    else:
        feedback = leaderboard_text(pd.read_csv(board_path))

    start = int(pd.read_csv(board_path).gen.max()) + 1
    best_t, stale, total = -1e9, 0, 0.0
    for gen in range(start, start + generations):
        data, usage = ask_claude(client, feedback, gen, per_gen)
        total += log_usage(results_dir / "agent_usage.csv", gen, usage)
        for a in data["alphas"]:
            a["name"] = f"g{gen}_" + re.sub(r"[^a-z0-9_]", "_", a["name"].lower())[:40]
        path = alphas_dir / f"gen_{gen:02d}.json"
        path.write_text(json.dumps(data["alphas"], ensure_ascii=False, indent=2), encoding="utf-8")
        notes[str(gen)] = data["reflection"]
        notes_path.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")

        report = score_generation(str(path), results_dir)
        board = pd.read_csv(board_path)
        feedback = f"Bạn đã nhận xét: {data['reflection']}\n\n{report}\n\n{leaderboard_text(board)}"

        top = board.query("status == 'ok'").t_stat.max()
        stale = stale + 1 if top < best_t + 0.1 else 0
        best_t = max(best_t, top)
        yield {"gen": gen, "reflection": data["reflection"], "alphas": data["alphas"], "report": report,
               "usd": total, "best_t": best_t, "stop": stale >= patience}
        if stale >= patience:
            break


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--per-gen", type=int, default=8)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--workdir", default=str(ROOT / "demo_run"),
                    help="thư mục chứa alphas/ và results/ của lần chạy (mặc định: demo_run)")
    args = ap.parse_args()
    for step in evolve(args.workdir, args.generations, args.per_gen, args.patience):
        print(f"\n### Thế hệ {step['gen']} — suy nghĩ của agent: {step['reflection']}")
        print(f"(chi phí API tích luỹ: ${step['usd']:.2f})")
        if step["stop"]:
            print(f"Dừng: {args.patience} thế hệ liên tiếp không cải thiện t-stat quá 0.1")
