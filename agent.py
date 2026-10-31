"""AI Quant Researcher running autonomously on the Claude API.

Each generation: Claude reads the leaderboard (TRAIN stats only) -> proposes new alphas (schema-constrained JSON)
-> run_generation scores them -> the feedback goes back to Claude in the next generation.

Run:   pip install anthropic
       export ANTHROPIC_API_KEY=...     (Windows: set ANTHROPIC_API_KEY=..., or log in with `ant auth login`)
       python agent.py --generations 8 --per-gen 8          # writes to demo_run/
       streamlit run demo_app.py                            # or run it from the demo UI

The report's results live in alphas/ and results/; by default the agent writes to demo_run/ so it never overwrites them.
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
PRICE_IN, PRICE_OUT = 4.00, 20.00  # USD / 1M tokens (Claude Opus 5.5)

SYSTEM = f"""You are a quantitative researcher looking for alphas on the VN30 basket (the 30 largest-cap stocks on HOSE, Vietnam).

Each alpha is ONE Python expression returning a DataFrame (date x ticker); a higher value = higher expected return over the next 5 sessions.
Available variables: open, high, low, close, volume, vwap, returns (daily return), value (traded value in VND), adv20.
Allowed operators: {", ".join(sorted(OPS))}.
- rank/zscore/demean: compare tickers within the same day. ts_*(x, d), delay, delta, pct, decay_linear: over time, a window of d sessions.
- Arithmetic + - * / and comparisons (>, <) are allowed; NO builtins, imports, or ticker names.
- decay_linear needs d non-NaN values; prefer ts_mean when data may be missing.

Evaluation (train period 2019-2022 only): IC = rank correlation with the 5-session forward return (entry at the next session's close),
t_stat of the IC (the main objective), excess_ann = return in excess of the equal-weight basket for a long-only 6-stock portfolio after 0.2% fees per side,
turnover_ann = portfolio turnovers per year (high turnover lets fees eat the profit), max_corr = overlap with the leading alphas.

How to work: every alpha needs a clear economic/behavioural hypothesis (money flow, retail investor sentiment, liquidity...).
Combine parameter mutation, crossover of good alphas, flipping the sign of alphas with a stable negative IC, and exploring new low-correlation ideas.
Check robustness (vary the parameters) before trusting an alpha. Avoid overfitting: prefer simple formulas."""

SCHEMA = {
    "type": "object",
    "properties": {
        "reflection": {"type": "string", "description": "What you learned from the previous generation's feedback (2-4 sentences)"},
        "alphas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "short snake_case, no generation prefix"},
                    "expr": {"type": "string"},
                    "hypothesis": {"type": "string"},
                    "parent": {"type": "string", "description": "name of the parent alpha, or an empty string"},
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
    prompt = (f"Generation {gen}. Current feedback:\n\n{feedback}\n\n"
              f"Propose exactly {per_gen} new alphas for generation {gen}.")
    resp = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # if refused by a safety filter, the API retries on the fallback model
        system=SYSTEM,
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"Claude refused: {resp.stop_details}")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("Hit max_tokens before the JSON was complete")
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
    return f"Current leaderboard ({len(board)} alphas tried), top 10:\n" + top[cols].round(3).to_string(index=False)


def evolve(workdir=ROOT, generations=8, per_gen=8, patience=2, api_key=None):
    """Generator: yields one dict per generation for the CLI or the demo UI to display.

    workdir holds its own alphas/ and results/ -> a demo run never overwrites the report's results.
    """
    workdir = Path(workdir)
    alphas_dir, results_dir = workdir / "alphas", workdir / "results"
    alphas_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    board_path = results_dir / "leaderboard.csv"
    notes_path = results_dir / "notes.json"
    notes = json.loads(notes_path.read_text(encoding="utf-8")) if notes_path.exists() else {}
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    if not board_path.exists():  # no baseline yet -> score the hand-written generation 0
        base = alphas_dir / "gen_00.json"
        if not base.exists():
            shutil.copy(ROOT / "alphas" / "gen_00.json", base)
        report = score_generation(str(base), results_dir)
        yield {"gen": 0, "reflection": "Hand-written baseline", "report": report,
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
        feedback = f"Your reflection: {data['reflection']}\n\n{report}\n\n{leaderboard_text(board)}"

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
                    help="directory holding this run's alphas/ and results/ (default: demo_run)")
    args = ap.parse_args()
    for step in evolve(args.workdir, args.generations, args.per_gen, args.patience):
        print(f"\n### Generation {step['gen']} — agent's reflection: {step['reflection']}")
        print(f"(cumulative API cost: ${step['usd']:.2f})")
        if step["stop"]:
            print(f"Stopping: {args.patience} consecutive generations without improving the t-stat by more than 0.1")
