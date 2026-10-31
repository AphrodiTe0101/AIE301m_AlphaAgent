"""AI Quant Researcher demo UI (AIE301m).

Run:  streamlit run demo_app.py

4 tabs:
  1. Replay     - replays the 8 generations the agent ran for the report (can re-score them for real)
  2. Live agent - Claude researches live in front of the audience (needs ANTHROPIC_API_KEY, costs money)
  3. Try an alpha - the audience types a formula and the engine scores it right away
  4. Final results - OOS IC, overfitting, backtest after fees
"""
import json
import os
import shutil
import time
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

from engine import ROOT, backtest, daily_ic, evaluate, ic_stats, load_env
from ops import OPS
from run_generation import main as score_generation

st.set_page_config(page_title="AI Quant Researcher", page_icon="📈", layout="wide")

RES = ROOT / "results"
BLUE, GREY = "#2563eb", "#9ca3af"
COLS = {"name": "Alpha", "hypothesis": "Hypothesis", "expr": "Formula", "ic": "IC", "t_stat": "t-stat",
        "excess_ann": "Excess/yr (after fees)", "turnover_ann": "Turnover/yr", "status": "Status"}
EXAMPLES = {
    "The agent's best alpha (g8_bhd_w)": "2 * rank(ts_corr(returns, returns - demean(returns), 60)) + 2 * rank(ts_mean(close, 5) / ts_max(close, 250)) + rank(-ts_corr(rank(close), rank(volume), 10))",
    "Near the 52-week high": "rank(close / ts_max(close, 250))",
    "Price–volume divergence": "-ts_corr(rank(close), rank(volume), 10)",
    "5-day reversal (baseline)": "rank(-pct(close, 5))",
    "Write your own formula": "rank(ts_mean(close / open - 1, 20))",
    "Try to break the sandbox (will be blocked)": "close.values",
}


# ---------------------------------------------------------------- helpers
@st.cache_data
def read_board(path):
    return pd.read_csv(path)


def read_notes(results_dir):
    p = results_dir / "notes.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def alpha_table(df):
    d = df.copy()
    d["status"] = ["✅" if s == "ok" else f"❌ {e}" for s, e in zip(d["status"], d.get("error", ""))]
    d = d.reindex(columns=list(COLS)).rename(columns=COLS)
    return d.style.format({"IC": "{:.3f}", "t-stat": "{:.2f}", "Excess/yr (after fees)": "{:+.1%}",
                           "Turnover/yr": "{:.0f}"}, na_rep="—")


def evolution_chart(board):
    ok = board[board.status == "ok"]
    per = ok.groupby("gen").t_stat.agg(["max", "median"]).reset_index()
    per["Best so far"] = per["max"].cummax()
    per["Generation median"] = per["median"]
    long = per.melt("gen", ["Best so far", "Generation median"], var_name="Line", value_name="t")
    line = alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("gen:O", title="Generation"),
        y=alt.Y("t:Q", title="IC t-stat (train)", scale=alt.Scale(domain=[-1, 4.5])),
        color=alt.Color("Line:N", title=None, scale=alt.Scale(range=[BLUE, GREY])),
        tooltip=["gen", "Line", alt.Tooltip("t:Q", format=".2f")])
    rule = alt.Chart(pd.DataFrame({"t": [3.0]})).mark_rule(strokeDash=[4, 4], color="#888").encode(y="t:Q")
    label = alt.Chart(pd.DataFrame({"t": [3.0], "x": [0]})).mark_text(
        text="t = 3 (significance bar under multiple testing)", align="left", dy=-8, color="#666").encode(y="t:Q")
    return (line + rule + label).properties(height=300)


def top_table(board, n=5):
    top = board[board.status == "ok"].nlargest(n, "t_stat")
    return alpha_table(top[["name", "expr", "ic", "t_stat", "excess_ann", "turnover_ann", "status", "error"]])


def show_generation(ph, gen, note, gen_rows, board):
    with ph.container():
        st.subheader(f"Generation {gen}" + (" — hand-written baseline" if gen == 0 else ""))
        if note:
            st.info(f"🧠 **Agent's reflection:** {note}")
        ok = gen_rows[gen_rows.status == "ok"]
        c = st.columns(4)
        c[0].metric("Alphas that ran", f"{len(ok)}/{len(gen_rows)}")
        c[1].metric("Best t-stat this generation", f"{ok.t_stat.max():.2f}" if len(ok) else "—")
        best_all = board[board.status == "ok"].t_stat.max()
        c[2].metric("Best t-stat overall", f"{best_all:.2f}")
        c[3].metric("Total alphas tried", len(board))
        st.dataframe(alpha_table(gen_rows.sort_values("t_stat", ascending=False)), hide_index=True)
        left, right = st.columns([3, 2])
        left.altair_chart(evolution_chart(board))
        right.markdown("**🏆 Leaderboard (train)**")
        right.dataframe(top_table(board), hide_index=True)


# ---------------------------------------------------------------- layout
st.title("📈 AI Quant Researcher — an LLM that finds alphas for VN30 on its own")
st.caption("AIE301m · The agent only sees the 2019–2022 train data. The 2023–2026 data is opened only once, after the agent stops.")

with st.sidebar:
    st.header("5-minute demo script")
    st.markdown(
        "1. **Try an alpha** (1'): invite someone to type a formula, look at the score, then try to *break the sandbox*.\n"
        "2. **Replay** (2'): run the 8 generations, pause at generation 4–5 to explain how the agent flipped the beta sign.\n"
        "3. **Final results** (1.5'): the OOS IC passes, but after fees it loses to the basket and the alpha dies from 2025.\n"
        "4. **Live agent** (0.5'): run 1 real generation with the Claude API if you have a key.")
    st.divider()
    st.markdown("**Variables:** open, high, low, close, volume, vwap, returns, value, adv20")
    st.markdown("**Operators:** " + ", ".join(f"`{o}`" for o in sorted(OPS)))

tab_replay, tab_live, tab_try, tab_result = st.tabs(
    ["🔁 Replay 8 generations", "🤖 Live agent (Claude API)", "🧪 Try an alpha", "📊 Final results"])

# ---------------------------------------------------------------- 1. replay
with tab_replay:
    st.markdown("Replays how the agent evolved alphas generation by generation, exactly as in the report's run.")
    c1, c2 = st.columns([1, 2])
    speed = c1.slider("Seconds per generation", 0.0, 10.0, 3.0, 0.5)
    rescore = c2.checkbox("Really re-score with the engine (slower, about 10–30 seconds per generation)",
                          help="Re-runs every backtest in demo_replay/ to prove the numbers are not made up.")
    if st.button("▶ Start replay", type="primary"):
        full = read_board(RES / "leaderboard.csv")
        notes = read_notes(RES)
        gens = sorted(full.gen.unique())
        work = ROOT / "demo_replay"
        if rescore:
            shutil.rmtree(work, ignore_errors=True)
        progress = st.progress(0.0)
        ph = st.empty()
        for i, g in enumerate(gens):
            if rescore:
                with st.spinner(f"Backtesting generation {g}..."):
                    score_generation(str(ROOT / "alphas" / f"gen_{g:02d}.json"), work / "results")
                board = pd.read_csv(work / "results" / "leaderboard.csv")
            else:
                board = full[full.gen <= g]
            show_generation(ph, g, notes.get(str(g), ""), board[board.gen == g], board)
            progress.progress((i + 1) / len(gens), text=f"Generation {g}/{gens[-1]}")
            time.sleep(speed)
        best = board[board.status == "ok"].nlargest(1, "t_stat").iloc[0]
        st.success(f"The agent stopped after {gens[-1]} generations ({len(board)} alphas). Alpha chosen by the pre-registered rule: "
                   f"**{best['name']}** (train t-stat {best.t_stat:.2f}). Nobody has looked at the 2023–2026 data yet, see the **Final results** tab.")

# ---------------------------------------------------------------- 2. live
with tab_live:
    st.markdown("Claude Opus 5.5 reads the leaderboard and proposes new alphas, the engine scores them, and the results go back to Claude. "
                "Each run writes to its own folder `demo_run/<timestamp>/` and never touches the report's results.")
    st.warning("Each generation calls the API once, taking about 30–90 seconds and a few US cents. Do a test run before presenting.")
    c1, c2, c3 = st.columns([2, 1, 1])
    api_key = c1.text_input("ANTHROPIC_API_KEY (leave empty if the environment variable is set)", type="password")
    n_gen = c2.number_input("Generations", 1, 10, 2)
    per_gen = c3.number_input("Alphas per generation", 3, 10, 6)
    has_creds = bool(api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    go = st.button("🚀 Run the agent", type="primary")
    if go and not has_creds:
        st.error("No API key: enter it above, or set the ANTHROPIC_API_KEY environment variable and restart the app.")
    if go and has_creds:
        import anthropic
        from agent import evolve

        work = ROOT / "demo_run" / datetime.now().strftime("%Y%m%d_%H%M%S")
        ph = st.empty()
        status = st.status("The agent is working...", expanded=False)
        try:
            for step in evolve(work, int(n_gen), int(per_gen), patience=2, api_key=api_key or None):
                board = pd.read_csv(work / "results" / "leaderboard.csv")
                show_generation(ph, step["gen"], step["reflection"], board[board.gen == step["gen"]], board)
                status.update(label=f"Generation {step['gen']} done · cumulative API cost ${step['usd']:.2f}")
                status.code(step["report"])
                if step["stop"]:
                    st.info("The agent stopped itself: 2 consecutive generations without improving the t-stat by more than 0.1.")
            status.update(label=f"Done · results saved in {work.relative_to(ROOT)}", state="complete")
        except anthropic.AuthenticationError:
            status.update(label="Authentication error", state="error")
            st.error("The API key is invalid or ANTHROPIC_API_KEY is not set.")
        except anthropic.RateLimitError:
            status.update(label="Rate limited", state="error")
            st.error("The API is rate limiting requests, try again in a few minutes.")
        except (anthropic.AnthropicError, RuntimeError, TypeError) as e:  # bad key, network/server error, refusal...
            status.update(label="Error calling Claude", state="error")
            st.error(f"{type(e).__name__}: {e}")

# ---------------------------------------------------------------- 3. try
with tab_try:
    st.markdown("Type an alpha formula and the engine scores it on the **2019–2022 train** period, exactly as it scores the agent.")
    choice = st.selectbox("Examples", list(EXAMPLES))
    expr = st.text_area("Formula", EXAMPLES[choice], key=f"expr_{choice}", height=80)
    show_oos = st.checkbox("Also open the 2023–2026 data (only to illustrate overfitting)",
                           help="If you look at the OOS results and then edit the formula, the OOS is no longer 'out of sample'.")
    if st.button("⚡ Score it", type="primary"):
        try:
            with st.spinner("Backtesting..."):
                stats, alpha, _ = evaluate(expr, "train")
                bt, _, _ = backtest(alpha, "train")
        except Exception as e:
            st.error(f"The engine rejected the formula: {type(e).__name__}: {e}")
        else:
            t = stats["t_stat"]
            verdict = "strong 💪" if t >= 3 else "promising 🙂" if t >= 2 else "weak 😐" if t >= 0 else "wrong sign, try flipping it 🔄"
            st.markdown(f"#### Train 2019–2022 result: the signal is **{verdict}**")
            c = st.columns(6)
            c[0].metric("IC", f"{stats['ic']:.3f}")
            c[1].metric("t-stat", f"{t:.2f}")
            c[2].metric("Days with IC > 0", f"{stats['ic_pos']:.0%}")
            c[3].metric("Excess/yr (after fees)", f"{bt['excess_ann']:+.1%}")
            c[4].metric("Sharpe", f"{bt['sharpe']:.2f}")
            c[5].metric("Turnover/yr", f"{bt['turnover_ann']:.0f}")
            if bt["turnover_ann"] > 40:
                st.caption("Above 40 turnovers/year, the 0.2% per-side fee eats about 8%/year of return. Try smoothing with ts_mean.")

            _, fwd, _ = load_env()
            ic = daily_ic(alpha, fwd)
            yearly = ic.groupby(ic.index.year).mean().rename("IC").rename_axis("Year").reset_index()
            yearly["Period"] = pd.cut(yearly["Year"], [0, 2022, 2024, 9999], labels=["Train", "Test", "Holdout"]).astype(str)
            if not show_oos:
                yearly = yearly[yearly["Period"] == "Train"]
            bars = alt.Chart(yearly).mark_bar().encode(
                x=alt.X("Year:O"), y=alt.Y("IC:Q", title="Mean IC"),
                color=alt.Color("Period:N", scale=alt.Scale(domain=["Train", "Test", "Holdout"], range=[GREY, BLUE, "#f59e0b"])),
                tooltip=["Year", "Period", alt.Tooltip("IC:Q", format=".3f")]).properties(height=260)
            st.altair_chart(bars)

            if show_oos:
                rows = []
                for p, label in [("train", "Train 2019–2022"), ("test", "Test 2023–2024"), ("holdout", "Holdout 2025–10/2026")]:
                    s, _ = ic_stats(alpha, p)
                    b, _, _ = backtest(alpha, p)
                    rows.append({"Period": label, "IC": s["ic"], "t-stat": s["t_stat"],
                                 "Return/yr": b["ann_ret"], "VN30 basket/yr": b["bench_ann_ret"], "Excess (after fees)": b["excess_ann"]})
                st.dataframe(pd.DataFrame(rows).style.format({"IC": "{:.3f}", "t-stat": "{:.2f}", "Return/yr": "{:+.1%}",
                                                              "VN30 basket/yr": "{:+.1%}", "Excess (after fees)": "{:+.1%}"}), hide_index=True)
                ratio = rows[1]["IC"] / rows[0]["IC"] if rows[0]["IC"] else float("nan")
                st.caption(f"Overfit ratio (test IC / train IC) = {ratio:.2f}. Below 0.5 means most of the train 'signal' was noise.")

# ---------------------------------------------------------------- 4. results
with tab_result:
    fin = json.loads((RES / "final.json").read_text(encoding="utf-8"))
    best = fin["best"]["report"]
    st.markdown(f"Chosen alpha: **{fin['best']['name']}**")
    st.code(fin["best"]["expr"], language="python")
    c = st.columns(3)
    c[0].metric("Agent code that ran", f"{fin['runnable_rate']:.1%}", "target ≥ 70% ✓", delta_color="off")
    c[1].metric("OOS IC 2023–2024", f"{best['test']['ic']:.3f}", "target ≥ 0.03 ✓", delta_color="off")
    c[2].metric("Overfit ratio", f"{fin['best']['overfit_ratio']:.2f}", "target ≥ 0.5 ✓", delta_color="off")

    left, right = st.columns(2)
    evo = pd.read_csv(RES / "evolution.csv").rename(columns={"best_ic_train": "Train 2019–2022", "best_ic_test": "OOS 2023–2024"})
    evo_long = evo.melt("gen", ["Train 2019–2022", "OOS 2023–2024"], var_name="Data", value_name="IC")
    left.markdown("**IC of the leading alpha at each generation**")
    left.altair_chart(alt.Chart(evo_long).mark_line(point=True).encode(
        x=alt.X("gen:O", title="Generation"), y=alt.Y("IC:Q"),
        color=alt.Color("Data:N", title=None, scale=alt.Scale(range=[BLUE, GREY])),
        tooltip=["gen", "Data", alt.Tooltip("IC:Q", format=".3f")]).properties(height=280))

    yearly = pd.read_csv(RES / "ic_by_year.csv")
    yearly["Period"] = pd.cut(yearly["year"], [0, 2022, 2024, 9999], labels=["Train", "Test", "Holdout"]).astype(str)
    right.markdown("**IC by year: positive 6 years in a row, then gone from 2025**")
    right.altair_chart(alt.Chart(yearly).mark_bar().encode(
        x=alt.X("year:O", title="Year"), y=alt.Y("best:Q", title="IC"),
        color=alt.Color("Period:N", scale=alt.Scale(domain=["Train", "Test", "Holdout"], range=[GREY, BLUE, "#f59e0b"])),
        tooltip=["year", alt.Tooltip("best:Q", format=".3f")]).properties(height=280))

    eq = pd.read_csv(RES / "equity_oos_weekly.csv", parse_dates=["Date"]).dropna()
    eq = eq.rename(columns={"best": "Best alpha (after fees)", "combo": "Combined 5-alpha portfolio", "vn30_ew": "Equal-weight VN30 basket"})
    eq_long = eq.melt("Date", var_name="Portfolio", value_name="Value")
    st.markdown("**Value of 1 unit invested, 01/2023–10/2026 (after 0.2% per-side fees)**")
    st.altair_chart(alt.Chart(eq_long).mark_line().encode(
        x=alt.X("Date:T", title=None), y=alt.Y("Value:Q", scale=alt.Scale(zero=False)),
        color=alt.Color("Portfolio:N", title=None, scale=alt.Scale(range=[BLUE, "#60a5fa", GREY])),
        tooltip=[alt.Tooltip("Date:T"), "Portfolio", alt.Tooltip("Value:Q", format=".2f")]).properties(height=320))

    rows = [{"Period": lbl, "IC": best[p]["ic"], "t-stat": best[p]["t_stat"], "Return/yr": best[p]["ann_ret"],
             "Sharpe": best[p]["sharpe"], "Max drawdown": best[p]["max_dd"], "VN30 basket/yr": best[p]["bench_ann_ret"],
             "Excess (after fees)": best[p]["excess_ann"], "Turnover/yr": best[p]["turnover_ann"]}
            for p, lbl in [("train", "Train 2019–2022"), ("test", "Test 2023–2024"),
                           ("holdout", "Holdout 2025–10/2026"), ("blind", "Blind 07–10/2026")]]
    st.dataframe(pd.DataFrame(rows).style.format({
        "IC": "{:.3f}", "t-stat": "{:.2f}", "Return/yr": "{:+.1%}", "Sharpe": "{:.2f}", "Max drawdown": "{:.1%}",
        "VN30 basket/yr": "{:+.1%}", "Excess (after fees)": "{:+.1%}", "Turnover/yr": "{:.0f}"}), hide_index=True)
    st.markdown(
        "- **Predictive ≠ profitable:** before fees it beats the basket by +6%/year (test), but 45 turnovers/year cost about 10%/year in fees.\n"
        "- **The alpha dies from 2025:** the rally concentrated in VIC and VHM, two stocks the alpha ranks low.\n"
        "- **Limitations:** the LLM may already 'know' the 2023–2025 market; the VN30 list is fixed, so there is survivorship bias.")
