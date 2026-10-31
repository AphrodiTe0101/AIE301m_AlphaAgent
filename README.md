# AI Quant Researcher — AIE301m

An LLM agent that proposes alpha formulas for the VN30 basket (Vietnam's 30 largest stocks), backtests them, and improves
them on its own over several generations.

## Installation

```bash
git clone https://github.com/AphrodiTe0101/AIE301m_AlphaAgent.git
cd AIE301m_AlphaAgent
pip install -r requirements.txt
```

The repo already includes the price data (`data/panel.pkl`) and every result from the report, so the demo works
right after cloning, with no download and no API key.

## Demo

```bash
streamlit run demo_app.py        # opens http://localhost:8501
```

| Tab | Purpose |
| --- | --- |
| 🔁 Replay 8 generations | Replays the evolution (tick "really re-score" to re-run the backtests, about 3 minutes) |
| 🤖 Live agent | Claude Opus 5.5 researches live; needs `ANTHROPIC_API_KEY`, writes to `demo_run/<timestamp>/` |
| 🧪 Try an alpha | Type a formula and the engine scores it on train right away; try `close.values` to see the sandbox block it |
| 📊 Final results | The 3 targets met, IC by year, equity curve after fees vs the VN30 basket |

A 5-minute demo script is in the app's left sidebar. Try the Live agent tab before presenting:
each generation takes about 30–90 seconds and costs a few US cents.

## Command line

```bash
python data.py                               # download VN30 data -> data/panel.pkl
python run_generation.py alphas/gen_00.json  # hand-written baseline
python agent.py --generations 8 --per-gen 8  # autonomous loop (needs ANTHROPIC_API_KEY), writes to demo_run/
python final_eval.py                         # opens the OOS data, run EXACTLY ONCE after the agent stops
```

To reproduce the report's results without calling the API: delete `results/leaderboard.csv`, run
`run_generation.py` on `alphas/gen_00.json` … `gen_08.json` in order, then `final_eval.py`.

## Structure

| File | Role |
| --- | --- |
| `data.py` | Downloads OHLCV from Yahoo Finance, fixes moves beyond the price limit (unadjusted splits) |
| `ops.py` | DSL: the alpha operators (backward-looking only) |
| `engine.py` | Computes alphas, IC, t-stat, and a long-only top-6 backtest after fees |
| `run_generation.py` | Scores one generation on train 2019-2022, writes `results/leaderboard.csv` |
| `agent.py` | Autonomous loop on the Claude API (structured output, stops by itself); `evolve()` is a generator shared by the CLI and the app |
| `demo_app.py` | 4-tab Streamlit demo UI |
| `final_eval.py` | Pre-registered selection rule + test/holdout/blind evaluation |
| `alphas/gen_XX.json` | The alphas the agent proposed, with their hypotheses |
| `results/` | Leaderboard, final results, chart data |

## Notes

- `engine.check_expr` only allows DSL variable/operator names, numbers and arithmetic (an AST whitelist: no attributes,
  no indexing, no lambdas), and only then calls `eval` with builtins blocked. The app only listens on `localhost`
  (`.streamlit/config.toml`); do not expose it to a public network.
- The VN30 list is fixed (2024 review) for the whole period -> survivorship bias.
- Continuing to run the agent on the old leaderboard after looking at the OOS data makes the OOS no longer "clean".
