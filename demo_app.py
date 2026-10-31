"""Giao diện demo AI Quant Researcher (AIE301m).

Chạy:  streamlit run demo_app.py

4 tab:
  1. Replay     - phát lại 8 thế hệ agent đã chạy cho báo cáo (có thể chấm điểm lại thật)
  2. Agent live - Claude tự nghiên cứu ngay trước lớp (cần ANTHROPIC_API_KEY, tốn phí)
  3. Tự thử alpha - khán giả gõ công thức, engine chấm điểm ngay
  4. Kết quả cuối - IC OOS, overfit, backtest sau phí
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
COLS = {"name": "Alpha", "hypothesis": "Giả thuyết", "expr": "Công thức", "ic": "IC", "t_stat": "t-stat",
        "excess_ann": "Vượt rổ/năm (sau phí)", "turnover_ann": "Turnover/năm", "status": "Trạng thái"}
EXAMPLES = {
    "Alpha tốt nhất của agent (g8_bhd_w)": "2 * rank(ts_corr(returns, returns - demean(returns), 60)) + 2 * rank(ts_mean(close, 5) / ts_max(close, 250)) + rank(-ts_corr(rank(close), rank(volume), 10))",
    "Gần đỉnh 52 tuần": "rank(close / ts_max(close, 250))",
    "Phân kỳ giá – khối lượng": "-ts_corr(rank(close), rank(volume), 10)",
    "Đảo chiều 5 phiên (baseline)": "rank(-pct(close, 5))",
    "Tự viết công thức mới": "rank(ts_mean(close / open - 1, 20))",
    "Thử phá sandbox (sẽ bị chặn)": "close.values",
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
    return d.style.format({"IC": "{:.3f}", "t-stat": "{:.2f}", "Vượt rổ/năm (sau phí)": "{:+.1%}",
                           "Turnover/năm": "{:.0f}"}, na_rep="—")


def evolution_chart(board):
    ok = board[board.status == "ok"]
    per = ok.groupby("gen").t_stat.agg(["max", "median"]).reset_index()
    per["Tốt nhất đến nay"] = per["max"].cummax()
    per["Trung vị thế hệ"] = per["median"]
    long = per.melt("gen", ["Tốt nhất đến nay", "Trung vị thế hệ"], var_name="Đường", value_name="t")
    line = alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("gen:O", title="Thế hệ"),
        y=alt.Y("t:Q", title="t-stat IC (train)", scale=alt.Scale(domain=[-1, 4.5])),
        color=alt.Color("Đường:N", title=None, scale=alt.Scale(range=[BLUE, GREY])),
        tooltip=["gen", "Đường", alt.Tooltip("t:Q", format=".2f")])
    rule = alt.Chart(pd.DataFrame({"t": [3.0]})).mark_rule(strokeDash=[4, 4], color="#888").encode(y="t:Q")
    label = alt.Chart(pd.DataFrame({"t": [3.0], "x": [0]})).mark_text(
        text="t = 3 (ngưỡng tin cậy khi thử nhiều)", align="left", dy=-8, color="#666").encode(y="t:Q")
    return (line + rule + label).properties(height=300)


def top_table(board, n=5):
    top = board[board.status == "ok"].nlargest(n, "t_stat")
    return alpha_table(top[["name", "expr", "ic", "t_stat", "excess_ann", "turnover_ann", "status", "error"]])


def show_generation(ph, gen, note, gen_rows, board):
    with ph.container():
        st.subheader(f"Thế hệ {gen}" + (" — baseline do người viết" if gen == 0 else ""))
        if note:
            st.info(f"🧠 **Agent suy nghĩ:** {note}")
        ok = gen_rows[gen_rows.status == "ok"]
        c = st.columns(4)
        c[0].metric("Alpha chạy được", f"{len(ok)}/{len(gen_rows)}")
        c[1].metric("t-stat tốt nhất thế hệ", f"{ok.t_stat.max():.2f}" if len(ok) else "—")
        best_all = board[board.status == "ok"].t_stat.max()
        c[2].metric("t-stat tốt nhất toàn cục", f"{best_all:.2f}")
        c[3].metric("Tổng alpha đã thử", len(board))
        st.dataframe(alpha_table(gen_rows.sort_values("t_stat", ascending=False)), hide_index=True)
        left, right = st.columns([3, 2])
        left.altair_chart(evolution_chart(board))
        right.markdown("**🏆 Bảng xếp hạng (train)**")
        right.dataframe(top_table(board), hide_index=True)


# ---------------------------------------------------------------- layout
st.title("📈 AI Quant Researcher — LLM tự tìm alpha cho VN30")
st.caption("AIE301m · Agent chỉ được thấy dữ liệu train 2019–2022. Dữ liệu 2023–2026 chỉ mở một lần sau khi agent dừng.")

with st.sidebar:
    st.header("Kịch bản demo 5 phút")
    st.markdown(
        "1. **Tự thử alpha** (1'): mời một bạn gõ công thức, xem điểm, rồi thử *phá sandbox*.\n"
        "2. **Replay** (2'): chạy 8 thế hệ, dừng ở thế hệ 4–5 để giải thích agent đảo dấu beta.\n"
        "3. **Kết quả cuối** (1,5'): IC OOS đạt, nhưng sau phí thua rổ và alpha tắt từ 2025.\n"
        "4. **Agent live** (0,5'): bấm chạy 1 thế hệ thật bằng Claude API nếu có key.")
    st.divider()
    st.markdown("**Biến:** open, high, low, close, volume, vwap, returns, value, adv20")
    st.markdown("**Toán tử:** " + ", ".join(f"`{o}`" for o in sorted(OPS)))

tab_replay, tab_live, tab_try, tab_result = st.tabs(
    ["🔁 Replay 8 thế hệ", "🤖 Agent live (Claude API)", "🧪 Tự thử alpha", "📊 Kết quả cuối"])

# ---------------------------------------------------------------- 1. replay
with tab_replay:
    st.markdown("Phát lại quá trình agent tiến hoá alpha qua các thế hệ, đúng như lần chạy trong báo cáo.")
    c1, c2 = st.columns([1, 2])
    speed = c1.slider("Giây mỗi thế hệ", 0.0, 10.0, 3.0, 0.5)
    rescore = c2.checkbox("Chấm điểm lại thật bằng engine (chậm hơn, khoảng 10–30 giây mỗi thế hệ)",
                          help="Chạy lại toàn bộ backtest trong thư mục demo_replay/ để chứng minh số liệu không bịa.")
    if st.button("▶ Bắt đầu replay", type="primary"):
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
                with st.spinner(f"Đang backtest thế hệ {g}..."):
                    score_generation(str(ROOT / "alphas" / f"gen_{g:02d}.json"), work / "results")
                board = pd.read_csv(work / "results" / "leaderboard.csv")
            else:
                board = full[full.gen <= g]
            show_generation(ph, g, notes.get(str(g), ""), board[board.gen == g], board)
            progress.progress((i + 1) / len(gens), text=f"Thế hệ {g}/{gens[-1]}")
            time.sleep(speed)
        best = board[board.status == "ok"].nlargest(1, "t_stat").iloc[0]
        st.success(f"Agent dừng sau {gens[-1]} thế hệ ({len(board)} alpha). Alpha được chọn theo quy tắc định trước: "
                   f"**{best['name']}** (t-stat train {best.t_stat:.2f}). Chưa ai nhìn dữ liệu 2023–2026, xem tab **Kết quả cuối**.")

# ---------------------------------------------------------------- 2. live
with tab_live:
    st.markdown("Claude Opus 5.5 đọc bảng xếp hạng, đề xuất alpha mới, engine chấm điểm, rồi kết quả quay lại cho Claude. "
                "Mỗi lần chạy ghi vào một thư mục riêng `demo_run/<thời gian>/`, không đụng tới kết quả của báo cáo.")
    st.warning("Mỗi thế hệ gọi API một lần, khoảng 30–90 giây và vài cent USD. Nên chạy thử ở nhà trước buổi thuyết trình.")
    c1, c2, c3 = st.columns([2, 1, 1])
    api_key = c1.text_input("ANTHROPIC_API_KEY (để trống nếu đã đặt biến môi trường)", type="password")
    n_gen = c2.number_input("Số thế hệ", 1, 10, 2)
    per_gen = c3.number_input("Alpha mỗi thế hệ", 3, 10, 6)
    has_creds = bool(api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    go = st.button("🚀 Cho agent chạy", type="primary")
    if go and not has_creds:
        st.error("Chưa có API key: nhập vào ô trên, hoặc đặt biến môi trường ANTHROPIC_API_KEY rồi mở lại app.")
    if go and has_creds:
        import anthropic
        from agent import evolve

        work = ROOT / "demo_run" / datetime.now().strftime("%Y%m%d_%H%M%S")
        ph = st.empty()
        status = st.status("Agent đang làm việc...", expanded=False)
        try:
            for step in evolve(work, int(n_gen), int(per_gen), patience=2, api_key=api_key or None):
                board = pd.read_csv(work / "results" / "leaderboard.csv")
                show_generation(ph, step["gen"], step["reflection"], board[board.gen == step["gen"]], board)
                status.update(label=f"Xong thế hệ {step['gen']} · chi phí API tích luỹ ${step['usd']:.2f}")
                status.code(step["report"])
                if step["stop"]:
                    st.info("Agent tự dừng: 2 thế hệ liên tiếp không cải thiện t-stat quá 0,1.")
            status.update(label=f"Hoàn tất · kết quả lưu ở {work.relative_to(ROOT)}", state="complete")
        except anthropic.AuthenticationError:
            status.update(label="Lỗi xác thực", state="error")
            st.error("API key không hợp lệ hoặc chưa đặt ANTHROPIC_API_KEY.")
        except anthropic.RateLimitError:
            status.update(label="Bị giới hạn tốc độ", state="error")
            st.error("API đang giới hạn tốc độ, thử lại sau ít phút.")
        except (anthropic.AnthropicError, RuntimeError, TypeError) as e:  # key sai, lỗi mạng/server, từ chối...
            status.update(label="Lỗi khi gọi Claude", state="error")
            st.error(f"{type(e).__name__}: {e}")

# ---------------------------------------------------------------- 3. try
with tab_try:
    st.markdown("Gõ một công thức alpha và engine sẽ chấm điểm trên giai đoạn **train 2019–2022**, giống hệt cách nó chấm cho agent.")
    choice = st.selectbox("Ví dụ", list(EXAMPLES))
    expr = st.text_area("Công thức", EXAMPLES[choice], key=f"expr_{choice}", height=80)
    show_oos = st.checkbox("Mở cả dữ liệu 2023–2026 (chỉ để minh hoạ overfit)",
                           help="Nếu nhìn kết quả OOS rồi sửa công thức thì OOS không còn là 'ngoài mẫu' nữa.")
    if st.button("⚡ Chấm điểm", type="primary"):
        try:
            with st.spinner("Đang backtest..."):
                stats, alpha, _ = evaluate(expr, "train")
                bt, _, _ = backtest(alpha, "train")
        except Exception as e:
            st.error(f"Engine từ chối công thức: {type(e).__name__}: {e}")
        else:
            t = stats["t_stat"]
            verdict = "mạnh 💪" if t >= 3 else "hứa hẹn 🙂" if t >= 2 else "yếu 😐" if t >= 0 else "ngược dấu, thử đảo dấu 🔄"
            st.markdown(f"#### Kết quả train 2019–2022: tín hiệu **{verdict}**")
            c = st.columns(6)
            c[0].metric("IC", f"{stats['ic']:.3f}")
            c[1].metric("t-stat", f"{t:.2f}")
            c[2].metric("Ngày IC > 0", f"{stats['ic_pos']:.0%}")
            c[3].metric("Vượt rổ/năm (sau phí)", f"{bt['excess_ann']:+.1%}")
            c[4].metric("Sharpe", f"{bt['sharpe']:.2f}")
            c[5].metric("Turnover/năm", f"{bt['turnover_ann']:.0f}")
            if bt["turnover_ann"] > 40:
                st.caption("Turnover trên 40 lần/năm thì phí 0,2%/chiều ăn khoảng 8%/năm lợi nhuận. Thử làm mượt bằng ts_mean.")

            _, fwd, _ = load_env()
            ic = daily_ic(alpha, fwd)
            yearly = ic.groupby(ic.index.year).mean().rename("IC").rename_axis("Năm").reset_index()
            yearly["Giai đoạn"] = pd.cut(yearly["Năm"], [0, 2022, 2024, 9999], labels=["Train", "Test", "Holdout"]).astype(str)
            if not show_oos:
                yearly = yearly[yearly["Giai đoạn"] == "Train"]
            bars = alt.Chart(yearly).mark_bar().encode(
                x=alt.X("Năm:O"), y=alt.Y("IC:Q", title="IC trung bình"),
                color=alt.Color("Giai đoạn:N", scale=alt.Scale(domain=["Train", "Test", "Holdout"], range=[GREY, BLUE, "#f59e0b"])),
                tooltip=["Năm", "Giai đoạn", alt.Tooltip("IC:Q", format=".3f")]).properties(height=260)
            st.altair_chart(bars)

            if show_oos:
                rows = []
                for p, label in [("train", "Train 2019–2022"), ("test", "Test 2023–2024"), ("holdout", "Holdout 2025–10/2026")]:
                    s, _ = ic_stats(alpha, p)
                    b, _, _ = backtest(alpha, p)
                    rows.append({"Giai đoạn": label, "IC": s["ic"], "t-stat": s["t_stat"],
                                 "Lợi nhuận/năm": b["ann_ret"], "Rổ VN30/năm": b["bench_ann_ret"], "Vượt rổ (sau phí)": b["excess_ann"]})
                st.dataframe(pd.DataFrame(rows).style.format({"IC": "{:.3f}", "t-stat": "{:.2f}", "Lợi nhuận/năm": "{:+.1%}",
                                                              "Rổ VN30/năm": "{:+.1%}", "Vượt rổ (sau phí)": "{:+.1%}"}), hide_index=True)
                ratio = rows[1]["IC"] / rows[0]["IC"] if rows[0]["IC"] else float("nan")
                st.caption(f"Tỷ lệ overfit (IC test / IC train) = {ratio:.2f}. Dưới 0,5 nghĩa là phần lớn 'tín hiệu' train là nhiễu.")

# ---------------------------------------------------------------- 4. results
with tab_result:
    fin = json.loads((RES / "final.json").read_text(encoding="utf-8"))
    best = fin["best"]["report"]
    st.markdown(f"Alpha được chọn: **{fin['best']['name']}**")
    st.code(fin["best"]["expr"], language="python")
    c = st.columns(3)
    c[0].metric("Code agent chạy được", f"{fin['runnable_rate']:.1%}", "ngưỡng ≥ 70% ✓", delta_color="off")
    c[1].metric("IC OOS 2023–2024", f"{best['test']['ic']:.3f}", "ngưỡng ≥ 0,03 ✓", delta_color="off")
    c[2].metric("Tỷ lệ overfit", f"{fin['best']['overfit_ratio']:.2f}", "ngưỡng ≥ 0,5 ✓", delta_color="off")

    left, right = st.columns(2)
    evo = pd.read_csv(RES / "evolution.csv").rename(columns={"best_ic_train": "Train 2019–2022", "best_ic_test": "OOS 2023–2024"})
    evo_long = evo.melt("gen", ["Train 2019–2022", "OOS 2023–2024"], var_name="Dữ liệu", value_name="IC")
    left.markdown("**IC của alpha dẫn đầu qua từng thế hệ**")
    left.altair_chart(alt.Chart(evo_long).mark_line(point=True).encode(
        x=alt.X("gen:O", title="Thế hệ"), y=alt.Y("IC:Q"),
        color=alt.Color("Dữ liệu:N", title=None, scale=alt.Scale(range=[BLUE, GREY])),
        tooltip=["gen", "Dữ liệu", alt.Tooltip("IC:Q", format=".3f")]).properties(height=280))

    yearly = pd.read_csv(RES / "ic_by_year.csv")
    yearly["Giai đoạn"] = pd.cut(yearly["year"], [0, 2022, 2024, 9999], labels=["Train", "Test", "Holdout"]).astype(str)
    right.markdown("**IC theo năm: dương 6 năm liền rồi tắt từ 2025**")
    right.altair_chart(alt.Chart(yearly).mark_bar().encode(
        x=alt.X("year:O", title="Năm"), y=alt.Y("best:Q", title="IC"),
        color=alt.Color("Giai đoạn:N", scale=alt.Scale(domain=["Train", "Test", "Holdout"], range=[GREY, BLUE, "#f59e0b"])),
        tooltip=["year", alt.Tooltip("best:Q", format=".3f")]).properties(height=280))

    eq = pd.read_csv(RES / "equity_oos_weekly.csv", parse_dates=["Date"]).dropna()
    eq = eq.rename(columns={"best": "Alpha tốt nhất (sau phí)", "combo": "Danh mục kết hợp 5 alpha", "vn30_ew": "Rổ VN30 chia đều"})
    eq_long = eq.melt("Date", var_name="Danh mục", value_name="Giá trị")
    st.markdown("**Giá trị 1 đồng đầu tư, 01/2023–10/2026 (sau phí 0,2%/chiều)**")
    st.altair_chart(alt.Chart(eq_long).mark_line().encode(
        x=alt.X("Date:T", title=None), y=alt.Y("Giá trị:Q", scale=alt.Scale(zero=False)),
        color=alt.Color("Danh mục:N", title=None, scale=alt.Scale(range=[BLUE, "#60a5fa", GREY])),
        tooltip=[alt.Tooltip("Date:T"), "Danh mục", alt.Tooltip("Giá trị:Q", format=".2f")]).properties(height=320))

    rows = [{"Giai đoạn": lbl, "IC": best[p]["ic"], "t-stat": best[p]["t_stat"], "Lợi nhuận/năm": best[p]["ann_ret"],
             "Sharpe": best[p]["sharpe"], "Max drawdown": best[p]["max_dd"], "Rổ VN30/năm": best[p]["bench_ann_ret"],
             "Vượt rổ (sau phí)": best[p]["excess_ann"], "Turnover/năm": best[p]["turnover_ann"]}
            for p, lbl in [("train", "Train 2019–2022"), ("test", "Test 2023–2024"),
                           ("holdout", "Holdout 2025–10/2026"), ("blind", "Blind 07–10/2026")]]
    st.dataframe(pd.DataFrame(rows).style.format({
        "IC": "{:.3f}", "t-stat": "{:.2f}", "Lợi nhuận/năm": "{:+.1%}", "Sharpe": "{:.2f}", "Max drawdown": "{:.1%}",
        "Rổ VN30/năm": "{:+.1%}", "Vượt rổ (sau phí)": "{:+.1%}", "Turnover/năm": "{:.0f}"}), hide_index=True)
    st.markdown(
        "- **Dự báo được ≠ kiếm tiền được:** trước phí vượt rổ +6%/năm (test), nhưng turnover 45 lần/năm tốn khoảng 10%/năm phí.\n"
        "- **Alpha tắt từ 2025:** đà tăng dồn vào VIC và VHM, hai mã alpha xếp hạng thấp.\n"
        "- **Giới hạn:** LLM có thể đã 'biết' thị trường 2023–2025; danh sách VN30 cố định nên có survivorship bias.")
