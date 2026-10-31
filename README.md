# AI Quant Researcher — AIE301m

LLM agent tự đề xuất công thức alpha cho rổ VN30, tự backtest và tự cải tiến qua nhiều thế hệ.

## Cài đặt

```bash
pip install yfinance pandas scipy anthropic streamlit
```

## Demo trên lớp

```bash
streamlit run demo_app.py        # mở http://localhost:8501
```

| Tab | Dùng để |
| --- | --- |
| 🔁 Replay 8 thế hệ | Phát lại quá trình tiến hoá (tick "chấm điểm lại thật" để chạy lại backtest, khoảng 3 phút) |
| 🤖 Agent live | Claude Opus 5.5 nghiên cứu ngay trước lớp; cần `ANTHROPIC_API_KEY`, ghi vào `demo_run/<thời gian>/` |
| 🧪 Tự thử alpha | Khán giả gõ công thức, engine chấm điểm train ngay; thử `close.values` để thấy sandbox chặn |
| 📊 Kết quả cuối | 3 ngưỡng đạt, IC theo năm, đường vốn sau phí so với rổ VN30 |

Kịch bản 5 phút có sẵn ở thanh bên trái của app. Nên chạy thử tab Agent live ở nhà trước buổi thuyết trình,
vì mỗi thế hệ mất khoảng 30–90 giây và vài cent USD.

## Chạy từ dòng lệnh

```bash
python data.py                               # tải dữ liệu VN30 -> data/panel.pkl
python run_generation.py alphas/gen_00.json  # baseline do người viết
python agent.py --generations 8 --per-gen 8  # vòng lặp tự động (cần ANTHROPIC_API_KEY), ghi vào demo_run/
python final_eval.py                         # mở dữ liệu OOS, chạy ĐÚNG MỘT LẦN sau khi agent dừng
```

Muốn tái hiện kết quả trong báo cáo mà không gọi API: xoá `results/leaderboard.csv`, chạy
`run_generation.py` lần lượt cho `alphas/gen_00.json` … `gen_08.json`, rồi `final_eval.py`.

## Cấu trúc

| File | Vai trò |
| --- | --- |
| `data.py` | Tải OHLCV từ Yahoo Finance, sửa biến động vượt biên độ (chia tách chưa điều chỉnh) |
| `ops.py` | DSL: các toán tử alpha (chỉ nhìn về quá khứ) |
| `engine.py` | Tính alpha, IC, t-stat, backtest long-only top-6 sau phí |
| `run_generation.py` | Chấm một thế hệ trên train 2019-2022, ghi `results/leaderboard.csv` |
| `agent.py` | Vòng lặp tự động với Claude API (structured output, tự dừng); `evolve()` là generator dùng chung cho CLI và app |
| `demo_app.py` | Giao diện demo Streamlit 4 tab |
| `final_eval.py` | Quy tắc chọn định trước + đánh giá test/holdout/blind |
| `alphas/gen_XX.json` | Các alpha agent đã đề xuất, kèm giả thuyết |
| `results/` | Bảng xếp hạng, kết quả cuối, dữ liệu biểu đồ |

## Lưu ý

- `engine.check_expr` chỉ cho phép tên biến/toán tử DSL, số và phép toán (whitelist AST, không thuộc tính,
  không index, không lambda), sau đó mới `eval` với builtins bị chặn. App chỉ mở trên `localhost`
  (`.streamlit/config.toml`); đừng mở ra mạng công cộng.
- Danh sách VN30 cố định (kỳ 2024) cho cả giai đoạn -> có survivorship bias.
- Chạy tiếp agent trên bảng xếp hạng cũ sau khi đã xem OOS sẽ làm OOS không còn "sạch".
