"""Tải và làm sạch dữ liệu giá ngày của rổ VN30 từ Yahoo Finance.

Chạy: python data.py  -> tạo data/panel.pkl (dict các DataFrame ngày x mã).
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

# Thành phần VN30 (kỳ cơ cấu 2024). Dùng danh sách hiện tại cho cả giai đoạn 2019-2026
# nên có survivorship bias -- đã ghi rõ trong báo cáo.
VN30 = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
]
START, END = "2019-01-01", "2026-10-31"
OUT = Path(__file__).parent / "data" / "panel.pkl"


def download() -> dict:
    raw = yf.download([f"{t}.VN" for t in VN30], start=START, end=END,
                      auto_adjust=True, progress=False, group_by="column")
    panel = {}
    for field in ["Open", "High", "Low", "Close", "Volume"]:
        df = raw[field].copy()
        df.columns = [c.replace(".VN", "") for c in df.columns]
        panel[field.lower()] = df[VN30]
    return panel


def clean(panel: dict) -> dict:
    vol = panel["volume"]
    # Yahoo chèn ngày nghỉ lễ với volume = 0 cho mọi mã -> bỏ những ngày mà >80% mã không có giao dịch
    traded = (vol > 0).sum(axis=1) / vol.notna().sum(axis=1).clip(lower=1)
    keep = traded > 0.2
    out = {k: v.loc[keep] for k, v in panel.items()}
    # Ngày một mã không giao dịch (volume 0) -> giá coi như thiếu, không nội suy để tránh tín hiệu giả
    no_trade = out["volume"] <= 0
    for k in ["open", "high", "low", "close"]:
        out[k] = out[k].mask(no_trade)
    out["volume"] = out["volume"].mask(no_trade)
    # Biên độ HOSE ±7% (HNX ±10%): biến động ngày > 10.5% gần như luôn là chia tách/thưởng cổ phiếu
    # Yahoo chưa điều chỉnh (vd VCB -33% ngày 2025-03-03). Coi lợi nhuận ngày đó = 0 và
    # dựng lại chuỗi giá từ lợi nhuận đã sửa; OHLC nhân cùng hệ số điều chỉnh.
    close = out["close"]
    r = close / close.ffill().shift(1) - 1
    bad = r.abs() > 0.105
    r = r.mask(bad, 0.0)
    rebuilt = (1 + r.fillna(0)).cumprod().where(close.notna())
    factor = (rebuilt / close)
    # chuẩn hoá để giá ngày cuối = giá thật (dễ đọc), hệ số không đổi giữa hai lần sửa
    factor = factor / factor.ffill().iloc[-1]
    for k in ["open", "high", "low", "close"]:
        out[k] = out[k] * factor
    out["volume"] = out["volume"] / factor  # giữ giá trị giao dịch (giá x KL) không đổi
    out["n_fixed"] = int(bad.sum().sum())
    # bỏ các phiên quá ít mã có dữ liệu
    enough = out["close"].count(axis=1) >= 10
    for k in ["open", "high", "low", "close", "volume"]:
        out[k] = out[k].loc[enough]
    return out


if __name__ == "__main__":
    panel = clean(download())
    OUT.parent.mkdir(exist_ok=True)
    pd.to_pickle(panel, OUT)
    close = panel["close"]
    print(f"{close.shape[0]} phiên, {close.index[0].date()} -> {close.index[-1].date()}")
    first = close.apply(lambda s: s.first_valid_index())
    print("Ngày có dữ liệu đầu tiên của từng mã:")
    print(first.dt.date.sort_values().to_string())
    print("Tỷ lệ ô thiếu:", round(close.isna().mean().mean(), 3))
    print("Số biến động bất thường đã sửa:", panel["n_fixed"])
    r = close.pct_change(fill_method=None).stack()
    print("Lợi nhuận ngày lớn nhất sau sửa:", round(r.max(), 3), "nhỏ nhất:", round(r.min(), 3))
