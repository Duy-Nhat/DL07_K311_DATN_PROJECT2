"""
Script 1: Làm sạch dữ liệu bất động sản từ 3 quận
Chạy: python data_cleaning.py
Output: data_cleaned.csv
"""

import pandas as pd
import numpy as np
import re
import os

# ─── 1. ĐỌC & GỘP DỮ LIỆU ────────────────────────────────────────────────────

DATA_DIR = "data"  # Đặt 3 file CSV vào thư mục data/

files = {
    "Bình Thạnh": "quan-binh-thanh.csv",
    "Gò Vấp":     "quan-go-vap.csv",
    "Phú Nhuận":  "quan-phu-nhuan.csv",
}

dfs = []
for quan, filename in files.items():
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path, low_memory=False)
    df["quan"] = quan
    dfs.append(df)
    print(f"✅ Đọc {filename}: {len(df):,} dòng")

df = pd.concat(dfs, ignore_index=True)
print(f"\n📦 Tổng cộng sau khi gộp: {len(df):,} dòng, {df.shape[1]} cột")

# ─── 2. HÀM PARSE CÁC CỘT ────────────────────────────────────────────────────

def parse_gia_ban(s):
    """Chuyển '3,85 tỷ' → 3.85 (đơn vị: tỷ VNĐ)"""
    if pd.isna(s): return np.nan
    s = str(s).replace(",", ".").strip()
    m = re.search(r"([\d.]+)\s*tỷ", s)
    if m: return float(m.group(1))
    m = re.search(r"([\d.]+)\s*triệu", s)
    if m: return float(m.group(1)) / 1000
    return np.nan

def parse_number(s):
    """Trích số từ chuỗi như '36 m²', '4.5 m', '2 phòng'"""
    if pd.isna(s): return np.nan
    s = str(s).replace(",", ".")
    m = re.search(r"([\d.]+)", s)
    return float(m.group(1)) if m else np.nan

# ─── 3. PARSE CÁC CỘT QUAN TRỌNG ─────────────────────────────────────────────

print("\n🔧 Đang parse các cột số...")
df["gia_ban_ty"]       = df["gia_ban"].apply(parse_gia_ban)
df["dien_tich_m2"]     = df["dien_tich"].apply(parse_number)
df["so_phong_ngu_num"] = df["so_phong_ngu"].apply(parse_number)
df["so_phong_vs_num"]  = df["so_phong_ve_sinh"].apply(parse_number)
df["chieu_ngang_num"]  = df["chieu_ngang"].apply(parse_number)
df["chieu_dai_num"]    = df["chieu_dai"].apply(parse_number)
df["tong_so_tang"]     = pd.to_numeric(df["tong_so_tang"], errors="coerce")

# ─── 4. LÀM SẠCH DỮ LIỆU ─────────────────────────────────────────────────────

raw_count = len(df)
print(f"\n🧹 Bắt đầu làm sạch từ {raw_count:,} dòng...")

# 4a. Xóa trùng lặp theo tiêu đề + địa chỉ
df = df.drop_duplicates(subset=["tieu_de", "dia_chi"], keep="first")
print(f"  Sau xóa trùng lặp (tieu_de + dia_chi): {len(df):,} dòng")

# 4b. Xóa các dòng thiếu giá trị ở cột thiết yếu
essential_cols = ["gia_ban_ty", "dien_tich_m2", "quan"]
df = df.dropna(subset=essential_cols)
print(f"  Sau xóa NaN thiết yếu: {len(df):,} dòng")

# 4c. Lọc giá trị hợp lệ (loại bỏ outlier cực đoan)
df = df[
    (df["gia_ban_ty"]   >= 0.5)  & (df["gia_ban_ty"]   <= 200) &  # 0.5 – 200 tỷ
    (df["dien_tich_m2"] >= 10)   & (df["dien_tich_m2"] <= 500)    # 10 – 500 m²
]
print(f"  Sau lọc outlier: {len(df):,} dòng")

# 4d. Thêm cột phái sinh
df["gia_per_m2"] = df["gia_ban_ty"] * 1000 / df["dien_tich_m2"]  # triệu/m²

# ─── 5. RESET INDEX & LƯU ─────────────────────────────────────────────────────

df = df.reset_index(drop=True)
out_path = "data_cleaned.csv"
df.to_csv(out_path, index=False, encoding="utf-8-sig")

print(f"\n✅ Hoàn tất! Đã lưu {len(df):,} dòng → {out_path}")
print(f"   Đã loại bỏ {raw_count - len(df):,} dòng không hợp lệ/trùng lặp")
print("\nThống kê theo quận:")
print(df.groupby("quan")[["gia_ban_ty", "dien_tich_m2", "gia_per_m2"]].agg(["mean", "count"]).round(2))
