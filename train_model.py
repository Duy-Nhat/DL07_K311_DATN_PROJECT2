"""
Script 2: Huấn luyện mô hình KMeans phân cụm bất động sản
Chạy: python train_model.py
Output: model/kmeans_model.pkl, model/scaler.pkl, model/model_info.pkl
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# ─── 1. ĐỌC DỮ LIỆU ĐÃ LÀM SẠCH ─────────────────────────────────────────────

df = pd.read_csv("data_cleaned.csv")
print(f"📂 Đọc dữ liệu: {len(df):,} dòng")

# ─── 2. CHỌN ĐẶC TRƯNG ĐỂ PHÂN CỤM ──────────────────────────────────────────

FEATURES = [
    "gia_ban_ty",        # Giá bán (tỷ VNĐ)
    "dien_tich_m2",      # Diện tích (m²)
    "gia_per_m2",        # Giá/m² (triệu)
    "so_phong_ngu_num",  # Số phòng ngủ
    "so_phong_vs_num",   # Số phòng vệ sinh
    "tong_so_tang",      # Tổng số tầng
]

# Lấy tập con có đủ các features
df_feat = df[FEATURES].copy()
df_feat = df_feat.dropna()
df_valid = df.loc[df_feat.index].copy()

print(f"📊 Số mẫu có đủ đặc trưng: {len(df_feat):,}")
print(f"   Đặc trưng sử dụng: {FEATURES}")

# ─── 3. CHUẨN HÓA DỮ LIỆU ─────────────────────────────────────────────────────

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_feat)
print("\n✅ Đã chuẩn hóa dữ liệu (StandardScaler)")

# ─── 4. TÌM SỐ CỤM TỐI ƯU (ELBOW + SILHOUETTE) ───────────────────────────────

print("\n🔍 Đang tìm số cụm tối ưu (k=2 đến 10)...")
inertia_list = []
silhouette_list = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertia_list.append(km.inertia_)
    sil = silhouette_score(X_scaled, labels, sample_size=min(3000, len(X_scaled)))
    silhouette_list.append(sil)
    print(f"   k={k}: inertia={km.inertia_:.1f}, silhouette={sil:.4f}")

best_k_sil = k_range[np.argmax(silhouette_list)]
print(f"\n⭐ Số cụm tốt nhất theo Silhouette Score: k={best_k_sil}")

# ─── 5. HUẤN LUYỆN MÔ HÌNH CHÍNH ─────────────────────────────────────────────

N_CLUSTERS = best_k_sil  # Hoặc thay bằng số cụm bạn muốn

print(f"\n🚀 Huấn luyện KMeans với k={N_CLUSTERS}...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10, max_iter=300)
labels = kmeans.fit_predict(X_scaled)

# Gán nhãn cụm vào dataframe
df_valid["cum"] = labels
df_valid["cum_label"] = df_valid["cum"].apply(lambda x: f"Cụm {x}")

# ─── 6. PCA ĐỂ VISUALIZE ──────────────────────────────────────────────────────

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df_valid["pca_x"] = X_pca[:, 0]
df_valid["pca_y"] = X_pca[:, 1]

# ─── 7. THỐNG KÊ MÔ TẢ CÁC CỤM ──────────────────────────────────────────────

print(f"\n📈 Thống kê các cụm:")
cluster_stats = df_valid.groupby("cum")[FEATURES + ["gia_per_m2"]].mean().round(2)
cluster_stats["so_luong"] = df_valid.groupby("cum").size()
print(cluster_stats)

# ─── 8. LƯU MODEL ─────────────────────────────────────────────────────────────

os.makedirs("model", exist_ok=True)

# Lưu KMeans model
with open("model/kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)

# Lưu Scaler
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Lưu PCA
with open("model/pca.pkl", "wb") as f:
    pickle.dump(pca, f)

# Lưu thông tin model (metadata)
model_info = {
    "n_clusters":      N_CLUSTERS,
    "features":        FEATURES,
    "inertia_list":    list(inertia_list),
    "silhouette_list": list(silhouette_list),
    "k_range":         list(k_range),
    "best_k":          best_k_sil,
    "cluster_stats":   cluster_stats.to_dict(),
    "pca_variance":    pca.explained_variance_ratio_.tolist(),
}
with open("model/model_info.pkl", "wb") as f:
    pickle.dump(model_info, f)

# Lưu dữ liệu đã gán nhãn
df_valid.to_csv("data_with_clusters.csv", index=False, encoding="utf-8-sig")

print("\n✅ Đã lưu:")
print("   model/kmeans_model.pkl  ← Mô hình KMeans")
print("   model/scaler.pkl        ← Scaler chuẩn hóa")
print("   model/pca.pkl           ← PCA 2D visualization")
print("   model/model_info.pkl    ← Metadata & thống kê")
print("   data_with_clusters.csv  ← Dữ liệu có nhãn cụm")
print(f"\n🎉 Hoàn tất! Mô hình phân {N_CLUSTERS} cụm bất động sản.")
