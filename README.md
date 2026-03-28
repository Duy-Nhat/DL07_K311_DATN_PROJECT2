# 🏠 Phân Cụm Bất Động Sản – KMeans Clustering

Phân tích và phân cụm dữ liệu bất động sản tại Bình Thạnh, Gò Vấp, Phú Nhuận (TP.HCM)
bằng thuật toán **KMeans**, với giao diện trực quan bằng **Streamlit**.

---

## 📁 Cấu trúc Project

```
realestate_kmeans/
├── data/
│   ├── quan-binh-thanh.csv
│   ├── quan-go-vap.csv
│   └── quan-phu-nhuan.csv
├── model/                     ← Tự động tạo sau khi train
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   ├── pca.pkl
│   └── model_info.pkl
├── data_cleaning.py           ← Bước 1: Làm sạch dữ liệu
├── train_model.py             ← Bước 2: Huấn luyện KMeans
├── app.py                     ← Bước 3: Streamlit app
├── requirements.txt
└── README.md
```

---

## ⚙️ Cài đặt

### 1. Cài Python packages

```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu

Tạo thư mục `data/` và đặt 3 file CSV vào đó:

```
data/quan-binh-thanh.csv
data/quan-go-vap.csv
data/quan-phu-nhuan.csv
```

---

## 🚀 Chạy theo thứ tự

### Bước 1 – Làm sạch dữ liệu

```bash
python data_cleaning.py
```

**Output:** `data_cleaned.csv`  
Bao gồm: xóa trùng lặp, xử lý giá trị thiếu, parse cột số, lọc outlier.

---

### Bước 2 – Huấn luyện mô hình KMeans

```bash
python train_model.py
```

**Output:**
- `model/kmeans_model.pkl` – Mô hình KMeans đã train
- `model/scaler.pkl`       – StandardScaler
- `model/pca.pkl`          – PCA 2D để visualize
- `model/model_info.pkl`   – Metadata (elbow, silhouette, stats)
- `data_with_clusters.csv` – Dữ liệu đã gán nhãn cụm

---

### Bước 3 – Khởi động Streamlit App

```bash
streamlit run app.py
```

Mở trình duyệt tại: **http://localhost:8501**

---

## 📊 Tính năng App

| Tab | Nội dung |
|-----|----------|
| 🗺️ Phân cụm 2D | Biểu đồ scatter PCA – xem phân bố các cụm |
| 📈 Elbow & Silhouette | Biểu đồ chọn số cụm tối ưu |
| 📊 So sánh cụm | Box plot, bảng thống kê, radar chart |
| 🏘️ Phân tích quận | Phân bố cụm theo từng quận |
| 🔮 Dự đoán cụm | Nhập thông tin BĐS → dự đoán thuộc cụm nào |

---

## 🔧 Đặc trưng phân cụm (Features)

| Đặc trưng | Mô tả |
|-----------|-------|
| `gia_ban_ty` | Giá bán (tỷ VNĐ) |
| `dien_tich_m2` | Diện tích (m²) |
| `gia_per_m2` | Giá/m² (triệu/m²) |
| `so_phong_ngu_num` | Số phòng ngủ |
| `so_phong_vs_num` | Số phòng vệ sinh |
| `tong_so_tang` | Tổng số tầng |

---

## 📌 Ghi chú

- Số cụm `k` được chọn tự động dựa trên **Silhouette Score** cao nhất
- Dữ liệu được chuẩn hóa bằng **StandardScaler** trước khi đưa vào KMeans
- **PCA 2D** dùng để visualize không gian nhiều chiều lên 2D
