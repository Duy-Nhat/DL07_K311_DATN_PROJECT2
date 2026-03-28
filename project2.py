import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# ─── CẤU HÌNH TRANG ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Data Science and Machine Learning",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS CHUNG ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .header-container {
        background: linear-gradient(135deg, #4c1d95 0%, #5b21b6 50%, #7c3aed 100%);
        padding: 2rem 1rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(139, 92, 246, 0.3);
        border: 1px solid rgba(139, 92, 246, 0.2);
    }
    .main-title {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .feature-title {
        color: white;
        font-size: 2rem;
        font-weight: bold;
    }
    .section-header {
        color: #f7fafc;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #4a5568;
    }
    .breadcrumb {
        background: linear-gradient(90deg, #2d3748 0%, #374151 100%);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #8b5cf6;
        margin: 1rem 0 1.5rem 0;
        font-weight: 500;
        color: #e5e7eb;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    .footer-container {
        width: 100%;
        margin: 40px auto;
        padding: 25px;
        border-radius: 12px;
        background-color: #2c3e50;
        text-align: center;
        font-family: 'Segoe UI', sans-serif;
        color: #ecf0f1;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .footer-container h4 { font-size: 18px; font-weight: 600; }
    .footer-container .title {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #f1c40f;
    }
    .footer-container a { color: #1abc9c; text-decoration: none; }
    .footer-container a:hover { text-decoration: underline; }
    .footer-container hr { margin: 18px auto; width: 60%; border: 0.5px solid #7f8c8d; }
    .footer-container p { margin: 6px 0; font-size: 15px; }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ──────────────────────────────────────────────────────────────────

st.image("banner_nhatot.png", width="stretch")
st.markdown("""
<div class="header-container">
    <div class="main-title">Đồ án tốt nghiệp - Data Science and Machine Learning</div>
    <div class="feature-title">Dự đoán giá nhà & phát hiện bất thường</div>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR – MENU ──────────────────────────────────────────────────────────

st.sidebar.markdown("### MENU")
choice = st.sidebar.selectbox(
    'Chọn chức năng',
    ['Tổng quan', 'Hệ thống gợi ý nhà dựa trên nội dung', 'Hệ thống phân cụm nhà'],
    help="Chọn phần bạn muốn khám phá"
)

# Breadcrumb
st.markdown(f'<div class="breadcrumb">{choice}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TRANG 1: TỔNG QUAN
# ══════════════════════════════════════════════════════════════════════════════

if choice == 'Tổng quan':
    st.markdown('<h1 class="section-header">Chào mừng đến với ứng dụng phân tích</h1>',
                unsafe_allow_html=True)
    st.subheader("[Trang chủ](https://csc.edu.vn)")
    st.write('''
    ### Chào mừng bạn đến với khóa học
    ##### Đồ án TN''')

# ══════════════════════════════════════════════════════════════════════════════
# TRANG 2: HỆ THỐNG GỢI Ý NHÀ
# ══════════════════════════════════════════════════════════════════════════════

elif choice == 'Hệ thống gợi ý nhà dựa trên nội dung':
    st.subheader("[Đồ án TN Data Science](https://csc.edu.vn/data-science-machine-learning/Do-An-Tot-Nghiep-Data-Science---Machine-Learning_229)")
    st.write("##### Gợi ý điều khiển project 2: Recommender System")
    st.write("##### Dữ liệu mẫu")

    # Đọc và gộp dữ liệu
    df_bt = pd.read_csv("./data/quan-binh-thanh.csv")
    df_gv = pd.read_csv("./data/quan-go-vap.csv")
    df_pn = pd.read_csv("./data/quan-phu-nhuan.csv")
    df = pd.concat([df_bt, df_gv, df_pn], ignore_index=True)

    # Parse giá và diện tích sang số
    df["gia_ban_num"] = pd.to_numeric(
        df["gia_ban"].astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce"
    )
    df["dien_tich_num"] = pd.to_numeric(
        df["dien_tich"].astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce"
    )
    df = df.dropna(subset=["gia_ban_num", "dien_tich_num"])
    df['ma_can'] = df['ma_can'].astype(str)

    st.dataframe(df, width="stretch")

    # ── Tìm nhà tương tự theo dropdown ──────────────────────────────────────
    st.write("### 1. Tìm kiếm nhà tương tự")
    selected_house = st.selectbox("Chọn nhà", df['tieu_de'])
    idx = df[df['tieu_de'] == selected_house].index[0]

    st.write("### Thông tin nhà đã chọn:")
    st.dataframe(df.iloc[[idx]][["tieu_de", "gia_ban", "dien_tich"]], width="stretch")

    st.write("Thông tin các ngôi nhà tương tự:")
    selected_row = df.iloc[idx]
    similar_houses = df[
        (abs(df["gia_ban_num"] - selected_row["gia_ban_num"]) < 50) &
        (abs(df["dien_tich_num"] - selected_row["dien_tich_num"]) < 20)
    ].drop(index=idx).head(5)
    st.dataframe(similar_houses[["tieu_de", "gia_ban", "dien_tich"]], width="stretch")

    # ── Tìm kiếm bằng từ khoá ────────────────────────────────────────────────
    st.write("### 2. Tìm kiếm bằng từ khoá")
    search = st.text_input("Nhập thông tin tìm kiếm")
    if st.button("Tìm kiếm"):
        result = df[df['tieu_de'].str.lower().str.contains(search.lower(), na=False)]
        st.write("Danh sách nhà tìm được:")
        st.dataframe(result[["tieu_de", "gia_ban", "dien_tich"]], width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# TRANG 3: HỆ THỐNG PHÂN CỤM NHÀ (tích hợp toàn bộ app.py)
# ══════════════════════════════════════════════════════════════════════════════

elif choice == 'Hệ thống phân cụm nhà':
    st.subheader("[Đồ án TN Data Science](https://csc.edu.vn/data-science-machine-learning/Do-An-Tot-Nghiep-Data-Science---Machine-Learning_229)")

    # ── Tải dữ liệu & model ──────────────────────────────────────────────────

    @st.cache_data
    def load_data():
        df = pd.read_csv("data_with_clusters.csv")
        
        # Fix: ép tất cả cột object về str thuần để tránh lỗi PyArrow
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str)
        
        return df

    @st.cache_resource
    def load_model():
        with open("model/kmeans_model.pkl", "rb") as f:
            kmeans = pickle.load(f)
        with open("model/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("model/pca.pkl", "rb") as f:
            pca = pickle.load(f)
        with open("model/model_info.pkl", "rb") as f:
            info = pickle.load(f)
        return kmeans, scaler, pca, info

    try:
        df_cluster = load_data()
        kmeans, scaler, pca, model_info = load_model()
    except FileNotFoundError:
        st.error(
            "⚠️ Chưa tìm thấy file dữ liệu hoặc model. Hãy chạy:\n\n"
            "```\npython data_cleaning.py\npython train_model.py\n```"
        )
        st.stop()

    CLUSTER_COLORS = px.colors.qualitative.Bold
    n_clusters = model_info["n_clusters"]
    COLOR_MAP = {f"Cụm {i}": CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(n_clusters)}

    # ── Bộ lọc trong sidebar (chỉ hiện khi ở trang phân cụm) ─────────────────

    with st.sidebar:
        st.markdown("---")
        st.header("🔧 Bộ lọc phân cụm")

        all_quan = sorted(df_cluster["quan"].unique().tolist())
        selected_quan = st.multiselect("📍 Quận", all_quan, default=all_quan)

        all_clusters = sorted(df_cluster["cum_label"].unique().tolist())
        selected_clusters_filter = st.multiselect("🎯 Cụm", all_clusters, default=all_clusters)

        gia_min = float(df_cluster["gia_ban_ty"].min())
        gia_max = float(df_cluster["gia_ban_ty"].max())
        price_range = st.slider("💰 Giá bán (tỷ VNĐ)", gia_min, gia_max, (gia_min, gia_max), step=0.5)

        dt_min = float(df_cluster["dien_tich_m2"].min())
        dt_max = float(df_cluster["dien_tich_m2"].max())
        area_range = st.slider("📐 Diện tích (m²)", dt_min, dt_max, (dt_min, dt_max), step=5.0)

        st.divider()
        st.caption(f"🤖 Mô hình: KMeans (k={n_clusters})")
        st.caption(f"📊 Tổng dữ liệu: {len(df_cluster):,} BĐS")

    # Lọc dữ liệu
    mask = (
        df_cluster["quan"].isin(selected_quan) &
        df_cluster["cum_label"].isin(selected_clusters_filter) &
        df_cluster["gia_ban_ty"].between(*price_range) &
        df_cluster["dien_tich_m2"].between(*area_range)
    )
    df_filtered = df_cluster[mask].copy()

    # ── KPI Cards ─────────────────────────────────────────────────────────────

    st.markdown("### 📊 Tổng quan")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("🏘️ Tổng BĐS",     f"{len(df_filtered):,}")
    k2.metric("💰 Giá TB",        f"{df_filtered['gia_ban_ty'].mean():.2f} tỷ"  if len(df_filtered) else "—")
    k3.metric("📐 Diện tích TB",  f"{df_filtered['dien_tich_m2'].mean():.1f} m²" if len(df_filtered) else "—")
    k4.metric("💵 Giá/m² TB",     f"{df_filtered['gia_per_m2'].mean():.1f} tr"  if len(df_filtered) else "—")
    k5.metric("🎯 Số cụm",        n_clusters)

    st.divider()

    # ── 5 Tabs chính ─────────────────────────────────────────────────────────

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️ Phân cụm 2D",
        "📈 Elbow & Silhouette",
        "📊 So sánh cụm",
        "🏘️ Phân tích quận",
        "🔮 Dự đoán cụm",
    ])

    # ════ TAB 1: SCATTER PCA ═══════════════════════════════════════════════════
    with tab1:
        st.subheader("Phân cụm BĐS theo không gian PCA 2D")
        st.caption(f"PCA giải thích {sum(model_info['pca_variance'])*100:.1f}% phương sai dữ liệu")

        if len(df_filtered) == 0:
            st.warning("Không có dữ liệu phù hợp với bộ lọc.")
        else:
            col_scatter, col_info = st.columns([3, 1])

            with col_scatter:
                hover_cols = {c: True for c in ["tieu_de", "quan", "cum_label"]
                              if c in df_filtered.columns}
                hover_cols.update({
                    "gia_ban_ty":   ":.2f",
                    "dien_tich_m2": ":.1f",
                    "gia_per_m2":   ":.1f",
                    "pca_x": False,
                    "pca_y": False,
                })
                fig_pca = px.scatter(
                    df_filtered, x="pca_x", y="pca_y",
                    color="cum_label",
                    color_discrete_map=COLOR_MAP,
                    hover_data=hover_cols,
                    labels={
                        "pca_x": f"PC1 ({model_info['pca_variance'][0]*100:.1f}%)",
                        "pca_y": f"PC2 ({model_info['pca_variance'][1]*100:.1f}%)",
                        "cum_label": "Cụm",
                    },
                    title="Biểu đồ phân cụm (PCA 2D)",
                    opacity=0.7, height=520,
                )
                fig_pca.update_traces(marker=dict(size=5))
                fig_pca.update_layout(legend_title="Cụm BĐS")
                st.plotly_chart(fig_pca, width="stretch")

            with col_info:
                st.markdown("**Số lượng theo cụm:**")
                for cum_label in sorted(df_filtered["cum_label"].unique()):
                    cnt = (df_filtered["cum_label"] == cum_label).sum()
                    pct = cnt / len(df_filtered) * 100
                    color = COLOR_MAP.get(cum_label, "#888")
                    st.markdown(
                        f'<span style="background:{color};color:white;padding:3px 10px;'
                        f'border-radius:12px;font-size:0.85rem">{cum_label}</span>'
                        f' &nbsp; **{cnt:,}** ({pct:.1f}%)',
                        unsafe_allow_html=True,
                    )
                    st.markdown("")

                st.divider()
                st.markdown("**Giá trung bình theo cụm:**")
                for label, val in df_filtered.groupby("cum_label")["gia_ban_ty"].mean().sort_index().items():
                    st.markdown(f"**{label}**: {val:.2f} tỷ")

    # ════ TAB 2: ELBOW & SILHOUETTE ════════════════════════════════════════════
    with tab2:
        st.subheader("Phương pháp Elbow & Silhouette – Chọn số cụm tối ưu")

        k_range    = model_info["k_range"]
        inertia    = model_info["inertia_list"]
        silhouette = model_info["silhouette_list"]
        best_k     = model_info["best_k"]

        col_e, col_s = st.columns(2)

        with col_e:
            fig_elbow = go.Figure()
            fig_elbow.add_trace(go.Scatter(
                x=k_range, y=inertia,
                mode="lines+markers",
                line=dict(color="#1f4e79", width=2),
                marker=dict(size=8, color=["#e74c3c" if k == best_k else "#1f4e79" for k in k_range]),
                name="Inertia",
            ))
            fig_elbow.add_vline(x=best_k, line_dash="dash", line_color="red",
                                annotation_text=f"k={best_k}", annotation_position="top right")
            fig_elbow.update_layout(
                title="📉 Elbow Method (Inertia)",
                xaxis_title="Số cụm k",
                yaxis_title="Inertia (Within-cluster SSE)",
                height=350,
            )
            st.plotly_chart(fig_elbow, width="stretch")

        with col_s:
            fig_sil = go.Figure()
            fig_sil.add_trace(go.Scatter(
                x=k_range, y=silhouette,
                mode="lines+markers",
                line=dict(color="#27ae60", width=2),
                marker=dict(size=8, color=["#e74c3c" if k == best_k else "#27ae60" for k in k_range]),
                name="Silhouette",
            ))
            fig_sil.add_vline(x=best_k, line_dash="dash", line_color="red",
                              annotation_text=f"k={best_k} (tốt nhất)", annotation_position="top right")
            fig_sil.update_layout(
                title="📈 Silhouette Score",
                xaxis_title="Số cụm k",
                yaxis_title="Silhouette Score (cao hơn = tốt hơn)",
                height=350,
            )
            st.plotly_chart(fig_sil, width="stretch")

        st.info(
            f"✅ **Số cụm được chọn: k = {best_k}** "
            f"(Silhouette Score = {max(silhouette):.4f}). "
            "Silhouette Score gần 1 nghĩa là các cụm phân tách tốt."
        )

    # ════ TAB 3: SO SÁNH CỤM ══════════════════════════════════════════════════
    with tab3:
        st.subheader("So sánh đặc trưng giữa các cụm")

        if len(df_filtered) == 0:
            st.warning("Không có dữ liệu.")
        else:
            features_show = {
                "gia_ban_ty":       "Giá bán (tỷ)",
                "dien_tich_m2":     "Diện tích (m²)",
                "gia_per_m2":       "Giá/m² (triệu)",
                "so_phong_ngu_num": "Phòng ngủ",
                "so_phong_vs_num":  "Phòng vệ sinh",
                "tong_so_tang":     "Số tầng",
            }

            col_feat, col_chart = st.columns([1, 3])
            with col_feat:
                selected_feat = st.radio(
                    "Chọn đặc trưng", list(features_show.keys()),
                    format_func=lambda x: features_show[x]
                )
            with col_chart:
                fig_box = px.box(
                    df_filtered, x="cum_label", y=selected_feat,
                    color="cum_label", color_discrete_map=COLOR_MAP,
                    labels={"cum_label": "Cụm", selected_feat: features_show[selected_feat]},
                    title=f"Phân phối '{features_show[selected_feat]}' theo cụm",
                    height=400,
                )
                fig_box.update_layout(showlegend=False)
                st.plotly_chart(fig_box, width="stretch")

            # Bảng thống kê trung bình
            st.markdown("#### Bảng thống kê trung bình theo cụm")
            stats_df = df_filtered.groupby("cum_label")[list(features_show.keys())].mean().round(2)
            stats_df.columns = list(features_show.values())
            stats_df["Số lượng"] = df_filtered.groupby("cum_label").size()
            st.dataframe(stats_df.style.background_gradient(cmap="Blues"), width="stretch")

            # Radar chart
            st.markdown("#### Biểu đồ Radar so sánh cụm")
            base = stats_df.drop("Số lượng", axis=1)
            norm_stats = (base - base.min()) / (base.max() - base.min() + 1e-9)
            categories = list(features_show.values())
            fig_radar = go.Figure()
            for i, cum_label in enumerate(norm_stats.index):
                vals = norm_stats.loc[cum_label].tolist() + [norm_stats.loc[cum_label].tolist()[0]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals,
                    theta=categories + [categories[0]],
                    fill="toself",
                    name=cum_label,
                    line_color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                    opacity=0.7,
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                height=420,
                title="Radar Chart – Đặc trưng chuẩn hóa theo cụm",
            )
            st.plotly_chart(fig_radar, width="stretch")

    # ════ TAB 4: PHÂN TÍCH THEO QUẬN ══════════════════════════════════════════
    with tab4:
        st.subheader("Phân bố cụm theo quận")

        if len(df_filtered) == 0:
            st.warning("Không có dữ liệu.")
        else:
            col_a, col_b = st.columns(2)

            with col_a:
                cross = pd.crosstab(df_filtered["quan"], df_filtered["cum_label"])
                fig_bar = px.bar(
                    cross.reset_index().melt(id_vars="quan", var_name="Cụm", value_name="Số lượng"),
                    x="quan", y="Số lượng", color="Cụm",
                    color_discrete_map=COLOR_MAP,
                    barmode="stack",
                    title="Phân bố số lượng BĐS theo quận & cụm",
                    height=380,
                )
                st.plotly_chart(fig_bar, width="stretch")

            with col_b:
                avg_price = df_filtered.groupby("quan")["gia_ban_ty"].mean().reset_index()
                fig_price = px.bar(
                    avg_price, x="quan", y="gia_ban_ty", color="quan",
                    title="Giá bán trung bình theo quận (tỷ VNĐ)",
                    labels={"gia_ban_ty": "Giá TB (tỷ)", "quan": "Quận"},
                    text_auto=".2f", height=380,
                )
                fig_price.update_traces(textposition="outside")
                fig_price.update_layout(showlegend=False)
                st.plotly_chart(fig_price, width="stretch")

            fig_scatter = px.scatter(
                df_filtered.sample(min(2000, len(df_filtered)), random_state=42),
                x="dien_tich_m2", y="gia_ban_ty",
                color="quan", symbol="cum_label", opacity=0.6,
                title="Giá bán vs Diện tích (màu = quận, ký hiệu = cụm)",
                labels={"dien_tich_m2": "Diện tích (m²)", "gia_ban_ty": "Giá bán (tỷ)"},
                height=420,
            )
            st.plotly_chart(fig_scatter, width="stretch")

    # ════ TAB 5: DỰ ĐOÁN CỤM ══════════════════════════════════════════════════
    with tab5:
        st.subheader("🔮 Dự đoán cụm cho BĐS mới")
        st.caption("Nhập thông tin bất động sản để dự đoán thuộc cụm nào")

        col1, col2 = st.columns(2)
        with col1:
            inp_gia    = st.number_input("💰 Giá bán (tỷ VNĐ)",   min_value=0.5,  max_value=200.0, value=5.0,   step=0.5)
            inp_dt     = st.number_input("📐 Diện tích (m²)",      min_value=10.0, max_value=500.0, value=50.0,  step=5.0)
            inp_gia_m2 = st.number_input("💵 Giá/m² (triệu/m²)",  min_value=10.0, max_value=500.0, value=100.0, step=5.0)
        with col2:
            inp_pn   = st.number_input("🛏️ Số phòng ngủ",         min_value=1, max_value=10, value=3, step=1)
            inp_vs   = st.number_input("🚽 Số phòng vệ sinh",      min_value=1, max_value=10, value=2, step=1)
            inp_tang = st.number_input("🏢 Tổng số tầng",          min_value=1, max_value=20, value=3, step=1)

        if st.button("🚀 Dự đoán cụm", type="primary", width="stretch"):
            input_data     = np.array([[inp_gia, inp_dt, inp_gia_m2, inp_pn, inp_vs, inp_tang]])
            input_scaled   = scaler.transform(input_data)
            pred_cluster   = kmeans.predict(input_scaled)[0]
            cum_label_pred = f"Cụm {pred_cluster}"
            color          = COLOR_MAP.get(cum_label_pred, "#667eea")

            st.markdown(
                f"""
                <div style="background:{color};color:white;padding:1.5rem;
                            border-radius:16px;text-align:center;margin:1rem 0">
                    <div style="font-size:1.2rem;opacity:0.9">BĐS này thuộc</div>
                    <div style="font-size:2.5rem;font-weight:700">{cum_label_pred}</div>
                </div>
                """, unsafe_allow_html=True
            )

            cum_data = df_cluster[df_cluster["cum"] == pred_cluster]
            st.markdown(f"**Thống kê {cum_label_pred} (toàn bộ dữ liệu):**")
            cs1, cs2, cs3, cs4 = st.columns(4)
            cs1.metric("Số lượng BĐS", f"{len(cum_data):,}")
            cs2.metric("Giá TB",       f"{cum_data['gia_ban_ty'].mean():.2f} tỷ")
            cs3.metric("Diện tích TB", f"{cum_data['dien_tich_m2'].mean():.1f} m²")
            cs4.metric("Giá/m² TB",    f"{cum_data['gia_per_m2'].mean():.1f} tr")

            distances = kmeans.transform(input_scaled)[0]
            fig_dist = px.bar(
                x=[f"Cụm {i}" for i in range(n_clusters)],
                y=distances,
                color=[f"Cụm {i}" for i in range(n_clusters)],
                color_discrete_map=COLOR_MAP,
                title="Khoảng cách đến các trung tâm cụm (càng nhỏ càng gần)",
                labels={"x": "Cụm", "y": "Khoảng cách"},
            )
            st.plotly_chart(fig_dist, width="stretch")

    # ── Xem dữ liệu thô & tải CSV ─────────────────────────────────────────────
    st.divider()
    with st.expander("📋 Xem dữ liệu thô (sau lọc)", expanded=False):
        rename_map = {
            "tieu_de": "Tiêu đề", "quan": "Quận", "cum_label": "Cụm",
            "gia_ban_ty": "Giá (tỷ)", "dien_tich_m2": "DT (m²)",
            "gia_per_m2": "Giá/m² (tr)", "so_phong_ngu_num": "P.Ngủ",
            "tong_so_tang": "Số tầng",
        }
        display_cols = [c for c in rename_map if c in df_filtered.columns]
        st.dataframe(
            df_filtered[display_cols].rename(columns=rename_map),
            width="stretch", height=300,
        )
        st.download_button(
            "⬇️ Tải CSV",
            df_filtered.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            "bds_filtered.csv", "text/csv",
        )

# ─── FOOTER ──────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div class="footer-container">
    <p class="title">🎓 Đồ án tốt nghiệp – Data Science & Machine Learning</p>
    <h4>Phát triển bởi</h4>
    <p>• <strong>Trương Trường Giang</strong> – <a href="mailto:truonggiang210195@gmail.com">truonggiang210195@gmail.com</a></p>
    <p>• <strong>Phạm Công Đoàn</strong> – <a href="mailto:phamcongdoan1702@gmail.com">phamcongdoan1702@gmail.com</a></p>
    <p>• <strong>Võ Duy Nhật</strong> – <a href="mailto:duynhathm530@gmail.com">duynhathm530@gmail.com</a></p>
    <hr>
    <p><em>Made with ❤️ using <strong>Streamlit</strong> & <strong>Machine Learning</strong></em></p>
</div>
""", unsafe_allow_html=True)
