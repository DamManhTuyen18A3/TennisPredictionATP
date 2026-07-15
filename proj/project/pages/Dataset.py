"""
Dataset Explorer Page (MỚI)
============================

Trước đây trang này chỉ có dòng "🔍 Dataset explorer is being built...".
Bản mới cho phép lọc/xem trực tiếp dữ liệu thật đã qua xử lý (05_clean_data),
phù hợp để hội đồng chấm NCKH kiểm tra ngay tính minh bạch của dữ liệu.
"""

import pandas as pd
import streamlit as st

from project.services.data_fetcher import load_datasets


def show():
    st.title("📂 Dataset Explorer")
    st.markdown("Duyệt trực tiếp dữ liệu thật đã qua xử lý (`data/processed/05_clean_data.parquet`).")

    df_clean, df_features = load_datasets()
    if df_clean.empty:
        st.error("Không tải được dữ liệu.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        surfaces = ["Tất cả"] + sorted(df_clean["surface"].dropna().unique().tolist())
        surface_filter = st.selectbox("Mặt sân", surfaces)
    with col2:
        min_year, max_year = int(df_clean["year"].min()), int(df_clean["year"].max())
        year_range = st.slider("Khoảng năm", min_year, max_year, (min_year, max_year))
    with col3:
        search_name = st.text_input("Tìm theo tên tay vợt (winner hoặc loser)")

    filtered = df_clean.copy()
    if surface_filter != "Tất cả":
        filtered = filtered[filtered["surface"] == surface_filter]
    filtered = filtered[(filtered["year"] >= year_range[0]) & (filtered["year"] <= year_range[1])]
    if search_name:
        mask = (
            filtered["winner_name"].str.contains(search_name, case=False, na=False)
            | filtered["loser_name"].str.contains(search_name, case=False, na=False)
        )
        filtered = filtered[mask]

    st.markdown(f"**Số trận khớp bộ lọc: {len(filtered):,} / {len(df_clean):,} tổng số trận**")

    show_cols = [
        "tourney_date", "tourney_name", "surface", "tourney_level", "round",
        "winner_name", "winner_rank", "loser_name", "loser_rank", "best_of",
    ]
    show_cols = [c for c in show_cols if c in filtered.columns]
    st.dataframe(
        filtered[show_cols].sort_values("tourney_date", ascending=False).head(500),
        use_container_width=True, height=450,
    )
    st.caption("Hiển thị tối đa 500 dòng gần nhất khớp bộ lọc.")

    with st.expander("📦 Xem schema đầy đủ dữ liệu (05_clean_data & 08_selected_features)"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**05_clean_data.parquet** — {df_clean.shape[0]:,} dòng × {df_clean.shape[1]} cột")
            st.code("\n".join(df_clean.columns.tolist()), language="text")
        with c2:
            if not df_features.empty:
                st.markdown(f"**08_selected_features.parquet** — {df_features.shape[0]:,} dòng × {df_features.shape[1]} cột")
                st.code("\n".join(df_features.columns.tolist()), language="text")


if __name__ == "__main__":
    show()
