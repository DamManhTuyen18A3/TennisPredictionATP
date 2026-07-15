"""
Player Profile Page (TÍNH NĂNG MỚI)
=====================================

Trang này KHÔNG tồn tại ở bản trước. Đây là bổ sung tăng tính ứng dụng thực
tế: một HLV/nhà phân tích/khán giả có thể tra cứu diễn biến phong độ (Elo),
tỷ lệ thắng theo mặt sân, và lịch sử thi đấu gần đây của BẤT KỲ tay vợt nào
trong 6.187 tay vợt thật của dữ liệu — không chỉ dùng để dự đoán 1 trận.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from project.services.data_fetcher import load_datasets, load_player_profiles, get_unique_players
from project.services.player_profiles import (
    build_elo_match_log, get_player_timeline, get_surface_breakdown, get_player_row,
)
from project.utils.theme import ColorPalette, ChartTheme


def show():
    st.title("🔎 Player Profile Explorer")
    st.markdown("Tra cứu diễn biến phong độ THẬT (Elo, thứ hạng, tỷ lệ thắng theo mặt sân) của bất kỳ tay vợt nào.")

    df_clean, _ = load_datasets()
    if df_clean.empty:
        st.error("Không tải được dữ liệu.")
        return

    profiles = load_player_profiles(df_clean)
    player_names = get_unique_players(df_clean)

    player_name = st.selectbox(f"Chọn tay vợt ({len(player_names)} người)", player_names)

    row = get_player_row(profiles, player_name)
    if row is None:
        st.warning("Không có hồ sơ cho tay vợt này.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Elo hiện tại", f"{row['elo']:.0f}")
    with col2:
        rank_val = row.get("rank")
        st.metric("Thứ hạng gần nhất", f"#{int(rank_val)}" if pd.notna(rank_val) else "—")
    with col3:
        st.metric("Số trận trong dữ liệu", f"{int(row['matches_played']):,}")
    with col4:
        st.metric("Trận gần nhất", str(pd.Timestamp(row["last_match_date"]).date()))

    days_since = (pd.Timestamp.today() - pd.Timestamp(row["last_match_date"])).days
    if days_since > 180:
        st.warning(
            f"⚠️ Tay vợt này không có trận nào trong dữ liệu suốt {days_since} ngày gần đây — "
            "có thể đã giải nghệ / chấn thương / chuyển hạng đấu khác. Hồ sơ (tuổi, Elo, rank) "
            "dùng để dự đoán có thể không còn phản ánh đúng phong độ hiện tại."
        )

    elo_log = build_elo_match_log(df_clean)
    timeline = get_player_timeline(elo_log, row["player_id"])

    st.markdown("### 📈 Diễn biến Elo Rating theo thời gian (thật, tính tuần tự từ lịch sử)")
    if not timeline.empty:
        palette = ColorPalette()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timeline["tourney_date"], y=timeline["elo_pre"],
            mode="lines", line=dict(color=palette.SECONDARY, width=2),
            name="Elo trước mỗi trận",
        ))
        fig.update_layout(
            **ChartTheme.layout_defaults(palette),
            xaxis=ChartTheme.axis_defaults(palette),
            yaxis=dict(title="Elo", **ChartTheme.axis_defaults(palette)),
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Không có dữ liệu lịch sử để vẽ.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 🎾 Tỷ lệ thắng theo mặt sân")
        surface_df = get_surface_breakdown(df_clean, row["player_id"])
        if not surface_df.empty:
            surface_df_display = surface_df.copy()
            surface_df_display["win_rate"] = (surface_df_display["win_rate"] * 100).round(1).astype(str) + "%"
            st.dataframe(surface_df_display, use_container_width=True, hide_index=True)
        else:
            st.info("Không có dữ liệu.")

    with col_b:
        st.markdown("### 📋 10 trận gần nhất")
        if not timeline.empty:
            recent = timeline.sort_values("tourney_date", ascending=False).head(10)
            st.dataframe(
                recent[["tourney_date", "tourney_name", "surface", "opponent", "result"]],
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("Không có dữ liệu.")


if __name__ == "__main__":
    show()
