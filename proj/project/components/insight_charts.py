"""
Insight Charts Component
========================

Thay thế 2 file cũ:
- radar_chart.py: vẽ radar "Serve/Return/Forehand/Backhand/Stamina/Mental" với
  số liệu HOÀN TOÀN bịa (85,70,90,80,85,95 / 95,80,85,85,75,80) — dữ liệu
  không hề tồn tại trong bộ ATP/WTA (không có chỉ số serve/backhand per-match).
- shap_plot.py: vẽ bar chart SHAP với 6 giá trị cứng, giống nhau cho mọi trận.

Ở đây thay bằng 2 biểu đồ dùng đúng số liệu thật đã tính cho cặp đấu đang xét.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from project.utils.theme import ColorPalette, ChartTheme


def render_comparison_radar(player_a: str, player_b: str, p1: dict, p2: dict):
    """Radar so sánh các chỉ số THẬT đã biết của 2 tay vợt (Elo, điểm ATP, chiều
    cao...), chuẩn hoá về thang 0-100 để so sánh trực quan. Không có chỉ số nào
    bị bịa thêm."""
    palette = ColorPalette()

    def norm(value, lo, hi):
        if value is None or pd.isna(value):
            return 0
        return max(0, min(100, (value - lo) / (hi - lo) * 100))

    categories = ["Elo Rating", "Điểm ATP", "Thứ hạng (đảo ngược)", "Chiều cao", "Tuổi (trẻ hơn tốt hơn)"]

    r1 = [
        norm(p1.get("elo"), 1200, 2400),
        norm(p1.get("rank_points"), 0, 12000),
        norm(-(p1.get("rank") or 2000), -2000, 0),
        norm(p1.get("ht"), 165, 210),
        norm(-(p1.get("age_at_last_match") or 40), -40, -16),
    ]
    r2 = [
        norm(p2.get("elo"), 1200, 2400),
        norm(p2.get("rank_points"), 0, 12000),
        norm(-(p2.get("rank") or 2000), -2000, 0),
        norm(p2.get("ht"), 165, 210),
        norm(-(p2.get("age_at_last_match") or 40), -40, -16),
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=r1, theta=categories, fill="toself", name=player_a,
                                   line_color=palette.SECONDARY))
    fig.add_trace(go.Scatterpolar(r=r2, theta=categories, fill="toself", name=player_b,
                                   line_color=palette.ERROR))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor=palette.NEUTRAL_DARK),
                   bgcolor=palette.PRIMARY),
        showlegend=True,
        legend=dict(font=dict(color=palette.TEXT_PRIMARY)),
        **ChartTheme.layout_defaults(palette),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("Các trục là số liệu hồ sơ THẬT (Elo, điểm ATP, hạng, chiều cao, tuổi) đã "
               "chuẩn hoá 0-100, không phải chỉ số kỹ năng bịa.")


def render_shap_bar(shap_top: dict):
    """Bar chart SHAP THẬT cho đúng dự đoán này (nhận dict từ inference.get_shap_explanation)."""
    palette = ColorPalette()

    if not shap_top:
        st.warning("Chưa có SHAP values để hiển thị.")
        return

    df_shap = pd.DataFrame(
        {"Feature": list(shap_top.keys()), "SHAP value": list(shap_top.values())}
    ).sort_values("SHAP value")

    colors = [palette.SUCCESS if v > 0 else palette.ERROR for v in df_shap["SHAP value"]]

    fig = px.bar(df_shap, x="SHAP value", y="Feature", orientation="h")
    fig.update_traces(marker_color=colors)
    fig.update_layout(**ChartTheme.layout_defaults(palette), showlegend=False)
    x_axis = ChartTheme.axis_defaults(palette)
    x_axis["zerolinecolor"] = palette.NEUTRAL_LIGHT  # override: zeroline rõ hơn cho biểu đồ SHAP quanh mốc 0
    fig.update_xaxes(zeroline=True, **x_axis)
    fig.update_yaxes(**ChartTheme.axis_defaults(palette))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
