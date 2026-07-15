import streamlit as st
import plotly.graph_objects as go

from project.utils.theme import ColorPalette, ChartTheme


def _hex_to_rgba(hex_color: str, alpha: float = 0.13) -> str:
    """Chuyển '#RRGGBB' -> 'rgba(r,g,b,a)'. Bản cũ nối chuỗi kiểu '#EF444420'
    (hex 8 ký tự) mà Plotly bản mới không chấp nhận -> crash toàn bộ gauge."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def render_gauge_chart(prob: float) -> None:
    """Vẽ Gauge Confidence chart (đồng bộ Design System)."""
    palette = ColorPalette()

    prob = max(0.0, min(100.0, float(prob)))

    if prob <= 33:
        bar_color = palette.ERROR
    elif prob <= 66:
        bar_color = palette.WARNING
    else:
        bar_color = palette.SUCCESS

    step1 = _hex_to_rgba(palette.ERROR)
    step2 = _hex_to_rgba(palette.WARNING)
    step3 = _hex_to_rgba(palette.SUCCESS)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "AI Confidence (%)"},
            gauge={
                "axis": {"range": [None, 100], "tickcolor": palette.NEUTRAL_DARK, "tickwidth": 1},
                "bar": {"color": bar_color},
                "steps": [
                    {"range": [0, 40], "color": step1},
                    {"range": [40, 60], "color": step2},
                    {"range": [60, 100], "color": step3},
                ],
            },
        )
    )

    fig.update_layout(height=250, **ChartTheme.layout_defaults(palette))

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})


