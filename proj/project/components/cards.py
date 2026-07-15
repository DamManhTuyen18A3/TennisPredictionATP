"""
Card Components

Reusable card components for displaying metrics, statistics, and other data.
These components follow the professional design system and provide consistent styling.
"""

from typing import Optional, Callable
import textwrap
import streamlit as st
from project.utils.theme import ColorPalette, Spacing, BorderRadius, Shadow, GlassmorphismStyle, Typography
from project.utils.formatters import NumberFormatter, PercentageBar


def _render_html(html: str) -> None:
    """Render 1 khối HTML an toàn qua st.markdown.

    LỖI THẬT ĐÃ TÌM RA (sâu hơn lỗi thụt lề đã sửa trước đó): khi 1 phần nội
    dung f-string là điều kiện (vd. `{'...' if icon else ''}`) trả về chuỗi
    rỗng, dòng đó trở thành DÒNG TRỐNG nằm giữa khối `<div>...</div>`. Theo
    quy tắc CommonMark, một dòng trống sẽ CẮT NGANG khối HTML đang mở giữa
    chừng — phần `</div></div>` còn lại sau dòng trống bị xử lý như một khối
    Markdown MỚI và hiển thị NGUYÊN VĂN dạng chữ trên giao diện (đúng như lỗi
    "hộp </div></div>" người dùng thấy). `textwrap.dedent()` một mình không
    đủ để sửa lỗi này. Cách sửa triệt để: xoá hết các dòng trống bên trong
    khối HTML trước khi render.
    """
    dedented = textwrap.dedent(html)
    cleaned = "\n".join(line for line in dedented.split("\n") if line.strip() != "")
    st.markdown(cleaned, unsafe_allow_html=True)


def metric_card(label: str, value: str, 
               change: Optional[str] = None,
               color: str = None,
               icon: str = None,
               help_text: Optional[str] = None) -> None:
    """
    Display a metric card — thiết kế "bảng điểm sân vận động" (scoreboard):
    số liệu lớn dùng font Bebas Neue (condensed, đậm) thay vì Inter thông
    thường, kèm 1 vạch màu mỏng ở mép trên card làm điểm nhấn (thay vì tô
    màu toàn bộ chữ số như bản cũ — giúp nhất quán hơn khi nhiều card đặt
    cạnh nhau, mắt tập trung vào bản thân con số thay vì bị phân tán bởi
    nhiều màu chữ khác nhau).

    Args:
        label: Card label/title
        value: Main metric value
        change: Optional change indicator (e.g., "+0.8%")
        color: Màu vạch nhấn trên đầu card (hex code)
        icon: Emoji icon to display
        help_text: Hover help text

    Example:
        metric_card("Accuracy", f"{real_accuracy:.1%}", "+0.8%", color="#22C55E", icon="✅")  # ví dụ: dùng số liệu thật, không viết tay
    """
    colors = ColorPalette()

    if color is None:
        color = colors.ACCENT_SPORT

    change_html = (
        f'<span style="font-family:{Typography.FONT_MONO}; font-size:12px; font-weight:600; '
        f'color:{color}; background:{color}1A; padding:2px 8px; border-radius:999px;">{change}</span>'
        if change else ''
    )
    icon_html = f'<span style="font-size:18px; line-height:1;">{icon}</span>' if icon else ''

    card_html = f"""
    <div style="
        background: {colors.SURFACE};
        border: 1px solid {colors.NEUTRAL_DARK};
        border-radius: {BorderRadius.LG}px;
        padding: {Spacing.LG}px;
        box-shadow: {Shadow.GLASS};
        transition: all 200ms ease-in-out;
        position: relative;
        overflow: hidden;
        cursor: {'pointer' if help_text else 'default'};
    " class="hover-lift">
        <div style="position:absolute; top:0; left:0; width:100%; height:3px; background:{color};"></div>
        <div style="display:flex; align-items:center; gap:6px; margin-bottom:{Spacing.SM}px;">
            {icon_html}
            <span style="
                font-size: 11px;
                color: {colors.TEXT_TERTIARY};
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 1px;
            ">{label}</span>
        </div>
        <div style="
            font-family: {Typography.FONT_DISPLAY};
            font-size: 40px;
            line-height: 1;
            color: {colors.TEXT_PRIMARY};
            letter-spacing: 0.5px;
            margin-bottom: {Spacing.XS}px;
        ">{value}</div>
        {change_html}
    </div>
    """
    
    if help_text:
        _render_html(card_html)  # loai dong trong + dedent, tranh Markdown hieu nham thanh code block
        st.caption(help_text)
    else:
        _render_html(card_html)  # loai dong trong + dedent, tranh Markdown hieu nham thanh code block


def stat_card(title: str, value: str, subtitle: Optional[str] = None,
             metric: Optional[dict] = None) -> None:
    """
    Display a statistics card with title, value, and optional additional metric.
    
    Args:
        title: Card title
        value: Main statistic value
        subtitle: Optional subtitle text
        metric: Optional metric dict with 'label' and 'value' keys
    
    Example:
        stat_card("Training Matches", "42,350", "as of 2025-06-15")
    """
    colors = ColorPalette()

    card_html = f"""
    <div style="
        background: {colors.SURFACE};
        border: 1px solid {colors.NEUTRAL_DARK};
        border-radius: {BorderRadius.LG}px;
        padding: {Spacing.LG}px;
        box-shadow: {Shadow.GLASS};
    " class="hover-lift">
        <div style="
            color: {colors.TEXT_TERTIARY};
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: {Spacing.SM}px;
        ">{title}</div>
        <div style="
            font-family: {Typography.FONT_DISPLAY};
            color: {colors.TEXT_PRIMARY};
            font-size: 32px;
            line-height: 1;
            letter-spacing: 0.5px;
            margin-bottom: {Spacing.SM}px;
        ">{value}</div>
        {f'<div style=\"color: {colors.TEXT_TERTIARY}; font-size: 12px;\">{subtitle}</div>' if subtitle else ''}
        {f'''<div style="
            font-family: {Typography.FONT_MONO};
            color: {colors.ACCENT_SPORT};
            font-size: 12px;
            margin-top: {Spacing.MD}px;
            padding-top: {Spacing.SM}px;
            border-top: 1px solid {colors.NEUTRAL_DARK};
        ">{metric['label']}: {metric['value']}</div>''' if metric else ''}
    </div>
    """
    
    _render_html(card_html)  # loai dong trong + dedent, tranh Markdown hieu nham thanh code block


def kpi_grid(metrics: list, columns: int = 4) -> None:
    """
    Display multiple KPI cards in a grid layout.
    
    Args:
        metrics: List of dicts with keys: 'label', 'value', 'change' (optional), 'color' (optional), 'icon' (optional)
        columns: Number of columns in grid
    
    Example:
        kpi_grid([
            {'label': 'Accuracy', 'value': f'{real_accuracy:.1%}', 'change': '+0.8%', 'color': '#22C55E'},  # số liệu thật
            {'label': 'ROC AUC', 'value': f'{real_auc:.3f}', 'color': '#2563EB'},
        ], columns=2)
    """
    cols = st.columns(columns)
    
    for idx, metric in enumerate(metrics):
        with cols[idx % columns]:
            metric_card(
                label=metric['label'],
                value=metric['value'],
                change=metric.get('change'),
                color=metric.get('color'),
                icon=metric.get('icon'),
                help_text=metric.get('help_text'),
            )


def info_card(title: str, content: str, icon: str = "ℹ️",
             style: str = "info") -> None:
    """
    Display an information card.
    
    Args:
        title: Card title
        content: Card content (markdown supported)
        icon: Emoji icon
        style: Style type - 'info', 'warning', 'success', 'error'
    """
    colors = ColorPalette()
    
    color_map = {
        "info": colors.SECONDARY,
        "warning": colors.WARNING,
        "success": colors.SUCCESS,
        "error": colors.ERROR,
    }
    
    bg_color = color_map.get(style, colors.SECONDARY)
    
    card_html = f"""
    <div style="
        background: rgba(37, 99, 235, 0.15);
        border: 1px solid {bg_color}40;
        border-radius: {BorderRadius.MD}px;
        padding: {Spacing.LG}px;
        border-left: 4px solid {bg_color};
    ">
        <div style="font-weight: 600; margin-bottom: {Spacing.SM}px; font-size: 14px;">
            {icon} {title}
        </div>
        <div style="font-size: 13px; line-height: 1.6; color: {colors.TEXT_SECONDARY};">
            {content}
        </div>
    </div>
    """
    
    _render_html(card_html)  # loai dong trong + dedent, tranh Markdown hieu nham thanh code block


def progress_card(title: str, value: float, max_value: float = 1.0,
                 show_percentage: bool = True, color: str = None) -> None:
    """
    Display a progress card with percentage bar.
    
    Args:
        title: Card title
        value: Current value
        max_value: Maximum value
        show_percentage: Show percentage text
        color: Progress bar color
    """
    if color is None:
        colors = ColorPalette()
        color = colors.SUCCESS
    
    percentage = min(100, (value / max_value * 100)) if max_value > 0 else 0
    
    card_html = f"""
    <div style="
        background: rgba(31, 41, 55, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: {BorderRadius.MD}px;
        padding: {Spacing.LG}px;
        box-shadow: {Shadow.GLASS};
    ">
        <div style="
            display: flex;
            justify-content: space-between;
            margin-bottom: {Spacing.SM}px;
        ">
            <span style="font-weight: 600; color: {ColorPalette().TEXT_SECONDARY};">{title}</span>
            {f'<span style="color: {color}; font-weight: 700;">{percentage:.1f}%</span>' if show_percentage else ''}
        </div>
        <div style="
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            height: 8px;
            overflow: hidden;
        ">
            <div style="
                background: {color};
                height: 100%;
                width: {percentage}%;
                transition: width 500ms ease-out;
                box-shadow: 0 0 10px {color}40;
            "></div>
        </div>
    </div>
    """
    
    _render_html(card_html)  # loai dong trong + dedent, tranh Markdown hieu nham thanh code block


def comparison_card(label_a: str, value_a: str, 
                   label_b: str, value_b: str,
                   winner: Optional[str] = None) -> None:
    """
    Display a side-by-side comparison card.
    
    Args:
        label_a: Left label
        value_a: Left value
        label_b: Right label
        value_b: Right value
        winner: 'A', 'B', or None for draw
    """
    colors = ColorPalette()
    
    winner_color_a = colors.SUCCESS if winner == "A" else colors.NEUTRAL_LIGHT
    winner_color_b = colors.SUCCESS if winner == "B" else colors.NEUTRAL_LIGHT
    
    card_html = f"""
    <div style="
        background: rgba(31, 41, 55, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: {BorderRadius.MD}px;
        padding: {Spacing.LG}px;
        box-shadow: {Shadow.GLASS};
    ">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: {Spacing.LG}px;">
            <div style="text-align: center; padding-right: {Spacing.LG}px; border-right: 1px solid rgba(255, 255, 255, 0.1);">
                <div style="color: {colors.TEXT_SECONDARY}; font-size: 12px; margin-bottom: {Spacing.SM}px;">
                    {label_a}
                </div>
                <div style="
                    color: {winner_color_a};
                    font-size: 24px;
                    font-weight: 700;
                    {'color: ' + colors.SUCCESS + '; text-shadow: 0 0 10px ' + colors.SUCCESS + '40;' if winner == 'A' else ''}
                ">
                    {value_a}
                </div>
            </div>
            <div style="text-align: center; padding-left: {Spacing.LG}px;">
                <div style="color: {colors.TEXT_SECONDARY}; font-size: 12px; margin-bottom: {Spacing.SM}px;">
                    {label_b}
                </div>
                <div style="
                    color: {winner_color_b};
                    font-size: 24px;
                    font-weight: 700;
                    {'color: ' + colors.SUCCESS + '; text-shadow: 0 0 10px ' + colors.SUCCESS + '40;' if winner == 'B' else ''}
                ">
                    {value_b}
                </div>
            </div>
        </div>
    </div>
    """
    
    _render_html(card_html)  # loai dong trong + dedent, tranh Markdown hieu nham thanh code block


__all__ = [
    "metric_card",
    "stat_card",
    "kpi_grid",
    "info_card",
    "progress_card",
    "comparison_card",
]
