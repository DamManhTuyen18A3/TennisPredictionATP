"""
Design System & Theme Configuration
Provides centralized styling, colors, and visual utilities for the application.

This module implements the professional design language:
- Dark theme with glassmorphism
- Apple/TradingView inspired aesthetics
- Consistent spacing and typography
"""

from typing import Literal
from dataclasses import dataclass
import textwrap


@dataclass
class ColorPalette:
    """Bảng màu 'Night Court' — sân đấu về đêm, thiết kế riêng cho ứng dụng
    quần vợt thay vì bảng màu SaaS chung chung (slate-gray + màu random cho
    từng số liệu). Nguyên tắc: nền đen ánh xanh sân cỏ, 1 accent tín hiệu duy
    nhất (vàng bóng tennis) dùng cho điểm nhấn số liệu quan trọng nhất, và 1
    accent phụ (đất nện/clay) cho các điểm nhấn thứ cấp — thay vì rải ngẫu
    nhiên xanh/lục/cam/đỏ như trước."""

    # Nền — đen ánh xanh sân cỏ về đêm, không phải charcoal chung chung
    PRIMARY: str = "#0B120F"        # Nền chính — "sân đấu về đêm"
    SURFACE: str = "#141F1B"        # Nền card/bề mặt nổi lên trên PRIMARY

    # Accent tín hiệu — DUY NHẤT 1 màu vàng bóng tennis cho điểm nhấn quan
    # trọng nhất mỗi màn hình (không rải nhiều màu ngẫu nhiên cho mọi số liệu)
    SECONDARY: str = "#2563EB"      # Xanh dữ liệu — chỉ dùng cho link/info, KHÔNG dùng cho KPI chính
    ACCENT_CLAY: str = "#C4623F"    # Đất nện (clay court) — accent phụ, dùng cho điểm nhấn thứ cấp

    SUCCESS: str = "#34D399"        # Xanh cỏ (grass court) — tích cực/thắng
    WARNING: str = "#F5A623"        # Vàng cam — cảnh báo
    ERROR: str = "#F0553F"          # Đỏ vạch kẻ sân — lỗi/thua

    # Neutral — ánh xanh nhẹ thay vì xám thuần, đồng bộ tông "sân đấu"
    NEUTRAL_LIGHT: str = "#3A4A42"  # Viền/đường kẻ sáng
    NEUTRAL_DARK: str = "#1E2A24"   # Viền/nền card tối
    NEUTRAL_DARKER: str = "#0B120F" # = PRIMARY

    # Text — trắng ánh xanh nhẹ thay vì trắng xám thuần
    TEXT_PRIMARY: str = "#EDF2EF"   # Chữ chính
    TEXT_SECONDARY: str = "#A8B8AE" # Chữ phụ
    TEXT_TERTIARY: str = "#6B7C72"  # Chữ mờ nhất

    # Overlay colors (for glassmorphism)
    OVERLAY_LIGHT: str = "rgba(237, 242, 239, 0.08)"
    OVERLAY_BORDER: str = "rgba(237, 242, 239, 0.14)"

    # Sport accent colors (tong the thao - san/bong tennis)
    ACCENT_SPORT: str = "#D4FF3F"        # Vang bong tennis - accent tin hieu DUY NHAT, dung cho diem nhan quan trong nhat
    COURT_GREEN_DARK: str = "#081210"    # Xanh san dam hon PRIMARY - gradient sidebar (toi hon nen chinh 1 bac)
    
    def to_css_variable(self, name: str) -> str:
        """Get CSS variable reference."""
        return f"var(--{name.lower()})"


class Spacing:
    """Spacing system (base 4px)."""
    XS: int = 4
    SM: int = 8
    MD: int = 12
    LG: int = 16
    XL: int = 24
    XXL: int = 32
    XXXL: int = 48


class BorderRadius:
    """Border radius system."""
    SM: int = 6
    MD: int = 12
    LG: int = 16
    FULL: int = 999


class Shadow:
    """Shadow definitions."""
    SMALL: str = "0 2px 4px rgba(0, 0, 0, 0.1)"
    MEDIUM: str = "0 4px 12px rgba(0, 0, 0, 0.15)"
    LARGE: str = "0 8px 24px rgba(0, 0, 0, 0.2)"
    
    # For cards with glassmorphism
    GLASS: str = "0 8px 32px rgba(0, 0, 0, 0.3)"


class Typography:
    """Typography specifications.

    3 vai trò font riêng biệt (thay vì chỉ dùng Inter cho mọi thứ — nhìn
    chung chung như mọi SaaS dashboard khác):
    - FONT_FAMILY (Inter): chữ nội dung, đọc rõ, trung tính.
    - FONT_DISPLAY (Bebas Neue): chữ số lớn kiểu "bảng điểm sân vận động"
      (scoreboard) — dùng cho số KPI chính và tiêu đề hero, tạo bản sắc thị
      giác riêng cho 1 app THỂ THAO thay vì chữ số Inter thông thường.
    - FONT_MONO (JetBrains Mono): số liệu dạng bảng/thống kê (Elo, tỷ lệ
      cược, rank) — cảm giác "terminal dữ liệu thể thao" chính xác, rõ ràng.
    """
    FONT_FAMILY: str = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    FONT_DISPLAY: str = "'Bebas Neue', 'Inter', sans-serif"
    FONT_MONO: str = "'JetBrains Mono', 'SF Mono', Consolas, monospace"
    GOOGLE_FONTS_IMPORT: str = (
        "@import url('https://fonts.googleapis.com/css2?"
        "family=Bebas+Neue&family=Inter:wght@400;500;600;700;800&"
        "family=JetBrains+Mono:wght@400;500;700&display=swap');"
    )

    # Heading sizes
    H1_SIZE: int = 32
    H1_WEIGHT: int = 700
    
    H2_SIZE: int = 24
    H2_WEIGHT: int = 700
    
    H3_SIZE: int = 20
    H3_WEIGHT: int = 700
    
    # Body text
    BODY_SIZE: int = 16
    BODY_WEIGHT: int = 400
    
    # Small text
    SMALL_SIZE: int = 12
    SMALL_WEIGHT: int = 400
    
    # Monospace (for stats)
    MONOSPACE: str = "'Monaco', 'Courier New', monospace"


class IconSize:
    """Kich thuoc icon chuan (px) - Design Token con thieu, da bo sung theo TODO.md."""
    SM: int = 16
    MD: int = 20
    LG: int = 24
    XL: int = 32


class ButtonStyle:
    """Chuan hoa style button (bo/border-radius/font-weight) - dung chung
    cho moi noi thay vi moi cho tu viet 1 kieu CSS button rieng."""
    BORDER_RADIUS: int = 999   # bo tron kieu "pill" - dong bo voi .stButton trong main.py
    FONT_WEIGHT: int = 600
    PADDING_Y: int = 8
    PADDING_X: int = 20


class TableStyle:
    """Chuan hoa style bang (st.dataframe/st.table)."""
    HEADER_BG: str = "#1F2937"       # = ColorPalette.NEUTRAL_DARK
    ROW_HOVER_BG: str = "rgba(37, 99, 235, 0.08)"   # tint nhe cua SECONDARY
    BORDER_COLOR: str = "rgba(255, 255, 255, 0.1)"
    ROW_HEIGHT: int = 36


class ChartTheme:
    """Theme dung chung cho moi bieu do Plotly trong app - dam bao Analytics,
    Prediction, Bracket, PlayerProfile deu co giao dien bieu do dong nhat."""
    @staticmethod
    def layout_defaults(colors: "ColorPalette" = None) -> dict:
        colors = colors or ColorPalette()
        return dict(
            paper_bgcolor=colors.PRIMARY,
            plot_bgcolor=colors.PRIMARY,
            font=dict(color=colors.TEXT_PRIMARY, family=Typography.FONT_FAMILY),
            margin=dict(l=10, r=10, t=30, b=10),
        )

    @staticmethod
    def axis_defaults(colors: "ColorPalette" = None) -> dict:
        colors = colors or ColorPalette()
        return dict(gridcolor=colors.NEUTRAL_DARK, zerolinecolor=colors.NEUTRAL_LIGHT)


class AlertStyle:
    """Mau nen/vien chuan cho st.info/st.warning/st.success/st.error, dung khi
    can lam alert box HTML tuy chinh thay vi component mac dinh cua Streamlit."""
    INFO_BG: str = "rgba(37, 99, 235, 0.1)"
    INFO_BORDER: str = "#2563EB"      # = ColorPalette.SECONDARY
    SUCCESS_BG: str = "rgba(34, 197, 94, 0.1)"
    SUCCESS_BORDER: str = "#22C55E"   # = ColorPalette.SUCCESS
    WARNING_BG: str = "rgba(245, 158, 11, 0.1)"
    WARNING_BORDER: str = "#F59E0B"   # = ColorPalette.WARNING
    ERROR_BG: str = "rgba(239, 68, 68, 0.1)"
    ERROR_BORDER: str = "#EF4444"     # = ColorPalette.ERROR


class Animation:
    """Animation specifications (Streamlit custom CSS)."""
    DURATION_FAST: str = "150ms"
    DURATION_NORMAL: str = "200ms"
    DURATION_SLOW: str = "300ms"
    
    EASE_IN_OUT: str = "ease-in-out"
    EASE_OUT: str = "ease-out"
    EASE_CUBIC: str = "cubic-bezier(0.4, 0, 0.2, 1)"
    
    # Preset transitions
    FADE: str = f"opacity {DURATION_NORMAL} {EASE_IN_OUT}"
    SLIDE: str = f"transform {DURATION_SLOW} {EASE_CUBIC}"
    SCALE: str = f"transform {DURATION_FAST} {EASE_OUT}"
    
    # Animation keyframes (for use in custom CSS)
    @staticmethod
    def pulse_keyframe() -> str:
        return """
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        """
    
    @staticmethod
    def slide_in_keyframe() -> str:
        return """
        @keyframes slideIn {
            from { transform: translateY(10px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        """
    
    @staticmethod
    def fade_in_keyframe() -> str:
        return """
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        """


class GlassmorphismStyle:
    """Glassmorphism effect specifications."""
    
    @staticmethod
    def card_style(color_bg: str = "rgba(31, 41, 55, 0.8)") -> dict:
        """Get glassmorphism card style."""
        return {
            "background": color_bg,
            "backdrop_filter": "blur(10px)",
            "border": f"1px solid {ColorPalette.OVERLAY_BORDER}",
            "box_shadow": Shadow.GLASS,
        }
    
    @staticmethod
    def get_css() -> str:
        """Get complete glassmorphism CSS."""
        colors = ColorPalette()
        return f"""
        <style>
        {Typography.GOOGLE_FONTS_IMPORT}

        :root {{
            --primary: {colors.PRIMARY};
            --secondary: {colors.SECONDARY};
            --success: {colors.SUCCESS};
            --warning: {colors.WARNING};
            --error: {colors.ERROR};
            --neutral-light: {colors.NEUTRAL_LIGHT};
            --text-primary: {colors.TEXT_PRIMARY};
            --accent-sport: {colors.ACCENT_SPORT};
        }}
        
        * {{
            font-family: {Typography.FONT_FAMILY};
        }}

        /* Tiêu đề dùng font Bebas Neue - bản sắc "bảng điểm sân vận động",
           tách biệt rõ với chữ nội dung Inter thay vì dùng chung 1 font */
        h1, h2, h3 {{
            font-family: {Typography.FONT_DISPLAY};
            letter-spacing: 0.5px;
        }}

        /* Số liệu bảng/thống kê dùng font mono - cảm giác terminal dữ liệu */
        [data-testid="stMetricValue"], code, .stDataFrame {{
            font-family: {Typography.FONT_MONO};
        }}
        
        /* Glassmorphic cards */
        .glass-card {{
            background: rgba(31, 41, 55, 0.8);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: {BorderRadius.MD}px;
            box-shadow: {Shadow.GLASS};
            padding: {Spacing.LG}px;
        }}
        
        /* Smooth transitions */
        .smooth-transition {{
            transition: all 200ms ease-in-out;
        }}
        
        /* Animations */
        {Animation.pulse_keyframe()}
        {Animation.slide_in_keyframe()}
        {Animation.fade_in_keyframe()}
        
        .animate-pulse {{
            animation: pulse 2s infinite;
        }}
        
        .animate-slide-in {{
            animation: slideIn 300ms cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        .animate-fade-in {{
            animation: fadeIn 200ms ease-in-out;
        }}
        
        /* Hover effects */
        .hover-lift {{
            transition: transform 150ms ease-out, box-shadow 150ms ease-out;
        }}
        
        .hover-lift:hover {{
            transform: translateY(-4px);
            box-shadow: {Shadow.LARGE};
        }}
        
        /* Text gradients */
        .gradient-text {{
            background: linear-gradient(135deg, {colors.SECONDARY} 0%, {colors.SUCCESS} 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        /* Utility classes */
        .text-primary {{ color: {colors.TEXT_PRIMARY}; }}
        .text-secondary {{ color: {colors.TEXT_SECONDARY}; }}
        .text-success {{ color: {colors.SUCCESS}; }}
        .text-warning {{ color: {colors.WARNING}; }}
        .text-error {{ color: {colors.ERROR}; }}
        
        .bg-primary {{ background-color: {colors.PRIMARY}; }}
        .bg-secondary {{ background-color: {colors.SECONDARY}; }}
        .bg-neutral-dark {{ background-color: {colors.NEUTRAL_DARK}; }}
        
        /* Button/Alert/Tabs chuan hoa - dung ButtonStyle/AlertStyle tokens
           thay vi moi noi tu viet 1 kieu rieng (theo TODO.md buoc 3) */
        .stButton > button {{
            border-radius: 999px !important;
            font-weight: 600 !important;
            padding: 8px 20px !important;
            transition: all 200ms ease-in-out !important;
        }}
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }}

        div[data-testid="stTabs"] button[role="tab"] {{
            font-weight: 600;
            border-radius: 6px 6px 0 0;
        }}

        div[data-testid="stDataFrame"] {{
            border-radius: 12px;
            overflow: hidden;
        }}

        /* Scrollbar styling */
        ::-webkit-scrollbar {{
            width: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: {colors.PRIMARY};
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: {colors.NEUTRAL_DARK};
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: {colors.SECONDARY};
        }}
        </style>
        """


def inject_theme_css() -> str:
    """Get complete CSS for theme injection into Streamlit."""
    import streamlit as st
    # LỖI ĐÃ SỬA: get_css() trả về CSS với các dòng thụt lề 8 dấu cách (do format
    # tự nhiên của code Python). Markdown coi đoạn text thụt lề ≥4 dấu cách là
    # "code block" và hiển thị NGUYÊN VĂN ra màn hình thay vì áp dụng CSS — đây
    # chính là nguyên nhân xuất hiện các "hộp" chứa văn bản HTML/CSS thô trên
    # giao diện. `textwrap.dedent()` loại bỏ phần thụt lề chung, đưa <style> về
    # cột 0 để Markdown nhận đúng là khối HTML/CSS cần render, không phải code.
    st.markdown(textwrap.dedent(GlassmorphismStyle.get_css()), unsafe_allow_html=True)


# Preset color schemes for different contexts
COLOR_SCHEMES = {
    "win": ColorPalette.SUCCESS,
    "loss": ColorPalette.ERROR,
    "high_confidence": ColorPalette.SUCCESS,
    "medium_confidence": ColorPalette.WARNING,
    "low_confidence": ColorPalette.ERROR,
    "neutral": ColorPalette.NEUTRAL_LIGHT,
}


def get_color_by_value(value: float, min_val: float = 0, max_val: float = 1) -> str:
    """
    Get color based on value between min and max.
    
    Args:
        value: The value to evaluate (0-1 typically)
        min_val: Minimum value (represents error color)
        max_val: Maximum value (represents success color)
    
    Returns:
        Hex color code
    """
    colors = ColorPalette()
    normalized = (value - min_val) / (max_val - min_val)
    
    if normalized < 0.33:
        return colors.ERROR
    elif normalized < 0.66:
        return colors.WARNING
    else:
        return colors.SUCCESS


# Export commonly used objects
__all__ = [
    "ColorPalette",
    "Spacing",
    "BorderRadius",
    "Shadow",
    "Typography",
    "IconSize",
    "ButtonStyle",
    "TableStyle",
    "ChartTheme",
    "AlertStyle",
    "Animation",
    "GlassmorphismStyle",
    "inject_theme_css",
    "COLOR_SCHEMES",
    "get_color_by_value",
]
