"""
Main Application Entry Point — PHIÊN BẢN SỬA LỖI
====================================================

BẢN CŨ: đây là file chạy thật (qua `runapp.py`), nhưng TỰ ĐỊNH NGHĨA các hàm
render_dashboard/render_prediction/... là các trang STUB rỗng, hoàn toàn tách
biệt khỏi `project/pages/Home.py` và `project/pages/Prediction.py` (là 2 file
có giao diện đầy đủ hơn nhưng lại KHÔNG được gọi ở đâu cả — dead code). Vì
vậy khi chạy app, người dùng luôn thấy "🚧 đang được xây dựng..." và số liệu
cứng (74.3%, 0.812, "42,350 trận", "2,847 tay vợt", "156 giải") không khớp
số liệu thật.

BẢN MỚI: router gọi đúng show() của từng trang trong project/pages/, đã được
viết lại để dùng dữ liệu + model thật. Sidebar cũng hiển thị số liệu thật
(đọc từ experiments/test_evaluation.json), không còn số viết tay.

LỖI ĐÃ SỬA: `ModuleNotFoundError: No module named 'project'`. Nguyên nhân:
khi chạy `streamlit run project/main.py` (hoặc mở trực tiếp file này bằng
`python`), Python/Streamlit chỉ thêm THƯ MỤC CHỨA FILE NÀY (`.../project/`)
vào sys.path, KHÔNG thêm thư mục gốc dự án (`.../` — thư mục cha của
`project/`). Vì các import trong toàn bộ code đều dùng dạng tuyệt đối
(`from project.utils... import ...`), Python cần thấy thư mục gốc dự án
trong sys.path để tìm ra package `project`. Đoạn bootstrap bên dưới tự thêm
thư mục gốc dự án vào sys.path TRƯỚC khi import bất cứ gì từ `project`, nên
file này chạy đúng bất kể được gọi từ đâu (double-click, `python main.py`,
`streamlit run project/main.py` từ thư mục khác, v.v.).
"""

import sys
from pathlib import Path
import textwrap

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from project.utils.theme import inject_theme_css, ColorPalette
from project.utils.logger import get_logger

logger = get_logger(__name__)


def configure_page():
    """Cấu hình trang + giao diện tông thể thao (xanh sân/ vàng bóng tennis)."""
    st.set_page_config(
        page_title="ATP Match Prediction System",
        page_icon="🎾",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get help": "https://github.com/yourusername/atp-predictor",
            "Report a bug": "https://github.com/yourusername/atp-predictor/issues",
            "About": "ATP Tennis Match Prediction - Nhóm 9, DHKL18A3HN - NCKH",
        },
    )

    inject_theme_css()  # CSS chuẩn (glassmorphism + button/tabs/table) — xem project/utils/theme.py

    colors = ColorPalette()

    # Chỉ còn phần CSS ĐẶC THÙ tông thể thao (sidebar/h1-h2) ở đây — phần
    # button/tabs/table dùng chung đã chuyển hẳn vào theme.py để tránh viết
    # trùng 2 nơi (đúng tinh thần "chuẩn hoá design system" trong TODO.md).
    st.markdown(
        textwrap.dedent(f"""
    <style>
    .main {{ padding: 0; }}
    .block-container {{ padding: 2rem 1rem; }}
    * {{ transition: all 200ms ease-in-out; }}
    footer {{ display: none; }}

    /* Sidebar tông xanh sân tennis — dùng ColorPalette.COURT_GREEN_DARK/ACCENT_SPORT */
    [data-testid="stSidebar"] {{
        background: linear-gradient(160deg, {colors.COURT_GREEN_DARK} 0%, {colors.PRIMARY} 55%, {colors.PRIMARY} 100%);
        border-right: 1px solid {colors.ACCENT_SPORT}26;
    }}

    /* Viền vàng bóng tennis nhấn nhẹ cho header */
    h1, h2 {{
        border-bottom: 2px solid {colors.ACCENT_SPORT}59;
        padding-bottom: 0.3rem;
    }}
    </style>
    """),
        unsafe_allow_html=True,
    )


def init_session_state():
    defaults = {
        "current_page": "🏠 Trang chủ",
        "theme_mode": "dark",
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def _load_sidebar_metrics():
    """Đọc số liệu THẬT cho sidebar (không còn 74.3% / 0.812 viết tay)."""
    import json
    from pathlib import Path

    eval_path = Path(__file__).resolve().parents[1] / "experiments" / "test_evaluation.json"
    if not eval_path.exists():
        return None
    data = json.loads(eval_path.read_text())
    return data.get("CatBoost_tuned") or data.get("CatBoost_baseline")


def render_sidebar():
    """Sidebar điều hướng — tông xanh sân/ vàng bóng tennis, số liệu thật."""
    colors = ColorPalette()
    st.sidebar.markdown(
        textwrap.dedent(f"""
    <div style='text-align:center; padding:1rem; margin-bottom:1.5rem;
                border-bottom:1px solid {colors.ACCENT_SPORT}33;'>
        <h2 style='color:{colors.ACCENT_SPORT}; margin:0; font-size:28px;'>🎾</h2>
        <h3 style='margin:0.4rem 0 0 0; font-size:15px; color:{colors.NEUTRAL_LIGHT}; letter-spacing:1px;'>
            ATP PREDICTOR
        </h3>
        <p style='margin:0.2rem 0 0 0; font-size:10px; color:{colors.TEXT_TERTIARY}; text-transform:uppercase;'>
            Nhóm 9 · DHKL18A3HN · NCKH
        </p>
    </div>
    """),
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### MENU")

    pages = [
        ("🏠 Trang chủ", "dashboard"),
        ("🎾 Dự đoán trận đấu", "prediction"),
        ("🏆 Mô phỏng giải đấu", "bracket"),
        ("🔎 Hồ sơ tay vợt", "player_profile"),
        ("📊 Thống kê & Phân tích", "analytics"),
        ("📂 Khám phá dữ liệu", "dataset"),
        ("ℹ️ Giới thiệu", "about"),
    ]

    selected_page = st.sidebar.radio(
        "Navigate to:", [p[0] for p in pages], label_visibility="collapsed",
        index=[p[0] for p in pages].index(st.session_state.current_page)
        if st.session_state.current_page in [p[0] for p in pages] else 0,
    )
    st.session_state.current_page = selected_page
    current_page_key = next((key for label, key in pages if label == selected_page), "dashboard")

    st.sidebar.divider()
    st.sidebar.markdown("### MODEL STATUS")

    real_metrics = _load_sidebar_metrics()
    with st.sidebar:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model", "CatBoost")
        with col2:
            st.metric("Status", "Ready" if real_metrics else "No metrics")

    if real_metrics:
        st.sidebar.markdown("**Performance (test set thật)**")
        st.sidebar.caption(f"Accuracy: **{real_metrics.get('accuracy', 0):.1%}**")
        st.sidebar.caption(f"ROC AUC: **{real_metrics.get('auc', 0):.3f}**")
        st.sidebar.caption(f"F1-Score: **{real_metrics.get('f1', 0):.3f}**")
    else:
        st.sidebar.caption("⚠️ Chưa đọc được experiments/test_evaluation.json")

    st.sidebar.divider()
    st.sidebar.markdown(
        textwrap.dedent(f"""
    <div style='text-align:center; padding-top:1rem; border-top:1px solid {colors.OVERLAY_LIGHT};
                font-size:11px; color:{colors.TEXT_TERTIARY};'>
        <p style='margin:0.4rem 0;'>ATP Prediction Research<br>Nhóm 9 · 2025-2026</p>
    </div>
    """),
        unsafe_allow_html=True,
    )

    return current_page_key


def render_about():
    from pathlib import Path
    import json
    import pandas as pd
    from project.services.data_fetcher import load_datasets

    st.title("ℹ️ Giới thiệu ATP Match Predictor")

    df_clean, _ = load_datasets()
    n_matches = f"{len(df_clean):,}" if not df_clean.empty else "—"
    n_players = (
        f"{pd.concat([df_clean['winner_name'], df_clean['loser_name']]).nunique():,}"
        if not df_clean.empty else "—"
    )
    n_tourneys = f"{df_clean['tourney_name'].nunique():,}" if not df_clean.empty else "—"

    eval_path = Path(__file__).resolve().parents[1] / "experiments" / "test_evaluation.json"
    metrics = {}
    if eval_path.exists():
        data = json.loads(eval_path.read_text())
        metrics = data.get("CatBoost_tuned") or data.get("CatBoost_baseline") or {}

    st.markdown(
        f"""
## Project Overview
Hệ thống **Decision Support System** dự đoán kết quả trận đấu ATP bằng Machine
Learning + Explainable AI, thực hiện bởi **Nhóm 9 - DHKL18A3HN** cho đề tài
NCKH sinh viên.

### Key Features
- 🤖 **Đa mô hình**: CatBoost (chính), so sánh Logistic Regression, Random Forest, XGBoost
- 📊 **Analytics**: EDA gallery, model evaluation, dataset explorer — dùng dữ liệu thật
- 🧠 **Explainable AI**: SHAP values tính trực tiếp cho từng dự đoán
- ⏱️ **Time-based validation**: chia tập theo thời gian, không rò rỉ dữ liệu

### Dataset (số liệu thật, tính trực tiếp từ dữ liệu đã xử lý)
- **Số trận**: {n_matches}
- **Số tay vợt**: {n_players}
- **Số giải đấu**: {n_tourneys}
- **Nguồn dữ liệu**: ATP/WTA match data + betting odds

### Model Performance (đo trên tập test, thật)
| Metric | Value |
|--------|-------|
| Accuracy | {metrics.get('accuracy', 0):.1%} |
| ROC AUC | {metrics.get('auc', 0):.3f} |
| Precision | {metrics.get('precision', 0):.1%} |
| Recall | {metrics.get('recall', 0):.1%} |
| F1-Score | {metrics.get('f1', 0):.3f} |
| Log Loss | {metrics.get('log_loss', 0):.3f} |

---
**Project**: ATP Match Prediction System — Nhóm 9, DHKL18A3HN
**Status**: NCKH Sinh viên
        """
    )


def main():
    configure_page()
    init_session_state()
    current_page = render_sidebar()

    if current_page == "dashboard":
        from project.pages import Home
        Home.show()
    elif current_page == "prediction":
        from project.pages import Prediction
        Prediction.show()
    elif current_page == "bracket":
        from project.pages import Bracket
        Bracket.show()
    elif current_page == "player_profile":
        from project.pages import PlayerProfile
        PlayerProfile.show()
    elif current_page == "analytics":
        from project.pages import Analytics
        Analytics.show()
    elif current_page == "dataset":
        from project.pages import Dataset
        Dataset.show()
    elif current_page == "about":
        render_about()
    else:
        st.error(f"Page not found: {current_page}")
        logger.warning(f"Unknown page requested: {current_page}")


if __name__ == "__main__":
    main()
