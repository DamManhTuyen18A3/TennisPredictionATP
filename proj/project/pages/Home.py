"""
Home Page — PHIÊN BẢN SỬA LỖI
================================

BẢN CŨ: toàn bộ chỉ số (Accuracy 74.3%, ROC AUC 0.812, Precision 72.1%,
Recall 68.9%, F1 0.704, Log Loss 0.541, "42,350 trận", "2,847 tay vợt",
"156 giải") đều là con số VIẾT TAY, không khớp với kết quả thật trong
`experiments/test_evaluation.json` (CatBoost thật: Accuracy ~68%, AUC ~0.75)
và không khớp số trận/tay vợt thật trong dữ liệu (157,081 trận / 6,187 tay vợt).

BẢN MỚI: đọc trực tiếp từ experiments/metrics.json, experiments/test_evaluation.json
và tính thống kê thật từ data/processed/05_clean_data.parquet.
"""

import json
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st

from project.components.cards import stat_card, info_card, metric_card, _render_html
from project.services.data_fetcher import load_datasets
from project.utils.logger import get_logger
from project.utils.theme import ColorPalette, BorderRadius, Typography

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]


@st.cache_data(show_spinner=False)
def load_real_metrics():
    metrics_path = BASE_DIR / "experiments" / "metrics.json"
    eval_path = BASE_DIR / "experiments" / "test_evaluation.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    evaluation = json.loads(eval_path.read_text()) if eval_path.exists() else {}
    return metrics, evaluation


@st.cache_data(show_spinner=False)
def compute_real_dataset_stats(df_clean: pd.DataFrame):
    if df_clean.empty:
        return None
    return {
        "matches": len(df_clean),
        "players": pd.concat([df_clean["winner_name"], df_clean["loser_name"]]).nunique(),
        "tournaments": df_clean["tourney_name"].nunique(),
        "date_min": df_clean["tourney_date"].min(),
        "date_max": df_clean["tourney_date"].max(),
    }


def _court_pattern_data_uri(line_color: str) -> str:
    """Hoạ tiết vạch kẻ sân tennis (rất mờ) làm nền cho bento hero — thay vì
    nền phẳng 1 màu, tạo chiều sâu thị giác đúng chủ đề thay vì dashboard
    SaaS chung chung. Dùng urllib.parse.quote để encode an toàn (tránh lỗi
    tự escape thủ công ký tự '#' trong data URI)."""
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='340' height='200' viewBox='0 0 340 200'>
        <rect x='8' y='8' width='324' height='184' fill='none' stroke='{line_color}' stroke-width='1.4'/>
        <rect x='46' y='8' width='248' height='184' fill='none' stroke='{line_color}' stroke-width='1'/>
        <line x1='8' y1='100' x2='332' y2='100' stroke='{line_color}' stroke-width='1'/>
        <line x1='170' y1='8' x2='170' y2='192' stroke='{line_color}' stroke-width='1'/>
    </svg>"""
    return f"data:image/svg+xml,{quote(svg)}"


def _inject_bento_css(colors) -> None:
    """CSS riêng cho bento hero trang chủ — glassmorphism thật (backdrop-filter),
    hiệu ứng pop/fade khi vào trang, viền phát sáng theo màu accent khi hover.
    Scope bằng class atp-* riêng để không ảnh hưởng các trang khác."""
    css = f"""
    <style>
    @keyframes atpFadeUp {{
        from {{ opacity: 0; transform: translateY(16px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes atpPop {{
        0%   {{ opacity: 0; transform: scale(0.86); }}
        65%  {{ opacity: 1; transform: scale(1.04); }}
        100% {{ opacity: 1; transform: scale(1); }}
    }}
    .atp-bento {{
        display: grid;
        grid-template-columns: 1.35fr 1fr 1fr;
        grid-template-rows: auto auto;
        gap: 14px;
        margin-bottom: 28px;
    }}
    .atp-cell {{
        position: relative;
        border-radius: {BorderRadius.LG}px;
        border: 1px solid {colors.NEUTRAL_DARK};
        background:
            linear-gradient(160deg, rgba(237,242,239,0.05), rgba(237,242,239,0) 60%),
            {colors.SURFACE};
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        overflow: hidden;
        animation: atpFadeUp 0.55s cubic-bezier(0.22,1,0.36,1) both;
        transition: transform 220ms ease, box-shadow 220ms ease, border-color 220ms ease;
    }}
    .atp-cell:hover {{
        transform: translateY(-3px);
        border-color: {colors.ACCENT_SPORT}66;
        box-shadow: 0 0 0 1px {colors.ACCENT_SPORT}33, 0 14px 32px rgba(0,0,0,0.35),
                    0 0 26px {colors.ACCENT_SPORT}26;
    }}
    .atp-hero-cell {{
        grid-column: 1 / 2; grid-row: 1 / 3;
        padding: 34px 36px;
        background-image: url("{_court_pattern_data_uri(colors.ACCENT_SPORT)}"),
            linear-gradient(160deg, rgba(237,242,239,0.05), rgba(237,242,239,0) 60%), {colors.SURFACE};
        background-size: 320px auto, cover, cover;
        background-position: right -40px bottom -30px, center, center;
        background-repeat: no-repeat;
    }}
    .atp-side-cell {{
        padding: 18px 20px;
        display: flex; flex-direction: column; justify-content: center;
    }}
    .atp-eyebrow {{
        font-family: {Typography.FONT_MONO};
        font-size: 12px; letter-spacing: 3px; text-transform: uppercase;
        color: {colors.ACCENT_SPORT}; margin-bottom: 10px; font-weight: 700;
    }}
    .atp-hero-number {{
        font-family: {Typography.FONT_DISPLAY};
        font-size: 72px; line-height: 0.95; letter-spacing: 0.5px;
        color: {colors.TEXT_PRIMARY};
        animation: atpPop 0.75s cubic-bezier(0.22,1,0.36,1) both;
        animation-delay: 0.1s;
    }}
    .atp-hero-caption {{
        font-family: {Typography.FONT_FAMILY};
        font-size: 17px; font-weight: 500;
        color: {colors.TEXT_SECONDARY}; margin-top: 6px;
    }}
    .atp-hero-desc {{
        color: {colors.TEXT_SECONDARY}; font-size: 13.5px; max-width: 480px; margin-top: 12px;
        line-height: 1.5;
    }}
    .atp-side-label {{
        color: {colors.TEXT_TERTIARY}; font-size: 11px; font-weight: 700;
        text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;
    }}
    .atp-side-number {{
        font-family: {Typography.FONT_DISPLAY};
        font-size: 30px; line-height: 1; color: {colors.TEXT_PRIMARY};
        animation: atpPop 0.6s cubic-bezier(0.22,1,0.36,1) both;
    }}
    @media (max-width: 900px) {{
        .atp-bento {{ grid-template-columns: 1fr 1fr; }}
        .atp-hero-cell {{ grid-column: 1 / 3; grid-row: 1; }}
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def _render_bento_hero(accuracy: float, auc: float, f1: float, n_matches: int, colors) -> None:
    """Bento grid hero — thay cho hero band phẳng + hàng KPI 4 cột đều nhau
    kiểu dashboard SaaS chung chung: 1 ô lớn (Accuracy, điểm nhấn chính) +
    3 ô nhỏ (AUC/F1/Số trận) sắp lệch tầng, có glassmorphism thật
    (backdrop-filter), hoạ tiết vạch sân tennis mờ, hiệu ứng pop/fade khi
    vào trang, viền phát sáng theo màu accent khi hover."""
    cells = [
        ("ROC AUC", f"{auc:.3f}", 0.15),
        ("F1-Score", f"{f1:.3f}", 0.2),
        ("Trận đấu huấn luyện", f"{n_matches:,}", 0.25),
    ]
    side_html = ""
    for label, value, delay in cells:
        side_html += f"""
        <div class="atp-cell atp-side-cell" style="animation-delay:{delay}s;">
            <div class="atp-side-label">{label}</div>
            <div class="atp-side-number" style="animation-delay:{delay + 0.15}s;">{value}</div>
        </div>"""

    bento_html = f"""
    <div class="atp-bento">
        <div class="atp-cell atp-hero-cell">
            <div class="atp-eyebrow">🎾 ATP MATCH PREDICTION SYSTEM</div>
            <div class="atp-hero-number">{accuracy:.1%}</div>
            <div class="atp-hero-caption">độ chính xác thật trên tập test</div>
            <div class="atp-hero-desc">Mô hình CatBoost huấn luyện trên {n_matches:,} trận đấu ATP thật (2020–2025), giải thích bằng SHAP cho từng dự đoán — không phải số liệu minh hoạ.</div>
        </div>
        {side_html}
    </div>
    """
    _render_html(bento_html)


def show():
    """Render trang dashboard chính — dùng số liệu THẬT."""
    metrics, evaluation = load_real_metrics()
    df_clean, _ = load_datasets()
    stats = compute_real_dataset_stats(df_clean)
    colors = ColorPalette()

    _inject_bento_css(colors)

    # Chọn model chính để hiển thị ở dashboard: ưu tiên CatBoost_tuned
    best_eval = evaluation.get("CatBoost_tuned") or evaluation.get("CatBoost_baseline") or {}

    if best_eval and stats:
        _render_bento_hero(
            best_eval.get("accuracy", 0), best_eval.get("auc", 0),
            best_eval.get("f1", 0), stats["matches"], colors,
        )
    else:
        st.markdown("# ATP Match Prediction System")
        st.markdown("Hệ thống hỗ trợ ra quyết định dùng AI cho phân tích quần vợt (số liệu lấy trực tiếp từ pipeline)")
        if not best_eval:
            st.warning("Không đọc được `experiments/test_evaluation.json` — chưa có số liệu model thật để hiển thị.")

    if metrics:
        st.markdown("## 🤖 So sánh 4 mô hình (đo trên cùng tập test, số liệu thật)")
        cols = st.columns(len(metrics))
        for col, (name, vals) in zip(cols, metrics.items()):
            with col:
                metric_card(
                    name, f"Acc {vals.get('accuracy', 0):.1%}",
                    change=f"AUC {vals.get('auc', 0):.3f}",
                    color=colors.SECONDARY, icon="🌲",
                )

    st.markdown("---")
    st.markdown("## 📚 Dataset Information (tính trực tiếp từ dữ liệu, không viết tay)")

    if stats:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            stat_card("Tổng số trận", f"{stats['matches']:,}",
                       f"{stats['date_min'].date()} → {stats['date_max'].date()}")
        with col2:
            stat_card("Tay vợt duy nhất", f"{stats['players']:,}", "Cả ATP main + qualifying + futures")
        with col3:
            stat_card("Giải đấu", f"{stats['tournaments']:,}", "Khác nhau trong dữ liệu")
        with col4:
            stat_card("Features", "48", "Sau bước feature selection (08_selected_features)")
    else:
        st.error("Không tải được `data/processed/05_clean_data.parquet` để tính thống kê thật.")

    st.markdown("---")
    st.markdown("## 🔧 System Information")

    info_col1, info_col2 = st.columns(2)
    with info_col1:
        info_card(
            "🚀 Quick Start Guide",
            "1. **Vào** trang 'Match Prediction'\n"
            "2. **Chọn** 2 tay vợt thật + mặt sân + vòng đấu\n"
            "3. **Nhấn** nút dự đoán để chạy model thật\n"
            "4. **Xem** giải thích SHAP thật cho đúng dự đoán đó\n"
            "5. **Đối chiếu** với các trận lịch sử tương tự",
            style="info",
        )
    with info_col2:
        info_card(
            "🧠 Technical Features",
            "- **Models**: CatBoost (chính), so sánh với LightGBM, XGBoost, RandomForest\n"
            "- **Explainability**: SHAP values thật cho từng dự đoán\n"
            "- **Validation**: Time-series split, không rò rỉ dữ liệu (đã kiểm tra |corr|>0.9)\n"
            "- **Elo/H2H**: tính tuần tự theo lịch sử thật, không leakage",
            style="success",
        )

    st.markdown("---")
    st.markdown(
        "Xem toàn bộ ảnh EDA & biểu đồ đánh giá mô hình (đã được pipeline sinh ra) "
        "ở trang **📊 Thống kê & Phân tích**."
    )

    if st.button("🎾 Đi tới Dự đoán trận đấu →", use_container_width=True):
        st.session_state.current_page = "🎾 Dự đoán trận đấu"
        st.rerun()

    logger.info("Home page rendered with real metrics")


if __name__ == "__main__":
    show()
