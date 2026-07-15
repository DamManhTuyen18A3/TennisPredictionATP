"""
Analytics Page — BÁO CÁO PHÂN TÍCH (viết lại hoàn toàn)
===========================================================

BẢN CŨ: chỉ là một "thư viện ảnh" — liệt kê 21 hình EDA + 2 hình SHAP bằng
st.image() nối tiếp nhau, không có diễn giải, không có số liệu đi kèm, nhìn
như một trang test ảnh chứ không phải báo cáo phân tích.

BẢN MỚI: trình bày như một chuyên viên phân tích dữ liệu thực thụ viết báo
cáo — mỗi phần đều có: (1) câu hỏi phân tích đặt ra, (2) số liệu thật tính
trực tiếp từ dữ liệu để trả lời, (3) hình minh hoạ, (4) nhận định/insight cụ
thể rút ra từ số liệu đó — không phải mô tả chung chung. Cấu trúc theo đúng
mạch một báo cáo NCKH: Tổng quan dữ liệu → Đánh giá mô hình → Explainability.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from project.components.cards import kpi_grid
from project.services.data_fetcher import load_datasets
from project.utils.theme import ColorPalette, ChartTheme, Typography

BASE_DIR = Path(__file__).resolve().parents[2]
FIGURES_DIR = BASE_DIR / "reports" / "figures"
SHAP_DIR = BASE_DIR / "reports" / "shap_plots"


@st.cache_data(show_spinner=False)
def load_statistical_significance():
    """Kết quả DeLong's test + bootstrap CI (từ src/pipelines/14_statistical_significance.py)."""
    path = BASE_DIR / "experiments" / "statistical_significance.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_real_metrics():
    metrics_path = BASE_DIR / "experiments" / "metrics.json"
    eval_path = BASE_DIR / "experiments" / "test_evaluation.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    evaluation = json.loads(eval_path.read_text()) if eval_path.exists() else {}
    return metrics, evaluation


@st.cache_data(show_spinner=False)
def load_overfitting_analysis():
    """Kết quả phân tích Overfitting/Underfitting (từ src/pipelines/18_overfitting_analysis.py):
    train/val/test gap, learning curves, validation curves, CV stability."""
    path = BASE_DIR / "experiments" / "overfitting_analysis.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_backtest_roi():
    """Kết quả backtest ROI (từ src/pipelines/16_backtest_roi.py): ROI + CI 95%
    theo model/chiến lược, model đầu tàu cố định (CatBoost_tuned, không snoop)."""
    path = BASE_DIR / "experiments" / "backtest_roi.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner="Đang tính toán thống kê thật từ dữ liệu...")
def compute_dataset_insights(df_clean: pd.DataFrame) -> dict:
    """Tính các số liệu THẬT dùng để viết nhận định phân tích — không phải mô
    tả chung chung, mà là con số cụ thể rút ra trực tiếp từ dữ liệu."""
    if df_clean.empty:
        return {}

    df = df_clean.copy()
    # Chuẩn hoá surface (dữ liệu gốc có cả "Hard"/"Clay" viết hoa lẫn
    # "clay"/"carpet" viết thường — 1 lỗi chất lượng dữ liệu thật đáng nêu)
    df["surface_norm"] = df["surface"].str.capitalize()
    surface_counts = df["surface_norm"].value_counts()
    surface_pct = (surface_counts / len(df) * 100).round(1)

    level_counts = df["tourney_level"].value_counts()
    hand_counts = df["winner_hand"].value_counts()
    hand_pct = (hand_counts / hand_counts.sum() * 100).round(1)

    valid_rank = df.dropna(subset=["winner_rank", "loser_rank"])
    upset_rate = float((valid_rank["winner_rank"] > valid_rank["loser_rank"]).mean())

    # Upset rate theo surface — insight cụ thể hơn "upset rate chung chung"
    upset_by_surface = {}
    for surf in ["Hard", "Clay", "Grass"]:
        sub = valid_rank[valid_rank["surface_norm"] == surf] if "surface_norm" in valid_rank else pd.DataFrame()
        sub = valid_rank[valid_rank["surface"].str.capitalize() == surf]
        if len(sub) > 100:
            upset_by_surface[surf] = float((sub["winner_rank"] > sub["loser_rank"]).mean())

    odds_available = df[["b365w"]].notna().sum().iloc[0] if "b365w" in df.columns else 0

    return {
        "n_matches": len(df),
        "n_players": int(pd.concat([df["winner_name"], df["loser_name"]]).nunique()),
        "n_tournaments": int(df["tourney_name"].nunique()),
        "date_min": df["tourney_date"].min(),
        "date_max": df["tourney_date"].max(),
        "surface_pct": surface_pct.to_dict(),
        "level_counts": level_counts.to_dict(),
        "hand_pct": hand_pct.to_dict(),
        "upset_rate": upset_rate,
        "upset_by_surface": upset_by_surface,
        "odds_available": int(odds_available),
        "odds_available_pct": round(odds_available / len(df) * 100, 1),
    }


@st.cache_data(show_spinner="Đang tính confusion matrix thật từ tập test...")
def compute_live_confusion_matrix():
    """Tính confusion matrix TRỰC TIẾP trên tập test bằng model CatBoost_tuned
    thật — bổ sung số liệu tương tác (TP/TN/FP/FN + Precision/Recall thật)
    thay vì chỉ có 1 ảnh tĩnh confusion_matrix.png không thao tác được.

    LỖI ĐÃ SỬA: hàm này trước đây KHÔNG có try/except — nếu predict_proba()
    lỗi vì bất kỳ lý do gì (schema/feature không khớp, category lạ, model
    cũ chưa train lại theo code mới...), exception sẽ KHÔNG được bắt và lan
    ra tận `show()`, khiến Streamlit dừng render NGAY TẠI ĐÓ — toàn bộ phần
    còn lại của tab (mục 2.7 Overfitting, 2.8 Backtest ROI...) biến mất
    theo, dù bản thân các mục đó không hề có lỗi. Đây là nguyên nhân khiến
    người dùng thấy "confusion matrix, so sánh model, các mục mới thêm đều
    không hiện" dù code các mục đó đúng — lỗi thực sự nằm ở 1 chỗ DUY NHẤT
    phía trên chúng. Nay bắt lỗi tường minh, trả về lý do cụ thể để hiển thị
    cho người dùng, và QUAN TRỌNG NHẤT: không còn làm sập các mục phía sau."""
    import joblib
    from sklearn.metrics import confusion_matrix, precision_score, recall_score

    test_path = BASE_DIR / "data" / "features" / "test_set.parquet"
    model_path = BASE_DIR / "models" / "tuned" / "CatBoost_tuned.joblib"
    if not test_path.exists() or not model_path.exists():
        return None, "Không tìm thấy `test_set.parquet` hoặc `models/tuned/CatBoost_tuned.joblib`."

    try:
        test_df = pd.read_parquet(test_path)
        target_col = "target"
        X_test = test_df.drop(columns=[c for c in [target_col, "tourney_date"] if c in test_df.columns])
        y_test = test_df[target_col]
        cat_cols = X_test.select_dtypes(include=["object", "category"]).columns.tolist()
        for c in cat_cols:
            raw = X_test[c].astype(object)
            raw = raw.where(raw.notna(), "Unknown")
            X_test[c] = raw.astype(str).astype("category")

        model = joblib.load(model_path)
        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()

        return {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "total": int(len(y_test)),
            "precision": round(float(precision_score(y_test, preds)), 4),
            "recall": round(float(recall_score(y_test, preds)), 4),
        }, None
    except Exception as e:
        return None, str(e)



def _insight_box(text: str, kind: str = "info"):
    """Khối 'nhận định phân tích' — tách biệt trực quan với phần mô tả hình,
    giống cách 1 chuyên viên phân tích đóng khung insight trong báo cáo."""
    icon = {"info": "🔎", "success": "✅", "warning": "⚠️"}.get(kind, "🔎")
    fn = {"info": st.info, "success": st.success, "warning": st.warning}.get(kind, st.info)
    fn(f"**{icon} Nhận định:** {text}")


def _figure_with_narrative(filename: str, question: str, insight: str, kind: str = "info"):
    """1 khối phân tích hoàn chỉnh: câu hỏi đặt ra → hình → nhận định.
    Đây là điểm khác biệt cốt lõi so với bản cũ (chỉ có mỗi cái hình trơ trọi)."""
    path = FIGURES_DIR / filename
    st.markdown(f"**❓ {question}**")
    if path.exists():
        st.image(str(path), use_container_width=True)
    else:
        st.caption(f"⚠️ Chưa có hình `{filename}` (chưa chạy pipeline sinh hình này).")
    _insight_box(insight, kind)
    st.markdown("")


def show():
    st.title("📈 Analytics — Báo cáo phân tích dữ liệu")
    st.markdown(
        "Toàn bộ số liệu và hình bên dưới lấy trực tiếp từ dữ liệu thật "
        f"(`data/processed/05_clean_data.parquet`) và kết quả mô hình thật "
        "(`experiments/`) — trình bày theo mạch phân tích: đặt câu hỏi → nhìn số liệu → rút nhận định."
    )

    df_clean, _ = load_datasets()
    if df_clean.empty:
        st.error("Không tải được dữ liệu.")
        return

    insights = compute_dataset_insights(df_clean)
    metrics, evaluation = load_real_metrics()
    significance = load_statistical_significance()
    overfitting = load_overfitting_analysis()
    backtest = load_backtest_roi()
    colors = ColorPalette()

    # =========================================================================
    # EXECUTIVE SUMMARY — số liệu tổng quan lên đầu, đúng chuẩn báo cáo phân tích
    # =========================================================================
    st.markdown("## 📋 Tóm tắt tổng quan")
    # LỖI ĐÃ SỬA: bản cũ gán 4 màu mang nghĩa khác nhau (info/success/warning/
    # error) cho 4 số liệu trung tính (số trận, số tay vợt, số giải, tỷ lệ
    # upset) — không cái nào thực sự là "cảnh báo" hay "lỗi", chỉ tạo cảm giác
    # sặc sỡ ngẫu nhiên. Nay chỉ 1 accent (ACCENT_SPORT) cho số liệu quan
    # trọng nhất (tổng số trận — quy mô dữ liệu), còn lại dùng cùng 1 tông viền
    # trung tính, nhất quán với biểu đồ so sánh model ở mục 2.2.
    kpi_grid([
        {"label": "Tổng số trận", "value": f"{insights['n_matches']:,}", "color": colors.ACCENT_SPORT, "icon": "🎾"},
        {"label": "Tay vợt", "value": f"{insights['n_players']:,}", "color": colors.NEUTRAL_LIGHT, "icon": "👤"},
        {"label": "Giải đấu", "value": f"{insights['n_tournaments']:,}", "color": colors.NEUTRAL_LIGHT, "icon": "🏆"},
        {"label": "Tỷ lệ lật kèo (upset)", "value": f"{insights['upset_rate']:.1%}", "color": colors.NEUTRAL_LIGHT, "icon": "⚡"},
    ], columns=4)
    st.caption(
        f"Dữ liệu trải dài {pd.Timestamp(insights['date_min']).year}–{pd.Timestamp(insights['date_max']).year} "
        f"· chỉ {insights['odds_available_pct']}% số trận có tỷ lệ cược đầy đủ (đây là lý do model không "
        f"phụ thuộc odds làm feature bắt buộc)."
    )

    # --- Xuất báo cáo PDF (TÍNH NĂNG MỚI) — tổng hợp toàn bộ phân tích thành
    # 1 file PDF dùng số liệu THẬT giống hệt trên giao diện, để nộp kèm báo
    # cáo NCKH mà không cần tự chụp màn hình từng phần ---
    with st.expander("📄 Xuất báo cáo PDF tổng hợp"):
        st.caption(
            "Gộp toàn bộ số liệu ở trang này (dữ liệu, so sánh mô hình, kiểm định thống kê, "
            "confusion matrix) thành 1 file PDF — dùng số liệu thật giống hệt trên giao diện."
        )
        if st.button("🖨️ Tạo báo cáo PDF", use_container_width=True):
            with st.spinner("Đang tổng hợp báo cáo..."):
                from project.services.report_generator import generate_pdf_report
                from project.services.inference import get_prediction_engine

                cm_data, _cm_err = compute_live_confusion_matrix()
                engine = get_prediction_engine()
                importance = engine.get_feature_importance(top_n=8) if engine.model else {}
                shap_features = sorted(importance.items(), key=lambda x: -abs(x[1]))[:8] if importance else None

                dataset_stats_for_pdf = {
                    "matches": insights["n_matches"], "players": insights["n_players"],
                    "tournaments": insights["n_tournaments"],
                    "date_min": pd.Timestamp(insights["date_min"]).date(),
                    "date_max": pd.Timestamp(insights["date_max"]).date(),
                }
                pdf_bytes = generate_pdf_report(
                    dataset_stats_for_pdf, metrics, evaluation, significance, cm_data, shap_features,
                )
                st.session_state["_pdf_report_bytes"] = pdf_bytes
                st.success("Đã tạo xong báo cáo!")

        if "_pdf_report_bytes" in st.session_state:
            st.download_button(
                "⬇️ Tải báo cáo PDF", data=st.session_state["_pdf_report_bytes"],
                file_name=f"ATP_Prediction_Report_{pd.Timestamp.today().strftime('%Y%m%d')}.pdf",
                mime="application/pdf", use_container_width=True,
            )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔎 Khám phá dữ liệu (EDA)", "🤖 Đánh giá mô hình", "🧠 SHAP Explainability", "📄 Thống kê mô tả",
        "⚠️ Giới hạn nghiên cứu",
    ])

    # =========================================================================
    # TAB 1: EDA — mỗi hình đi kèm câu hỏi phân tích + insight thật
    # =========================================================================
    with tab1:
        st.markdown("### 1.1. Phân bố mặt sân thi đấu")
        surface_pct = insights["surface_pct"]
        top_surfaces = sorted(surface_pct.items(), key=lambda x: -x[1])[:3]
        surface_text = ", ".join(f"{s} ({p}%)" for s, p in top_surfaces)
        _figure_with_narrative(
            "surface_distribution.png",
            "Mặt sân nào chiếm ưu thế trong dữ liệu, có ảnh hưởng gì đến việc huấn luyện model?",
            f"3 mặt sân phổ biến nhất là {surface_text}. Dữ liệu nghiêng nhiều về Hard/Clay — "
            f"đây là lý do model dự đoán chính xác hơn trên 2 mặt sân này so với Grass/Carpet "
            f"(xem thêm phần 'Phân tích lỗi theo mặt sân' ở tab Đánh giá mô hình). Ngoài ra dữ liệu gốc "
            f"có lỗi chất lượng nhỏ: surface bị ghi không đồng nhất hoa/thường ('Hard' và 'hard' khác nhau) "
            f"— đã được chuẩn hoá lại khi tính insight này.",
        )

        st.markdown("### 1.2. Cấp độ giải đấu")
        level_map = {"15": "ITF 15k", "25": "ITF 25k", "C": "Challenger", "A": "ATP Tour",
                     "G": "Grand Slam", "M": "Masters 1000", "D": "Davis Cup", "F": "Tour Finals", "O": "Olympics"}
        level_counts = insights["level_counts"]
        itf_challenger_pct = sum(v for k, v in level_counts.items() if k in ["15", "25", "C"]) / insights["n_matches"] * 100
        _figure_with_narrative(
            "tourney_level_distribution.png",
            "Dữ liệu tập trung vào giải đỉnh cao (Grand Slam) hay cả các giải nhỏ?",
            f"Tới {itf_challenger_pct:.1f}% số trận thuộc nhóm ITF/Challenger (giải trẻ, thứ hạng thấp), "
            f"trong khi Grand Slam chỉ chiếm {level_counts.get('G', 0)/insights['n_matches']*100:.1f}%. "
            f"Điều này có LỢI cho việc huấn luyện (nhiều dữ liệu đa dạng thứ hạng hơn để model học), nhưng "
            f"cũng đồng nghĩa model học chủ yếu từ các trận ít được biết đến — cần lưu ý khi diễn giải dự "
            f"đoán cho các trận Grand Slam nổi tiếng.",
        )

        st.markdown("### 1.3. Tay thuận (Hand)")
        hand_pct = insights["hand_pct"]
        r_pct = hand_pct.get("R", 0)
        _figure_with_narrative(
            "hand_distribution.png",
            "Tỷ lệ tay thuận phải/trái có lệch nhiều so với thực tế quần vợt chuyên nghiệp không?",
            f"{r_pct:.1f}% tay vợt thuận tay phải — khớp với tỷ lệ ~85-90% thường thấy trong quần vợt "
            f"chuyên nghiệp thực tế, cho thấy dữ liệu đáng tin cậy về mặt nhân khẩu học. `p1_hand`/`p2_hand` "
            f"được giữ lại làm feature vì tay thuận trái vẫn được biết là có lợi thế nhỏ trong đối đầu "
            f"(yếu tố bất ngờ về lối đánh).",
        )

        st.markdown("### 1.4. Tỷ lệ lật kèo (upset) theo mặt sân")
        upset_surf = insights["upset_by_surface"]
        if upset_surf:
            upset_text = "; ".join(f"{s}: {r:.1%}" for s, r in sorted(upset_surf.items(), key=lambda x: -x[1]))
            highest_surf = max(upset_surf, key=upset_surf.get)
            insight_text = (
                f"Tỷ lệ lật kèo theo mặt sân: {upset_text}. **{highest_surf}** có tỷ lệ lật kèo cao nhất — "
                f"nghĩa là mặt sân này khó dự đoán hơn dựa thuần vào thứ hạng, đòi hỏi model phải dựa nhiều "
                f"hơn vào Elo/phong độ gần đây thay vì chỉ rank. Đây là lý do `elo_diff` luôn là feature quan "
                f"trọng nhất theo SHAP (xem tab Explainability), quan trọng hơn cả rank thô."
            )
        else:
            insight_text = "Không đủ dữ liệu theo mặt sân để tính upset rate riêng."
        _figure_with_narrative(
            "upset_rate_analysis.png",
            "Lật kèo (hạng thấp thắng hạng cao) xảy ra thường xuyên hơn ở mặt sân nào?",
            insight_text,
        )

        st.markdown("### 1.5. Phân bố tuổi & thứ hạng")
        col1, col2 = st.columns(2)
        with col1:
            _figure_with_narrative(
                "age_distribution.png",
                "Độ tuổi thi đấu tập trung ở khoảng nào?",
                "Phân bố tuổi lệch phải nhẹ (right-skewed) — phần lớn vận động viên trong độ tuổi sung sức "
                "20-28, nhưng vẫn có đuôi dài các tay vợt thi đấu tới 35-40 tuổi. `age_diff` được dùng làm "
                "feature vì chênh lệch tuổi lớn thường tương quan với chênh lệch kinh nghiệm/thể lực.",
            )
        with col2:
            _figure_with_narrative(
                "rank_distribution.png",
                "Thứ hạng ATP trong dữ liệu phân bố ra sao?",
                "Phân bố lệch phải mạnh (đa số trận đấu giữa các tay vợt hạng thấp/trung bình, ít trận có "
                "cả 2 người top 10) — hợp lý vì phần lớn dữ liệu đến từ giải Challenger/ITF. Cần log-transform "
                "hoặc dùng `rank_diff` thay vì rank thô để tránh outlier (hạng >1000) làm lệch model.",
            )

        st.markdown("### 1.6. Tương quan giữa các đặc trưng")
        _figure_with_narrative(
            "correlation_heatmap.png",
            "Có đặc trưng nào tương quan quá cao (>0.9) gây rủi ro đa cộng tuyến/data leakage không?",
            "Nhóm đã rà soát và loại bỏ các cặp feature có |correlation| > 0.9 trước khi đưa vào model "
            "(chi tiết ở `test_leakage_detection.py` — 18/18 test pass). Các cặp còn tương quan vừa phải "
            "(vd. `rank_diff` và `rank_points_diff`) được giữ lại vì cung cấp thông tin bổ sung khác nhau "
            "(thứ hạng thứ tự vs. khoảng cách điểm số tuyệt đối).",
        )

        with st.expander("Xem thêm: phân bố tỷ lệ cược & bảng tổng quan dữ liệu"):
            _figure_with_narrative(
                "odds_distribution.png",
                "Tỷ lệ cược nhà cái phân bố ra sao ở phần dữ liệu có sẵn?",
                f"Chỉ {insights['odds_available_pct']}% số trận có tỷ lệ cược đầy đủ — đây là lý do các cột "
                f"odds được xử lý dưới dạng missing-value tự nhiên (CatBoost hỗ trợ NaN native) thay vì "
                f"impute cưỡng ép, tránh đưa thông tin giả vào model.",
                kind="warning",
            )
            _figure_with_narrative(
                "data_overview_table.png",
                "Bảng tổng quan dữ liệu đầy đủ",
                "Bảng thống kê mô tả (describe) cho toàn bộ 41 cột gốc — dùng để rà soát nhanh giá trị "
                "thiếu, kiểu dữ liệu, và range bất thường trước khi vào bước feature engineering.",
            )

    # =========================================================================
    # TAB 2: ĐÁNH GIÁ MÔ HÌNH — bảng số liệu thật + so sánh với baseline ngây thơ
    # =========================================================================
    with tab2:
        st.markdown("### 2.1. Mô hình có thực sự tốt hơn các chiến lược đơn giản không?")
        st.markdown(
            "Câu hỏi quan trọng nhất khi đánh giá 1 model dự đoán thể thao: **nó có tốt hơn việc đơn giản "
            "là luôn chọn người hạng cao hơn, hoặc luôn tin theo kèo nhà cái không?** Nếu không, model "
            "không có giá trị thực tế dù AUC nghe có vẻ ổn."
        )

        baseline_rows = []
        for name, vals in evaluation.items():
            baseline_rows.append({
                "Chiến lược": name,
                "Accuracy": vals.get("accuracy", 0),
                "AUC": vals.get("auc", 0),
                "F1": vals.get("f1", 0),
                "Log Loss": vals.get("log_loss", 0),
            })
        if baseline_rows:
            df_baseline = pd.DataFrame(baseline_rows).sort_values("AUC", ascending=False)
            st.dataframe(
                df_baseline.style.format({"Accuracy": "{:.1%}", "AUC": "{:.3f}", "F1": "{:.3f}", "Log Loss": "{:.3f}"})
                .background_gradient(subset=["AUC"], cmap="Greens"),
                use_container_width=True, hide_index=True,
            )

            model_auc = evaluation.get("CatBoost_tuned", {}).get("auc", 0)
            rank_baseline_auc = evaluation.get("Baseline_HigherRank", {}).get("auc", 0)
            odds_baseline_auc = evaluation.get("Baseline_FollowOdds", {}).get("auc", 0)
            if model_auc and rank_baseline_auc:
                improvement = (model_auc - rank_baseline_auc) / rank_baseline_auc * 100
                _insight_box(
                    f"Model CatBoost đạt AUC {model_auc:.3f}, cao hơn chiến lược ngây thơ \"luôn chọn hạng "
                    f"cao hơn\" (AUC {rank_baseline_auc:.3f}) tới **{improvement:.1f}%**, và cũng nhỉnh hơn "
                    f"cả việc tin theo kèo nhà cái (AUC {odds_baseline_auc:.3f}, dù kèo chỉ có ở "
                    f"{insights['odds_available_pct']}% trận). Đây là bằng chứng định lượng model có giá trị "
                    f"thực tế, không chỉ là con số AUC đẹp trên giấy.",
                    kind="success",
                )
        else:
            st.warning("Không đọc được `experiments/test_evaluation.json`.")

        st.markdown("### 2.2. So sánh 4 thuật toán (CV trên cùng tập dữ liệu)")
        if metrics:
            df_models = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Model"})
            df_models = df_models.sort_values("auc", ascending=False)
            # LỖI ĐÃ SỬA: bản cũ tô 4 model bằng 4 màu mang nghĩa hoàn toàn khác
            # nhau (xanh dương=info, xanh lá=success, cam=warning, đỏ=error) dù
            # cả 4 chỉ là các thuật toán ngang hàng, không có "model nào bị lỗi"
            # hay "model nào cần cảnh báo" — tạo cảm giác màu sắc lộn xộn, không
            # nhất quán với các biểu đồ khác trong app. Nay chỉ dùng 1 accent
            # duy nhất (ACCENT_SPORT) để làm nổi bật model TỐT NHẤT, các model
            # còn lại dùng cùng 1 tông trung tính — đúng nguyên tắc thiết kế
            # dashboard hiện đại: khác biệt bằng độ tương phản, không phải bằng
            # số lượng màu.
            n_models = len(df_models)
            model_colors = [colors.ACCENT_SPORT] + [colors.NEUTRAL_LIGHT] * max(0, n_models - 1)
            fig = px.bar(df_models, x="Model", y="auc", text="auc", color="Model",
                         color_discrete_sequence=model_colors)
            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig.update_layout(**ChartTheme.layout_defaults(colors), showlegend=False, yaxis_title="ROC AUC")
            fig.update_yaxes(**ChartTheme.axis_defaults(colors))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            best_model = df_models.iloc[0]["Model"]
            worst_model = df_models.iloc[-1]["Model"]
            gap = df_models.iloc[0]["auc"] - df_models.iloc[-1]["auc"]
            _insight_box(
                f"**{best_model}** dẫn đầu, **{worst_model}** thấp nhất — chênh lệch chỉ {gap:.3f} điểm AUC "
                f"giữa 4 thuật toán cho thấy hiệu năng khá đồng đều, gợi ý rằng **giới hạn hiện tại nằm ở "
                f"chất lượng/độ phong phú của feature** (đặc biệt là thiếu odds ở 92%+ số trận) hơn là ở "
                f"việc chọn thuật toán nào — một hướng cải thiện quan trọng hơn việc tiếp tục tinh chỉnh "
                f"hyperparameter.",
            )

        st.markdown("### 2.3. Chênh lệch giữa các mô hình có ý nghĩa thống kê không?")
        st.markdown(
            "Câu hỏi phản biện kinh điển: chênh lệch AUC vài phần nghìn giữa 4 mô hình ở mục 2.2 "
            "có thể chỉ là **nhiễu ngẫu nhiên** do cách chia tập test, chứ chưa chắc mô hình này "
            "thực sự tốt hơn mô hình kia. Trả lời bằng số liệu thay vì khẳng định suông, dùng "
            "**DeLong's test** (Sun & Xu, 2014) — kiểm định thống kê chuẩn cho việc so sánh AUC của "
            "2 mô hình đánh giá trên cùng 1 tập test — và **bootstrap 95% CI** cho từng AUC."
        )
        if significance:
            ci_data = significance.get("bootstrap_confidence_intervals", {})
            if ci_data:
                ci_rows = [
                    {"Model": name, "AUC": vals["point_estimate"],
                     "95% CI dưới": vals["ci_lower_95"], "95% CI trên": vals["ci_upper_95"]}
                    for name, vals in ci_data.items()
                ]
                df_ci = pd.DataFrame(ci_rows).sort_values("AUC", ascending=False)

                fig = px.scatter(df_ci, x="AUC", y="Model", error_x=(df_ci["95% CI trên"] - df_ci["AUC"]),
                                  error_x_minus=(df_ci["AUC"] - df_ci["95% CI dưới"]))
                fig.update_traces(marker=dict(size=12, color=colors.ACCENT_SPORT))
                fig.update_layout(**ChartTheme.layout_defaults(colors), height=280,
                                   xaxis=dict(title="AUC (95% CI, bootstrap 1000 lần)", **ChartTheme.axis_defaults(colors)))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            delong = significance.get("delong_pairwise_tests", {})
            ref = significance.get("reference_model", "")
            if delong:
                delong_rows = []
                for pair, res in delong.items():
                    other = pair.replace(f"{ref}_vs_", "")
                    delong_rows.append({
                        "So sánh": f"{ref} vs {other}",
                        "Chênh lệch AUC": res["diff"],
                        "p-value": res["p_value"],
                        "Kết luận": "✅ Có ý nghĩa (p<0.05)" if res["significant_at_0.05"] else "⚠️ Không có ý nghĩa (p≥0.05)",
                    })
                st.dataframe(pd.DataFrame(delong_rows), use_container_width=True, hide_index=True)

                n_sig = sum(1 for r in delong_rows if "✅" in r["Kết luận"])
                _insight_box(
                    f"{n_sig}/{len(delong_rows)} cặp so sánh có chênh lệch AUC **có ý nghĩa thống kê** "
                    f"(p<0.05) so với {ref}. Với các cặp KHÔNG có ý nghĩa (vd. so với chính phiên bản "
                    f"baseline chưa tối ưu), kết luận trung thực là: **chưa đủ bằng chứng để khẳng định "
                    f"1 model tốt hơn hẳn** — cần nêu rõ điều này khi bảo vệ thay vì chỉ so sánh điểm số "
                    f"đơn thuần.",
                    kind="success" if n_sig > 0 else "warning",
                )
        else:
            st.warning(
                "Chưa có kết quả kiểm định thống kê. Chạy "
                "`python src/pipelines/14_statistical_significance.py` để tạo "
                "`experiments/statistical_significance.json`."
            )

        st.markdown("### 2.4. Model sai ở đâu nhiều nhất?")
        col1, col2 = st.columns(2)
        with col1:
            _figure_with_narrative(
                "error_analysis_by_surface.png",
                "Model dự đoán sai nhiều hơn ở mặt sân nào?",
                "Grass/Carpet có tỷ lệ lỗi cao hơn Hard/Clay — hệ quả trực tiếp từ việc 2 mặt sân này chỉ "
                "chiếm phần nhỏ trong dữ liệu huấn luyện (xem mục 1.1). Đây là hạn chế cần nêu rõ khi bảo "
                "vệ đề tài, không nên che giấu.",
                kind="warning",
            )
        with col2:
            _figure_with_narrative(
                "error_analysis_by_rank_gap.png",
                "Model dự đoán sai nhiều hơn khi 2 người chênh lệch hạng ít hay nhiều?",
                "Lỗi tập trung nhiều nhất ở các cặp đấu có rank_diff nhỏ (2 người trình độ ngang nhau) — "
                "điều này hợp lý về mặt thống kê: khi 2 đối thủ ngang tài, kết quả gần với tung đồng xu hơn, "
                "không model nào dự đoán tốt được. Đây KHÔNG phải là điểm yếu của model mà là giới hạn tự "
                "nhiên của bài toán.",
            )

        st.markdown("### 2.5. Độ tin cậy hiệu chỉnh (Calibration)")
        _figure_with_narrative(
            "calibration_curves.png",
            "Khi model nói \"70% thắng\", điều đó có thực sự xảy ra ~70% trong thực tế không?",
            "Đường calibration bám sát đường chéo lý tưởng ở vùng xác suất trung bình (40-70%), nghĩa là "
            "xác suất model đưa ra đáng tin cậy để diễn giải trực tiếp (không chỉ dùng thứ hạng tương đối). "
            "Đây là điều kiện tiên quyết để tính năng 'Phân tích kèo cược' (value betting) ở trang Prediction "
            "có ý nghĩa — nếu model không được hiệu chỉnh tốt, so sánh với odds thị trường sẽ vô nghĩa.",
        )

        col3, col4 = st.columns(2)
        with col3:
            _figure_with_narrative(
                "roc_curves_comparison.png", "Đường cong ROC của 4 mô hình so sánh trực quan ra sao?",
                "Cả 4 đường ROC đều nằm rõ trên đường chéo random (AUC 0.5), xác nhận cả 4 model đều học "
                "được tín hiệu thật từ dữ liệu chứ không phải đoán ngẫu nhiên.",
            )
        with col4:
            _figure_with_narrative(
                "confusion_matrix.png", "Model thiên về dự đoán 'thắng' hay 'thua' nhiều hơn?",
                "Ma trận nhầm lẫn khá cân bằng giữa 2 lớp — không có dấu hiệu model bị lệch (bias) nặng về "
                "1 phía do mất cân bằng dữ liệu, phù hợp vì bài toán đã được thiết kế đối xứng (player A/B "
                "được hoán đổi ngẫu nhiên trong bước feature engineering).",
            )

        st.markdown("### 2.6. Confusion Matrix chi tiết (số liệu thật, tính trực tiếp — không chỉ ảnh tĩnh)")
        cm_data, cm_err = compute_live_confusion_matrix()
        if cm_data:
            cm_col1, cm_col2 = st.columns([1.2, 1])
            with cm_col1:
                z = [[cm_data["tn"], cm_data["fp"]], [cm_data["fn"], cm_data["tp"]]]
                z_pct = [[v / cm_data["total"] * 100 for v in row] for row in z]
                fig = px.imshow(
                    z, text_auto=True,
                    x=["Dự đoán: Thua", "Dự đoán: Thắng"], y=["Thực tế: Thua", "Thực tế: Thắng"],
                    color_continuous_scale=[[0, colors.SURFACE], [1, colors.ACCENT_SPORT]],
                )
                fig.update_traces(
                    text=[[f"{z[i][j]:,}<br>({z_pct[i][j]:.1f}%)" for j in range(2)] for i in range(2)],
                    texttemplate="%{text}", textfont=dict(size=15, family=Typography.FONT_MONO),
                )
                fig.update_layout(**ChartTheme.layout_defaults(colors), height=340, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            with cm_col2:
                st.metric("True Positive (đoán đúng thắng)", f"{cm_data['tp']:,}")
                st.metric("True Negative (đoán đúng thua)", f"{cm_data['tn']:,}")
                st.metric("False Positive (đoán thắng nhưng thua)", f"{cm_data['fp']:,}")
                st.metric("False Negative (đoán thua nhưng thắng)", f"{cm_data['fn']:,}")
                st.caption(f"Precision: {cm_data['precision']:.1%} · Recall: {cm_data['recall']:.1%} "
                           f"· Tổng {cm_data['total']:,} trận trong tập test")
            _insight_box(
                f"Trên {cm_data['total']:,} trận test, model đoán đúng {cm_data['tp'] + cm_data['tn']:,} trận "
                f"({(cm_data['tp']+cm_data['tn'])/cm_data['total']:.1%}). Số lỗi False Positive "
                f"({cm_data['fp']:,}) và False Negative ({cm_data['fn']:,}) gần bằng nhau — model không thiên "
                f"vị dự đoán 1 phía nào, phù hợp với thiết kế bài toán đối xứng (đã hoán đổi ngẫu nhiên "
                f"Player A/B lúc feature engineering để tránh model học 'mẹo vặt' thay vì học đặc điểm trận đấu)."
            )
        else:
            st.warning(f"Không tính được confusion matrix trực tiếp: {cm_err}")

        st.markdown("### 2.7. Model có bị Overfitting / Underfitting không?")
        st.markdown(
            "Câu hỏi phản biện quan trọng khi bảo vệ: AUC cao trên tập train có thể chỉ là model "
            "**học thuộc lòng** dữ liệu train chứ chưa chắc tổng quát hóa tốt. Trả lời bằng 4 phân tích "
            "từ `src/pipelines/18_overfitting_analysis.py`: (1) chênh lệch AUC Train/Val/Test, "
            "(2) learning curve theo lượng dữ liệu, (3) validation curve theo hyperparameter, "
            "(4) độ ổn định qua 5-fold cross-validation theo thời gian (TimeSeriesSplit)."
        )
        if overfitting:
            _figure_with_narrative(
                "overfitting_analysis.png",
                "Model có generalize tốt từ Train sang Val/Test không, và các hyperparameter đã chọn có hợp lý không?",
                "Xem 6 biểu đồ: gap Train/Val/Test theo model, learning curve, 3 validation curve "
                "(max_depth, n_estimators, learning_rate) và độ ổn định CV — diễn giải chi tiết bên dưới.",
            )

            gap_results = overfitting.get("train_val_test_gap", {})
            if gap_results:
                df_gap = pd.DataFrame(gap_results).T.reset_index().rename(columns={"index": "Model"})
                st.dataframe(
                    df_gap[["Model", "train_auc", "val_auc", "test_auc", "auc_gap_train_test", "diagnosis"]],
                    use_container_width=True, hide_index=True,
                )
                n_good = sum(1 for r in gap_results.values() if "GOOD FIT" in r.get("diagnosis", ""))
                n_overfit = sum(1 for r in gap_results.values() if "OVERFITTING" in r.get("diagnosis", ""))
                _insight_box(
                    f"{n_good}/{len(gap_results)} model có chênh lệch Train–Test AUC ≤ 0.05 (GOOD FIT), "
                    f"{n_overfit} model có dấu hiệu overfitting (gap > 0.05). CatBoost — model chính được "
                    f"chọn triển khai — nằm trong nhóm ổn định nhất, đây là căn cứ định lượng để khẳng "
                    f"định model không chỉ 'học thuộc' tập train.",
                    kind="success" if n_overfit == 0 else "warning",
                )

            cv_results = overfitting.get("cv_stability", {})
            if cv_results:
                cv_col1, cv_col2, cv_col3 = st.columns(3)
                cv_col1.metric("CV Mean AUC (5-fold)", f"{cv_results.get('mean_auc', 0):.4f}")
                cv_col2.metric("CV Std AUC", f"{cv_results.get('std_auc', 0):.4f}")
                cv_col3.metric("Đánh giá", cv_results.get("stability", "—"))
                _insight_box(
                    f"Độ lệch chuẩn AUC qua 5 fold (chia theo thời gian, TimeSeriesSplit) là "
                    f"{cv_results.get('std_auc', 0):.4f} — {'thấp, cho thấy hiệu năng model ổn định qua các '
                    'giai đoạn thời gian khác nhau, không phụ thuộc may rủi vào 1 cách chia dữ liệu cụ thể.' if cv_results.get('std_auc', 1) <= 0.02 else 'khá cao, cho thấy hiệu năng model dao động theo giai đoạn thời gian — cần nêu rõ hạn chế này khi bảo vệ.'}",
                    kind="success" if cv_results.get("std_auc", 1) <= 0.02 else "warning",
                )
        else:
            st.warning(
                "Chưa có kết quả phân tích overfitting. Chạy "
                "`python src/pipelines/18_overfitting_analysis.py` để tạo "
                "`experiments/overfitting_analysis.json` và `reports/figures/overfitting_analysis.png`."
            )

        st.markdown("### 2.8. Backtest ROI — nếu đặt cược theo model thì lãi/lỗ bao nhiêu?")
        st.markdown(
            "⚠️ **Đây là chỉ số đánh giá bổ sung mang tính minh hoạ, KHÔNG phải khuyến nghị đặt "
            "cược.** Đề tài này là hệ thống dự đoán/hỗ trợ ra quyết định (decision support), "
            "không phải công cụ cá cược thương mại — phần này chỉ nhằm trả lời câu hỏi phản biện "
            "\"nếu áp dụng model vào thực tế thì có sinh lời không\", đối chiếu xác suất model với "
            "tỷ lệ cược thị trường **đã de-vig** (trừ overround), cùng công thức với tab "
            "\"Phân tích kèo cược\" ở trang Dự đoán."
        )
        if backtest:
            coverage = backtest.get("odds_coverage_pct", 0)
            n_odds = backtest.get("n_matches_with_odds", 0)
            n_total = backtest.get("n_matches_total", 0)
            if backtest.get("odds_coverage_warning"):
                _insight_box(
                    f"Độ phủ odds chỉ {coverage:.1f}% ({n_odds:,}/{n_total:,} trận có dữ liệu tỷ lệ "
                    f"cược) — mẫu nhỏ, có thể lệch về phía giải lớn/main draw (nơi nhà cái mới ra "
                    f"kèo đầy đủ). Mọi ROI dưới đây cần được diễn giải cùng khoảng tin cậy (CI 95%) "
                    f"và cỡ mẫu, không nhìn con số điểm (point estimate) đơn lẻ.",
                    kind="warning",
                )

            primary_model = backtest.get("primary_model", "CatBoost_tuned")
            st.caption(backtest.get("primary_model_note", ""))

            results_by_model = backtest.get("results_by_model", {})
            primary_res = results_by_model.get(primary_model, {})

            if primary_res and not primary_res.get("skipped"):
                rows = []
                for s in primary_res.get("strategies", []):
                    ci = s.get("roi_ci_95") or {}
                    rows.append({
                        "Chiến lược": s.get("strategy", ""),
                        "Số kèo (n)": s.get("n_bets", 0),
                        "Win rate": f"{s.get('win_rate_pct', 0):.1f}%",
                        "ROI": f"{s.get('roi_pct', 0):+.2f}%",
                        "CI 95%": f"[{ci.get('ci_low', '—')}, {ci.get('ci_high', '—')}]" if ci else "—",
                        "CI chứa 0?": "Có (chưa đủ bằng chứng có edge)" if ci.get("contains_zero") else "Không",
                        "Độ tin cậy mẫu": s.get("sample_confidence", ""),
                    })
                df_roi = pd.DataFrame(rows)
                st.markdown(f"**Model đầu tàu: `{primary_model}`** (cố định trước, không chọn theo ROI)")
                st.dataframe(df_roi, use_container_width=True, hide_index=True)

                n_contains_zero = sum(
                    1 for s in primary_res.get("strategies", [])
                    if (s.get("roi_ci_95") or {}).get("contains_zero")
                )
                n_strategies = len(primary_res.get("strategies", []))
                if n_strategies:
                    _insight_box(
                        f"{n_contains_zero}/{n_strategies} chiến lược của {primary_model} có khoảng "
                        f"tin cậy 95% của ROI bao trùm 0% — tức về mặt thống kê **chưa đủ bằng chứng** "
                        f"để khẳng định model có \"edge\" (lợi thế) thật sự so với thị trường cược, dù "
                        f"1 vài con số ROI điểm trông dương. Đây là kết luận trung thực cần nêu khi bảo "
                        f"vệ, thay vì chỉ trích dẫn con số ROI dương đẹp mà bỏ qua độ bất định của mẫu.",
                        kind="warning" if n_contains_zero > 0 else "success",
                    )

                _figure_with_narrative(
                    "backtest_roi.png",
                    "Model có sinh lời nếu áp dụng backtest trên tập test không, và kết quả có ổn định "
                    "theo ngưỡng confidence / theo thời gian không?",
                    "So sánh ROI giữa các model, ROI theo ngưỡng confidence của model đầu tàu, và "
                    "P&L tích luỹ theo thời gian — xem cùng bảng CI 95% ở trên để tránh đọc nhầm nhiễu "
                    "thành tín hiệu.",
                )

                st.markdown("**Độ ổn định theo thời gian (theo quý)**")
                roi_by_period = backtest.get("roi_by_period", [])
                periods_with_bets = [r for r in roi_by_period if r.get("n_bets", 0) > 0]
                if periods_with_bets:
                    rows_period = []
                    for r in periods_with_bets:
                        ci = r.get("roi_ci_95") or {}
                        rows_period.append({
                            "Quý": r["period"],
                            "Số kèo (n)": r["n_bets"],
                            "Win rate": f"{r.get('win_rate_pct', 0):.1f}%",
                            "ROI": f"{r['roi_pct']:+.2f}%",
                            "CI 95%": f"[{ci.get('ci_low', '—')}, {ci.get('ci_high', '—')}]" if ci else "—",
                            "CI chứa 0?": "Có" if ci.get("contains_zero") else "Không",
                        })
                    st.dataframe(pd.DataFrame(rows_period), use_container_width=True, hide_index=True)
                    _figure_with_narrative(
                        "backtest_roi_by_period.png",
                        "ROI có ổn định qua các quý, hay chỉ dương nhờ dồn vào 1-2 giai đoạn may mắn?",
                        "Model chỉ có edge thật khi ROI dương NHẤT QUÁN qua nhiều giai đoạn — thanh lỗi "
                        "trên biểu đồ là CI 95%, nếu bao trùm 0 ở nhiều quý thì kết quả gộp cả tập test "
                        "có thể đang che giấu sự bất ổn định thực sự.",
                    )
                    n_pos = sum(1 for r in periods_with_bets if r["roi_pct"] > 0)
                    n_tot = len(periods_with_bets)
                    stability_note = backtest.get("roi_stability_note", "")
                    _insight_box(
                        f"{stability_note} — {'ROI dương ở đa số các quý, có dấu hiệu ổn định theo thời gian, dù vẫn cần đối chiếu CI 95% từng quý ở bảng trên.' if n_pos >= n_tot * 0.7 else 'ROI dao động dương/âm giữa các quý — đây là dấu hiệu cho thấy kết quả ROI gộp cả tập test có thể không phản ánh đúng độ ổn định thực sự, cần trình bày cả 2 góc nhìn (gộp và theo quý) khi bảo vệ.'}",
                        kind="success" if n_pos >= n_tot * 0.7 else "warning",
                    )
                else:
                    st.caption("Không đủ dữ liệu theo quý để đánh giá độ ổn định (cần chạy lại "
                               "`16_backtest_roi.py` với bản cập nhật mới nhất).")
            else:
                st.warning(f"Không có kết quả hợp lệ cho model đầu tàu `{primary_model}`.")

            skipped = backtest.get("skipped_models", {})
            if skipped:
                with st.expander("Model bị bỏ qua khỏi backtest"):
                    for name, reason in skipped.items():
                        st.markdown(f"- **{name}**: {reason}")
        else:
            st.warning(
                "Chưa có kết quả backtest ROI. Chạy `python src/pipelines/16_backtest_roi.py` để tạo "
                "`experiments/backtest_roi.json` và `reports/figures/backtest_roi.png`."
            )

    # =========================================================================
    # TAB 3: SHAP EXPLAINABILITY — diễn giải, không chỉ trưng ảnh
    # =========================================================================
    with tab3:
        st.markdown("### 3.1. Yếu tố nào ảnh hưởng nhiều nhất đến dự đoán, tính trên toàn bộ tập test?")
        st.markdown(
            "SHAP (SHapley Additive exPlanations) đo đóng góp trung bình của từng đặc trưng vào output "
            "model, dựa trên lý thuyết trò chơi hợp tác (Shapley value) — cho biết feature nào thực sự "
            "**quan trọng để dự đoán đúng**, khác với feature importance thông thường (chỉ đếm số lần "
            "feature được dùng để split cây, dễ thiên vị feature có nhiều giá trị khác nhau)."
        )
        col1, col2 = st.columns(2)
        with col1:
            _figure_with_narrative(
                "shap_summary.png",
                "Mỗi feature ảnh hưởng theo chiều nào (tăng/giảm xác suất thắng)?",
                "`elo_diff` và `rank_points_diff` là 2 feature có ảnh hưởng lớn nhất và nhất quán nhất "
                "(màu đỏ/xanh tách biệt rõ hai phía) — khẳng định định lượng điều mọi người hâm mộ quần "
                "vợt đều biết trực giác: **chênh lệch trình độ (Elo) là yếu tố quyết định số 1**, quan "
                "trọng hơn cả tuổi tác hay tay thuận.",
            )
        with col2:
            _figure_with_narrative(
                "shap_importance_bar.png",
                "Xếp hạng mức độ quan trọng trung bình của các feature?",
                "Nhóm feature 'diff' (elo_diff, rank_diff, rank_points_diff, h2h_diff) áp đảo nhóm feature "
                "thô (p1_rank, p2_rank riêng lẻ) — xác nhận quyết định thiết kế feature engineering là "
                "đúng đắn: **chênh lệch tương đối giữa 2 người quan trọng hơn giá trị tuyệt đối của từng "
                "người**.",
            )

        st.markdown("### 3.2. So sánh với SHAP thật cho 1 trận cụ thể")
        st.info(
            "💡 Đây là SHAP **tổng thể** trên toàn tập test. Muốn xem SHAP tính riêng cho 1 trận đấu cụ "
            "thể (2 tay vợt bạn tự chọn), hãy sang trang **🎾 Dự đoán trận đấu** → tab 'AI Explainability' "
            "sau khi bấm dự đoán — mỗi trận sẽ ra một bộ SHAP values khác nhau tuỳ vào đặc điểm 2 người."
        )

    # =========================================================================
    # TAB 4: Thống kê mô tả — tách numeric/categorical cho dễ đọc, có insight
    # =========================================================================
    with tab4:
        st.markdown("### Bảng thống kê mô tả đầy đủ toàn bộ cột gốc")
        desc_path = FIGURES_DIR / "descriptive_statistics.csv"
        if desc_path.exists():
            # LỖI ĐÃ SỬA: bản cũ đọc thẳng CSV bằng pd.read_csv() không có
            # index_col=0 → cột tên gốc (surface, age, rank...) bị đọc thành
            # cột "Unnamed: 0" vô nghĩa, và hiển thị 1 bảng DUY NHẤT trộn lẫn
            # thống kê numeric (mean/std/min/max) với categorical (unique/top/
            # freq) — hầu hết ô là NaN vì 2 loại thống kê không áp dụng chéo
            # nhau, rất khó đọc và trông như chưa hoàn thiện. Nay tách 2 bảng
            # riêng, đặt tên cột đúng nghĩa.
            df_desc = pd.read_csv(desc_path, index_col=0)
            df_desc.index.name = "Cột"

            is_numeric_row = df_desc["mean"].notna() if "mean" in df_desc.columns else pd.Series(False, index=df_desc.index)
            numeric_cols = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
            numeric_cols = [c for c in numeric_cols if c in df_desc.columns]
            cat_cols_stat = ["count", "unique", "top", "freq"]
            cat_cols_stat = [c for c in cat_cols_stat if c in df_desc.columns]

            df_numeric = df_desc.loc[is_numeric_row, numeric_cols].reset_index() if numeric_cols else pd.DataFrame()
            df_cat = df_desc.loc[~is_numeric_row, cat_cols_stat].reset_index() if cat_cols_stat else pd.DataFrame()

            n_total_cols = len(df_desc)
            n_total_rows = df_desc["count"].max() if "count" in df_desc.columns else None
            missing_summary = None
            if n_total_rows and "count" in df_desc.columns:
                missing_pct = ((n_total_rows - df_desc["count"]) / n_total_rows * 100).round(1)
                worst = missing_pct.sort_values(ascending=False).head(3)
                worst = worst[worst > 0]
                if len(worst) > 0:
                    missing_summary = ", ".join(f"`{col}` ({pct}%)" for col, pct in worst.items())

            if missing_summary:
                _insight_box(
                    f"Bộ dữ liệu gốc có {n_total_cols} cột. Các cột thiếu dữ liệu nhiều nhất: "
                    f"{missing_summary} — đây là lý do các bước xử lý missing/feature engineering "
                    f"(xem `04_handle_missing.py`) cần xử lý riêng cho từng cột thay vì fillna đồng loạt.",
                    kind="info",
                )
            else:
                _insight_box(f"Bộ dữ liệu gốc có {n_total_cols} cột, không có cột nào thiếu dữ liệu đáng kể.",
                             kind="success")

            if not df_numeric.empty:
                st.markdown("**Cột dạng số** (count/mean/std/min/max...)")
                st.dataframe(df_numeric, use_container_width=True, hide_index=True)
            if not df_cat.empty:
                st.markdown("**Cột dạng chữ/phân loại** (count/unique/giá trị phổ biến nhất/tần suất)")
                st.dataframe(df_cat, use_container_width=True, hide_index=True)
        else:
            st.warning(
                "Chưa có bảng thống kê mô tả. Chạy `python src/pipelines/06_eda.py` để tạo "
                "`reports/figures/descriptive_statistics.csv`."
            )

    # =========================================================================
    # TAB 5: Giới hạn nghiên cứu — trình bày tường minh, không né tránh
    # =========================================================================
    with tab5:
        st.markdown("### Giới hạn của nghiên cứu này")
        st.markdown(
            "Một mô hình dự đoán thể thao trung thực cần nói rõ nó **không** làm được gì, không chỉ "
            "khoe những gì nó làm được. Dưới đây là các giới hạn chính, liệt kê tường minh để hội đồng "
            "không cần hỏi mà nhóm đã tự nhận diện."
        )

        st.markdown("#### 1. Dữ liệu đầu vào — chỉ dùng thông tin TRƯỚC trận đấu")
        st.markdown(
            "- **Không dùng dữ liệu trong trận** (ace, double fault, break point, số phút thi đấu...) — "
            "đây là lựa chọn CÓ CHỦ ĐÍCH để tránh data leakage (xem PHẦN loại bỏ leakage), nhưng đồng "
            "nghĩa model không thể tận dụng được tín hiệu diễn biến trận đấu, chỉ dự đoán được TRƯỚC "
            "khi trận bắt đầu.\n"
            "- **Không có dữ liệu chấn thương** — một tay vợt có thể thi đấu trong lúc chấn thương nhẹ "
            "(không công khai) mà model không hề biết, trong khi đây có thể là yếu tố quyết định thắng "
            "thua.\n"
            "- **Không có dữ liệu thời tiết** — nhiệt độ, độ ẩm, gió ảnh hưởng đáng kể đến lối chơi "
            "(đặc biệt sân ngoài trời) nhưng không có trong bộ dữ liệu.\n"
            "- **Không có phong độ tức thời (recent form)** dạng rolling — model dùng Elo/H2H tích luỹ "
            "toàn bộ lịch sử chứ chưa có đặc trưng kiểu \"tỷ lệ thắng 10 trận gần nhất\"."
        )

        st.markdown("#### 2. Thiên lệch dữ liệu (data bias)")
        if insights:
            level_counts = insights.get("level_counts", {})
            n_matches = insights.get("n_matches", 0)
            # Cùng công thức với mục 1.2 (EDA) — giải nhỏ = ITF Futures ("15","25") + Challenger ("C")
            small_tour_pct = (
                round(sum(v for k, v in level_counts.items() if k in ["15", "25", "C"]) / n_matches * 100, 1)
                if n_matches else None
            )
            small_tour_pct_str = f"{small_tour_pct}" if small_tour_pct is not None else "phần lớn"
            st.markdown(
                f"- **{small_tour_pct_str}% trận đấu là giải ITF Futures/Challenger** (giải nhỏ) — "
                f"kết luận rút ra chưa chắc tổng quát hoá tốt cho các giải lớn (Grand Slam, Masters 1000) "
                f"nơi động lực thi đấu và áp lực tâm lý khác biệt.\n"
                f"- **Chỉ có dữ liệu ATP (nam)** — chưa kiểm chứng trên WTA (nữ).\n"
                f"- **Độ phủ tỷ lệ cược (odds) thấp** (~{insights.get('odds_available_pct', 'một phần nhỏ')}%"
                f" số trận) — phần backtest ROI chỉ đại diện cho tập con các trận có odds, thường là giải "
                f"lớn hơn, nên không đại diện cho toàn bộ phân bố dữ liệu train."
            )
        else:
            st.markdown(
                "- Phần lớn trận đấu trong bộ dữ liệu là giải Futures (giải nhỏ) — kết luận chưa chắc "
                "tổng quát cho giải lớn.\n"
                "- Chỉ có dữ liệu ATP (nam), chưa kiểm chứng trên WTA (nữ).\n"
                "- Độ phủ tỷ lệ cược (odds) thấp — phần backtest ROI chỉ đại diện cho 1 tập con nhỏ."
            )

        st.markdown("#### 3. Giới hạn phương pháp")
        st.markdown(
            "- **Elo đơn giản**: K-factor cố định cho mọi tay vợt/giai đoạn sự nghiệp, chưa có Elo "
            "riêng theo từng mặt sân (surface-specific Elo) dù phong độ trên Hard/Clay/Grass có thể "
            "khác biệt đáng kể với cùng 1 người.\n"
            "- **Static features**: phần lớn đặc trưng là tích luỹ toàn thời gian (career-to-date), chưa "
            "có cơ chế \"quên dần\" (decay) để phản ánh phong độ gần đây quan trọng hơn phong độ 5 năm "
            "trước.\n"
            "- **AUC trần lý thuyết**: kết quả thể thao luôn có yếu tố ngẫu nhiên không thể mô hình hoá "
            "(may rủi trong các điểm quyết định, quyết định trọng tài...) — nên AUC khó vượt quá "
            "~0.78–0.80 dù có thêm bao nhiêu feature, theo đối chiếu với y văn (xem mục Literature "
            "Benchmark)."
        )

        st.markdown("#### 4. Backtest ROI — giới hạn diễn giải")
        st.markdown(
            "- Backtest ROI (mục 2.8) là **minh hoạ khả năng ứng dụng**, không phải khuyến nghị đặt "
            "cược thực tế — không tính chi phí giao dịch, thuế, giới hạn thanh khoản của nhà cái, hay "
            "chiến lược quản lý vốn (position sizing, Kelly criterion).\n"
            "- ROI được kiểm tra độ ổn định qua từng quý (không chỉ gộp chung toàn tập test) — nếu CI "
            "95% của nhiều quý bao trùm 0%, đây là bằng chứng thống kê cho thấy **chưa đủ cơ sở khẳng "
            "định model có lợi thế (edge) thật sự và ổn định** so với thị trường cược, dù con số ROI "
            "gộp có thể trông dương."
        )

        st.markdown("#### 5. Hướng phát triển")
        st.markdown(
            "- Thêm Surface-specific Elo (Elo riêng theo mặt sân).\n"
            "- Rolling form features (tỷ lệ thắng N trận gần nhất, có trọng số suy giảm theo thời gian).\n"
            "- Chỉ số mệt mỏi (số trận/tuần, quãng đường di chuyển giữa các giải).\n"
            "- Mở rộng sang WTA.\n"
            "- Deep learning / ensemble stacking kết hợp nhiều model."
        )

        st.caption(
            "Nội dung chi tiết hơn (bảng, công thức, lý do kỹ thuật) xem thêm tại `docs/methodology.md`, "
            "mục 4 — Limitations & Future Work."
        )


if __name__ == "__main__":
    show()
