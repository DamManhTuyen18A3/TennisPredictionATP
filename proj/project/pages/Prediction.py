"""
Prediction Page — PHIÊN BẢN SỬA LỖI
=====================================

BẢN CŨ (lỗi nghiêm trọng):
- Dropdown chỉ có 4 tên cầu thủ cứng (do lỗi ở data_fetcher.get_unique_players).
- Nhấn "Predict" chỉ time.sleep(1.0) rồi gán cứng `prob_a = 62.5` — KHÔNG hề
  gọi tới model đã train, bất kể chọn ai kết quả luôn là 62.5% / 37.5%.
- H2H "4-2", Surface Win Rate "78% vs 65%", Elo "+125" đều là text cứng,
  giống nhau cho MỌI cặp đấu.
- Radar chart "Serve/Return/Forehand..." và SHAP bar chart là số bịa.
- "Top Similar Matches" là 2 dòng dữ liệu viết tay.

BẢN MỚI: chọn từ hơn 6.000 tay vợt thật trong dữ liệu, ghép feature vector
thật từ hồ sơ (Elo/rank/H2H tính từ lịch sử), gọi đúng model CatBoost đã
train, và giải thích bằng SHAP values thật cho từng dự đoán.
"""

import pandas as pd
import streamlit as st

from project.components.prediction_card import render_prediction_card
from project.components.insight_charts import render_comparison_radar, render_shap_bar
from project.components.explain_card import render_explain_card
from project.services.data_fetcher import (
    load_datasets, load_player_profiles, get_unique_players, prepare_input_features,
)
from project.services.player_profiles import (
    TOURNEY_LEVEL_LABELS, ROUND_LABELS, find_similar_historical_matches,
)
from project.services.inference import get_prediction_engine


def show():
    st.title("🎯 Prediction Dashboard")
    st.markdown("Chọn 2 tay vợt thật từ dữ liệu ATP để nhận dự đoán từ mô hình CatBoost đã huấn luyện.")

    df_clean, _df_features = load_datasets()
    if df_clean.empty:
        st.error(
            "Không tìm thấy `data/processed/05_clean_data.parquet`. "
            "Hãy chạy pipeline (`src/pipelines/`) trước khi chạy app."
        )
        return

    profiles = load_player_profiles(df_clean)
    player_names = get_unique_players(df_clean)

    if not player_names:
        st.error("Không tìm thấy tên tay vợt nào trong dữ liệu.")
        return

    engine = get_prediction_engine()
    if engine.model is None:
        st.warning(
            "Không tìm thấy model đã train ở `models/`. Kết quả bên dưới sẽ không chính xác "
            "cho tới khi có file model hợp lệ."
        )

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            player_a = st.selectbox(f"Player A ({len(player_names)} tay vợt)", player_names, index=0)
        with col2:
            default_b_idx = 1 if len(player_names) > 1 else 0
            player_b = st.selectbox("Player B", player_names, index=default_b_idx)

        col3, col4, col5, col6 = st.columns(4)
        with col3:
            surface = st.selectbox("Surface", ["Hard", "Clay", "Grass", "Carpet"])
        with col4:
            level_code = st.selectbox(
                "Tournament Level", list(TOURNEY_LEVEL_LABELS.keys()),
                format_func=lambda c: TOURNEY_LEVEL_LABELS[c], index=0,
            )
        with col5:
            round_options = list(ROUND_LABELS.keys())
            round_code = st.selectbox(
                "Round", round_options,
                format_func=lambda c: f"{c} - {ROUND_LABELS[c]}",
                index=round_options.index("QF"),
            )
        with col6:
            best_of = st.selectbox("Best of", [3, 5], index=0)

        match_date = st.date_input("Ngày thi đấu (dự kiến)", value=pd.Timestamp.today())

        submitted = st.form_submit_button("🔮 Predict Match", use_container_width=True)

    if submitted:
        if player_a == player_b:
            st.error("Vui lòng chọn 2 tay vợt khác nhau!")
            return

        with st.spinner("Đang chạy model CatBoost thật trên feature vector vừa ghép..."):
            try:
                feature_columns = engine.model.feature_names_ if engine.model is not None else None
                if feature_columns is None:
                    st.error("Không có model để dự đoán. Vui lòng kiểm tra thư mục `models/`.")
                    return

                X_row, h2h, p1, p2 = prepare_input_features(
                    feature_columns, profiles, df_clean,
                    player_a, player_b, surface, level_code, round_code, best_of, match_date,
                )
                proba_a = float(engine.predict(X_row)[0])
                shap_top = engine.get_shap_explanation(X_row, top_n=8)
                consensus = engine.get_multi_model_consensus(X_row)
            except Exception as e:
                st.error(f"Lỗi khi ghép feature / dự đoán: {e}")
                return

        # LƯU VÀO session_state: tính năng dự đoán (model + SHAP + đồng thuận đa
        # mô hình) chỉ chạy MỘT LẦN khi bấm nút. Các tương tác khác trên trang
        # (vd. gõ số vào máy tính kèo cược ở tab 5) sẽ kích hoạt Streamlit chạy
        # lại toàn bộ script — nếu không lưu kết quả, kết quả dự đoán sẽ biến
        # mất và toàn bộ pipeline (bao gồm cả 4 mô hình) bị tính lại không cần
        # thiết mỗi lần gõ phím.
        st.session_state["last_prediction"] = {
            "player_a": player_a, "player_b": player_b,
            "surface": surface, "X_row": X_row, "h2h": h2h, "p1": p1, "p2": p2,
            "proba_a": proba_a, "shap_top": shap_top, "consensus": consensus,
        }

    if "last_prediction" not in st.session_state:
        return

    pred = st.session_state["last_prediction"]
    player_a, player_b = pred["player_a"], pred["player_b"]
    surface = pred["surface"]
    X_row, h2h, p1, p2 = pred["X_row"], pred["h2h"], pred["p1"], pred["p2"]
    proba_a, shap_top, consensus = pred["proba_a"], pred["shap_top"], pred["consensus"]

    prob_a_pct = proba_a * 100
    prob_b_pct = 100 - prob_a_pct
    winner = player_a if prob_a_pct > prob_b_pct else player_b

    st.success("Dự đoán hoàn tất (dùng model + dữ liệu thật)!")

    # Cảnh báo độ mới dữ liệu (TÍNH NĂNG MỚI) — tăng tính trung thực khoa học:
    # nếu 1 trong 2 tay vợt lâu không thi đấu, hồ sơ (tuổi/rank/Elo) được dùng
    # để dự đoán có thể đã lỗi thời (vd. đã giải nghệ).
    today = pd.Timestamp.today()
    for label, prof in [(player_a, p1), (player_b, p2)]:
        last_date = prof.get("last_match_date")
        if pd.notna(last_date):
            days_stale = (today - pd.Timestamp(last_date)).days
            if days_stale > 180:
                st.warning(
                    f"⚠️ **{label}** không có trận nào trong dữ liệu suốt {days_stale} ngày gần đây "
                    f"(trận cuối: {pd.Timestamp(last_date).date()}). Hồ sơ dùng để dự đoán có thể "
                    f"không còn phản ánh đúng phong độ/tuổi hiện tại (vd. có thể đã giải nghệ)."
                )

    render_prediction_card(player_a, player_b, prob_a_pct, prob_b_pct, winner)

    st.divider()

    elo_diff = float(p1.get("elo", 0) or 0) - float(p2.get("elo", 0) or 0)
    render_explain_card(winner, max(prob_a_pct, prob_b_pct), shap_top, h2h, elo_diff)

    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Hồ sơ & H2H", "🧠 AI Explainability", "📉 Trận tương tự trong lịch sử",
        "🤝 Đồng thuận đa mô hình", "🎰 Phân tích kèo cược",
    ])

    with tab1:
        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            st.markdown("#### 📡 So sánh hồ sơ 2 tay vợt (số liệu thật)")
            render_comparison_radar(player_a, player_b, p1, p2)
        with stat_col2:
            st.markdown("#### ⚡ H2H & Hồ sơ")
            st.metric(label=f"H2H {player_a} vs {player_b}", value=f"{h2h['a_wins']} - {h2h['b_wins']}",
                      delta=f"Tổng {h2h['total']} trận trong dữ liệu")
            rank_a = p1.get("rank")
            rank_b = p2.get("rank")
            rank_a_str = f"#{int(rank_a)}" if pd.notna(rank_a) else "—"
            rank_b_str = f"#{int(rank_b)}" if pd.notna(rank_b) else "—"
            st.metric(label="Thứ hạng ATP gần nhất trong dữ liệu", value=f"{rank_a_str} vs {rank_b_str}")
            st.metric(label="Elo Rating Difference", value=f"{elo_diff:+.0f} Elo")

    with tab2:
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.markdown("#### 🎯 AI Confidence")
            from project.components.gauge import render_gauge_chart
            render_gauge_chart(max(prob_a_pct, prob_b_pct))
        with exp_col2:
            st.markdown("#### 🔍 SHAP Feature Importance (thật, cho đúng trận này)")
            render_shap_bar(shap_top)

    with tab3:
        st.markdown("#### 🎾 Các trận lịch sử có chênh lệch thứ hạng & mặt sân tương tự")
        rank_diff = X_row["rank_diff"].iloc[0]
        similar = find_similar_historical_matches(df_clean, surface, rank_diff, n=5)
        if similar.empty:
            st.info("Không tìm thấy trận lịch sử tương tự để đối chiếu.")
        else:
            st.dataframe(similar, use_container_width=True)
            st.caption(
                "Đây là các trận THẬT trong dữ liệu có mặt sân và chênh lệch thứ hạng gần "
                "giống trận đang dự đoán — dùng để đối chiếu trực quan, không phải mô phỏng."
            )

        st.markdown("#### 📋 Model đang sử dụng")
        st.caption(f"`{engine.model_name}` — xem chỉ số đánh giá đầy đủ ở trang Thống kê & Phân tích.")

    with tab4:
        st.markdown("#### 🤝 So sánh dự đoán giữa 4 thuật toán khác nhau (TÍNH NĂNG MỚI)")
        st.caption(
            "Thay vì chỉ tin vào 1 mô hình, so sánh dự đoán của CatBoost, LightGBM, XGBoost và "
            "Random Forest cho ĐÚNG cặp đấu này giúp đánh giá mức độ đồng thuận — nếu các mô "
            "hình lệch nhau nhiều, kết quả nên được diễn giải thận trọng hơn."
        )
        # consensus đã được tính 1 lần duy nhất lúc bấm nút Predict (lưu trong
        # session_state) — không gọi lại engine.get_multi_model_consensus() ở
        # đây để tránh tính toán lại không cần thiết mỗi khi trang render lại.
        valid_results = {k: v for k, v in consensus.items() if v is not None}
        if valid_results:
            cons_df = pd.DataFrame({
                "Mô hình": list(valid_results.keys()),
                f"Xác suất {player_a} thắng": list(valid_results.values()),
            }).sort_values(f"Xác suất {player_a} thắng", ascending=False).set_index("Mô hình")

            st.bar_chart(cons_df, use_container_width=True)

            vals = list(valid_results.values())
            spread = (max(vals) - min(vals)) * 100
            if spread < 10:
                st.success(f"✅ Các mô hình khá đồng thuận (chênh lệch chỉ {spread:.1f} điểm %).")
            elif spread < 25:
                st.info(f"ℹ️ Các mô hình đồng thuận vừa phải (chênh lệch {spread:.1f} điểm %).")
            else:
                st.warning(f"⚠️ Các mô hình KHÔNG đồng thuận (chênh lệch {spread:.1f} điểm %) — nên diễn giải kết quả thận trọng.")

        skipped = [k for k, v in consensus.items() if v is None]
        if skipped:
            st.caption(f"Không khả dụng cho trận này: {', '.join(skipped)}.")

    with tab5:
        st.markdown("#### 🎰 Phân tích kèo cược (Value Betting Analysis)")
        st.info(
            "**Mục đích nghiên cứu/giáo dục**: so sánh xác suất mô hình dự đoán với tỷ lệ cược "
            "thị trường để minh hoạ khái niệm \"value bet\" trong phân tích thể thao định lượng. "
            "Đây KHÔNG phải lời khuyên tài chính hay khuyến khích cá cược — cá cược tiềm ẩn rủi "
            "ro tài chính thực sự."
        )
        odds_col1, odds_col2 = st.columns(2)
        with odds_col1:
            odds_a = st.number_input(f"Tỷ lệ cược thập phân (decimal odds) cho {player_a}",
                                      min_value=1.01, value=2.00, step=0.05)
        with odds_col2:
            odds_b = st.number_input(f"Tỷ lệ cược thập phân (decimal odds) cho {player_b}",
                                      min_value=1.01, value=2.00, step=0.05)

        implied_a_raw = 1 / odds_a
        implied_b_raw = 1 / odds_b
        overround = implied_a_raw + implied_b_raw
        implied_a = implied_a_raw / overround
        implied_b = implied_b_raw / overround

        edge_a = (proba_a - implied_a) * 100
        edge_b = ((1 - proba_a) - implied_b) * 100

        c1, c2 = st.columns(2)
        with c1:
            st.metric(f"Xác suất ngụ ý thị trường — {player_a}", f"{implied_a:.1%}",
                      delta=f"Model edge: {edge_a:+.1f} điểm %")
        with c2:
            st.metric(f"Xác suất ngụ ý thị trường — {player_b}", f"{implied_b:.1%}",
                      delta=f"Model edge: {edge_b:+.1f} điểm %")

        st.caption(
            f"Biên lợi thế nhà cái (overround/vig) trong tỷ lệ cược bạn nhập: {(overround - 1) * 100:.1f}%. "
            "\"Model edge\" dương nghĩa là mô hình đánh giá xác suất thắng cao hơn thị trường ngụ ý — "
            "đây là khái niệm thống kê thuần tuý để minh hoạ phương pháp, không phải khuyến nghị đặt cược."
        )


if __name__ == "__main__":
    show()
