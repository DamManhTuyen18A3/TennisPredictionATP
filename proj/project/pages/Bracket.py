"""
Tournament Bracket Simulator (TÍNH NĂNG MỚI)
================================================

Đây là tính năng có giá trị ứng dụng thực tế rõ rệt nhất: thay vì chỉ dự
đoán 1 trận đơn lẻ, trang này mô phỏng CẢ MỘT NHÁNH ĐẤU (bracket) — đúng như
cách các đài truyền hình thể thao / nền tảng cá cược thể thao dự đoán nhà vô
địch trước giải. Dùng phương pháp Monte Carlo: chạy lại toàn bộ giải đấu
hàng trăm lần, mỗi lần các trận được "tung xúc xắc" theo đúng xác suất mô
hình đã dự đoán, rồi tổng hợp tần suất vô địch / vào chung kết của từng tay
vợt.
"""

import random

import pandas as pd
import plotly.express as px
import streamlit as st

from project.services.data_fetcher import load_datasets, load_player_profiles, get_unique_players, prepare_input_features
from project.services.inference import get_prediction_engine
from project.services.player_profiles import TOURNEY_LEVEL_LABELS
from project.utils.theme import ColorPalette, ChartTheme

BRACKET_SIZES = [4, 8, 16]


def _simulate_round(pairs, engine, feature_columns, profiles, df_clean, surface, level_code, round_name, best_of, match_date, rng):
    """Chạy 1 vòng đấu: trả về (winners, round_log) — round_log để hiển thị minh hoạ."""
    winners = []
    log = []
    for p1_name, p2_name in pairs:
        if p2_name is None:  # bye (không xảy ra vì bracket_size luôn là luỹ thừa của 2)
            winners.append(p1_name)
            continue
        X_row, h2h, p1, p2 = prepare_input_features(
            feature_columns, profiles, df_clean, p1_name, p2_name,
            surface, level_code, round_name, best_of, match_date,
        )
        prob_a = float(engine.predict(X_row)[0])
        winner = p1_name if rng.random() < prob_a else p2_name
        winners.append(winner)
        log.append({"Trận": f"{p1_name} vs {p2_name}", f"Tỉ lệ thắng {p1_name}": f"{prob_a:.1%}", "Người thắng": winner})
    return winners, log


def show():
    st.title("🏆 Tournament Bracket Simulator")
    st.markdown(
        "Mô phỏng **cả một giải đấu** bằng phương pháp Monte Carlo: mô hình dự đoán xác suất "
        "từng trận, sau đó \"tung xúc xắc\" theo đúng xác suất đó hàng trăm lần để ước tính "
        "khả năng vô địch của mỗi tay vợt — cách tiếp cận dùng thực tế bởi các nền tảng phân "
        "tích thể thao (vd. FiveThirtyEight, Opta)."
    )

    df_clean, _ = load_datasets()
    if df_clean.empty:
        st.error("Không tải được dữ liệu.")
        return

    profiles = load_player_profiles(df_clean)
    player_names = get_unique_players(df_clean)
    engine = get_prediction_engine()
    if engine.model is None:
        st.error("Không có model để mô phỏng.")
        return
    feature_columns = engine.model.feature_names_

    top_by_elo = profiles.sort_values("elo", ascending=False)["player_name"].tolist()

    with st.form("bracket_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            bracket_size = st.selectbox("Số người trong nhánh đấu", BRACKET_SIZES, index=1)
        with col2:
            surface = st.selectbox("Mặt sân", ["Hard", "Clay", "Grass", "Carpet"])
        with col3:
            level_code = st.selectbox(
                "Cấp độ giải", list(TOURNEY_LEVEL_LABELS.keys()),
                format_func=lambda c: TOURNEY_LEVEL_LABELS[c], index=0,
            )

        default_players = top_by_elo[:bracket_size]
        selected_players = st.multiselect(
            f"Chọn đúng {bracket_size} tay vợt (mặc định: top Elo cao nhất hiện có trong dữ liệu)",
            player_names, default=default_players,
        )

        n_sims = st.slider("Số lần mô phỏng Monte Carlo", 50, 500, 150, step=50)

        run = st.form_submit_button("🎲 Chạy mô phỏng giải đấu", use_container_width=True)

    if not run:
        return

    if len(selected_players) != bracket_size:
        st.error(f"Vui lòng chọn đúng {bracket_size} tay vợt (đang chọn {len(selected_players)}).")
        return
    if len(set(selected_players)) != len(selected_players):
        st.error("Có tay vợt bị chọn trùng, vui lòng chọn lại.")
        return

    match_date = pd.Timestamp.today()
    round_names_by_size = {4: ["SF", "F"], 8: ["QF", "SF", "F"], 16: ["R16", "QF", "SF", "F"]}
    round_names = round_names_by_size[bracket_size]

    # --- 1) Minh hoạ 1 lần chạy chi tiết ---
    rng_demo = random.Random(42)
    st.markdown("### 📋 Minh hoạ chi tiết 1 lượt mô phỏng")
    current_round_players = list(selected_players)
    with st.spinner("Đang mô phỏng minh hoạ..."):
        for rname in round_names:
            pairs = [(current_round_players[i], current_round_players[i + 1])
                     for i in range(0, len(current_round_players), 2)]
            winners, log = _simulate_round(
                pairs, engine, feature_columns, profiles, df_clean,
                surface, level_code, rname, 3, match_date, rng_demo,
            )
            with st.expander(f"Vòng {rname} ({len(pairs)} trận)", expanded=(rname == round_names[-1])):
                st.dataframe(pd.DataFrame(log), use_container_width=True, hide_index=True)
            current_round_players = winners
    st.success(f"🏆 Nhà vô địch (lượt minh hoạ này): **{current_round_players[0]}**")

    # --- 2) Monte Carlo: chạy nhiều lần để tổng hợp xác suất vô địch ---
    st.markdown(f"### 🎲 Kết quả tổng hợp sau {n_sims} lần mô phỏng Monte Carlo")
    champion_counts = {p: 0 for p in selected_players}

    progress = st.progress(0, text="Đang chạy mô phỏng...")
    rng = random.Random()
    for sim_i in range(n_sims):
        current_round_players = list(selected_players)
        for rname in round_names:
            pairs = [(current_round_players[i], current_round_players[i + 1])
                     for i in range(0, len(current_round_players), 2)]
            winners, _ = _simulate_round(
                pairs, engine, feature_columns, profiles, df_clean,
                surface, level_code, rname, 3, match_date, rng,
            )
            current_round_players = winners
        champion_counts[current_round_players[0]] += 1
        if sim_i % max(1, n_sims // 20) == 0:
            progress.progress(min(1.0, (sim_i + 1) / n_sims), text=f"Đang chạy mô phỏng... {sim_i+1}/{n_sims}")
    progress.progress(1.0, text="Hoàn tất!")

    result_df = pd.DataFrame({
        "Tay vợt": list(champion_counts.keys()),
        "Xác suất vô địch": [v / n_sims for v in champion_counts.values()],
    }).sort_values("Xác suất vô địch", ascending=False)

    palette = ColorPalette()
    fig = px.bar(result_df, x="Xác suất vô địch", y="Tay vợt", orientation="h")
    fig.update_traces(marker_color=palette.SECONDARY)
    fig.update_layout(
        **ChartTheme.layout_defaults(palette),
        xaxis=dict(tickformat=".0%", **ChartTheme.axis_defaults(palette)),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.dataframe(
        result_df.assign(**{"Xác suất vô địch": (result_df["Xác suất vô địch"] * 100).round(1).astype(str) + "%"}),
        use_container_width=True, hide_index=True,
    )

    st.caption(
        f"Mỗi trận trong mỗi lượt mô phỏng được quyết định bằng cách lấy mẫu ngẫu nhiên theo đúng "
        f"xác suất mà model CatBoost dự đoán cho cặp đấu đó (không phải luôn chọn kèo cao hơn), "
        f"nên chạy {n_sims} lần cho ra phân phối xác suất vô địch thực tế hơn — đây chính là "
        f"phương pháp Monte Carlo simulation dùng phổ biến trong phân tích thể thao thực tế."
    )


if __name__ == "__main__":
    show()
