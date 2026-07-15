"""
Explain Card Component
======================

BẢN CŨ: hardcode cứng "phong độ 80% vs 60%", "Elo +85", "H2H 7-3", "mệt hơn 45
phút/trận" — giống nhau cho MỌI cặp đấu, không liên quan gì tới dữ liệu thật.

BẢN MỚI: nhận đúng các con số đã tính thật (Elo, H2H, rank, SHAP) cho cặp đấu
đang xét và hiển thị lại, không tự bịa thêm chi tiết nào.
"""

import streamlit as st


FEATURE_LABELS_VI = {
    "elo_diff": "Chênh lệch Elo rating",
    "rank_diff": "Chênh lệch thứ hạng ATP",
    "rank_points_diff": "Chênh lệch điểm ATP",
    "age_diff": "Chênh lệch tuổi",
    "ht_diff": "Chênh lệch chiều cao",
    "h2h_diff": "Chênh lệch đối đầu (H2H)",
    "p1_elo": "Elo của Player A",
    "p2_elo": "Elo của Player B",
    "p1_rank": "Thứ hạng Player A",
    "p2_rank": "Thứ hạng Player B",
    "p1_entry": "Diện tham dự Player A",
    "p2_entry": "Diện tham dự Player B",
}


def render_explain_card(winner: str, prob: float, shap_top: dict, h2h: dict, elo_diff: float):
    """Hiển thị khối giải thích XAI dựa trên SHAP values THẬT của đúng dự đoán này."""
    st.markdown("### 🧠 Vì sao mô hình dự đoán như vậy?")

    st.info(
        f"Dựa trên SHAP values (Shapley Additive Explanations) tính trực tiếp trên "
        f"mô hình CatBoost cho đúng cặp đấu này, mô hình nghiêng về **{winner}** "
        f"với xác suất {prob:.1f}%."
    )

    if shap_top:
        st.markdown("**Các yếu tố ảnh hưởng nhiều nhất đến dự đoán này (SHAP values thật):**")
        for feat, val in shap_top.items():
            label = FEATURE_LABELS_VI.get(feat, feat)
            direction = "🟢 đẩy xác suất Player A thắng lên" if val > 0 else "🔴 kéo xác suất Player A thắng xuống"
            st.markdown(f"- **{label}** (`{feat}`): SHAP = `{val:+.3f}` — {direction}")
    else:
        st.warning("Không tính được SHAP values cho trận này (thiếu thư viện `shap` hoặc lỗi mô hình).")

    st.markdown(
        f"**Lịch sử đối đầu (H2H) thật:** {h2h.get('a_wins', 0)} - {h2h.get('b_wins', 0)} "
        f"(tổng {h2h.get('total', 0)} trận trong dữ liệu)  \n"
        f"**Chênh lệch Elo rating thật:** {elo_diff:+.1f} điểm"
    )

    st.caption(
        "Lưu ý: SHAP values thể hiện đóng góp của từng đặc trưng vào đầu ra mô hình cho "
        "riêng trận đấu này, không phải feature importance tổng thể toàn bộ dữ liệu."
    )
