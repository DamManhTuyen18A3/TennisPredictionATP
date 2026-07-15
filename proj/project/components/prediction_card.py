import streamlit as st

from project.utils.theme import ColorPalette


def render_prediction_card(player_a, player_b, prob_a, prob_b, winner):
    """Render prediction result card (đồng bộ Design System)."""
    palette = ColorPalette()

    st.markdown("### 🏆 Prediction Result")
    card_col1, card_col2, card_col3 = st.columns([2, 1, 2])

    winner_color = palette.SUCCESS
    neutral_text = palette.TEXT_TERTIARY

    with card_col1:
        name_color = winner_color if player_a == winner else neutral_text
        st.markdown(
            f"<h3 style='text-align: center; color: {name_color};'>{player_a}</h3>",
            unsafe_allow_html=True,
        )
        st.markdown(f"<h1 style='text-align: center;'>{prob_a:.1f}%</h1>", unsafe_allow_html=True)

    with card_col2:
        st.markdown(
            f"<h2 style='text-align: center; color: {neutral_text}; margin-top: 20px;'>VS</h2>",
            unsafe_allow_html=True,
        )

    with card_col3:
        name_color = winner_color if player_b == winner else neutral_text
        st.markdown(
            f"<h3 style='text-align: center; color: {name_color};'>{player_b}</h3>",
            unsafe_allow_html=True,
        )
        st.markdown(f"<h1 style='text-align: center;'>{prob_b:.1f}%</h1>", unsafe_allow_html=True)

    st.progress(prob_a / 100.0)

