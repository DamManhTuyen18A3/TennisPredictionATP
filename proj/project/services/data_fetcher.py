"""
Data Fetcher Service
=====================

TRƯỚC ĐÂY (lỗi): hàm get_unique_players() đọc cột `df_clean['p1_name']`, nhưng
dữ liệu 05_clean_data.parquet thực tế dùng cột `winner_name` / `loser_name`
(cột p1/p2 chỉ xuất hiện ở data/features sau bước feature engineering, và
KHÔNG chứa tên - chỉ chứa số liệu). Vì cột không tồn tại, hàm luôn rơi vào
nhánh fallback với danh sách cứng 4 cái tên, khiến app chỉ hiển thị được
"Djokovic, Alcaraz, Sinner, Medvedev" bất kể dữ liệu có hơn 6000 tay vợt thật.

`prepare_input_features()` cũ cũng chỉ lấy NGẪU NHIÊN 1 dòng feature có sẵn
(`df_features.sample(1)`), hoàn toàn không liên quan tới 2 tay vợt được chọn.

File này được viết lại: dùng project.services.player_profiles để xây hồ sơ
tay vợt THẬT (Elo, rank, H2H...) và ghép feature vector THẬT theo đúng lựa
chọn của người dùng.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from project.services import player_profiles as pp


@st.cache_data(show_spinner="Đang tải dữ liệu trận đấu...")
def load_datasets():
    """Load dữ liệu thô và dữ liệu đã qua xử lý (cache trên RAM)."""
    base_dir = Path(__file__).resolve().parents[2] / "data"

    clean_data_path = base_dir / "processed" / "05_clean_data.parquet"
    features_data_path = base_dir / "features" / "08_selected_features.parquet"

    df_clean = pd.read_parquet(clean_data_path) if clean_data_path.exists() else pd.DataFrame()
    df_features = pd.read_parquet(features_data_path) if features_data_path.exists() else pd.DataFrame()

    return df_clean, df_features


@st.cache_data(show_spinner=False)
def load_player_profiles(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Hồ sơ THẬT của toàn bộ tay vợt trong dữ liệu (thay cho danh sách giả 4 tên)."""
    return pp.build_player_profiles(df_clean)


def get_unique_players(df_clean: pd.DataFrame):
    """Lấy danh sách TẤT CẢ các tay vợt thật để đổ vào Dropdown (đã sửa lỗi cột)."""
    if df_clean.empty:
        return []
    players = pd.concat(
        [df_clean.get("winner_name"), df_clean.get("loser_name")]
    ).dropna().unique()
    return sorted(players)


def get_player_recent_form(df_clean: pd.DataFrame, player_name: str, n_matches: int = 10):
    """Phong độ THẬT n trận gần nhất của một tay vợt (tính bằng winner_name/loser_name)."""
    if df_clean.empty:
        return {"win_rate": None, "matches": 0}

    matches = df_clean[
        (df_clean["winner_name"] == player_name) | (df_clean["loser_name"] == player_name)
    ].sort_values(by="tourney_date", ascending=False).head(n_matches)

    if len(matches) == 0:
        return {"win_rate": None, "matches": 0}

    wins = int((matches["winner_name"] == player_name).sum())
    return {"win_rate": wins / len(matches), "matches": len(matches)}


def prepare_input_features(
    feature_columns,
    profiles: pd.DataFrame,
    df_clean: pd.DataFrame,
    p1_name: str,
    p2_name: str,
    surface: str,
    tourney_level: str,
    round_: str,
    best_of: int,
    match_date,
):
    """Ghép 1 dòng feature THẬT từ hồ sơ 2 tay vợt được chọn, thay cho việc lấy
    1 dòng ngẫu nhiên như trước đây."""
    return pp.build_match_features(
        feature_columns, profiles, df_clean,
        p1_name, p2_name, surface, tourney_level, round_, best_of, match_date,
    )
