"""
Player Profiles Service
=======================

Xây dựng hồ sơ THỰC cho từng tay vợt từ toàn bộ lịch sử trận đấu (05_clean_data),
thay cho các con số/danh sách giả trước đây.

- Elo rating: tính TUẦN TỰ theo thời gian, dùng đúng công thức trong
  `src/pipelines/07_feature_engineering.py::calculate_elo` (không leakage: giá trị
  Elo lưu lại luôn là Elo TRƯỚC trận đấu tương ứng).
- Rank / rank_points / chiều cao / tay thuận / quốc tịch / tuổi: lấy từ trận gần
  nhất của mỗi tay vợt trong dữ liệu (rồi ngoại suy tuổi tới ngày trận giả định).
- H2H: đếm trực tiếp trên toàn bộ lịch sử đối đầu giữa 2 ID cầu thủ.

Tất cả đều cache bằng st.cache_data để chỉ tính một lần mỗi phiên làm việc.
"""

from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Điểm draw_size trung vị theo tourney_level, tính từ chính dữ liệu 05_clean_data
DRAW_SIZE_BY_LEVEL = {
    "G": 128,  # Grand Slam
    "M": 96,   # Masters 1000
    "O": 64,   # Olympics
    "A": 32,   # ATP Tour (250/500)
    "C": 32,   # Challenger
    "15": 32,  # ITF 15k
    "25": 32,  # ITF 25k
    "F": 8,    # Tour Finals
    "D": 4,    # Davis Cup
}

TOURNEY_LEVEL_LABELS = {
    "G": "Grand Slam",
    "M": "Masters 1000",
    "A": "ATP Tour (250/500)",
    "C": "Challenger",
    "F": "Tour Finals",
    "O": "Olympics",
    "D": "Davis Cup",
    "15": "ITF 15k",
    "25": "ITF 25k",
}

ROUND_LABELS = {
    "R128": "Vòng 1/64", "R64": "Vòng 1/32", "R32": "Vòng 1/16",
    "R16": "Vòng 1/8", "QF": "Tứ kết", "SF": "Bán kết", "F": "Chung kết",
    "RR": "Vòng bảng", "Q1": "Vòng loại 1", "Q2": "Vòng loại 2",
    "Q3": "Vòng loại 3", "BR": "Tranh hạng 3",
}


@st.cache_data(show_spinner="Đang xây dựng hồ sơ tay vợt từ dữ liệu lịch sử...")
def build_player_profiles(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Trả về DataFrame hồ sơ mới nhất của mỗi tay vợt (1 dòng / player_id)."""
    if df_clean.empty or "winner_id" not in df_clean.columns:
        return pd.DataFrame()

    df = df_clean.sort_values("tourney_date").reset_index(drop=True)

    # --- Elo tuần tự (giống hệt pipeline 07_feature_engineering.py) ---
    k, base_rating = 32, 1500.0
    elo = defaultdict(lambda: base_rating)
    for w_id, l_id in zip(df["winner_id"], df["loser_id"]):
        w_elo, l_elo = elo[w_id], elo[l_id]
        expected_w = 1 / (1 + 10 ** ((l_elo - w_elo) / 400))
        elo[w_id] = w_elo + k * (1 - expected_w)
        elo[l_id] = l_elo + k * (0 - (1 - expected_w))
    final_elo: Dict[int, float] = dict(elo)

    # --- Gộp winner/loser thành 1 bảng dài (mỗi trận sinh 2 dòng: 1 cho mỗi VĐV) ---
    long_frames = []
    for side in ["winner", "loser"]:
        cols = {
            f"{side}_id": "player_id",
            f"{side}_name": "player_name",
            f"{side}_rank": "rank",
            f"{side}_rank_points": "rank_points",
            f"{side}_age": "age_at_match",
            f"{side}_ht": "ht",
            f"{side}_hand": "hand",
            f"{side}_ioc": "ioc",
            "tourney_date": "tourney_date",
        }
        available = {k: v for k, v in cols.items() if k in df.columns}
        sub = df[list(available.keys())].rename(columns=available)
        long_frames.append(sub)

    long_df = pd.concat(long_frames, ignore_index=True)
    long_df = long_df.dropna(subset=["player_id"]).sort_values("tourney_date")

    match_counts = long_df.groupby("player_id").size().rename("matches_played")

    latest = long_df.groupby("player_id").last().reset_index()
    latest = latest.merge(match_counts, on="player_id", how="left")
    latest["elo"] = latest["player_id"].map(final_elo).fillna(base_rating)
    latest = latest.rename(
        columns={"tourney_date": "last_match_date", "age_at_match": "age_at_last_match"}
    )

    # Nếu 1 tên trùng nhiều player_id (hiếm), giữ hồ sơ có nhiều trận nhất
    latest = latest.sort_values("matches_played", ascending=False)
    latest = latest.drop_duplicates(subset=["player_name"], keep="first")

    return latest.reset_index(drop=True)


@st.cache_data(show_spinner="Đang tính lịch sử Elo theo thời gian...")
def build_elo_match_log(df_clean: pd.DataFrame) -> pd.DataFrame:
    """TÍNH NĂNG MỚI: trả về toàn bộ lịch sử trận đấu kèm Elo TRƯỚC mỗi trận
    (winner_elo_pre / loser_elo_pre), dùng để vẽ biểu đồ diễn biến phong độ
    (Elo trend) thật của từng tay vợt theo thời gian — một tính năng có giá
    trị thực tế cho HLV/nhà phân tích, không chỉ phục vụ riêng dự đoán."""
    if df_clean.empty or "winner_id" not in df_clean.columns:
        return pd.DataFrame()

    df = df_clean.sort_values("tourney_date").reset_index(drop=True)
    k, base_rating = 32, 1500.0
    elo = defaultdict(lambda: base_rating)
    w_elo_pre = np.empty(len(df))
    l_elo_pre = np.empty(len(df))
    for i, (w_id, l_id) in enumerate(zip(df["winner_id"], df["loser_id"])):
        w_elo, l_elo = elo[w_id], elo[l_id]
        w_elo_pre[i], l_elo_pre[i] = w_elo, l_elo
        expected_w = 1 / (1 + 10 ** ((l_elo - w_elo) / 400))
        elo[w_id] = w_elo + k * (1 - expected_w)
        elo[l_id] = l_elo + k * (0 - (1 - expected_w))

    df["winner_elo_pre"] = w_elo_pre
    df["loser_elo_pre"] = l_elo_pre
    return df


def get_player_timeline(elo_log: pd.DataFrame, player_id) -> pd.DataFrame:
    """Dòng thời gian THẬT của 1 tay vợt: ngày, Elo trước trận, đối thủ, kết
    quả, mặt sân, thứ hạng — dùng cho trang Player Profile."""
    if elo_log.empty:
        return pd.DataFrame()

    as_winner = elo_log[elo_log["winner_id"] == player_id].copy()
    as_winner["elo_pre"] = as_winner["winner_elo_pre"]
    as_winner["result"] = "Thắng"
    as_winner["opponent"] = as_winner["loser_name"]
    as_winner["rank_at_match"] = as_winner["winner_rank"]

    as_loser = elo_log[elo_log["loser_id"] == player_id].copy()
    as_loser["elo_pre"] = as_loser["loser_elo_pre"]
    as_loser["result"] = "Thua"
    as_loser["opponent"] = as_loser["winner_name"]
    as_loser["rank_at_match"] = as_loser["loser_rank"]

    cols = ["tourney_date", "tourney_name", "surface", "opponent", "result", "elo_pre", "rank_at_match"]
    timeline = pd.concat([as_winner[cols], as_loser[cols]], ignore_index=True)
    return timeline.sort_values("tourney_date").reset_index(drop=True)


def get_surface_breakdown(df_clean: pd.DataFrame, player_id) -> pd.DataFrame:
    """Tỷ lệ thắng THẬT theo từng mặt sân cho 1 tay vợt."""
    if df_clean.empty:
        return pd.DataFrame()
    matches = df_clean[
        (df_clean["winner_id"] == player_id) | (df_clean["loser_id"] == player_id)
    ]
    rows = []
    for surface, group in matches.groupby("surface"):
        wins = int((group["winner_id"] == player_id).sum())
        total = len(group)
        rows.append({"surface": surface, "wins": wins, "total": total,
                      "win_rate": wins / total if total else 0})
    return pd.DataFrame(rows).sort_values("total", ascending=False)



    """Danh sách tên đầy đủ, THỰC, để đổ vào dropdown (không còn danh sách giả 4 tên)."""
    if profiles.empty:
        return []
    return sorted(profiles["player_name"].dropna().unique().tolist())


def get_player_row(profiles: pd.DataFrame, player_name: str):
    match = profiles.loc[profiles["player_name"] == player_name]
    if match.empty:
        return None
    return match.iloc[0]


def get_h2h(df_clean: pd.DataFrame, id_a, id_b) -> Dict[str, int]:
    """Đếm lịch sử đối đầu THỰC giữa 2 player_id từ toàn bộ dữ liệu."""
    if df_clean.empty:
        return {"a_wins": 0, "b_wins": 0, "total": 0}
    a_win_mask = (df_clean["winner_id"] == id_a) & (df_clean["loser_id"] == id_b)
    b_win_mask = (df_clean["winner_id"] == id_b) & (df_clean["loser_id"] == id_a)
    a_wins, b_wins = int(a_win_mask.sum()), int(b_win_mask.sum())
    return {"a_wins": a_wins, "b_wins": b_wins, "total": a_wins + b_wins}


def get_recent_form(df_clean: pd.DataFrame, player_id, n: int = 10) -> Dict[str, float]:
    """Phong độ THỰC n trận gần nhất (tính từ toàn bộ lịch sử, không giả lập)."""
    if df_clean.empty:
        return {"win_rate": None, "matches": 0}
    matches = df_clean[
        (df_clean["winner_id"] == player_id) | (df_clean["loser_id"] == player_id)
    ].sort_values("tourney_date", ascending=False).head(n)
    if len(matches) == 0:
        return {"win_rate": None, "matches": 0}
    wins = int((matches["winner_id"] == player_id).sum())
    return {"win_rate": wins / len(matches), "matches": len(matches)}


def find_similar_historical_matches(
    df_clean: pd.DataFrame, surface: str, rank_diff: float, n: int = 5
) -> pd.DataFrame:
    """Tìm các trận THỰC trong lịch sử có mặt sân giống & chênh lệch rank tương tự,
    dùng để minh hoạ 'các trận tương tự' một cách trung thực (không bịa)."""
    if df_clean.empty or rank_diff is None or np.isnan(rank_diff):
        return pd.DataFrame()

    df = df_clean.copy()
    df["match_rank_diff"] = df["winner_rank"] - df["loser_rank"]
    same_surface = df[df["surface"].str.lower() == str(surface).lower()]
    if same_surface.empty:
        same_surface = df

    same_surface = same_surface.dropna(subset=["match_rank_diff"])
    same_surface["gap_score"] = (same_surface["match_rank_diff"] - (-abs(rank_diff))).abs()
    top = same_surface.sort_values("gap_score").head(n)

    return top[
        ["tourney_date", "tourney_name", "surface", "winner_name", "loser_name", "winner_rank", "loser_rank"]
    ].reset_index(drop=True)


def build_match_features(
    feature_columns: list,
    profiles: pd.DataFrame,
    df_clean: pd.DataFrame,
    p1_name: str,
    p2_name: str,
    surface: str,
    tourney_level: str,
    round_: str,
    best_of: int,
    match_date,
) -> Tuple[pd.DataFrame, Dict, Dict, Dict]:
    """Ghép 1 dòng feature THỰC (từ hồ sơ tay vợt thật) đúng schema mà model đã học,
    thay cho việc lấy 1 dòng NGẪU NHIÊN như code cũ (`df_features.sample(1)`)."""

    p1 = get_player_row(profiles, p1_name)
    p2 = get_player_row(profiles, p2_name)
    if p1 is None or p2 is None:
        raise ValueError("Không tìm thấy hồ sơ của một trong hai tay vợt.")

    def extrapolate_age(row):
        if pd.isna(row["age_at_last_match"]) or pd.isna(row["last_match_date"]):
            return np.nan
        years_elapsed = (pd.Timestamp(match_date) - pd.Timestamp(row["last_match_date"])).days / 365.25
        return float(row["age_at_last_match"]) + years_elapsed

    h2h = get_h2h(df_clean, p1["player_id"], p2["player_id"])

    row = {
        "p1_b365": np.nan, "p2_b365": np.nan,
        "p1_ps": np.nan, "p2_ps": np.nan,
        "p1_max_odds": np.nan, "p2_max_odds": np.nan,
        "p1_avg_odds": np.nan, "p2_avg_odds": np.nan,
        "p1_bfe": np.nan, "p2_bfe": np.nan,
        "p1_h2h_wins": h2h["a_wins"], "p2_h2h_wins": h2h["b_wins"],
        "year": pd.Timestamp(match_date).year,
        "surface": surface,
        "draw_size": DRAW_SIZE_BY_LEVEL.get(tourney_level, 32),
        "tourney_level": tourney_level,
        "p1_seed": np.nan, "p2_seed": np.nan,
        "p1_entry": np.nan, "p2_entry": np.nan,
        "p1_hand": p1["hand"], "p2_hand": p2["hand"],
        "p1_ht": p1["ht"], "p2_ht": p2["ht"],
        "p1_ioc": p1["ioc"], "p2_ioc": p2["ioc"],
        "p1_age": extrapolate_age(p1), "p2_age": extrapolate_age(p2),
        "best_of": best_of,
        "round": round_,
        "p1_rank": p1["rank"], "p2_rank": p2["rank"],
        "p1_rank_points": p1["rank_points"], "p2_rank_points": p2["rank_points"],
        "match_category": "main",
        "p1_elo": p1["elo"], "p2_elo": p2["elo"],
    }
    row["rank_diff"] = row["p1_rank"] - row["p2_rank"]
    row["rank_points_diff"] = row["p1_rank_points"] - row["p2_rank_points"]
    row["age_diff"] = row["p1_age"] - row["p2_age"]
    row["ht_diff"] = row["p1_ht"] - row["p2_ht"]
    row["elo_diff"] = row["p1_elo"] - row["p2_elo"]
    row["h2h_diff"] = row["p1_h2h_wins"] - row["p2_h2h_wins"]
    for odds_name in ["b365", "ps", "max_odds", "avg_odds", "bfe"]:
        row[f"{odds_name}_implied_prob_diff"] = np.nan

    df_row = pd.DataFrame([row])
    df_row = df_row[feature_columns]

    # Ép các cột phân loại (categorical) về dtype string nullable giống hệt lúc
    # train (Arrow-backed "string"), để NaN được model xử lý đúng như lúc huấn
    # luyện thay vì gây lỗi kiểu dữ liệu ở CatBoost.
    cat_cols = [
        "surface", "tourney_level", "p1_entry", "p2_entry",
        "p1_hand", "p2_hand", "p1_ioc", "p2_ioc", "round", "match_category",
    ]
    for col in cat_cols:
        if col in df_row.columns:
            df_row[col] = df_row[col].apply(lambda v: "nan" if pd.isna(v) else str(v))


    return df_row, h2h, p1.to_dict(), p2.to_dict()

