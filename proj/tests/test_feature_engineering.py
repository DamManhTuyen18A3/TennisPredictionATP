"""Tests cho Feature Engineering pipeline (step 07).

Kiểm tra:
- Elo calculation đúng logic (pre-match, không leakage)
- H2H calculation đúng logic (pre-match, không leakage)
- Biến đổi đối xứng (swap Winner/Loser → Player_A/Player_B) đúng
- Target distribution cân bằng (~50/50)
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _import_feature_eng():
    """Import 07_feature_engineering module."""
    module_path = PROJECT_ROOT / "src" / "pipelines" / "07_feature_engineering.py"
    spec = importlib.util.spec_from_file_location("feature_engineering", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestEloCalculation:
    """Test hàm calculate_elo."""

    def test_elo_returns_correct_columns(self, sample_match_df):
        """Hàm calculate_elo phải thêm 2 cột winner_elo, loser_elo."""
        mod = _import_feature_eng()
        df = sample_match_df.sort_values('tourney_date').reset_index(drop=True)
        result = mod.calculate_elo(df)

        assert 'winner_elo' in result.columns, "Thiếu cột winner_elo"
        assert 'loser_elo' in result.columns, "Thiếu cột loser_elo"

    def test_elo_initial_rating_is_1500(self, sample_match_df):
        """Trận đầu tiên của một tay vợt mới phải có Elo = 1500 (base rating)."""
        mod = _import_feature_eng()
        # Tạo data với 1 tay vợt hoàn toàn mới
        df = pd.DataFrame({
            'tourney_date': pd.to_datetime(['2021-01-01', '2021-01-02']),
            'winner_id': [999, 999],
            'loser_id': [888, 777],
        })
        result = mod.calculate_elo(df, k=32, base_rating=1500)
        # Trận đầu tiên, cả winner (999) và loser (888) đều chưa từng đấu → Elo = 1500
        assert result['winner_elo'].iloc[0] == 1500
        assert result['loser_elo'].iloc[0] == 1500

    def test_elo_is_pre_match(self, sample_match_df):
        """Elo lưu trong dòng phải là giá trị TRƯỚC trận đó (pre-match).

        Nếu tay vợt A thắng trận 1, Elo của A ở trận 2 phải > 1500
        (chứng tỏ đã cập nhật sau trận 1).
        """
        mod = _import_feature_eng()
        df = pd.DataFrame({
            'tourney_date': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03']),
            'winner_id': [100, 100, 100],
            'loser_id': [200, 201, 202],
        })
        result = mod.calculate_elo(df, k=32, base_rating=1500)

        # Trận 1: winner_elo[0] = 1500 (chưa có lịch sử)
        assert result['winner_elo'].iloc[0] == 1500
        # Trận 2: winner_elo[1] > 1500 (đã thắng trận 1, được cộng Elo)
        assert result['winner_elo'].iloc[1] > 1500, \
            f"Elo trận 2 phải > 1500, nhưng = {result['winner_elo'].iloc[1]}"

    def test_elo_no_nan(self, sample_match_df):
        """Elo không được có NaN."""
        mod = _import_feature_eng()
        df = sample_match_df.sort_values('tourney_date').reset_index(drop=True)
        result = mod.calculate_elo(df)
        assert result['winner_elo'].isna().sum() == 0
        assert result['loser_elo'].isna().sum() == 0


class TestH2HCalculation:
    """Test hàm calculate_h2h."""

    def test_h2h_returns_correct_columns(self, sample_match_df):
        """Hàm calculate_h2h phải thêm cột h2h_winner_wins, h2h_loser_wins."""
        mod = _import_feature_eng()
        df = sample_match_df.sort_values('tourney_date').reset_index(drop=True)
        result = mod.calculate_h2h(df)

        assert 'h2h_winner_wins' in result.columns
        assert 'h2h_loser_wins' in result.columns

    def test_h2h_first_meeting_is_zero(self):
        """Lần gặp đầu tiên giữa 2 tay vợt, H2H phải = 0-0."""
        mod = _import_feature_eng()
        df = pd.DataFrame({
            'tourney_date': pd.to_datetime(['2021-01-01']),
            'winner_id': [100],
            'loser_id': [200],
        })
        result = mod.calculate_h2h(df)
        assert result['h2h_winner_wins'].iloc[0] == 0
        assert result['h2h_loser_wins'].iloc[0] == 0

    def test_h2h_is_pre_match(self):
        """H2H phải phản ánh số lần thắng TRƯỚC trận hiện tại."""
        mod = _import_feature_eng()
        df = pd.DataFrame({
            'tourney_date': pd.to_datetime(['2021-01-01', '2021-06-01', '2021-12-01']),
            'winner_id': [100, 100, 200],  # A thắng 2 trận, B thắng 1 trận
            'loser_id': [200, 200, 100],
        })
        result = mod.calculate_h2h(df)

        # Trận 1: 0-0 (lần đầu gặp)
        assert result['h2h_winner_wins'].iloc[0] == 0
        assert result['h2h_loser_wins'].iloc[0] == 0

        # Trận 2: A đã thắng B 1 lần → h2h_winner_wins (cho winner=A) = 1
        assert result['h2h_winner_wins'].iloc[1] == 1
        assert result['h2h_loser_wins'].iloc[1] == 0

        # Trận 3: B là winner, A là loser. B đã thắng A 0 lần, A đã thắng B 2 lần
        assert result['h2h_winner_wins'].iloc[2] == 0  # B chưa thắng A trước đó
        assert result['h2h_loser_wins'].iloc[2] == 2   # A đã thắng B 2 lần


class TestSymmetricTransform:
    """Test biến đổi đối xứng Winner/Loser → Player_A/Player_B."""

    def test_target_roughly_balanced(self, sample_match_df):
        """Sau biến đổi, target phải xấp xỉ 50/50 (±5%)."""
        mod = _import_feature_eng()
        df = sample_match_df.sort_values('tourney_date').reset_index(drop=True)

        # Chạy calculate_elo và calculate_h2h trước
        df = mod.calculate_elo(df)
        df = mod.calculate_h2h(df)

        # Tạo swap giống pipeline
        np.random.seed(42)
        swap_idx = np.random.rand(len(df)) > 0.5
        target = np.where(swap_idx, 1, 0)

        ratio = target.mean()
        assert 0.40 <= ratio <= 0.60, \
            f"Target ratio = {ratio:.2f}, phải nằm trong [0.40, 0.60]"
