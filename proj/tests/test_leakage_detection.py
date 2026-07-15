"""Tests kiểm tra Data Leakage.

Đây là test QUAN TRỌNG NHẤT — đảm bảo không feature nào có tương quan
gần tuyệt đối (|corr| > 0.9) với target, vì đó là dấu hiệu rò rỉ dữ liệu.

Ngoài ra kiểm tra các cột thống kê trong trận (w_ace, l_ace, minutes, score...)
đã bị loại bỏ hoàn toàn.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestLeakageOnProcessedData:
    """Kiểm tra leakage trên dữ liệu đã xử lý (file parquet thực tế)."""

    def _load_engineered_features(self, config):
        """Load file 07_engineered_features.parquet."""
        features_dir = Path(config['data']['features_dir'])
        path = features_dir / "07_engineered_features.parquet"
        if not path.exists():
            pytest.skip(f"File {path} chưa tồn tại — cần chạy pipeline trước")
        return pd.read_parquet(path)

    def _load_clean_data(self, config):
        """Load file 05_clean_data.parquet."""
        processed_dir = Path(config['data']['processed_dir'])
        path = processed_dir / "05_clean_data.parquet"
        if not path.exists():
            pytest.skip(f"File {path} chưa tồn tại — cần chạy pipeline trước")
        return pd.read_parquet(path)

    def test_no_in_match_stats_in_clean_data(self, config):
        """Sau step 05, các cột thống kê TRONG TRẬN phải bị loại bỏ.

        Các cột như w_ace, l_ace, w_svpt, l_svpt, minutes, score...
        là thông tin chỉ có SAU trận đấu — nếu còn lại sẽ gây leakage nghiêm trọng.
        """
        df = self._load_clean_data(config)

        leakage_cols = [
            'w_ace', 'w_df', 'w_svpt', 'w_1stin', 'w_1stwon', 'w_2ndwon',
            'l_ace', 'l_df', 'l_svpt', 'l_1stin', 'l_1stwon', 'l_2ndwon',
            'w_svgms', 'w_bpsaved', 'w_bpfaced',
            'l_svgms', 'l_bpsaved', 'l_bpfaced',
            'minutes', 'score'
        ]
        remaining = [col for col in leakage_cols if col in df.columns]
        assert len(remaining) == 0, \
            f"Các cột leakage vẫn còn trong dữ liệu sau step 05: {remaining}"

    def test_no_high_correlation_with_target(self, config):
        """KHÔNG feature nào được có |correlation| > 0.9 với target.

        Nếu có, đó là dấu hiệu leakage — model chỉ cần nhìn feature đó
        là đoán được gần như chắc chắn.
        """
        df = self._load_engineered_features(config)

        if 'target' not in df.columns:
            pytest.skip("File thiếu cột target")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('target', errors='ignore')
        correlations = df[numeric_cols].corrwith(df['target']).abs()
        suspicious = correlations[correlations > 0.9]

        assert len(suspicious) == 0, \
            f"CÓ LEAKAGE! Các cột có |corr| > 0.9 với target:\n{suspicious}"

    def test_no_moderate_leakage(self, config):
        """Cảnh báo nếu có feature nào có |corr| > 0.7 với target.

        Không nhất thiết là leakage, nhưng đáng nghi — cần kiểm tra thủ công.
        """
        df = self._load_engineered_features(config)

        if 'target' not in df.columns:
            pytest.skip("File thiếu cột target")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('target', errors='ignore')
        correlations = df[numeric_cols].corrwith(df['target']).abs()
        suspicious = correlations[correlations > 0.7]

        if len(suspicious) > 0:
            import warnings
            warnings.warn(
                f"Có {len(suspicious)} features với |corr| > 0.7 với target "
                f"(đáng nghi nhưng chưa chắc leakage):\n{suspicious}",
                UserWarning
            )

    def test_target_distribution_balanced(self, config):
        """Target phải cân bằng (40-60%) sau biến đổi đối xứng."""
        df = self._load_engineered_features(config)

        if 'target' not in df.columns:
            pytest.skip("File thiếu cột target")

        ratio = df['target'].mean()
        assert 0.40 <= ratio <= 0.60, \
            f"Target ratio = {ratio:.2f}, ngoài khoảng [0.40, 0.60] — có thể sai ở bước swap"


class TestLeakageOnTrainTestSplit:
    """Kiểm tra không có rò rỉ thông tin giữa train/test."""

    def test_train_before_test_chronologically(self, config):
        """Tất cả dates trong train phải < tất cả dates trong test (time-based split)."""
        features_dir = Path(config['data']['features_dir'])
        train_path = features_dir / "train_set.parquet"
        test_path = features_dir / "test_set.parquet"

        if not train_path.exists() or not test_path.exists():
            pytest.skip("Chưa có train/test split files")

        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)

        if 'tourney_date' not in train_df.columns or 'tourney_date' not in test_df.columns:
            pytest.skip("Không có cột tourney_date")

        train_max = train_df['tourney_date'].max()
        test_min = test_df['tourney_date'].min()

        assert train_max < test_min, \
            f"LEAKAGE! Train max date ({train_max}) >= Test min date ({test_min})"
