"""Pytest fixtures dùng chung cho tất cả test modules."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

# Đảm bảo project root nằm trong sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_match_df():
    """Tạo DataFrame mẫu mô phỏng dữ liệu trận đấu ATP đã clean (post step 05).

    Đủ nhỏ để test nhanh, nhưng đủ cột quan trọng để kiểm tra feature engineering.
    """
    np.random.seed(42)
    n = 200
    dates = pd.date_range('2021-01-01', periods=n, freq='3D')
    surfaces = np.random.choice(['Hard', 'Clay', 'Grass'], n)
    levels = np.random.choice(['F', 'C', 'A', 'G', 'M'], n, p=[0.5, 0.25, 0.15, 0.05, 0.05])

    df = pd.DataFrame({
        'tourney_date': dates,
        'tourney_id': [f'T{i//10}' for i in range(n)],
        'tourney_name': [f'Tournament_{i//10}' for i in range(n)],
        'surface': surfaces,
        'tourney_level': levels,
        'match_num': range(1, n + 1),
        'winner_id': np.random.randint(100, 200, n),
        'loser_id': np.random.randint(200, 300, n),
        'winner_name': [f'Player_W{i}' for i in range(n)],
        'loser_name': [f'Player_L{i}' for i in range(n)],
        'winner_hand': np.random.choice(['R', 'L'], n),
        'loser_hand': np.random.choice(['R', 'L'], n),
        'winner_ht': np.random.normal(185, 8, n).round(0),
        'loser_ht': np.random.normal(183, 8, n).round(0),
        'winner_age': np.random.uniform(18, 38, n).round(1),
        'loser_age': np.random.uniform(18, 38, n).round(1),
        'winner_rank': np.random.randint(1, 500, n),
        'loser_rank': np.random.randint(1, 500, n),
        'winner_rank_points': np.random.randint(10, 5000, n),
        'loser_rank_points': np.random.randint(10, 5000, n),
        'winner_ioc': np.random.choice(['USA', 'FRA', 'ESP', 'GER'], n),
        'loser_ioc': np.random.choice(['USA', 'FRA', 'ESP', 'GER'], n),
        'round': np.random.choice(['R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'F'], n),
        'b365w': np.random.uniform(1.1, 5.0, n).round(2),
        'b365l': np.random.uniform(1.1, 5.0, n).round(2),
        'psw': np.random.uniform(1.1, 5.0, n).round(2),
        'psl': np.random.uniform(1.1, 5.0, n).round(2),
        'maxw': np.random.uniform(1.1, 5.0, n).round(2),
        'maxl': np.random.uniform(1.1, 5.0, n).round(2),
        'avgw': np.random.uniform(1.1, 5.0, n).round(2),
        'avgl': np.random.uniform(1.1, 5.0, n).round(2),
    })
    return df


@pytest.fixture
def project_root():
    """Trả về đường dẫn project root."""
    return PROJECT_ROOT


@pytest.fixture
def config():
    """Load config từ project."""
    import yaml
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    # Absolutize paths
    for category in cfg:
        if isinstance(cfg[category], dict):
            for key, value in cfg[category].items():
                if key.endswith('_dir') or key == 'metrics_file':
                    cfg[category][key] = str(PROJECT_ROOT / value)
    return cfg
