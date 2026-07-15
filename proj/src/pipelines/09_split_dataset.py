import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger, load_config

logger = get_logger(__name__)

def split_dataset():
    """Chia tập dữ liệu thành Train, Validation và Test theo thời gian (Time-series split)."""
    config = load_config()
    features_dir = Path(config['data']['features_dir'])
    
    input_path = features_dir / "08_selected_features.parquet"
    
    logger.info(f"Bắt đầu Split Dataset từ: {input_path}")
    df = pd.read_parquet(input_path)
    
    if 'tourney_date' not in df.columns:
        logger.error("Không tìm thấy cột tourney_date để chia time-series.")
        raise ValueError("Missing tourney_date")
        
    df = df.sort_values('tourney_date').reset_index(drop=True)
    
    # Chiến lược chia:
    # Train: < 2024
    # Val: 2024
    # Test: >= 2025
    train_df = df[df['tourney_date'].dt.year < 2024]
    val_df = df[df['tourney_date'].dt.year == 2024]
    test_df = df[df['tourney_date'].dt.year >= 2025]
    
    # Drop tourney_date nếu model không cần đến nó nữa
    # train_df = train_df.drop(columns=['tourney_date']) ...
    
    logger.info(f"Kích thước tập Train: {train_df.shape}")
    logger.info(f"Kích thước tập Validation: {val_df.shape}")
    logger.info(f"Kích thước tập Test: {test_df.shape}")
    
    # Save splits
    train_df.to_parquet(features_dir / "train_set.parquet", index=False)
    val_df.to_parquet(features_dir / "val_set.parquet", index=False)
    test_df.to_parquet(features_dir / "test_set.parquet", index=False)
    
    logger.info("Đã chia xong và lưu vào thư mục features_dir.")

if __name__ == "__main__":
    split_dataset()
