import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger, load_config

logger = get_logger(__name__)

def handle_missing_values():
    """Xử lý các giá trị NaN/Null trong tập dữ liệu."""
    config = load_config()
    interim_dir = Path(config['data']['interim_dir'])
    
    input_path = interim_dir / "03_cleaned_schema.parquet"
    output_path = interim_dir / "04_handled_missing.parquet"
    
    logger.info(f"Bắt đầu xử lý missing values từ: {input_path}")
    df = pd.read_parquet(input_path)
    
    # Chiến lược xử lý missing (Minh họa, cần điều chỉnh theo dataset thực tế):
    
    # 1. Các cột categorical: Điền 'Unknown'
    cat_cols = ['surface', 'tourney_level', 'winner_hand', 'loser_hand']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            
    # 2. Các cột numeric thông tin cá nhân (tuổi, chiều cao): Điền bằng median
    personal_cols = ['winner_ht', 'loser_ht', 'winner_age', 'loser_age']
    for col in personal_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            
    # 3. Các cột thống kê trong trận (w_ace, l_ace...): Tùy bài toán, có thể điền 0 hoặc trung bình
    # Ở đây chúng ta sẽ giữ lại hoặc xử lý kỹ hơn ở Feature Engineering
    # Tuy nhiên để đảm bảo model không lỗi, fillna = 0 tạm thời
    # LƯU Ý: cột đã được lowercase ở bước 03_clean_schema.py (vd: w_1stIn -> w_1stin)
    stat_cols = ['w_ace', 'w_df', 'w_svpt', 'w_1stin', 'w_1stwon', 'w_2ndwon',
                 'w_svgms', 'w_bpsaved', 'w_bpfaced',
                 'l_ace', 'l_df', 'l_svpt', 'l_1stin', 'l_1stwon', 'l_2ndwon',
                 'l_svgms', 'l_bpsaved', 'l_bpfaced']
    for col in stat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            
    # Xoá các trận không có thông tin ngày đấu (cực kỳ quan trọng để chia train/test theo time)
    if 'tourney_date' in df.columns:
        before_drop = len(df)
        df.dropna(subset=['tourney_date'], inplace=True)
        if len(df) < before_drop:
            logger.info(f"Đã drop {before_drop - len(df)} dòng vì thiếu tourney_date.")
            
    # Lưu kết quả
    df.to_parquet(output_path, index=False)
    logger.info(f"Đã hoàn thành xử lý missing values và lưu tại: {output_path}")

if __name__ == "__main__":
    handle_missing_values()
