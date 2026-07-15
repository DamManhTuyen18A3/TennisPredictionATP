import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger, load_config

logger = get_logger(__name__)

def clean_schema():
    """Chuẩn hóa schema (tên cột, kiểu dữ liệu, định dạng ngày tháng)."""
    config = load_config()
    interim_dir = Path(config['data']['interim_dir'])
    
    input_path = interim_dir / "02_validated.parquet"
    output_path = interim_dir / "03_cleaned_schema.parquet"
    
    logger.info(f"Bắt đầu clean schema từ: {input_path}")
    df = pd.read_parquet(input_path)
    
    # 1. Chuẩn hóa tên cột (lower case, replace space)
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    
    # 2. Xử lý kiểu dữ liệu datetime
    # Giả định cột chứa ngày thi đấu là 'tourney_date' (định dạng YYYYMMDD theo chuẩn file tennis)
    if 'tourney_date' in df.columns:
        df['tourney_date'] = pd.to_datetime(df['tourney_date'], errors='coerce')
        logger.info("Đã parse thành công tourney_date sang datetime.")
            
    # 3. Chuẩn hóa các kiểu numeric nếu bị load nhầm thành object
    # Giả sử các cột chiều cao, tuổi...
    numeric_cols = ['winner_ht', 'loser_ht', 'winner_age', 'loser_age', 
                    'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
                    'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Lưu kết quả
    df.to_parquet(output_path, index=False)
    logger.info(f"Đã hoàn thành clean schema và lưu tại: {output_path}")

if __name__ == "__main__":
    clean_schema()
