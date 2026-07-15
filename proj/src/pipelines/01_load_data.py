import os
import sys
from pathlib import Path
import pandas as pd

# Add src to sys.path to allow importing from src
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.logger import get_logger, load_config

logger = get_logger(__name__)

def load_data():
    """Load raw data and save as initial parquet in interim folder for faster downstream processing."""
    config = load_config()
    
    raw_dir = Path(config['data']['raw_dir'])
    interim_dir = Path(config['data']['interim_dir'])
    raw_file = config['data']['raw_file']
    
    raw_path = raw_dir / raw_file
    interim_path = interim_dir / "01_loaded.parquet"
    
    logger.info(f"Bắt đầu tải dữ liệu thô từ: {raw_path}")
    
    if not raw_path.exists():
        logger.error(f"Không tìm thấy file dữ liệu gốc: {raw_path}")
        raise FileNotFoundError(f"Missing raw data: {raw_path}")
        
    try:
        # Tùy thuộc vào dữ liệu, có thể cần set low_memory=False
        df = pd.read_csv(raw_path, low_memory=False)
        logger.info(f"Đã tải thành công dữ liệu gốc. Kích thước (dòng, cột): {df.shape}")
        
        # Lưu vào interim dưới dạng parquet để tăng tốc độ load cho các bước sau
        df.to_parquet(interim_path, index=False)
        logger.info(f"Đã lưu dữ liệu tạm thời tại: {interim_path}")
        
    except Exception as e:
        logger.error(f"Lỗi trong quá trình tải dữ liệu: {str(e)}")
        raise e

if __name__ == "__main__":
    load_data()
