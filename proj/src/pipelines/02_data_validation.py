import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger, load_config

logger = get_logger(__name__)

def validate_data():
    """Validate data structure and contents."""
    config = load_config()
    interim_dir = Path(config['data']['interim_dir'])
    
    input_path = interim_dir / "01_loaded.parquet"
    output_path = interim_dir / "02_validated.parquet"
    
    logger.info(f"Bắt đầu validate dữ liệu từ: {input_path}")
    
    if not input_path.exists():
        logger.error(f"Không tìm thấy file: {input_path}")
        raise FileNotFoundError(f"Missing input data: {input_path}")
        
    df = pd.read_parquet(input_path)
    
    # Validation logic cơ bản (Ví dụ)
    # 1. Kiểm tra số lượng dòng
    if len(df) == 0:
        raise ValueError("Dữ liệu rỗng!")
        
    # 2. Kiểm tra các cột quan trọng (tùy thuộc vào dữ liệu ATP của bạn)
    # Ở đây chúng ta giả định một số cột bắt buộc, bạn có thể điều chỉnh sau
    expected_cols_subset = ['tourney_id', 'tourney_name', 'surface', 'draw_size', 'tourney_level', 'tourney_date', 'match_num', 'winner_id', 'loser_id', 'score']
    
    missing_cols = [col for col in expected_cols_subset if col not in df.columns]
    if missing_cols:
        logger.warning(f"Cảnh báo: Dữ liệu thiếu các cột quan trọng: {missing_cols}")
    else:
        logger.info("Dữ liệu đã chứa đủ các cột cơ bản được mong đợi.")
        
    # 3. Loại bỏ duplicate (nếu có)
    initial_shape = df.shape
    df.drop_duplicates(inplace=True)
    if df.shape != initial_shape:
        logger.info(f"Đã loại bỏ {initial_shape[0] - df.shape[0]} dòng trùng lặp.")
        
    # Lưu xuống interim cho bước tiếp theo
    df.to_parquet(output_path, index=False)
    logger.info(f"Đã hoàn thành validation và lưu tại: {output_path}")

if __name__ == "__main__":
    validate_data()
