import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger, load_config

logger = get_logger(__name__)

def feature_selection():
    """Lọc bỏ các trường định danh, chỉ giữ lại feature thực sự để train model."""
    config = load_config()
    features_dir = Path(config['data']['features_dir'])
    
    input_path = features_dir / "07_engineered_features.parquet"
    output_path = features_dir / "08_selected_features.parquet"
    
    logger.info(f"Bắt đầu Feature Selection từ: {input_path}")
    df = pd.read_parquet(input_path)
    
    # 1. Các cột loại bỏ (định danh, tên)
    cols_to_drop = [
        'tourney_id', 'tourney_name', 'p1_id', 'p2_id', 'p1_name', 'p2_name',
        'match_num', 'score' # nếu còn sót
    ]
    
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    
    # 2. Xử lý các biến Categorical còn lại (ví dụ: surface, round) bằng One-Hot Encoding hoặc chuyển type Category
    # Đối với Tree-based model (LightGBM, CatBoost), ta nên chuyển sang kiểu 'category'
    cat_cols = ['surface', 'tourney_level', 'round', 'p1_hand', 'p2_hand', 'p1_ioc', 'p2_ioc']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
            
    logger.info(f"Các features sẽ đưa vào mô hình: {df.columns.tolist()}")
    
    # Lưu
    df.to_parquet(output_path, index=False)
    logger.info(f"Đã hoàn thành Feature Selection. Shape: {df.shape}. Lưu tại: {output_path}")

if __name__ == "__main__":
    feature_selection()
