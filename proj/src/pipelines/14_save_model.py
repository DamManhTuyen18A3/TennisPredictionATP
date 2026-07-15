import sys
import shutil
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger, load_config

logger = get_logger(__name__)

def save_final_model():
    """Copy mô hình tốt nhất vào thư mục final và chuẩn bị artifacts cho Inference."""
    config = load_config()
    tuned_dir = Path(config['model']['tuned_dir'])
    final_dir = Path(config['model']['final_dir'])
    models_dir = Path(config['model']['models_dir'])
    
    logger.info("Đang đóng gói mô hình cuối cùng...")
    
    best_model_name = "CatBoost_tuned.joblib"
    source_path = tuned_dir / best_model_name
    
    if not source_path.exists():
        # Fallback
        source_path = models_dir / "CatBoost_baseline.joblib"
        
    if source_path.exists():
        target_path = final_dir / "final_model.joblib"
        shutil.copy(source_path, target_path)
        logger.info(f"Đã lưu mô hình production tại: {target_path}")
    else:
        logger.error("Không có mô hình nào để save thành final.")

if __name__ == "__main__":
    save_final_model()
