import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger, load_config

logger = get_logger(__name__)

def remove_data_leakage():
    """Loại bỏ các trường dữ liệu gây rò rỉ (những thông tin sinh ra sau/trong trận đấu)."""
    config = load_config()
    interim_dir = Path(config['data']['interim_dir'])
    processed_dir = Path(config['data']['processed_dir'])
    
    input_path = interim_dir / "04_handled_missing.parquet"
    output_path = processed_dir / "05_clean_data.parquet"
    
    logger.info(f"Bắt đầu remove data leakage từ: {input_path}")
    df = pd.read_parquet(input_path)
    
    # Những thông tin như số điểm ăn, số ace, double faults... đều xảy ra TRONG TRẬN
    # Nếu dùng nó để dự đoán kết quả TRƯỚC TRẬN, model sẽ đạt 99% accuracy ảo.
    # Vì thế phải drop toàn bộ.
    # QUAN TRỌNG: bước 03_clean_schema.py đã lowercase toàn bộ tên cột
    # (vd: w_1stIn -> w_1stin). Danh sách dưới đây trước đây dùng sai tên viết hoa
    # nên KHÔNG khớp được với df.columns -> các cột leakage này vẫn tồn tại
    # xuyên suốt tới model, đây chính là nguyên nhân chính gây leakage nghiêm trọng.
    leakage_cols = [
        'w_ace', 'w_df', 'w_svpt', 'w_1stin', 'w_1stwon', 'w_2ndwon',
        'l_ace', 'l_df', 'l_svpt', 'l_1stin', 'l_1stwon', 'l_2ndwon',
        'w_svgms', 'w_bpsaved', 'w_bpfaced',
        'l_svgms', 'l_bpsaved', 'l_bpfaced',
        'minutes', 'score' # score cũng là thông tin sau trận đấu
    ]
    
    cols_to_drop = [col for col in leakage_cols if col in df.columns]
    
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        logger.info(f"Đã loại bỏ các cột gây leakage: {cols_to_drop}")
    else:
        logger.info("Không phát hiện các cột rò rỉ dữ liệu đã liệt kê.")
        
    # Kiểm tra an toàn: đảm bảo không còn cột leakage nào sót lại (bắt lỗi sớm nếu tên cột đổi trong tương lai)
    remaining_leakage = [col for col in leakage_cols if col in df.columns]
    if remaining_leakage:
        logger.error(f"CẢNH BÁO: vẫn còn cột leakage sau khi drop: {remaining_leakage}")

    # Lưu vào processed data (dữ liệu sạch đã sẵn sàng để chuyển sang EDA và Feature Engineering)
    df.to_parquet(output_path, index=False)
    logger.info(f"Đã hoàn thành remove data leakage và lưu tại: {output_path}")

if __name__ == "__main__":
    remove_data_leakage()
