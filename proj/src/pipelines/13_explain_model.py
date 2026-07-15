import sys
from pathlib import Path
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger, load_config
from src.utils.categorical import get_unified_categories, apply_unified_categorical_dtype

logger = get_logger(__name__)

def explain_model():
    """Giải thích mô hình bằng SHAP values."""
    config = load_config()
    features_dir = Path(config['data']['features_dir'])
    models_dir = Path(config['model']['models_dir'])
    tuned_dir = Path(config['model']['tuned_dir'])
    shap_dir = Path(config['reports']['shap_dir'])
    
    logger.info("Bắt đầu giải thích mô hình bằng SHAP...")
    
    try:
        test_df_full = pd.read_parquet(features_dir / "test_set.parquet")
        # LỖI ĐÃ SỬA: sample cũ dùng replace=True vô điều kiện — nếu test set
        # có nhiều hơn 1000 dòng (thường là vậy), lấy mẫu CÓ hoàn lại là
        # không cần thiết, gây trùng lặp điểm dữ liệu làm sai lệch mật độ
        # hiển thị trên SHAP summary plot. Chỉ dùng replace=True khi test
        # set nhỏ hơn số lượng cần lấy mẫu.
        n_sample = min(1000, len(test_df_full))
        test_df = test_df_full.sample(n=n_sample, random_state=42,
                                       replace=(len(test_df_full) < 1000))
        target_col = 'target'
        exclude_cols = [target_col, 'tourney_date']
        
        X_test = test_df.drop(columns=[col for col in exclude_cols if col in test_df.columns])
        X_train_vocab = pd.read_parquet(features_dir / "train_set.parquet").drop(
            columns=[c for c in exclude_cols if c in test_df.columns], errors='ignore')
        X_val_vocab = pd.read_parquet(features_dir / "val_set.parquet").drop(
            columns=[c for c in exclude_cols if c in test_df.columns], errors='ignore')

        # Đồng nhất category dtype với train+val (cùng cách các script khác
        # đang dùng — xem src/utils/categorical.py) thay vì tự fit độc lập
        # chỉ từ X_test.
        cat_cols = X_test.select_dtypes(include=['object', 'category']).columns.tolist()
        unified_categories = get_unified_categories(X_train_vocab, X_val_vocab, X_test, cat_cols=cat_cols)
        X_test = apply_unified_categorical_dtype(X_test, unified_categories)

        # LỖI ĐÃ SỬA: trước đây luôn giải thích CatBoost_baseline, trong khi
        # TOÀN BỘ phần còn lại của đề tài (Home.py, Analytics.py, 14_save_
        # model.py, 16_backtest_roi.py...) đều thống nhất coi CatBoost_tuned
        # là model chính/model được triển khai. Biểu đồ SHAP trước đây vô
        # tình giải thích NHẦM model — ưu tiên tuned, fallback baseline nếu
        # chưa tune (khớp đúng logic fallback ở 14_save_model.py).
        model_path = tuned_dir / "CatBoost_tuned.joblib"
        if not model_path.exists():
            model_path = models_dir / "CatBoost_baseline.joblib"

        if model_path.exists():
            model = joblib.load(model_path)
            logger.info(f"Giải thích model: {model_path.name}")
            
            # Tính SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            # Summary Plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_test, show=False)
            shap_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(shap_dir / 'shap_summary.png', bbox_inches='tight')
            plt.close()
            
            # Feature Importance (Bar)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
            plt.savefig(shap_dir / 'shap_importance_bar.png', bbox_inches='tight')
            plt.close()
            
            logger.info(f"Đã lưu các biểu đồ SHAP tại {shap_dir}")
        else:
            logger.warning("Không tìm thấy mô hình để explain.")
            
    except Exception as e:
        logger.error(f"Lỗi khi chạy SHAP: {e}")

if __name__ == "__main__":
    explain_model()
