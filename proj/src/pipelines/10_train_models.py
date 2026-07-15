import sys
from pathlib import Path
import pandas as pd
import json
import joblib

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger, load_config
from src.utils.categorical import get_unified_categories, apply_unified_categorical_dtype, ordinal_encode

# Models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

logger = get_logger(__name__)

def train_models():
    """Huấn luyện baseline cho các mô hình."""
    config = load_config()
    features_dir = Path(config['data']['features_dir'])
    models_dir = Path(config['model']['models_dir'])
    metrics_file = Path(config['reports']['metrics_file'])
    
    logger.info("Bắt đầu huấn luyện các mô hình Baseline...")
    
    train_df = pd.read_parquet(features_dir / "train_set.parquet")
    val_df = pd.read_parquet(features_dir / "val_set.parquet")
    # LỖI ĐÃ SỬA (tận gốc): chỉ đọc test set để lấy VOCABULARY của các cột
    # categorical (KHÔNG dùng để train) — nếu không, model được fit ở đây sẽ
    # không hề biết tới 1 giá trị category (vd. mặt sân "Carpet") chỉ xuất
    # hiện ở test, và bất kỳ script nào sau này predict trên test đều lỗi
    # "Found a category not in the training set" (đã tái diễn ở 3 file khác
    # trước khi sửa đúng gốc ở đây). Xem src/utils/categorical.py.
    test_df = pd.read_parquet(features_dir / "test_set.parquet")
    
    # Chuẩn bị X, y
    target_col = 'target'
    exclude_cols = [target_col, 'tourney_date']
    
    X_train = train_df.drop(columns=[col for col in exclude_cols if col in train_df.columns])
    y_train = train_df[target_col]
    
    X_val = val_df.drop(columns=[col for col in exclude_cols if col in val_df.columns])
    y_val = val_df[target_col]

    X_test_vocab = test_df.drop(columns=[col for col in exclude_cols if col in test_df.columns])

    # Đồng nhất category dtype giữa train/val/test (hợp nhất TỪ CẢ 3 tập,
    # nhưng chỉ train/val được dùng làm dữ liệu huấn luyện thật sự).
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    unified_categories = get_unified_categories(X_train, X_val, X_test_vocab, cat_cols=cat_cols)
    X_train = apply_unified_categorical_dtype(X_train, unified_categories)
    X_val = apply_unified_categorical_dtype(X_val, unified_categories)
    
    # Định nghĩa các models
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42, enable_categorical=True),
        'LightGBM': LGBMClassifier(random_state=42),
        'CatBoost': CatBoostClassifier(iterations=100, verbose=0, random_state=42)
    }
    
    # Identify categorical features for models that support it (CatBoost, LightGBM)
    cat_features = X_train.select_dtypes(include=['category']).columns.tolist()

    # LỖI ĐÃ SỬA: RandomForest trước đây chỉ train trên các cột NUMERIC
    # (bỏ hết cat_cols — ghi chú cũ là "Placeholder"), trong khi
    # 12_evaluate_models.py lại đưa vào predict() TOÀN BỘ cột đã ordinal-
    # encode (bao gồm cả cat_cols) → lỗi "Feature names unseen at fit time".
    # Nay: encode đầy đủ cat_cols bằng ordinal_encode() dùng chung tập
    # category đã hợp nhất — RandomForest train trên ĐẦY ĐỦ feature, khớp
    # đúng những gì evaluate/backtest sẽ đưa vào sau này.
    X_train_rf = ordinal_encode(X_train, unified_categories)
    X_val_rf = ordinal_encode(X_val, unified_categories)
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"Đang huấn luyện {name}...")
        try:
            if name == 'CatBoost':
                model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val), early_stopping_rounds=20)
            elif name == 'LightGBM':
                # LightGBM tự nhận diện pandas 'category' type
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            elif name == 'XGBoost':
                # XGBoost từ version 1.5+ hỗ trợ category nếu bật enable_categorical=True.
                # Category dtype của X_train đã được hợp nhất với val+test ở trên nên
                # model sẽ nhận diện được mọi giá trị category có thể gặp lúc predict.
                model.fit(X_train, y_train)
            else:
                # RandomForest: dùng đầy đủ feature đã ordinal-encode (không bỏ cột
                # categorical nữa) — khớp đúng schema mà evaluate/backtest sẽ dùng.
                model.fit(X_train_rf, y_train)
                
            # Predict validation
            if name == 'RandomForest':
                val_preds = model.predict(X_val_rf)
                val_proba = model.predict_proba(X_val_rf)[:, 1]
            else:
                val_preds = model.predict(X_val)
                val_proba = model.predict_proba(X_val)[:, 1]
                
            acc = accuracy_score(y_val, val_preds)
            auc = roc_auc_score(y_val, val_proba)
            
            logger.info(f"{name} - Val Acc: {acc:.4f}, Val AUC: {auc:.4f}")
            results[name] = {'accuracy': acc, 'auc': auc}
            
            # Save model
            joblib.dump(model, models_dir / f"{name}_baseline.joblib")
            
        except Exception as e:
            logger.error(f"Lỗi khi huấn luyện {name}: {e}")
            
    # Save metrics
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    logger.info("Hoàn thành huấn luyện baseline models.")

if __name__ == "__main__":
    train_models()
