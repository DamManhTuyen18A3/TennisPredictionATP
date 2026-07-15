import sys
from pathlib import Path
import time
import pandas as pd
import numpy as np
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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

import optuna
from optuna.pruners import MedianPruner

# Suppress Optuna's internal logs to keep output clean
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Hằng số cấu hình tuning
# ---------------------------------------------------------------------------
# TRƯỚC ĐÂY (chạy rất chậm — có thể mất hàng giờ):
#   N_TRIALS=10, N_SPLITS=3, iterations/n_estimators tối đa 1000, KHÔNG early
#   stopping cho LightGBM/XGBoost (luôn train đủ 1000 cây mỗi lần thử), và
#   Optuna KHÔNG cắt sớm (pruning) các trial có tham số tệ.
#
# ĐÃ TỐI ƯU (nhanh hơn đáng kể, vẫn là tuning THẬT, không bịa số):
#   - Thêm early stopping thật cho LightGBM & XGBoost (giống CatBoost) — mô
#     hình tự dừng khi validation không cải thiện, thay vì luôn chạy hết
#     n_estimators tối đa.
#   - Giảm N_SPLITS 3 → 2 (TimeSeriesSplit vẫn hợp lệ về mặt phương pháp,
#     giảm ~35% khối lượng tính toán do fold cuối luôn là fold lớn nhất).
#   - Giảm trần tìm kiếm của iterations/n_estimators (1000 → 600) và
#     max_depth của RandomForest (30 → 20) — với early stopping, mô hình
#     hiếm khi cần tới trần này nên chất lượng tìm kiếm không đổi nhiều.
#   - Bật Optuna PRUNING (MedianPruner) cho cả 4 mô hình: nếu 1 trial cho AUC
#     tệ hơn trung vị các trial trước sau fold đầu tiên, Optuna dừng ngay
#     thay vì chạy tiếp fold thứ 2 — tiết kiệm thời gian đáng kể cho các
#     vùng tham số không triển vọng.
#   - Đặt rõ `thread_count=-1` / `n_jobs=-1` để đảm bảo dùng HẾT số nhân CPU
#     có sẵn trên máy (mặc định thường đã là -1, nhưng đặt tường minh để
#     chắc chắn, đặc biệt quan trọng trên máy nhiều nhân).
N_TRIALS = 10          # Số lần thử cho mỗi model (đủ cho nghiên cứu)
N_SPLITS = 2           # Số fold cho TimeSeriesSplit (giảm từ 3 → 2)
EARLY_STOPPING_ROUNDS = 30
RANDOM_STATE = 42


def _prepare_data(config):
    """Load train + val set, xử lý categorical, trả về X, y đã sẵn sàng."""
    features_dir = Path(config['data']['features_dir'])

    train_df = pd.read_parquet(features_dir / "train_set.parquet")
    val_df = pd.read_parquet(features_dir / "val_set.parquet")
    # LỖI ĐÃ SỬA (tận gốc, cùng nguyên nhân với 10_train_models.py): chỉ đọc
    # test set để lấy VOCABULARY category (KHÔNG dùng để tune) — nếu không,
    # model tuned cũng sẽ không biết tới giá trị category chỉ xuất hiện ở
    # test (vd. "Carpet") và lỗi "Found a category not in the training set"
    # sẽ tái diễn dù đã tune lại. Xem src/utils/categorical.py.
    test_df = pd.read_parquet(features_dir / "test_set.parquet")

    # Ghép train + val để chạy TimeSeriesSplit đúng cách
    # (không dùng val set cố định nữa — Optuna tự chia)
    df = pd.concat([train_df, val_df], ignore_index=True)

    # Sort theo thời gian (đảm bảo TimeSeriesSplit tôn trọng thứ tự)
    if 'tourney_date' in df.columns:
        df = df.sort_values('tourney_date').reset_index(drop=True)

    target_col = 'target'
    exclude_cols = [target_col, 'tourney_date']

    X = df.drop(columns=[c for c in exclude_cols if c in df.columns])
    y = df[target_col]
    X_test_vocab = test_df.drop(columns=[c for c in exclude_cols if c in test_df.columns])

    # Đồng nhất category dtype — hợp nhất TỪ CẢ train+val (X) VÀ test
    # (X_test_vocab), nhưng test chỉ đóng góp vocabulary, không phải dữ liệu
    # huấn luyện.
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    unified_categories = get_unified_categories(X, X_test_vocab, cat_cols=cat_cols)
    X = apply_unified_categorical_dtype(X, unified_categories)

    return X, y, cat_cols, unified_categories


def _get_ordinal_encoded(X, unified_categories):
    """Encode categorical columns cho RandomForest — dùng tập category đã
    hợp nhất (train+val+test vocab) để đảm bảo mã hoá nhất quán với lúc
    evaluate/backtest sau này (xem src/utils/categorical.py)."""
    return ordinal_encode(X, unified_categories)


# ===========================================================================
# Objective functions cho từng model
# ===========================================================================

def _objective_catboost(trial, X, y, cat_cols, tscv):
    params = {
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'iterations': trial.suggest_int('iterations', 100, 600, step=50),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 5.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 5.0),
        'verbose': 0,
        'random_state': RANDOM_STATE,
        'allow_writing_files': False,
        'thread_count': -1,  # dùng hết số nhân CPU có sẵn
    }
    scores = []
    for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostClassifier(**params)
        model.fit(X_tr, y_tr, cat_features=cat_cols,
                  eval_set=(X_va, y_va), early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=0)
        proba = model.predict_proba(X_va)[:, 1]
        scores.append(roc_auc_score(y_va, proba))

        # PRUNING: nếu sau fold đầu tiên trial này đã kém hơn hẳn trung vị
        # các trial trước, dừng ngay không chạy fold tiếp theo (tiết kiệm
        # thời gian đáng kể cho vùng tham số không triển vọng).
        trial.report(np.mean(scores), fold_i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(scores)


def _objective_lightgbm(trial, X, y, cat_cols, tscv):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 600, step=50),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': RANDOM_STATE,
        'verbose': -1,
        'n_jobs': -1,  # dùng hết số nhân CPU có sẵn
    }
    scores = []
    for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        # TRƯỚC ĐÂY: không có early stopping → luôn train đủ n_estimators
        # tối đa (tới 1000 cây) mỗi lần thử, rất chậm. Nay dừng sớm thật sự
        # khi validation AUC/logloss không cải thiện sau EARLY_STOPPING_ROUNDS
        # vòng liên tiếp.
        import lightgbm as lgb
        model = LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr, eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        proba = model.predict_proba(X_va)[:, 1]
        scores.append(roc_auc_score(y_va, proba))

        trial.report(np.mean(scores), fold_i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(scores)


def _objective_xgboost(trial, X, y, tscv):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 600, step=50),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'eval_metric': 'logloss',
        'random_state': RANDOM_STATE,
        'enable_categorical': True,
        'early_stopping_rounds': EARLY_STOPPING_ROUNDS,  # TRƯỚC ĐÂY: thiếu hoàn toàn
        'n_jobs': -1,  # dùng hết số nhân CPU có sẵn
    }
    scores = []
    for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        proba = model.predict_proba(X_va)[:, 1]
        scores.append(roc_auc_score(y_va, proba))

        trial.report(np.mean(scores), fold_i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(scores)


def _objective_rf(trial, X_enc, y, tscv):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': RANDOM_STATE,
        'n_jobs': -1,  # dùng hết số nhân CPU có sẵn
    }
    scores = []
    for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X_enc)):
        X_tr, X_va = X_enc.iloc[train_idx], X_enc.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        model = RandomForestClassifier(**params)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_va)[:, 1]
        scores.append(roc_auc_score(y_va, proba))

        trial.report(np.mean(scores), fold_i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(scores)


# ===========================================================================
# Main tuning function
# ===========================================================================

def tune_hyperparameters():
    """Tối ưu tham số cho tất cả models bằng Optuna + TimeSeriesSplit."""
    config = load_config()
    tuned_dir = Path(config['model']['tuned_dir'])
    tuned_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BẮT ĐẦU HYPERPARAMETER TUNING THẬT (Optuna + TimeSeriesCV)")
    logger.info(f"Cấu hình: {N_TRIALS} trials/model, {N_SPLITS} folds TimeSeriesSplit, "
                f"early stopping {EARLY_STOPPING_ROUNDS} vòng, có pruning.")
    logger.info("=" * 60)

    X, y, cat_cols, unified_categories = _prepare_data(config)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    # Chuẩn bị X đã encode cho RandomForest
    X_enc = _get_ordinal_encoded(X, unified_categories)

    # Pruner dùng chung cho cả 4 model: cắt sớm các trial có AUC (sau fold 1)
    # tệ hơn trung vị các trial trước — tiết kiệm nhiều thời gian.
    def _make_pruner():
        return MedianPruner(n_startup_trials=3, n_warmup_steps=0)

    all_results = {}
    total_t0 = time.time()

    # -----------------------------------------------------------------------
    # 1. CatBoost
    # -----------------------------------------------------------------------
    logger.info("\n[1/4] Tuning CatBoost...")
    t0 = time.time()
    study_cb = optuna.create_study(direction='maximize', study_name='CatBoost', pruner=_make_pruner())
    study_cb.optimize(
        lambda trial: _objective_catboost(trial, X, y, cat_cols, tscv),
        n_trials=N_TRIALS, show_progress_bar=False
    )
    logger.info(f"CatBoost best AUC: {study_cb.best_value:.4f} (mất {time.time()-t0:.0f}s, "
                f"{len(study_cb.trials)} trials, {sum(1 for t in study_cb.trials if t.state.name=='PRUNED')} bị cắt sớm)")
    logger.info(f"CatBoost best params: {study_cb.best_params}")

    # Retrain trên toàn bộ train+val với best params
    best_params_cb = study_cb.best_params.copy()
    best_params_cb.update({'verbose': 0, 'random_state': RANDOM_STATE,
                            'allow_writing_files': False, 'thread_count': -1})
    best_model_cb = CatBoostClassifier(**best_params_cb)
    best_model_cb.fit(X, y, cat_features=cat_cols, verbose=0)
    joblib.dump(best_model_cb, tuned_dir / "CatBoost_tuned.joblib")
    all_results['CatBoost'] = {
        'best_cv_auc': study_cb.best_value,
        'best_params': study_cb.best_params,
        'n_trials': N_TRIALS,
        'tuning_time_seconds': round(time.time() - t0, 1),
    }

    # -----------------------------------------------------------------------
    # 2. LightGBM
    # -----------------------------------------------------------------------
    logger.info("\n[2/4] Tuning LightGBM...")
    t0 = time.time()
    study_lgb = optuna.create_study(direction='maximize', study_name='LightGBM', pruner=_make_pruner())
    study_lgb.optimize(
        lambda trial: _objective_lightgbm(trial, X, y, cat_cols, tscv),
        n_trials=N_TRIALS, show_progress_bar=False
    )
    logger.info(f"LightGBM best AUC: {study_lgb.best_value:.4f} (mất {time.time()-t0:.0f}s, "
                f"{len(study_lgb.trials)} trials, {sum(1 for t in study_lgb.trials if t.state.name=='PRUNED')} bị cắt sớm)")
    logger.info(f"LightGBM best params: {study_lgb.best_params}")

    best_params_lgb = study_lgb.best_params.copy()
    best_params_lgb.update({'random_state': RANDOM_STATE, 'verbose': -1, 'n_jobs': -1})
    best_model_lgb = LGBMClassifier(**best_params_lgb)
    best_model_lgb.fit(X, y)
    joblib.dump(best_model_lgb, tuned_dir / "LightGBM_tuned.joblib")
    all_results['LightGBM'] = {
        'best_cv_auc': study_lgb.best_value,
        'best_params': study_lgb.best_params,
        'n_trials': N_TRIALS,
        'tuning_time_seconds': round(time.time() - t0, 1),
    }

    # -----------------------------------------------------------------------
    # 3. XGBoost
    # -----------------------------------------------------------------------
    logger.info("\n[3/4] Tuning XGBoost...")
    t0 = time.time()
    study_xgb = optuna.create_study(direction='maximize', study_name='XGBoost', pruner=_make_pruner())
    study_xgb.optimize(
        lambda trial: _objective_xgboost(trial, X, y, tscv),
        n_trials=N_TRIALS, show_progress_bar=False
    )
    logger.info(f"XGBoost best AUC: {study_xgb.best_value:.4f} (mất {time.time()-t0:.0f}s, "
                f"{len(study_xgb.trials)} trials, {sum(1 for t in study_xgb.trials if t.state.name=='PRUNED')} bị cắt sớm)")
    logger.info(f"XGBoost best params: {study_xgb.best_params}")

    # Lưu ý: bỏ 'early_stopping_rounds' cho lần fit CUỐI trên toàn bộ dữ liệu,
    # vì không còn eval_set riêng (train trên 100% train+val) — early stopping
    # chỉ có ý nghĩa trong lúc TÌM n_estimators tối ưu ở bước CV phía trên.
    best_params_xgb = study_xgb.best_params.copy()
    best_params_xgb.update({
        'eval_metric': 'logloss', 'random_state': RANDOM_STATE,
        'enable_categorical': True, 'n_jobs': -1,
    })
    best_model_xgb = XGBClassifier(**best_params_xgb)
    best_model_xgb.fit(X, y, verbose=False)
    joblib.dump(best_model_xgb, tuned_dir / "XGBoost_tuned.joblib")
    all_results['XGBoost'] = {
        'best_cv_auc': study_xgb.best_value,
        'best_params': study_xgb.best_params,
        'n_trials': N_TRIALS,
        'tuning_time_seconds': round(time.time() - t0, 1),
    }

    # -----------------------------------------------------------------------
    # 4. RandomForest (dùng OrdinalEncoder)
    # -----------------------------------------------------------------------
    logger.info("\n[4/4] Tuning RandomForest...")
    t0 = time.time()
    study_rf = optuna.create_study(direction='maximize', study_name='RandomForest', pruner=_make_pruner())
    study_rf.optimize(
        lambda trial: _objective_rf(trial, X_enc, y, tscv),
        n_trials=N_TRIALS, show_progress_bar=False
    )
    logger.info(f"RandomForest best AUC: {study_rf.best_value:.4f} (mất {time.time()-t0:.0f}s, "
                f"{len(study_rf.trials)} trials, {sum(1 for t in study_rf.trials if t.state.name=='PRUNED')} bị cắt sớm)")
    logger.info(f"RandomForest best params: {study_rf.best_params}")

    best_params_rf = study_rf.best_params.copy()
    best_params_rf.update({'random_state': RANDOM_STATE, 'n_jobs': -1})
    best_model_rf = RandomForestClassifier(**best_params_rf)
    best_model_rf.fit(X_enc, y)
    joblib.dump(best_model_rf, tuned_dir / "RandomForest_tuned.joblib")
    all_results['RandomForest'] = {
        'best_cv_auc': study_rf.best_value,
        'best_params': study_rf.best_params,
        'n_trials': N_TRIALS,
        'tuning_time_seconds': round(time.time() - t0, 1),
    }

    # -----------------------------------------------------------------------
    # Lưu tổng hợp kết quả tuning
    # -----------------------------------------------------------------------
    results_path = Path(config['reports']['metrics_file']).parent / "tuning_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python native for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = json.loads(json.dumps(all_results, default=_convert))
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=4, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("HOÀN THÀNH HYPERPARAMETER TUNING")
    logger.info(f"Tổng thời gian: {time.time() - total_t0:.0f}s (~{(time.time() - total_t0)/60:.1f} phút)")
    for name, res in all_results.items():
        logger.info(f"  {name}: CV AUC = {res['best_cv_auc']:.4f} ({res['tuning_time_seconds']:.0f}s)")
    logger.info(f"Kết quả chi tiết: {results_path}")
    logger.info(f"Tuned models: {tuned_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    tune_hyperparameters()
