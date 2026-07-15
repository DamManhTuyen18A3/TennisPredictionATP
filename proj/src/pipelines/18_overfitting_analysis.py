"""
18_overfitting_analysis.py — Kiểm tra Overfitting / Underfitting
=================================================================

MỤC ĐÍCH: Trả lời câu hỏi:
    "Model có overfit (quá khớp) hay underfit (dưới khớp) không?"

Phương pháp:
1. Train/Val/Test gap     — so sánh metrics trên 3 tập
2. Learning curves        — AUC theo lượng training data
3. Validation curves      — AUC theo hyperparameter quan trọng
4. Cross-validation std   — độ ổn định qua các fold
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import joblib
import warnings

warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger, load_config
from src.utils.categorical import get_unified_categories, apply_unified_categorical_dtype, ordinal_encode

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit

logger = get_logger(__name__)

RANDOM_STATE = 42


def _load_data(config):
    """Load train, val, test sets."""
    features_dir = Path(config['data']['features_dir'])

    train_df = pd.read_parquet(features_dir / "train_set.parquet")
    val_df = pd.read_parquet(features_dir / "val_set.parquet")
    test_df = pd.read_parquet(features_dir / "test_set.parquet")

    target_col = 'target'
    exclude_cols = [target_col, 'tourney_date']

    def split_xy(df):
        X = df.drop(columns=[c for c in exclude_cols if c in df.columns])
        y = df[target_col]
        return X, y

    X_train, y_train = split_xy(train_df)
    X_val, y_val = split_xy(val_df)
    X_test, y_test = split_xy(test_df)

    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # Đồng nhất category dtype giữa train/val/test bằng utility dùng chung
    # (src/utils/categorical.py) — cùng cách 10_train_models.py/
    # 11_hyperparameter_tuning.py/12_evaluate_models.py đang dùng, tránh lặp
    # lại lỗi "category not in training set" / lỗi pd.concat hạ dtype về
    # object đã từng xảy ra ở đây.
    unified_categories = get_unified_categories(X_train, X_val, X_test, cat_cols=cat_cols)
    X_train = apply_unified_categorical_dtype(X_train, unified_categories)
    X_val = apply_unified_categorical_dtype(X_val, unified_categories)
    X_test = apply_unified_categorical_dtype(X_test, unified_categories)

    return X_train, y_train, X_val, y_val, X_test, y_test, cat_cols, unified_categories


def _get_ordinal_encoded(X, unified_categories):
    """Encode categorical cho RandomForest — dùng map tất định từ tập
    category đã hợp nhất (KHÔNG fit OrdinalEncoder riêng cho từng DataFrame
    — bản cũ fit độc lập 3 lần cho train/val/test khiến cùng 1 giá trị
    category có thể bị mã hoá ra 2 con số khác nhau giữa các tập, một lỗi
    ÂM THẦM không crash nhưng làm sai lệch kết quả RandomForest_tuned trong
    phân tích gap). Xem src/utils/categorical.py."""
    return ordinal_encode(X, unified_categories)


# ===========================================================================
# 1. Train/Val/Test Gap Analysis
# ===========================================================================

def _analyze_gap(config, X_train, y_train, X_val, y_val, X_test, y_test, cat_cols, unified_categories):
    """So sánh metrics trên train/val/test cho từng model."""
    logger.info("\n📊 PHÂN TÍCH TRAIN/VAL/TEST GAP...")

    models_dir = Path(config['model']['models_dir'])
    tuned_dir = Path(config['model']['tuned_dir'])

    X_train_enc = _get_ordinal_encoded(X_train, unified_categories)
    X_val_enc = _get_ordinal_encoded(X_val, unified_categories)
    X_test_enc = _get_ordinal_encoded(X_test, unified_categories)

    model_configs = {
        'CatBoost_baseline': (models_dir / "CatBoost_baseline.joblib", False),
        'LightGBM_baseline': (models_dir / "LightGBM_baseline.joblib", False),
        'CatBoost_tuned': (tuned_dir / "CatBoost_tuned.joblib", False),
        'LightGBM_tuned': (tuned_dir / "LightGBM_tuned.joblib", False),
        'XGBoost_tuned': (tuned_dir / "XGBoost_tuned.joblib", False),
        'RandomForest_tuned': (tuned_dir / "RandomForest_tuned.joblib", True),
    }

    gap_results = {}
    for name, (path, use_encoded) in model_configs.items():
        if not path.exists():
            continue
        try:
            model = joblib.load(path)
            if use_encoded:
                try:
                    rf_cols = list(model.feature_names_in_)
                    Xtr = X_train_enc[rf_cols]
                    Xva = X_val_enc[rf_cols]
                    Xte = X_test_enc[rf_cols]
                except Exception:
                    continue
            else:
                Xtr, Xva, Xte = X_train, X_val, X_test

            train_auc = roc_auc_score(y_train, model.predict_proba(Xtr)[:, 1])
            val_auc = roc_auc_score(y_val, model.predict_proba(Xva)[:, 1])
            test_auc = roc_auc_score(y_test, model.predict_proba(Xte)[:, 1])

            train_acc = accuracy_score(y_train, model.predict(Xtr))
            val_acc = accuracy_score(y_val, model.predict(Xva))
            test_acc = accuracy_score(y_test, model.predict(Xte))

            gap = train_auc - test_auc
            if gap > 0.05:
                diagnosis = 'OVERFITTING (gap > 0.05)'
            elif gap < -0.01:
                diagnosis = 'POSSIBLE DATA SHIFT'
            else:
                diagnosis = 'GOOD FIT ✓'

            gap_results[name] = {
                'train_auc': round(train_auc, 4),
                'val_auc': round(val_auc, 4),
                'test_auc': round(test_auc, 4),
                'train_acc': round(train_acc, 4),
                'val_acc': round(val_acc, 4),
                'test_acc': round(test_acc, 4),
                'auc_gap_train_test': round(gap, 4),
                'diagnosis': diagnosis,
            }
            logger.info(f"  {name}: Train AUC={train_auc:.4f}, Val={val_auc:.4f}, "
                        f"Test={test_auc:.4f}, Gap={gap:.4f} → {diagnosis}")
        except Exception as e:
            logger.error(f"  Lỗi {name}: {e}")

    return gap_results


# ===========================================================================
# 2. Learning Curves
# ===========================================================================

def _learning_curves(X_train, y_train, X_val, y_val, cat_cols):
    """Plot AUC theo lượng training data tăng dần."""
    logger.info("\n📈 ĐANG TẠO LEARNING CURVES...")

    fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    n_total = len(X_train)

    results = {'LightGBM': {'train_aucs': [], 'val_aucs': [], 'sizes': []}}

    for frac in fractions:
        n_samples = int(n_total * frac)
        X_sub = X_train.iloc[:n_samples]
        y_sub = y_train.iloc[:n_samples]

        model = LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            random_state=RANDOM_STATE, verbose=-1, n_jobs=-1
        )
        model.fit(X_sub, y_sub)

        train_auc = roc_auc_score(y_sub, model.predict_proba(X_sub)[:, 1])
        val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

        results['LightGBM']['train_aucs'].append(round(train_auc, 4))
        results['LightGBM']['val_aucs'].append(round(val_auc, 4))
        results['LightGBM']['sizes'].append(n_samples)

        logger.info(f"  n={n_samples}: Train AUC={train_auc:.4f}, Val AUC={val_auc:.4f}")

    return results


# ===========================================================================
# 3. Validation Curves (AUC vs hyperparameter)
# ===========================================================================

def _validation_curves(X_train, y_train, X_val, y_val, cat_cols):
    """Plot AUC theo giá trị hyperparameter quan trọng."""
    logger.info("\n📉 ĐANG TẠO VALIDATION CURVES...")

    results = {}

    # a) max_depth
    depths = [2, 3, 4, 5, 6, 8, 10, 12, 15]
    train_aucs, val_aucs = [], []
    for d in depths:
        model = LGBMClassifier(
            n_estimators=200, max_depth=d, learning_rate=0.05,
            random_state=RANDOM_STATE, verbose=-1, n_jobs=-1
        )
        model.fit(X_train, y_train)
        train_aucs.append(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
        val_aucs.append(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
    results['max_depth'] = {
        'param_values': depths,
        'train_aucs': [round(a, 4) for a in train_aucs],
        'val_aucs': [round(a, 4) for a in val_aucs],
    }
    logger.info(f"  max_depth: best val AUC = {max(val_aucs):.4f} at depth={depths[np.argmax(val_aucs)]}")

    # b) n_estimators
    n_ests = [50, 100, 150, 200, 300, 400, 500]
    train_aucs, val_aucs = [], []
    for n in n_ests:
        model = LGBMClassifier(
            n_estimators=n, max_depth=6, learning_rate=0.05,
            random_state=RANDOM_STATE, verbose=-1, n_jobs=-1
        )
        model.fit(X_train, y_train)
        train_aucs.append(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
        val_aucs.append(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
    results['n_estimators'] = {
        'param_values': n_ests,
        'train_aucs': [round(a, 4) for a in train_aucs],
        'val_aucs': [round(a, 4) for a in val_aucs],
    }
    logger.info(f"  n_estimators: best val AUC = {max(val_aucs):.4f} at n={n_ests[np.argmax(val_aucs)]}")

    # c) learning_rate
    lrs = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
    train_aucs, val_aucs = [], []
    for lr in lrs:
        model = LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=lr,
            random_state=RANDOM_STATE, verbose=-1, n_jobs=-1
        )
        model.fit(X_train, y_train)
        train_aucs.append(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
        val_aucs.append(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
    results['learning_rate'] = {
        'param_values': lrs,
        'train_aucs': [round(a, 4) for a in train_aucs],
        'val_aucs': [round(a, 4) for a in val_aucs],
    }
    logger.info(f"  learning_rate: best val AUC = {max(val_aucs):.4f} at lr={lrs[np.argmax(val_aucs)]}")

    return results


# ===========================================================================
# 4. Cross-validation stability
# ===========================================================================

def _cv_stability(X_train, y_train, X_val, y_val, cat_cols):
    """Tính CV scores trên nhiều fold để đánh giá stability."""
    logger.info("\n🔄 CROSS-VALIDATION STABILITY...")

    X_all = pd.concat([X_train, X_val], ignore_index=True)
    y_all = pd.concat([y_train, y_val], ignore_index=True)

    tscv = TimeSeriesSplit(n_splits=5)
    fold_scores = []

    for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X_all)):
        X_tr, X_va = X_all.iloc[train_idx], X_all.iloc[val_idx]
        y_tr, y_va = y_all.iloc[train_idx], y_all.iloc[val_idx]

        model = LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            random_state=RANDOM_STATE, verbose=-1, n_jobs=-1
        )
        model.fit(X_tr, y_tr)
        auc = roc_auc_score(y_va, model.predict_proba(X_va)[:, 1])
        fold_scores.append(round(auc, 4))
        logger.info(f"  Fold {fold_i+1}: AUC = {auc:.4f}")

    mean_auc = np.mean(fold_scores)
    std_auc = np.std(fold_scores)
    logger.info(f"  Mean AUC = {mean_auc:.4f} ± {std_auc:.4f}")

    if std_auc > 0.02:
        stability = 'UNSTABLE (std > 0.02) — possible overfitting'
    else:
        stability = 'STABLE ✓ (std ≤ 0.02)'

    return {
        'fold_scores': fold_scores,
        'mean_auc': round(mean_auc, 4),
        'std_auc': round(std_auc, 4),
        'stability': stability,
    }


# ===========================================================================
# Plotting
# ===========================================================================

def _plot_all(gap_results, learning_results, validation_results, cv_results, reports_dir):
    """Tạo biểu đồ tổng hợp overfitting analysis."""

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. Train/Val/Test Gap — bar chart
    ax = axes[0, 0]
    model_names = list(gap_results.keys())
    train_aucs = [gap_results[m]['train_auc'] for m in model_names]
    val_aucs = [gap_results[m]['val_auc'] for m in model_names]
    test_aucs = [gap_results[m]['test_auc'] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.25
    ax.bar(x - width, train_aucs, width, label='Train', color='#3498db', edgecolor='black')
    ax.bar(x, val_aucs, width, label='Validation', color='#f39c12', edgecolor='black')
    ax.bar(x + width, test_aucs, width, label='Test', color='#e74c3c', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace('_', '\n') for n in model_names], fontsize=8)
    ax.set_ylabel('AUC', fontsize=11)
    ax.set_title('Train/Val/Test AUC Gap', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0.65, 0.85)

    # 2. Learning Curve
    ax = axes[0, 1]
    if 'LightGBM' in learning_results:
        lr = learning_results['LightGBM']
        ax.plot(lr['sizes'], lr['train_aucs'], 'o-', color='#3498db',
                linewidth=2, markersize=6, label='Train AUC')
        ax.plot(lr['sizes'], lr['val_aucs'], 's-', color='#e74c3c',
                linewidth=2, markersize=6, label='Validation AUC')
        ax.fill_between(lr['sizes'], lr['train_aucs'], lr['val_aucs'],
                        alpha=0.1, color='gray')
    ax.set_xlabel('Training Set Size', fontsize=11)
    ax.set_ylabel('AUC', fontsize=11)
    ax.set_title('Learning Curve (LightGBM)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # 3. Validation Curve — max_depth
    ax = axes[0, 2]
    if 'max_depth' in validation_results:
        vr = validation_results['max_depth']
        ax.plot(vr['param_values'], vr['train_aucs'], 'o-', color='#3498db',
                linewidth=2, label='Train AUC')
        ax.plot(vr['param_values'], vr['val_aucs'], 's-', color='#e74c3c',
                linewidth=2, label='Validation AUC')
    ax.set_xlabel('max_depth', fontsize=11)
    ax.set_ylabel('AUC', fontsize=11)
    ax.set_title('Validation Curve: max_depth', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # 4. Validation Curve — n_estimators
    ax = axes[1, 0]
    if 'n_estimators' in validation_results:
        vr = validation_results['n_estimators']
        ax.plot(vr['param_values'], vr['train_aucs'], 'o-', color='#3498db',
                linewidth=2, label='Train AUC')
        ax.plot(vr['param_values'], vr['val_aucs'], 's-', color='#e74c3c',
                linewidth=2, label='Validation AUC')
    ax.set_xlabel('n_estimators', fontsize=11)
    ax.set_ylabel('AUC', fontsize=11)
    ax.set_title('Validation Curve: n_estimators', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # 5. Validation Curve — learning_rate
    ax = axes[1, 1]
    if 'learning_rate' in validation_results:
        vr = validation_results['learning_rate']
        ax.plot(vr['param_values'], vr['train_aucs'], 'o-', color='#3498db',
                linewidth=2, label='Train AUC')
        ax.plot(vr['param_values'], vr['val_aucs'], 's-', color='#e74c3c',
                linewidth=2, label='Validation AUC')
    ax.set_xlabel('learning_rate', fontsize=11)
    ax.set_ylabel('AUC', fontsize=11)
    ax.set_title('Validation Curve: learning_rate', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xscale('log')

    # 6. CV Stability — box/bar plot
    ax = axes[1, 2]
    if cv_results and 'fold_scores' in cv_results:
        fold_scores = cv_results['fold_scores']
        ax.bar(range(1, len(fold_scores) + 1), fold_scores,
               color='#2ecc71', edgecolor='black', alpha=0.8)
        ax.axhline(y=cv_results['mean_auc'], color='red', linestyle='--',
                    linewidth=2, label=f"Mean = {cv_results['mean_auc']:.4f}")
        ax.fill_between(
            [0.5, len(fold_scores) + 0.5],
            cv_results['mean_auc'] - cv_results['std_auc'],
            cv_results['mean_auc'] + cv_results['std_auc'],
            alpha=0.2, color='red', label=f"±1 std = {cv_results['std_auc']:.4f}"
        )
    ax.set_xlabel('Fold', fontsize=11)
    ax.set_ylabel('AUC', fontsize=11)
    ax.set_title('Cross-Validation Stability (5-Fold)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Overfitting / Underfitting Analysis', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(reports_dir / 'overfitting_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()


def run_overfitting_analysis():
    """Chạy toàn bộ phân tích overfitting/underfitting."""
    config = load_config()
    reports_dir = Path(config['reports']['figures_dir'])
    experiments_dir = Path(config['reports']['metrics_file']).parent
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PHÂN TÍCH OVERFITTING / UNDERFITTING")
    logger.info("=" * 60)

    X_train, y_train, X_val, y_val, X_test, y_test, cat_cols, unified_categories = _load_data(config)
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # 1. Gap analysis
    gap_results = _analyze_gap(config, X_train, y_train, X_val, y_val,
                                X_test, y_test, cat_cols, unified_categories)

    # 2. Learning curves
    learning_results = _learning_curves(X_train, y_train, X_val, y_val, cat_cols)

    # 3. Validation curves
    validation_results = _validation_curves(X_train, y_train, X_val, y_val, cat_cols)

    # 4. CV stability
    cv_results = _cv_stability(X_train, y_train, X_val, y_val, cat_cols)

    # 5. Tạo biểu đồ
    logger.info("\n🎨 Đang tạo biểu đồ tổng hợp...")
    _plot_all(gap_results, learning_results, validation_results, cv_results, reports_dir)
    logger.info(f"Biểu đồ: {reports_dir / 'overfitting_analysis.png'}")

    # 6. Tổng hợp kết luận
    overall_diagnosis = []
    for name, res in gap_results.items():
        overall_diagnosis.append(f"{name}: {res['diagnosis']}")

    report = {
        'analysis': 'Overfitting / Underfitting Analysis',
        'train_val_test_gap': gap_results,
        'learning_curves': learning_results,
        'validation_curves': validation_results,
        'cv_stability': cv_results,
        'overall_diagnosis': overall_diagnosis,
        'conclusion': (
            'Phân tích cho thấy mức độ overfitting của các model thông qua: '
            '(1) So sánh AUC trên Train/Val/Test — gap nhỏ cho thấy model generalize tốt, '
            '(2) Learning curves — nếu Train và Val AUC hội tụ khi tăng data thì model fit tốt, '
            '(3) Validation curves — xác định giá trị hyperparameter tối ưu tránh overfitting, '
            '(4) CV stability — std thấp cho thấy model ổn định.'
        ),
    }

    report_path = experiments_dir / "overfitting_analysis.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    logger.info("=" * 60)
    logger.info("KẾT LUẬN OVERFITTING ANALYSIS")
    for diag in overall_diagnosis:
        logger.info(f"  {diag}")
    logger.info(f"CV Stability: {cv_results['stability']}")
    logger.info(f"Report: {report_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_overfitting_analysis()
