import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, f1_score, log_loss, brier_score_loss, roc_curve
)
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger, load_config
from src.utils.categorical import get_unified_categories, apply_unified_categorical_dtype, ordinal_encode

logger = get_logger(__name__)

# ===========================================================================
# Utilities
# ===========================================================================

def _load_and_prepare_test(config):
    """Load test set và xử lý categorical giống khi train.

    LỖI ĐÃ SỬA (tận gốc, cùng nguyên nhân với 10_train_models.py và
    11_hyperparameter_tuning.py): trước đây category dtype của X_test được
    fit ĐỘC LẬP chỉ từ chính X_test — không nhất quán với category dtype mà
    model đã thấy lúc train (chỉ từ train+val). Nay đọc thêm train/val CHỈ
    để lấy vocabulary category hợp nhất (giống hệt cách 10/11 đã làm), đảm
    bảo category dtype ở đây khớp đúng với model. Xem src/utils/categorical.py."""
    features_dir = Path(config['data']['features_dir'])
    test_df = pd.read_parquet(features_dir / "test_set.parquet")
    train_df = pd.read_parquet(features_dir / "train_set.parquet")
    val_df = pd.read_parquet(features_dir / "val_set.parquet")

    target_col = 'target'
    exclude_cols = [target_col, 'tourney_date']

    # Lưu lại các cột phụ trợ cho error analysis
    meta_cols = {}
    for col in ['surface', 'tourney_level', 'p1_rank', 'p2_rank']:
        if col in test_df.columns:
            meta_cols[col] = test_df[col].copy()

    X_test = test_df.drop(columns=[c for c in exclude_cols if c in test_df.columns])
    y_test = test_df[target_col]
    X_train_vocab = train_df.drop(columns=[c for c in exclude_cols if c in train_df.columns])
    X_val_vocab = val_df.drop(columns=[c for c in exclude_cols if c in val_df.columns])

    # Đồng nhất category dtype — hợp nhất TỪ CẢ train+val+test, khớp đúng
    # với cách model đã được train ở 10_train_models.py/11_hyperparameter_
    # tuning.py.
    cat_cols = X_test.select_dtypes(include=['object', 'category']).columns.tolist()
    unified_categories = get_unified_categories(X_train_vocab, X_val_vocab, X_test, cat_cols=cat_cols)
    X_test = apply_unified_categorical_dtype(X_test, unified_categories)

    return X_test, y_test, meta_cols, cat_cols, unified_categories


def _get_ordinal_encoded(X, unified_categories):
    """Encode categorical cho RandomForest — dùng tập category đã hợp nhất,
    khớp đúng mã hoá với lúc train (xem src/utils/categorical.py)."""
    return ordinal_encode(X, unified_categories)


def _compute_metrics(y_true, y_pred, y_proba):
    """Tính toàn bộ metrics."""
    return {
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'auc': round(roc_auc_score(y_true, y_proba), 4),
        'f1': round(f1_score(y_true, y_pred), 4),
        'log_loss': round(log_loss(y_true, y_proba), 4),
        'brier_score': round(brier_score_loss(y_true, y_proba), 4),
    }


# ===========================================================================
# Baseline strategies (không dùng ML)
# ===========================================================================

def _baseline_higher_rank(X_test, y_test):
    """Chiến lược: luôn chọn người có rank thấp hơn (= hạng cao hơn)."""
    if 'p1_rank' not in X_test.columns or 'p2_rank' not in X_test.columns:
        return None
    mask_valid = X_test['p1_rank'].notna() & X_test['p2_rank'].notna()
    if not mask_valid.any():
        return None
    y_sub = y_test[mask_valid]
    preds = (X_test.loc[mask_valid, 'p1_rank'] < X_test.loc[mask_valid, 'p2_rank']).astype(int)
    rank_diff = X_test.loc[mask_valid, 'p2_rank'] - X_test.loc[mask_valid, 'p1_rank']
    proba = 1 / (1 + np.exp(-rank_diff / 100))  # sigmoid scaling
    proba = proba.clip(0.01, 0.99)
    metrics = _compute_metrics(y_sub, preds, proba)
    metrics['note'] = f'Evaluated on {mask_valid.sum()}/{len(y_test)} rows'
    return metrics


def _baseline_follow_odds(X_test, y_test):
    """Chiến lược: luôn tin theo odds thấp hơn (=xác suất cao hơn theo nhà cái)."""
    odds_col_p1, odds_col_p2 = None, None
    for name in ['b365', 'avg_odds', 'ps', 'max_odds']:
        if f'p1_{name}' in X_test.columns and f'p2_{name}' in X_test.columns:
            odds_col_p1, odds_col_p2 = f'p1_{name}', f'p2_{name}'
            break

    if odds_col_p1 is None:
        return None

    # Chọn p1 thắng nếu odds p1 < odds p2 (odds thấp = favored)
    mask_valid = X_test[odds_col_p1].notna() & X_test[odds_col_p2].notna()
    mask_valid = mask_valid & (X_test[odds_col_p1] > 0) & (X_test[odds_col_p2] > 0)

    y_sub = y_test[mask_valid]
    preds = (X_test.loc[mask_valid, odds_col_p1] < X_test.loc[mask_valid, odds_col_p2]).astype(int)
    # Implied probability
    proba = (1 / X_test.loc[mask_valid, odds_col_p1]) / (
        1 / X_test.loc[mask_valid, odds_col_p1] + 1 / X_test.loc[mask_valid, odds_col_p2]
    )
    proba = proba.clip(0.01, 0.99)
    metrics = _compute_metrics(y_sub, preds, proba)
    metrics['note'] = f'Evaluated on {mask_valid.sum()}/{len(y_test)} rows with valid odds'
    return metrics


# ===========================================================================
# Plotting functions
# ===========================================================================

def _plot_model_comparison_table(all_metrics, reports_dir):
    """Bảng so sánh metrics — dạng heatmap table."""
    df = pd.DataFrame(all_metrics).T
    metric_cols = ['accuracy', 'auc', 'f1', 'log_loss', 'brier_score']
    df = df[[c for c in metric_cols if c in df.columns]]

    fig, ax = plt.subplots(figsize=(12, max(4, len(df) * 0.6)))
    sns.heatmap(df.astype(float), annot=True, fmt='.4f', cmap='YlGnBu',
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Score'})
    ax.set_title('Model Comparison — Test Set Metrics', fontsize=14, fontweight='bold')
    ax.set_ylabel('Model')
    plt.tight_layout()
    plt.savefig(reports_dir / 'model_comparison_table.png', dpi=150)
    plt.close()


def _plot_roc_curves(model_probas, y_test, reports_dir):
    """ROC curves overlay cho tất cả models."""
    fig, ax = plt.subplots(figsize=(8, 8))
    for name, proba in model_probas.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc_val = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.4f})', linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC=0.5)')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — All Models on Test Set', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(reports_dir / 'roc_curves_comparison.png', dpi=150)
    plt.close()


def _plot_calibration_curves(model_probas, y_test, reports_dir):
    """Calibration curves."""
    fig, ax = plt.subplots(figsize=(8, 8))
    for name, proba in model_probas.items():
        prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10, strategy='uniform')
        ax.plot(prob_pred, prob_true, marker='o', label=name, linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfectly calibrated')
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title('Calibration Curves — All Models', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(reports_dir / 'calibration_curves.png', dpi=150)
    plt.close()


def _plot_confusion_matrices(model_preds, y_test, reports_dir):
    """Grid confusion matrix cho tất cả models."""
    n_models = len(model_preds)
    cols = 2
    rows = (n_models + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    axes = axes.flatten() if n_models > 1 else [axes]

    for idx, (name, preds) in enumerate(model_preds.items()):
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['P2 wins', 'P1 wins'], yticklabels=['P2 wins', 'P1 wins'])
        axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Actual')
        axes[idx].set_xlabel('Predicted')

    # Ẩn axes thừa
    for idx in range(len(model_preds), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Confusion Matrices — All Models on Test Set', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(reports_dir / 'confusion_matrix_all_models.png', dpi=150)
    plt.close()


def _plot_error_analysis(model_preds, model_probas, y_test, meta_cols, reports_dir, best_model_name):
    """Phân tích lỗi theo surface và rank gap cho model tốt nhất."""
    best_preds = model_preds.get(best_model_name)
    if best_preds is None:
        return

    correct = (best_preds == y_test.values).astype(int)

    # --- Error by Surface ---
    if 'surface' in meta_cols:
        surface = meta_cols['surface']
        surface_accuracy = pd.DataFrame({'correct': correct, 'surface': surface.values})
        surface_acc = surface_accuracy.groupby('surface')['correct'].agg(['mean', 'count']).reset_index()
        surface_acc.columns = ['Surface', 'Accuracy', 'Count']
        surface_acc = surface_acc.sort_values('Accuracy', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(surface_acc['Surface'], surface_acc['Accuracy'], color=sns.color_palette('viridis', len(surface_acc)))
        for bar, count in zip(bars, surface_acc['Count']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'n={count}', ha='center', va='bottom', fontsize=9)
        ax.set_xlabel('Surface', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'Error Analysis: {best_model_name} Accuracy by Surface', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.axhline(y=correct.mean(), color='red', linestyle='--', alpha=0.7, label=f'Overall: {correct.mean():.3f}')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(reports_dir / 'error_analysis_by_surface.png', dpi=150)
        plt.close()

    # --- Error by Rank Gap ---
    if 'p1_rank' in meta_cols and 'p2_rank' in meta_cols:
        rank_gap = (meta_cols['p1_rank'] - meta_cols['p2_rank']).abs()
        bins = [0, 10, 30, 50, 100, 200, 500, float('inf')]
        labels = ['0-10', '11-30', '31-50', '51-100', '101-200', '201-500', '500+']
        rank_gap_cat = pd.cut(rank_gap, bins=bins, labels=labels, right=True)

        rank_analysis = pd.DataFrame({'correct': correct, 'rank_gap': rank_gap_cat.values})
        rank_acc = rank_analysis.groupby('rank_gap', observed=True)['correct'].agg(['mean', 'count']).reset_index()
        rank_acc.columns = ['Rank Gap', 'Accuracy', 'Count']

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(rank_acc['Rank Gap'].astype(str), rank_acc['Accuracy'],
                      color=sns.color_palette('magma', len(rank_acc)))
        for bar, count in zip(bars, rank_acc['Count']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'n={count}', ha='center', va='bottom', fontsize=9)
        ax.set_xlabel('Absolute Rank Difference', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'Error Analysis: {best_model_name} Accuracy by Rank Gap', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.axhline(y=correct.mean(), color='red', linestyle='--', alpha=0.7, label=f'Overall: {correct.mean():.3f}')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(reports_dir / 'error_analysis_by_rank_gap.png', dpi=150)
        plt.close()


# ===========================================================================
# Main evaluation function
# ===========================================================================

def evaluate_models():
    """Đánh giá chi tiết tất cả models trên tập Test — baseline, tuned, và baseline strategies."""
    config = load_config()
    models_dir = Path(config['model']['models_dir'])
    tuned_dir = Path(config['model']['tuned_dir'])
    reports_dir = Path(config['reports']['figures_dir'])
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BẮT ĐẦU ĐÁNH GIÁ TOÀN DIỆN TRÊN TẬP TEST")
    logger.info("=" * 60)

    X_test, y_test, meta_cols, cat_cols, unified_categories = _load_and_prepare_test(config)
    X_test_enc = _get_ordinal_encoded(X_test, unified_categories)

    # Danh sách model cần evaluate
    model_configs = {
        'CatBoost_baseline': (models_dir / "CatBoost_baseline.joblib", False),
        'LightGBM_baseline': (models_dir / "LightGBM_baseline.joblib", False),
        'XGBoost_baseline': (models_dir / "XGBoost_baseline.joblib", False),
        'RandomForest_baseline': (models_dir / "RandomForest_baseline.joblib", True),
        'CatBoost_tuned': (tuned_dir / "CatBoost_tuned.joblib", False),
        'LightGBM_tuned': (tuned_dir / "LightGBM_tuned.joblib", False),
        'XGBoost_tuned': (tuned_dir / "XGBoost_tuned.joblib", False),
        'RandomForest_tuned': (tuned_dir / "RandomForest_tuned.joblib", True),
    }

    all_metrics = {}
    model_probas = {}
    model_preds = {}

    for name, (path, use_encoded) in model_configs.items():
        if not path.exists():
            logger.warning(f"Bỏ qua {name}: không tìm thấy {path}")
            continue

        try:
            model = joblib.load(path)
            X_input = X_test_enc if use_encoded else X_test

            preds = model.predict(X_input)
            proba = model.predict_proba(X_input)[:, 1]

            metrics = _compute_metrics(y_test, preds, proba)
            all_metrics[name] = metrics
            model_probas[name] = proba
            model_preds[name] = preds

            logger.info(f"{name}: AUC={metrics['auc']}, Acc={metrics['accuracy']}, "
                        f"F1={metrics['f1']}, LogLoss={metrics['log_loss']}, Brier={metrics['brier_score']}")
        except Exception as e:
            err_msg = str(e)
            if "unseen at fit time" in err_msg or "feature names" in err_msg.lower():
                logger.error(
                    f"Lỗi khi evaluate {name}: {e}\n"
                    f"  → Đây thường là dấu hiệu {name}.joblib là model CŨ, train từ trước khi "
                    f"sửa lỗi feature/category (xem CHANGELOG_FIXES.md). Chạy lại "
                    f"`python src/pipelines/10_train_models.py` (cho *_baseline) hoặc "
                    f"`python src/pipelines/11_hyperparameter_tuning.py` (cho *_tuned) để tạo "
                    f"lại model khớp đúng schema hiện tại, rồi chạy lại evaluate."
                )
            elif "category not in the training set" in err_msg.lower():
                logger.error(
                    f"Lỗi khi evaluate {name}: {e}\n"
                    f"  → {name}.joblib được train từ TRƯỚC khi category dtype được hợp nhất với "
                    f"test set (xem src/utils/categorical.py). Chạy lại "
                    f"`python src/pipelines/10_train_models.py` hoặc "
                    f"`python src/pipelines/11_hyperparameter_tuning.py` để train lại model, "
                    f"vấn đề sẽ không còn tái diễn sau khi có model mới."
                )
            else:
                logger.error(f"Lỗi khi evaluate {name}: {e}")

    # Baseline strategies
    logger.info("\n--- Baseline Strategies (không ML) ---")
    rank_baseline = _baseline_higher_rank(X_test, y_test)
    if rank_baseline:
        all_metrics['Baseline_HigherRank'] = rank_baseline
        logger.info(f"Always-pick-higher-rank: AUC={rank_baseline['auc']}, Acc={rank_baseline['accuracy']}")

    odds_baseline = _baseline_follow_odds(X_test, y_test)
    if odds_baseline:
        all_metrics['Baseline_FollowOdds'] = odds_baseline
        logger.info(f"Always-follow-lowest-odds: AUC={odds_baseline['auc']}, Acc={odds_baseline['accuracy']}")

    # --- Plots ---
    logger.info("\nĐang tạo biểu đồ đánh giá...")

    if all_metrics:
        _plot_model_comparison_table(all_metrics, reports_dir)
    if model_probas:
        _plot_roc_curves(model_probas, y_test, reports_dir)
        _plot_calibration_curves(model_probas, y_test, reports_dir)
    if model_preds:
        _plot_confusion_matrices(model_preds, y_test, reports_dir)

    # Tìm model tốt nhất (theo AUC) để làm error analysis
    if all_metrics:
        ml_metrics = {k: v for k, v in all_metrics.items() if not k.startswith('Baseline_')}
        if ml_metrics:
            best_name = max(ml_metrics, key=lambda k: ml_metrics[k]['auc'])
            logger.info(f"\nModel tốt nhất (AUC): {best_name}")
            _plot_error_analysis(model_preds, model_probas, y_test, meta_cols, reports_dir, best_name)

    # --- Save metrics ---
    metrics_path = Path(config['reports']['metrics_file']).parent / "test_evaluation.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=4, ensure_ascii=False, default=str)

    # Lưu dạng CSV cho dễ đọc
    csv_path = metrics_path.with_suffix('.csv')
    pd.DataFrame(all_metrics).T.to_csv(csv_path)

    logger.info(f"\nĐã lưu metrics tại: {metrics_path}")
    logger.info(f"Đã lưu metrics CSV tại: {csv_path}")
    logger.info("=" * 60)
    logger.info("HOÀN THÀNH ĐÁNH GIÁ TOÀN DIỆN")
    logger.info("=" * 60)


if __name__ == "__main__":
    evaluate_models()
