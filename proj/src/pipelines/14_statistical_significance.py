"""
14_statistical_significance.py — Kiểm định ý nghĩa thống kê (CẬP NHẬT)
====================================================================

VẤN ĐỀ HỌC THUẬT CẦN GIẢI QUYẾT: bảng so sánh 4 mô hình trong
`experiments/metrics.json` chỉ đưa ra 1 con số AUC duy nhất cho mỗi model
(vd. CatBoost 0.7498 vs LightGBM 0.7466). Câu hỏi mà bất kỳ hội đồng chấm
NCKH nghiêm túc nào cũng sẽ hỏi: **"Chênh lệch 0.0032 đó có ý nghĩa thống kê
hay chỉ là nhiễu ngẫu nhiên do cách chia tập test?"** — một đề tài rigor sẽ
trả lời được câu này bằng số liệu, không phải bằng lời khẳng định suông.

CẬP NHẬT:
- Thêm TẤT CẢ tuned models (LightGBM_tuned, XGBoost_tuned, RandomForest_tuned)
- So sánh pairwise TOÀN DIỆN giữa tất cả models
- Tạo heatmap p-value matrix
- Tạo forest plot cho bootstrap CI

Script này bổ sung 2 phân tích thống kê chuẩn cho bài toán so sánh mô hình:

1. **DeLong's test** — kiểm định thống kê chuẩn (Sun & Xu, 2014) để so sánh
   AUC của 2 mô hình được đánh giá trên CÙNG một tập test (correlated ROC
   curves) — cho ra p-value, khác với so sánh 2 AUC độc lập bằng mắt.
2. **Bootstrap confidence interval** — khoảng tin cậy 95% cho AUC của từng
   model bằng cách resample tập test 1000 lần, cho thấy độ bất định của mỗi
   con số AUC thay vì chỉ là 1 điểm ước lượng.

Kết quả lưu vào `experiments/statistical_significance.json` và hiển thị lại
ở trang Analytics của Streamlit app (project/pages/Analytics.py).
"""

import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger, load_config
from src.utils.categorical import get_unified_categories, apply_unified_categorical_dtype, ordinal_encode

logger = get_logger(__name__)

N_BOOTSTRAP = 1000
RANDOM_STATE = 42


# ===========================================================================
# DeLong's test — cài đặt chuẩn theo Sun & Xu (2014)
# "Fast Implementation of DeLong's Algorithm for Comparing the Areas Under
# Correlated Receiver Operating Characteristic Curves"
# ===========================================================================

def _compute_midrank(x):
    """Tính midrank — bước trung gian bắt buộc của thuật toán DeLong gốc."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N - 1 and Z[j] == Z[j + 1]:
            j += 1
        T[i:j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted_transposed, label_1_count):
    """Tính ma trận hiệp phương sai AUC theo thuật toán DeLong nhanh."""
    m = label_1_count
    n = preds_sorted_transposed.shape[1] - m
    positive_examples = preds_sorted_transposed[:, :m]
    negative_examples = preds_sorted_transposed[:, m:]
    k = preds_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(preds_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true: np.ndarray, proba_a: np.ndarray, proba_b: np.ndarray) -> dict:
    """So sánh AUC của 2 mô hình (proba_a, proba_b) trên CÙNG tập test bằng
    DeLong's test. Trả về AUC mỗi model, chênh lệch, và p-value 2 phía."""
    y_true = np.asarray(y_true)
    order = np.argsort(-y_true)  # positive examples trước
    y_sorted = y_true[order]
    m = int(y_sorted.sum())  # số lượng positive

    preds = np.vstack([np.asarray(proba_a)[order], np.asarray(proba_b)[order]])
    aucs, cov = _fast_delong(preds, m)

    auc_a, auc_b = float(aucs[0]), float(aucs[1])
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    var = max(var, 1e-12)  # tránh chia 0 nếu 2 model giống hệt nhau
    z = (auc_a - auc_b) / np.sqrt(var)

    from scipy import stats
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "auc_a": round(auc_a, 4),
        "auc_b": round(auc_b, 4),
        "diff": round(auc_a - auc_b, 4),
        "z_score": round(float(z), 4),
        "p_value": round(float(p_value), 6),
        "significant_at_0.05": bool(p_value < 0.05),
    }


def bootstrap_auc_ci(y_true: np.ndarray, proba: np.ndarray, n_bootstrap: int = N_BOOTSTRAP) -> dict:
    """Khoảng tin cậy 95% cho AUC bằng bootstrap resampling (percentile method)."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(RANDOM_STATE)
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    n = len(y_true)

    boot_aucs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        y_boot, p_boot = y_true[idx], proba[idx]
        if len(np.unique(y_boot)) < 2:
            continue  # bỏ qua nếu bootstrap sample chỉ có 1 lớp
        boot_aucs.append(roc_auc_score(y_boot, p_boot))

    boot_aucs = np.array(boot_aucs)
    return {
        "point_estimate": round(float(roc_auc_score(y_true, proba)), 4),
        "ci_lower_95": round(float(np.percentile(boot_aucs, 2.5)), 4),
        "ci_upper_95": round(float(np.percentile(boot_aucs, 97.5)), 4),
        "std": round(float(boot_aucs.std()), 4),
        "n_bootstrap": len(boot_aucs),
    }


# ===========================================================================
# Data & model loading — tái sử dụng logic giống 12_evaluate_models.py
# ===========================================================================

def _load_and_prepare_test(config):
    """LỖI ĐÃ SỬA (cùng nguyên nhân với 10/11/12/16/18): category dtype
    trước đây fit độc lập chỉ từ X_test, không khớp với model đã train.
    Nay hợp nhất với train+val (chỉ vocabulary) bằng src/utils/categorical.py."""
    features_dir = Path(config['data']['features_dir'])
    test_df = pd.read_parquet(features_dir / "test_set.parquet")
    train_df = pd.read_parquet(features_dir / "train_set.parquet")
    val_df = pd.read_parquet(features_dir / "val_set.parquet")
    target_col = 'target'
    exclude_cols = [target_col, 'tourney_date']
    X_test = test_df.drop(columns=[c for c in exclude_cols if c in test_df.columns])
    y_test = test_df[target_col]
    X_train_vocab = train_df.drop(columns=[c for c in exclude_cols if c in train_df.columns])
    X_val_vocab = val_df.drop(columns=[c for c in exclude_cols if c in val_df.columns])

    cat_cols = X_test.select_dtypes(include=['object', 'category']).columns.tolist()
    unified_categories = get_unified_categories(X_train_vocab, X_val_vocab, X_test, cat_cols=cat_cols)
    X_test = apply_unified_categorical_dtype(X_test, unified_categories)
    return X_test, y_test, cat_cols, unified_categories


def _get_ordinal_encoded(X, unified_categories):
    """Map tất định từ tập category đã hợp nhất — khớp đúng mã hoá với lúc
    train (xem src/utils/categorical.py)."""
    return ordinal_encode(X, unified_categories)


def _plot_results(ci_results, delong_results, model_names, reports_dir):
    """Tạo biểu đồ forest plot CI và heatmap p-value."""
    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, len(model_names) * 0.8)))

    # 1. Forest plot — Bootstrap CI
    ax = axes[0]
    y_pos = range(len(model_names))
    point_estimates = [ci_results[m]['point_estimate'] for m in model_names]
    ci_lowers = [ci_results[m]['ci_lower_95'] for m in model_names]
    ci_uppers = [ci_results[m]['ci_upper_95'] for m in model_names]
    errors_lower = [pe - cl for pe, cl in zip(point_estimates, ci_lowers)]
    errors_upper = [cu - pe for pe, cu in zip(point_estimates, ci_uppers)]

    ax.errorbar(point_estimates, y_pos, xerr=[errors_lower, errors_upper],
                fmt='o', color='#2c3e50', markersize=8, capsize=5, capthick=2,
                elinewidth=2)
    for i, (pe, cl, cu) in enumerate(zip(point_estimates, ci_lowers, ci_uppers)):
        ax.text(cu + 0.002, i, f'{pe:.4f} [{cl:.4f}, {cu:.4f}]',
                va='center', fontsize=9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names, fontsize=10)
    ax.set_xlabel('AUC (95% CI)', fontsize=12)
    ax.set_title('Bootstrap 95% Confidence Intervals for AUC',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()

    # 2. Heatmap — DeLong p-values
    ax = axes[1]
    n = len(model_names)
    p_matrix = np.ones((n, n))
    for pair_key, result in delong_results.items():
        parts = pair_key.split('_vs_')
        if len(parts) == 2:
            name_a, name_b = parts[0], parts[1]
            if name_a in model_names and name_b in model_names:
                i = model_names.index(name_a)
                j = model_names.index(name_b)
                p_matrix[i, j] = result['p_value']
                p_matrix[j, i] = result['p_value']

    # Tạo annotation: hiển thị p-value + đánh dấu ý nghĩa
    annot = np.empty_like(p_matrix, dtype=object)
    for i in range(n):
        for j in range(n):
            if i == j:
                annot[i, j] = '—'
            else:
                p = p_matrix[i, j]
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                annot[i, j] = f'{p:.4f}\n{sig}'

    short_names = [name.replace('_baseline', '\n(base)').replace('_tuned', '\n(tuned)')
                   for name in model_names]
    sns.heatmap(p_matrix, annot=annot, fmt='', cmap='RdYlGn', vmin=0, vmax=0.1,
                xticklabels=short_names, yticklabels=short_names, ax=ax,
                linewidths=1, cbar_kws={'label': 'p-value'})
    ax.set_title("DeLong's Test p-value Matrix\n(* p<0.05, ** p<0.01, *** p<0.001, ns = not significant)",
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(reports_dir / 'statistical_significance.png', dpi=150, bbox_inches='tight')
    plt.close()


def run_statistical_significance():
    config = load_config()
    models_dir = Path(config['model']['models_dir'])
    tuned_dir = Path(config['model']['tuned_dir'])
    experiments_dir = Path(config.get('experiments_dir', 'experiments'))
    reports_dir = Path(config['reports']['figures_dir'])
    experiments_dir.mkdir(exist_ok=True, parents=True)
    reports_dir.mkdir(exist_ok=True, parents=True)

    logger.info("=" * 60)
    logger.info("KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ — DeLong's Test + Bootstrap CI")
    logger.info("=" * 60)

    logger.info("Đang tải dữ liệu test + các mô hình đã train...")
    X_test, y_test, cat_cols, unified_categories = _load_and_prepare_test(config)
    X_enc = _get_ordinal_encoded(X_test, unified_categories)

    # CẬP NHẬT: Thêm TẤT CẢ tuned models + baseline
    model_specs = {
        'CatBoost_tuned': (tuned_dir / "CatBoost_tuned.joblib", False),
        'LightGBM_tuned': (tuned_dir / "LightGBM_tuned.joblib", False),
        'XGBoost_tuned': (tuned_dir / "XGBoost_tuned.joblib", False),
        'RandomForest_tuned': (tuned_dir / "RandomForest_tuned.joblib", True),
        'CatBoost_baseline': (models_dir / "CatBoost_baseline.joblib", False),
        'LightGBM_baseline': (models_dir / "LightGBM_baseline.joblib", False),
        'RandomForest_baseline': (models_dir / "RandomForest_baseline.joblib", True),
    }

    model_probas = {}
    for name, (path, use_encoded) in model_specs.items():
        if not path.exists():
            logger.warning(f"Bỏ qua {name}: không tìm thấy {path}")
            continue
        try:
            model = joblib.load(path)
            if use_encoded:
                try:
                    rf_cols = list(model.feature_names_in_)
                    X_input = X_enc[rf_cols]
                except Exception:
                    # Fallback: chỉ dùng numeric columns
                    X_input = X_enc
            else:
                X_input = X_test
            model_probas[name] = model.predict_proba(X_input)[:, 1]
            logger.info(f"Đã tính prediction cho {name}")
        except Exception as e:
            logger.error(f"Lỗi khi load/predict {name}: {e}")

    if len(model_probas) < 2:
        logger.error("Cần ít nhất 2 model để so sánh. Dừng lại.")
        return

    y_true = y_test.values
    model_names = list(model_probas.keys())

    # --- 1. Bootstrap CI cho từng model ---
    logger.info(f"\n📊 Bootstrap CI ({N_BOOTSTRAP} lần resample) cho từng model...")
    ci_results = {}
    for name, proba in model_probas.items():
        ci_results[name] = bootstrap_auc_ci(y_true, proba)
        ci = ci_results[name]
        logger.info(f"  {name}: AUC = {ci['point_estimate']} "
                     f"(95% CI: [{ci['ci_lower_95']}, {ci['ci_upper_95']}])")

    # --- 2. DeLong pairwise test — TẤT CẢ các cặp ---
    logger.info("\n🔬 DeLong's test — so sánh TẤT CẢ các cặp model...")
    delong_results = {}
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            name_a = model_names[i]
            name_b = model_names[j]
            result = delong_roc_test(y_true, model_probas[name_a], model_probas[name_b])
            pair_key = f"{name_a}_vs_{name_b}"
            delong_results[pair_key] = result
            sig = "CÓ Ý NGHĨA" if result["significant_at_0.05"] else "KHÔNG có ý nghĩa"
            logger.info(f"  {name_a} vs {name_b}: "
                         f"diff={result['diff']}, p={result['p_value']:.6f} ({sig} ở mức 5%)")

    # --- 3. Tạo biểu đồ ---
    logger.info("\n🎨 Đang tạo biểu đồ...")
    _plot_results(ci_results, delong_results, model_names, reports_dir)
    logger.info(f"Biểu đồ: {reports_dir / 'statistical_significance.png'}")

    # --- 4. Tìm model tốt nhất ---
    reference_model = max(ci_results, key=lambda k: ci_results[k]['point_estimate'])

    output = {
        "reference_model": reference_model,
        "bootstrap_confidence_intervals": ci_results,
        "delong_pairwise_tests": delong_results,
        "n_models_compared": len(model_probas),
        "n_pairwise_tests": len(delong_results),
        "methodology_note": (
            "DeLong's test (Sun & Xu, 2014) so sánh AUC của 2 mô hình được đánh giá trên "
            "CÙNG 1 tập test (correlated ROC curves), cho ra p-value kiểm định giả thuyết "
            "H0: 2 model có AUC bằng nhau. Bootstrap CI ước lượng khoảng tin cậy 95% cho AUC "
            f"của từng model bằng {N_BOOTSTRAP} lần resample có hoàn lại trên tập test. "
            "Ký hiệu: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant."
        ),
    }

    out_path = experiments_dir / "statistical_significance.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("=" * 60)
    logger.info("HOÀN THÀNH KIỂM ĐỊNH THỐNG KÊ")
    logger.info(f"  Model tốt nhất: {reference_model} "
                 f"(AUC = {ci_results[reference_model]['point_estimate']})")
    logger.info(f"  Số cặp so sánh: {len(delong_results)}")
    sig_count = sum(1 for r in delong_results.values() if r['significant_at_0.05'])
    logger.info(f"  Có ý nghĩa thống kê: {sig_count}/{len(delong_results)} cặp")
    logger.info(f"  Kết quả: {out_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_statistical_significance()
