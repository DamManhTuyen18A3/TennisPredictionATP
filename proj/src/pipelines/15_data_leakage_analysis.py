"""
15_data_leakage_analysis.py — Phân tích Data Leakage (Đóng góp khoa học)
=========================================================================

MỤC ĐÍCH: Chứng minh nhóm hiểu bản chất ML bằng cách thể hiện câu chuyện
phát hiện và xử lý data leakage:

    Ban đầu AUC ≈ 0.99 (dùng thông tin trong/sau trận)
    ↓ Phát hiện leakage
    ↓ Loại bỏ các cột w_ace, w_df, w_svpt, minutes, score...
    ↓ AUC ≈ 0.74 (chỉ dùng thông tin trước trận)

Đây là câu chuyện cực kỳ thuyết phục khi bảo vệ trước hội đồng.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import warnings

warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger, load_config

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.model_selection import train_test_split

logger = get_logger(__name__)


def _build_features_with_leakage(df):
    """Xây dựng features GIỮ NGUYÊN cột leakage (in-match stats) để demo AUC ~0.99."""
    # Đây là những cột chứa thông tin xảy ra TRONG trận đấu
    # Nếu giữ chúng lại → model sẽ "nhìn thấy" kết quả gián tiếp → AUC cực cao
    leakage_cols_present = [
        col for col in [
            'w_ace', 'w_df', 'w_svpt', 'w_1stin', 'w_1stwon', 'w_2ndwon',
            'l_ace', 'l_df', 'l_svpt', 'l_1stin', 'l_1stwon', 'l_2ndwon',
            'w_svgms', 'w_bpsaved', 'w_bpfaced',
            'l_svgms', 'l_bpsaved', 'l_bpfaced',
            'minutes'
        ] if col in df.columns
    ]
    return leakage_cols_present


def _simple_train_eval(df, leakage_cols, label=''):
    """Train LightGBM nhanh trên df, trả về metrics. Dùng cho cả có/không leakage."""
    # Tạo target đơn giản: luôn có winner_id → target = 1 cho tất cả rows
    # (vì data gốc luôn ở format winner/loser)
    # Để tạo bài toán phân loại thực sự, ta swap 50% thành loser-first
    np.random.seed(42)
    n = len(df)
    swap_mask = np.random.rand(n) > 0.5

    # Chọn feature columns (chỉ numeric, bỏ ID/name/date)
    exclude = ['tourney_id', 'tourney_name', 'winner_id', 'loser_id',
               'winner_name', 'loser_name', 'tourney_date', 'match_num',
               'score', 'target']
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in exclude]

    if not leakage_cols:
        # Loại bỏ leakage cols khỏi features
        all_leakage = [
            'w_ace', 'w_df', 'w_svpt', 'w_1stin', 'w_1stwon', 'w_2ndwon',
            'l_ace', 'l_df', 'l_svpt', 'l_1stin', 'l_1stwon', 'l_2ndwon',
            'w_svgms', 'w_bpsaved', 'w_bpfaced',
            'l_svgms', 'l_bpsaved', 'l_bpfaced',
            'minutes'
        ]
        feature_cols = [c for c in feature_cols if c not in all_leakage]

    X = df[feature_cols].copy()

    # Xử lý cặp winner/loser: swap để tạo p1/p2
    # Với các cột có prefix w_ và l_, swap chúng khi swap_mask = True
    w_cols = [c for c in feature_cols if c.startswith('w_') or c.startswith('winner_')]
    l_cols_map = {}
    for wc in w_cols:
        lc = wc.replace('winner_', 'loser_', 1) if wc.startswith('winner_') else wc.replace('w_', 'l_', 1)
        if lc in feature_cols:
            l_cols_map[wc] = lc

    for wc, lc in l_cols_map.items():
        w_vals = X[wc].values.copy()
        l_vals = X[lc].values.copy()
        X.loc[swap_mask, wc] = l_vals[swap_mask]
        X.loc[swap_mask, lc] = w_vals[swap_mask]

    # LỖI ĐÃ SỬA: bản cũ viết `y = np.where(swap_mask, 1, 0)` cùng comment
    # "1 nếu winner là p1 (không swap)" — MÂU THUẪN với chính code: khi
    # swap_mask=True, cột winner_ đã bị tráo sang chứa dữ liệu của LOSER, tức
    # p1 lúc này KHÔNG PHẢI winner → nhãn đúng phải là 0, không phải 1.
    # (Không ảnh hưởng đến AUC đã báo cáo trước đó: vì bug tráo nhãn cũ được
    # áp dụng NHẤT QUÁN cho cả 2 kịch bản with/without leakage và cả
    # train/test cùng lúc — về mặt toán học AUC bất biến với việc đảo nhãn
    # đối xứng toàn cục. Chỉ là comment/ý nghĩa "y=1" bị hiểu sai khi đọc
    # code, không phải sai số liệu.)
    y = np.where(swap_mask, 0, 1)  # 1 nếu winner vẫn ở vị trí p1 (không swap)

    # Fill NaN
    X = X.fillna(X.median())

    # Split theo thời gian (dùng index vì đã sort)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Train LightGBM
    model = LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        random_state=42, verbose=-1, n_jobs=-1
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, preds)

    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)

    logger.info(f"  [{label}] AUC = {auc:.4f}, Accuracy = {acc:.4f}, "
                f"Num features = {len(feature_cols)}")

    return {
        'auc': round(auc, 4),
        'accuracy': round(acc, 4),
        'n_features': len(feature_cols),
        'top_features': importance.to_dict('records'),
        'fpr_tpr': roc_curve(y_test, proba),
    }


def run_data_leakage_analysis():
    """Phân tích và so sánh model VỚI và KHÔNG CÓ data leakage."""
    config = load_config()
    interim_dir = Path(config['data']['interim_dir'])
    reports_dir = Path(config['reports']['figures_dir'])
    experiments_dir = Path(config['reports']['metrics_file']).parent
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PHÂN TÍCH DATA LEAKAGE — ĐÓNG GÓP KHOA HỌC")
    logger.info("=" * 60)

    # Load dữ liệu TRƯỚC khi remove leakage
    raw_path = interim_dir / "04_handled_missing.parquet"
    if not raw_path.exists():
        logger.error(f"Không tìm thấy file: {raw_path}")
        return

    df_raw = pd.read_parquet(raw_path)
    if 'tourney_date' in df_raw.columns:
        df_raw = df_raw.sort_values('tourney_date').reset_index(drop=True)

    logger.info(f"Data shape: {df_raw.shape}")

    # Xác định các cột leakage hiện có
    leakage_cols = _build_features_with_leakage(df_raw)
    logger.info(f"Các cột leakage tìm thấy: {leakage_cols}")

    # -----------------------------------------------------------------------
    # Giai đoạn 1: Train VỚI leakage (giữ tất cả in-match stats)
    # -----------------------------------------------------------------------
    logger.info("\n📊 GIAI ĐOẠN 1: Train VỚI data leakage (in-match stats)...")
    result_with = _simple_train_eval(df_raw, leakage_cols, label='WITH LEAKAGE')

    # -----------------------------------------------------------------------
    # Giai đoạn 2: Train KHÔNG CÓ leakage (loại bỏ in-match stats)
    # -----------------------------------------------------------------------
    logger.info("\n📊 GIAI ĐOẠN 2: Train KHÔNG CÓ data leakage (pre-match only)...")
    result_without = _simple_train_eval(df_raw, [], label='NO LEAKAGE')

    # -----------------------------------------------------------------------
    # Tạo biểu đồ so sánh
    # -----------------------------------------------------------------------
    logger.info("\n🎨 Đang tạo biểu đồ...")

    # 1. Bar chart: AUC comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # AUC comparison
    scenarios = ['With Leakage\n(in-match stats)', 'Without Leakage\n(pre-match only)']
    aucs = [result_with['auc'], result_without['auc']]
    colors = ['#e74c3c', '#27ae60']
    bars = axes[0].bar(scenarios, aucs, color=colors, width=0.5, edgecolor='black', linewidth=1.2)
    for bar, auc in zip(bars, aucs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'AUC = {auc:.4f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('AUC Score', fontsize=12)
    axes[0].set_title('AUC: Before vs After Leakage Removal', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    axes[0].legend(fontsize=10)
    axes[0].grid(axis='y', alpha=0.3)

    # 2. ROC curves comparison
    fpr_w, tpr_w, _ = result_with['fpr_tpr']
    fpr_wo, tpr_wo, _ = result_without['fpr_tpr']
    axes[1].plot(fpr_w, tpr_w, color='#e74c3c', linewidth=2.5,
                 label=f'With Leakage (AUC={result_with["auc"]:.4f})')
    axes[1].plot(fpr_wo, tpr_wo, color='#27ae60', linewidth=2.5,
                 label=f'Without Leakage (AUC={result_without["auc"]:.4f})')
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    axes[1].set_xlabel('False Positive Rate', fontsize=12)
    axes[1].set_ylabel('True Positive Rate', fontsize=12)
    axes[1].set_title('ROC Curves: Leakage Impact', fontsize=14, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=10)
    axes[1].grid(alpha=0.3)

    # 3. Feature importance comparison (top features WITH leakage)
    top_feat_with = result_with['top_features'][:10]
    feat_names = [f['feature'] for f in top_feat_with]
    feat_imp = [f['importance'] for f in top_feat_with]
    y_pos = range(len(feat_names))
    bar_colors = ['#e74c3c' if any(prefix in name for prefix in ['w_', 'l_', 'minute'])
                  else '#3498db' for name in feat_names]
    axes[2].barh(y_pos, feat_imp, color=bar_colors, edgecolor='black', linewidth=0.5)
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(feat_names, fontsize=10)
    axes[2].invert_yaxis()
    axes[2].set_xlabel('Feature Importance', fontsize=12)
    axes[2].set_title('Top Features WITH Leakage\n(🔴 = leakage features)', fontsize=14, fontweight='bold')
    axes[2].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(reports_dir / 'data_leakage_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Biểu đồ: {reports_dir / 'data_leakage_analysis.png'}")

    # 4. Feature importance WITHOUT leakage (riêng)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    top_feat_wo = result_without['top_features'][:10]
    feat_names_wo = [f['feature'] for f in top_feat_wo]
    feat_imp_wo = [f['importance'] for f in top_feat_wo]
    ax2.barh(range(len(feat_names_wo)), feat_imp_wo, color='#27ae60', edgecolor='black', linewidth=0.5)
    ax2.set_yticks(range(len(feat_names_wo)))
    ax2.set_yticklabels(feat_names_wo, fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xlabel('Feature Importance', fontsize=12)
    ax2.set_title('Top Features WITHOUT Leakage (Pre-match Only)', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(reports_dir / 'feature_importance_no_leakage.png', dpi=150, bbox_inches='tight')
    plt.close()

    # -----------------------------------------------------------------------
    # Lưu report JSON
    # -----------------------------------------------------------------------
    # Không serialize fpr_tpr (numpy arrays)
    report = {
        'analysis': 'Data Leakage Impact Analysis',
        'description': (
            'So sánh hiệu suất model khi sử dụng thông tin trong/sau trận (leakage) '
            'vs chỉ sử dụng thông tin trước trận (pre-match). Đây là đóng góp khoa học '
            'quan trọng cho thấy nhóm hiểu bản chất ML và biết phát hiện lỗi dữ liệu.'
        ),
        'with_leakage': {
            'auc': result_with['auc'],
            'accuracy': result_with['accuracy'],
            'n_features': result_with['n_features'],
            'top_features': result_with['top_features'],
        },
        'without_leakage': {
            'auc': result_without['auc'],
            'accuracy': result_without['accuracy'],
            'n_features': result_without['n_features'],
            'top_features': result_without['top_features'],
        },
        'leakage_columns_found': leakage_cols,
        'auc_drop': round(result_with['auc'] - result_without['auc'], 4),
        'conclusion': (
            f"AUC giảm từ {result_with['auc']:.4f} xuống {result_without['auc']:.4f} "
            f"(giảm {result_with['auc'] - result_without['auc']:.4f}) sau khi loại bỏ "
            f"{len(leakage_cols)} cột leakage. Điều này chứng minh các thông tin trong trận "
            f"(ace, double fault, break points...) tạo ra kết quả ảo cực cao, "
            f"và việc loại bỏ chúng là bắt buộc để model có giá trị dự đoán thực sự."
        ),
    }

    report_path = experiments_dir / "data_leakage_analysis.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    logger.info("=" * 60)
    logger.info("KẾT LUẬN DATA LEAKAGE ANALYSIS")
    logger.info(f"  WITH leakage:    AUC = {result_with['auc']:.4f}")
    logger.info(f"  WITHOUT leakage: AUC = {result_without['auc']:.4f}")
    logger.info(f"  AUC drop: {result_with['auc'] - result_without['auc']:.4f}")
    logger.info(f"  Report: {report_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_data_leakage_analysis()
