"""
16_backtest_roi.py — Backtest ROI (Return on Investment)
========================================================

MỤC ĐÍCH: Trả lời câu hỏi thực tế nhất mà hội đồng có thể hỏi:
    "Nếu dùng model này đặt cược thì lãi hay lỗ?"

Thay vì chỉ nói "Accuracy = 68%", ta chứng minh:
    "Nếu áp dụng model vào toàn bộ test set (2025+), ROI = +X%"

Các chiến lược backtest:
1. Flat betting     — đặt 1 đơn vị cho mỗi trận model dự đoán
2. Value betting    — chỉ đặt khi xác suất model > implied probability từ odds
3. Confidence tier  — đặt theo ngưỡng confidence (0.55, 0.60, 0.65)
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

logger = get_logger(__name__)


def _load_test_with_odds(config):
    """Load test set kèm cột odds để tính ROI.

    LỖI ĐÃ SỬA (tận gốc, cùng nguyên nhân với 10_train_models.py/
    11_hyperparameter_tuning.py/12_evaluate_models.py/18_overfitting_
    analysis.py): category dtype trước đây fit ĐỘC LẬP chỉ từ X_test, không
    khớp với category dtype model đã thấy lúc train. Nay hợp nhất với
    train+val (chỉ lấy vocabulary) bằng src/utils/categorical.py."""
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

    # Giữ tourney_date cho P&L timeline
    dates = test_df['tourney_date'] if 'tourney_date' in test_df.columns else None

    # Đồng nhất category dtype — hợp nhất từ cả train+val+test
    cat_cols = X_test.select_dtypes(include=['object', 'category']).columns.tolist()
    unified_categories = get_unified_categories(X_train_vocab, X_val_vocab, X_test, cat_cols=cat_cols)
    X_test = apply_unified_categorical_dtype(X_test, unified_categories)

    return X_test, y_test, dates, cat_cols, unified_categories


def _get_ordinal_encoded(X, unified_categories):
    """Encode categorical cho RandomForest — map tất định từ tập category
    đã hợp nhất (không fit OrdinalEncoder riêng, khớp đúng mã hoá với lúc
    train). Xem src/utils/categorical.py."""
    return ordinal_encode(X, unified_categories)


def _find_odds_columns(X_test):
    """Tìm cặp cột odds tốt nhất trong test set."""
    # Thứ tự ưu tiên: avg_odds > b365 > ps > max_odds
    for name in ['avg_odds', 'b365', 'ps', 'max_odds']:
        p1_col = f'p1_{name}'
        p2_col = f'p2_{name}'
        if p1_col in X_test.columns and p2_col in X_test.columns:
            # Kiểm tra có đủ data không
            valid = X_test[p1_col].notna() & X_test[p2_col].notna()
            valid = valid & (X_test[p1_col] > 1) & (X_test[p2_col] > 1)
            if valid.sum() > 100:
                return p1_col, p2_col, valid
    return None, None, None


def _confidence_label(n_bets: int) -> str:
    """Nhãn độ tin cậy mẫu — càng ít cược thì ROI càng dễ là nhiễu ngẫu nhiên,
    cần nêu rõ khi trình bày để tránh đọc nhầm ROI dương/âm là 'model có edge'."""
    if n_bets < 30:
        return "Rất thấp (n<30 — gần như chắc chắn là nhiễu, KHÔNG dùng để kết luận)"
    elif n_bets < 100:
        return "Thấp (n<100 — nên thận trọng khi diễn giải)"
    elif n_bets < 300:
        return "Trung bình"
    else:
        return "Đủ lớn"


def _bootstrap_roi_ci(y_bets, odds_bets, n_boot: int = 2000, ci: float = 0.95, seed: int = 42):
    """Bootstrap CI cho ROI trên đúng tập cược đã chọn — nhất quán với cách
    làm bootstrap CI cho AUC ở 14_statistical_significance.py, thay vì chỉ
    báo 1 con số ROI điểm (point estimate) không có khoảng tin cậy."""
    n = len(y_bets)
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    boot_rois = np.empty(n_boot)
    idx_range = np.arange(n)
    for b in range(n_boot):
        sample_idx = rng.choice(idx_range, size=n, replace=True)
        profit = np.where(y_bets[sample_idx] == 1, odds_bets[sample_idx] - 1, -1).sum()
        boot_rois[b] = profit / n * 100
    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_rois, alpha * 100))
    upper = float(np.percentile(boot_rois, (1 - alpha) * 100))
    return {"ci_low": round(lower, 2), "ci_high": round(upper, 2), "contains_zero": bool(lower <= 0 <= upper)}


def _backtest_flat(y_true, proba, odds_p1, threshold=0.5):
    """Flat betting: đặt 1 đơn vị cho p1 khi model dự đoán p1 thắng."""
    bet_mask = proba >= threshold
    if bet_mask.sum() == 0:
        return {'n_bets': 0, 'roi_pct': 0, 'profit': 0}

    n_bets = bet_mask.sum()
    y_bets = y_true[bet_mask]
    odds_bets = odds_p1[bet_mask]

    # Lợi nhuận: nếu p1 thắng (target=1) → lãi = odds - 1, nếu thua → lỗ = -1
    profit = np.where(y_bets == 1, odds_bets - 1, -1).sum()

    roi = (profit / n_bets) * 100
    win_rate = y_bets.mean() * 100
    ci = _bootstrap_roi_ci(y_bets, odds_bets)

    return {
        'strategy': 'Flat Betting',
        'threshold': threshold,
        'n_bets': int(n_bets),
        'wins': int(y_bets.sum()),
        'win_rate_pct': round(win_rate, 2),
        'total_profit_units': round(float(profit), 2),
        'roi_pct': round(float(roi), 2),
        'roi_ci_95': ci,
        'sample_confidence': _confidence_label(n_bets),
    }


def _backtest_value(y_true, proba, odds_p1, odds_p2, margin=0.0):
    """Value betting: chỉ đặt khi xác suất model > implied probability ĐÃ DE-VIG
    (trừ overround) + margin.

    LỖI ĐÃ SỬA: bản cũ dùng `implied_prob = 1/odds_p1` thô (implied probability
    CHƯA trừ biên lợi thế nhà cái/overround), không nhất quán với tab "Phân
    tích kèo cược" ở Prediction.py — nơi đã tính đúng:
        overround   = 1/odds_p1 + 1/odds_p2
        implied_p1  = (1/odds_p1) / overround   (de-vig)
    Vì overround luôn > 1 (thường 1.02–1.10 tuỳ nhà cái), implied_prob thô
    LUÔN cao hơn implied_prob thật → ngưỡng "value" bị đặt quá cao → bỏ lỡ
    nhiều kèo có value thật, đồng thời 2 nơi trong app tính cùng 1 khái niệm
    theo 2 công thức khác nhau (không nhất quán khi đối chiếu số liệu)."""
    implied_p1_raw = 1.0 / odds_p1
    implied_p2_raw = 1.0 / odds_p2
    overround = implied_p1_raw + implied_p2_raw
    implied_prob = implied_p1_raw / overround  # de-vig — khớp công thức Prediction.py

    value_mask = proba > (implied_prob + margin)

    if value_mask.sum() == 0:
        return {'n_bets': 0, 'roi_pct': 0, 'profit': 0, 'strategy': f'Value Betting (margin={margin})'}

    n_bets = value_mask.sum()
    y_bets = y_true[value_mask]
    odds_bets = odds_p1[value_mask]

    profit = np.where(y_bets == 1, odds_bets - 1, -1).sum()

    roi = (profit / n_bets) * 100
    win_rate = y_bets.mean() * 100
    ci = _bootstrap_roi_ci(y_bets, odds_bets)

    return {
        'strategy': f'Value Betting (margin={margin}, de-vig)',
        'n_bets': int(n_bets),
        'wins': int(y_bets.sum()),
        'win_rate_pct': round(win_rate, 2),
        'total_profit_units': round(float(profit), 2),
        'roi_pct': round(float(roi), 2),
        'roi_ci_95': ci,
        'sample_confidence': _confidence_label(n_bets),
    }


def _compute_cumulative_pnl(y_true, proba, odds_p1, dates, threshold=0.5):
    """Tính P&L tích lũy theo thời gian."""
    bet_mask = proba >= threshold
    pnl = np.where(y_true == 1, odds_p1 - 1, -1)
    pnl[~bet_mask] = 0
    cumulative = np.cumsum(pnl)
    return cumulative, bet_mask


def _roi_by_period(y_true, proba, odds_p1, dates, threshold=0.5, freq='Q'):
    """ROI + CI 95% theo từng giai đoạn thời gian (mặc định theo quý) — trả
    lời câu hỏi phản biện "ROI có ổn định theo thời gian không, hay chỉ là
    may rủi tập trung ở 1 giai đoạn?". Một model chỉ có edge thật nếu ROI
    dương NHẤT QUÁN qua nhiều giai đoạn, không phải dương "đẹp" nhờ dồn hết
    vào 1-2 quý may mắn rồi âm ở các quý khác (điều mà chỉ nhìn ROI gộp cả
    tập test sẽ không phát hiện ra)."""
    if dates is None:
        return []
    dates = pd.to_datetime(dates).reset_index(drop=True)
    periods = dates.dt.to_period(freq).astype(str)

    results = []
    for period in sorted(periods.unique()):
        mask = (periods == period).values
        bet_mask = mask & (proba >= threshold)
        n_bets = int(bet_mask.sum())
        if n_bets == 0:
            results.append({'period': period, 'n_bets': 0, 'roi_pct': None,
                             'roi_ci_95': None, 'sample_confidence': 'Không có kèo nào'})
            continue
        y_bets = y_true[bet_mask]
        odds_bets = odds_p1[bet_mask]
        profit = np.where(y_bets == 1, odds_bets - 1, -1).sum()
        roi = (profit / n_bets) * 100
        ci = _bootstrap_roi_ci(y_bets, odds_bets)
        results.append({
            'period': period,
            'n_bets': n_bets,
            'win_rate_pct': round(float(y_bets.mean() * 100), 2),
            'roi_pct': round(float(roi), 2),
            'roi_ci_95': ci,
            'sample_confidence': _confidence_label(n_bets),
        })
    return results


def run_backtest_roi():
    """Chạy backtest ROI cho model tốt nhất trên test set."""
    config = load_config()
    models_dir = Path(config['model']['models_dir'])
    tuned_dir = Path(config['model']['tuned_dir'])
    reports_dir = Path(config['reports']['figures_dir'])
    experiments_dir = Path(config['reports']['metrics_file']).parent
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BACKTEST ROI — ĐÁNH GIÁ GIÁ TRỊ ỨNG DỤNG THỰC TẾ")
    logger.info("=" * 60)

    X_test, y_test, dates, cat_cols, unified_categories = _load_test_with_odds(config)
    X_test_enc = _get_ordinal_encoded(X_test, unified_categories)

    # Tìm cột odds
    odds_p1_col, odds_p2_col, valid_mask = _find_odds_columns(X_test)
    if odds_p1_col is None:
        logger.error("Không tìm thấy cột odds hợp lệ trong test set. Dừng backtest.")
        return

    logger.info(f"Sử dụng odds: {odds_p1_col}, {odds_p2_col}")
    logger.info(f"Số trận có odds hợp lệ: {valid_mask.sum()}/{len(y_test)}")

    # Load tất cả models
    model_configs = {
        'CatBoost_tuned': (tuned_dir / "CatBoost_tuned.joblib", False),
        'LightGBM_tuned': (tuned_dir / "LightGBM_tuned.joblib", False),
        'XGBoost_tuned': (tuned_dir / "XGBoost_tuned.joblib", False),
        'RandomForest_tuned': (tuned_dir / "RandomForest_tuned.joblib", True),
        'CatBoost_baseline': (models_dir / "CatBoost_baseline.joblib", False),
        'LightGBM_baseline': (models_dir / "LightGBM_baseline.joblib", False),
    }

    all_results = {}
    # LỖI ĐÃ SỬA: bản cũ chọn "best_model_name" bằng ROI đo trên CHÍNH test
    # set — với 6 model × 7 chiến lược = 42 phép so sánh, gần như chắc chắn
    # có ≥1 kết quả "đẹp" chỉ do may rủi (multiple comparisons / data
    # snooping), rồi lại dùng model "thắng" đó để vẽ biểu đồ threshold +
    # cumulative P&L — tạo ảo giác "đã tìm ra chiến lược tốt nhất" trong khi
    # thực chất là overfit vào chính tập test dùng để đánh giá.
    # NAY: cố định model đầu tàu = CatBoost_tuned — nhất quán với toàn bộ
    # phần còn lại của báo cáo (CatBoost là model chính, đã chọn dựa trên
    # AUC/overfitting-gap từ TRƯỚC, không dựa trên ROI). Các model khác vẫn
    # được backtest đầy đủ để so sánh, nhưng KHÔNG dùng ROI để "bầu" ra 1
    # model tốt nhất.
    PRIMARY_MODEL = 'CatBoost_tuned'

    for name, (path, use_encoded) in model_configs.items():
        if not path.exists():
            logger.warning(f"Bỏ qua {name}: không tìm thấy {path}")
            continue

        try:
            model = joblib.load(path)
            X_input = X_test_enc if use_encoded else X_test

            # Predict trên subset có odds
            if use_encoded:
                try:
                    rf_cols = list(model.feature_names_in_)
                    X_sub = X_input[rf_cols].iloc[valid_mask.values]
                except Exception:
                    logger.warning(f"Bỏ qua {name}: không khớp features")
                    continue
            else:
                X_sub = X_input[valid_mask]

            try:
                proba = model.predict_proba(X_sub)[:, 1]
            except Exception as e:
                # LỖI ĐÃ BIẾT, TÁI DIỄN: XGBoost dùng native categorical
                # encoding (enable_categorical=True) — nếu category dtype của
                # X_test chứa 1 giá trị (vd. 'Carpet') mà model KHÔNG thấy
                # lúc train (vì được fit trên train+val ở
                # 11_hyperparameter_tuning.py, độc lập với tập test), XGBoost
                # raise lỗi thẳng thay vì tự động fallback như CatBoost/
                # LightGBM. Đây là hạn chế đã biết của native categorical
                # handling trong XGBoost, KHÔNG thể vá chỉ bằng cách định
                # dạng lại dữ liệu ở script này — cần fit lại model với
                # category dtype hợp nhất train+val+test (xem
                # 18_overfitting_analysis.py._load_data). Ghi nhận rõ ràng
                # trong report thay vì âm thầm biến mất khỏi bảng kết quả.
                logger.error(f"  {name}: bỏ qua — lỗi category không xác định lúc predict ({e})")
                all_results[name] = {
                    'model': name,
                    'skipped': True,
                    'skip_reason': (
                        "Model gặp giá trị category (vd. mặt sân 'Carpet') chưa từng thấy lúc "
                        "huấn luyện. Đây là hạn chế đã biết của XGBoost native categorical "
                        "encoding — không ảnh hưởng model chính (CatBoost)."
                    ),
                    'strategies': [],
                }
                continue

            y_sub = y_test[valid_mask].values
            odds_p1 = X_test.loc[valid_mask, odds_p1_col].values.astype(float)
            odds_p2 = X_test.loc[valid_mask, odds_p2_col].values.astype(float)

            logger.info(f"\n--- {name} ---")

            model_results = {'model': name, 'skipped': False, 'strategies': []}

            # 1. Flat betting với nhiều threshold
            for thresh in [0.50, 0.55, 0.60, 0.65]:
                result = _backtest_flat(y_sub, proba, odds_p1, threshold=thresh)
                result['strategy'] = f'Flat (threshold={thresh})'
                model_results['strategies'].append(result)
                logger.info(f"  {result['strategy']}: {result['n_bets']} bets, "
                            f"Win={result['win_rate_pct']:.1f}%, ROI={result['roi_pct']:.2f}%")

            # 2. Value betting (de-vig)
            for margin in [0.0, 0.05, 0.10]:
                result = _backtest_value(y_sub, proba, odds_p1, odds_p2, margin=margin)
                model_results['strategies'].append(result)
                logger.info(f"  {result['strategy']}: {result['n_bets']} bets, "
                            f"Win={result.get('win_rate_pct', 0):.1f}%, ROI={result['roi_pct']:.2f}%")

            all_results[name] = model_results

        except Exception as e:
            logger.error(f"Lỗi khi backtest {name}: {e}")
            all_results[name] = {'model': name, 'skipped': True, 'skip_reason': str(e), 'strategies': []}

    ran_results = {k: v for k, v in all_results.items() if not v.get('skipped') and v.get('strategies')}
    if not ran_results:
        logger.error("Không có model nào chạy được backtest. Dừng lại.")
        return

    best_model_name = PRIMARY_MODEL if PRIMARY_MODEL in ran_results else next(iter(ran_results))
    odds_coverage_pct = valid_mask.sum() / len(y_test) * 100
    if odds_coverage_pct < 20:
        logger.warning(
            f"⚠️ Độ phủ odds chỉ {odds_coverage_pct:.1f}% ({valid_mask.sum()}/{len(y_test)} trận) — "
            f"mẫu nhỏ, có thể lệch về phía main draw/giải lớn. Diễn giải ROI cần thận trọng."
        )

    # -----------------------------------------------------------------------
    # Biểu đồ
    # -----------------------------------------------------------------------
    logger.info("\n🎨 Đang tạo biểu đồ backtest...")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 1. ROI comparison by model (flat betting, threshold=0.5) — chỉ các model
    # chạy được; model bị skip (vd. XGBoost gặp category lạ) hiển thị riêng
    # trong bảng ở Analytics.py, không vẽ vào biểu đồ này.
    model_names = list(ran_results.keys())
    rois = [ran_results[m]['strategies'][0]['roi_pct'] for m in model_names]
    colors = ['#27ae60' if r > 0 else '#e74c3c' for r in rois]
    bars = axes[0].bar(range(len(model_names)), rois, color=colors,
                       edgecolor='black', linewidth=1)
    for bar, roi_val in zip(bars, rois):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + (0.3 if roi_val >= 0 else -0.8),
                     f'{roi_val:+.1f}%', ha='center', fontsize=11, fontweight='bold')
    axes[0].set_xticks(range(len(model_names)))
    axes[0].set_xticklabels([n.replace('_', '\n') for n in model_names],
                             fontsize=9, rotation=0)
    axes[0].axhline(y=0, color='black', linewidth=1)
    axes[0].set_ylabel('ROI (%)', fontsize=12)
    axes[0].set_title('ROI by Model (Flat Betting)', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # 2. ROI by confidence threshold (model đầu tàu cố định — CatBoost_tuned)
    if best_model_name:
        thresholds = [0.50, 0.55, 0.60, 0.65]
        best_strats = ran_results[best_model_name]['strategies'][:4]
        roi_by_thresh = [s['roi_pct'] for s in best_strats]
        n_bets_by_thresh = [s['n_bets'] for s in best_strats]

        color2 = ['#2ecc71' if r > 0 else '#e74c3c' for r in roi_by_thresh]
        bars2 = axes[1].bar(range(len(thresholds)), roi_by_thresh, color=color2,
                            edgecolor='black', linewidth=1)
        for bar, roi_v, nb in zip(bars2, roi_by_thresh, n_bets_by_thresh):
            axes[1].text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + (0.3 if roi_v >= 0 else -0.8),
                         f'{roi_v:+.1f}%\n(n={nb})', ha='center', fontsize=10)
        axes[1].set_xticks(range(len(thresholds)))
        axes[1].set_xticklabels([f'≥{t}' for t in thresholds], fontsize=11)
        axes[1].axhline(y=0, color='black', linewidth=1)
        axes[1].set_xlabel('Confidence Threshold', fontsize=12)
        axes[1].set_ylabel('ROI (%)', fontsize=12)
        axes[1].set_title(f'ROI by Threshold — {best_model_name}',
                          fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)

    # 3. Cumulative P&L cho model đầu tàu (CatBoost_tuned — cố định, không snoop)
    if best_model_name:
        path_best, use_enc = model_configs[best_model_name]
        model_best = joblib.load(path_best)
        X_input_best = X_test_enc if use_enc else X_test
        if use_enc:
            rf_cols = list(model_best.feature_names_in_)
            X_sub_best = X_input_best[rf_cols].iloc[valid_mask.values]
        else:
            X_sub_best = X_input_best[valid_mask]
        proba_best = model_best.predict_proba(X_sub_best)[:, 1]
        y_sub = y_test[valid_mask].values
        odds_p1 = X_test.loc[valid_mask, odds_p1_col].values.astype(float)

        cum_pnl, bet_mask = _compute_cumulative_pnl(y_sub, proba_best, odds_p1,
                                                     dates, threshold=0.5)
        axes[2].plot(range(len(cum_pnl)), cum_pnl, color='#2c3e50', linewidth=1.5)
        axes[2].fill_between(range(len(cum_pnl)), cum_pnl, 0,
                             where=cum_pnl >= 0, color='#27ae60', alpha=0.3)
        axes[2].fill_between(range(len(cum_pnl)), cum_pnl, 0,
                             where=cum_pnl < 0, color='#e74c3c', alpha=0.3)
        axes[2].axhline(y=0, color='black', linewidth=1)
        axes[2].set_xlabel('Match Number', fontsize=12)
        axes[2].set_ylabel('Cumulative P&L (units)', fontsize=12)
        axes[2].set_title(f'Cumulative P&L — {best_model_name}',
                          fontsize=14, fontweight='bold')
        axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(reports_dir / 'backtest_roi.png', dpi=150, bbox_inches='tight')
    plt.close()

    # -----------------------------------------------------------------------
    # ROI theo giai đoạn thời gian (độ ổn định) — model đầu tàu
    # -----------------------------------------------------------------------
    roi_by_period = []
    if best_model_name and dates is not None:
        dates_sub = dates[valid_mask].reset_index(drop=True)
        roi_by_period = _roi_by_period(y_sub, proba_best, odds_p1, dates_sub, threshold=0.5)

        periods_with_bets = [r for r in roi_by_period if r['n_bets'] > 0]
        if periods_with_bets:
            fig_p, ax_p = plt.subplots(figsize=(max(10, len(periods_with_bets) * 1.2), 6))
            period_labels = [r['period'] for r in periods_with_bets]
            period_rois = [r['roi_pct'] for r in periods_with_bets]
            period_colors = ['#27ae60' if r >= 0 else '#e74c3c' for r in period_rois]
            err_low = [r['roi_pct'] - r['roi_ci_95']['ci_low'] if r['roi_ci_95'] else 0 for r in periods_with_bets]
            err_high = [r['roi_ci_95']['ci_high'] - r['roi_pct'] if r['roi_ci_95'] else 0 for r in periods_with_bets]
            ax_p.bar(range(len(period_labels)), period_rois, color=period_colors,
                     edgecolor='black', linewidth=1,
                     yerr=[err_low, err_high], capsize=4, ecolor='#555555')
            for i, r in enumerate(periods_with_bets):
                ax_p.text(i, r['roi_pct'] + (2 if r['roi_pct'] >= 0 else -2),
                          f"n={r['n_bets']}", ha='center', fontsize=9)
            ax_p.set_xticks(range(len(period_labels)))
            ax_p.set_xticklabels(period_labels, fontsize=10, rotation=30, ha='right')
            ax_p.axhline(y=0, color='black', linewidth=1)
            ax_p.set_ylabel('ROI (%) — thanh lỗi = CI 95%', fontsize=12)
            ax_p.set_title(f'ROI theo giai đoạn (quý) — {best_model_name}\n'
                           f'Độ ổn định thực sự: model có edge chỉ khi ROI dương NHẤT QUÁN qua các quý',
                           fontsize=13, fontweight='bold')
            ax_p.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(reports_dir / 'backtest_roi_by_period.png', dpi=150, bbox_inches='tight')
            plt.close()

    # -----------------------------------------------------------------------
    # Save report
    # -----------------------------------------------------------------------
    primary_flat_roi = ran_results[best_model_name]['strategies'][0]['roi_pct'] if best_model_name in ran_results else None
    skipped_models = {k: v.get('skip_reason') for k, v in all_results.items() if v.get('skipped')}

    n_periods_positive = sum(1 for r in roi_by_period if r.get('roi_pct') is not None and r['roi_pct'] > 0)
    n_periods_negative = sum(1 for r in roi_by_period if r.get('roi_pct') is not None and r['roi_pct'] <= 0)
    n_periods_total = n_periods_positive + n_periods_negative

    report = {
        'analysis': 'Backtest ROI Analysis',
        'description': (
            'Đánh giá giá trị ứng dụng thực tế của model bằng cách mô phỏng '
            'chiến lược đặt cược trên tập test (dữ liệu model chưa thấy). '
            'ROI cho biết nếu đặt cược theo dự đoán của model thì lãi/lỗ bao nhiêu. '
            'Đây là chỉ số đánh giá bổ sung mang tính minh hoạ, KHÔNG phải khuyến '
            'nghị đặt cược — xem roi_ci_95/sample_confidence để đánh giá độ tin cậy '
            'trước khi diễn giải bất kỳ con số ROI nào.'
        ),
        'odds_columns_used': [odds_p1_col, odds_p2_col],
        'n_matches_with_odds': int(valid_mask.sum()),
        'n_matches_total': len(y_test),
        'odds_coverage_pct': round(odds_coverage_pct, 2),
        'odds_coverage_warning': (
            odds_coverage_pct < 20
        ),
        'results_by_model': all_results,
        'skipped_models': skipped_models,
        'primary_model': best_model_name,
        'primary_model_note': (
            'Model đầu tàu được CỐ ĐỊNH là CatBoost_tuned (model chính của toàn bộ '
            'đề tài, chọn theo AUC/overfitting-gap từ trước) — KHÔNG chọn theo ROI '
            'cao nhất trên test set để tránh data snooping (42 phép so sánh '
            '6 model × 7 chiến lược gần như chắc chắn có 1 kết quả "đẹp" do may rủi).'
        ),
        'primary_model_flat_roi_pct': primary_flat_roi,
        'roi_by_period': roi_by_period,
        'roi_stability_note': (
            f"{n_periods_positive}/{n_periods_total} quý có ROI dương" if n_periods_total else
            "Không đủ dữ liệu theo quý để đánh giá độ ổn định."
        ),
    }

    report_path = experiments_dir / "backtest_roi.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    logger.info("=" * 60)
    logger.info("KẾT QUẢ BACKTEST ROI")
    for name, res in ran_results.items():
        flat = res['strategies'][0]
        ci = flat.get('roi_ci_95') or {}
        logger.info(f"  {name}: ROI={flat['roi_pct']:+.2f}% "
                    f"(95% CI [{ci.get('ci_low', '—')}, {ci.get('ci_high', '—')}]), "
                    f"Win={flat['win_rate_pct']:.1f}%, N={flat['n_bets']}")
    for name, reason in skipped_models.items():
        logger.info(f"  {name}: BỎ QUA — {reason}")
    logger.info(f"Model đầu tàu (cố định, không snoop): {best_model_name} "
                f"(ROI={primary_flat_roi:+.2f}%)" if primary_flat_roi is not None else
                f"Model đầu tàu: {best_model_name}")
    logger.info(f"Report: {report_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_backtest_roi()
