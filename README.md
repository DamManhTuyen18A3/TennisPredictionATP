# 🎾 ATP Match Predictor — Dự đoán Kết quả Quần vợt bằng Machine Learning

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Tổng quan

Dự án **Nghiên cứu Khoa học (NCKH)** xây dựng hệ thống dự đoán kết quả trận đấu quần vợt nam (ATP) sử dụng các thuật toán Machine Learning: **CatBoost**, **LightGBM**, **XGBoost**, và **Random Forest**, kèm ứng dụng demo Streamlit trực quan.

### Mục tiêu nghiên cứu
1. Xây dựng pipeline ML hoàn chỉnh, có kiểm soát chất lượng, từ dữ liệu thô đến mô hình dự đoán.
2. So sánh hiệu quả của 4 thuật toán Gradient Boosting & Ensemble trên bài toán dự đoán thể thao.
3. Phân tích các yếu tố ảnh hưởng lớn nhất đến kết quả trận đấu (SHAP explainability).
4. Xác minh mô hình ML có vượt trội hơn các chiến lược đơn giản (luôn chọn hạng cao hơn, luôn theo tỷ lệ cược thấp nhất) và đối chiếu với kết quả trong y văn (literature benchmark).
5. Đánh giá độ tin cậy thống kê (DeLong's test, bootstrap CI), mức độ overfitting, và giá trị ứng dụng thực tế (backtest ROI có kiểm định độ ổn định theo thời gian) — không chỉ báo cáo 1 con số accuracy.

### Phạm vi dữ liệu

| Thuộc tính | Chi tiết |
|-----------|---------|
| **Nguồn** | ATP match data 2020–2025 kết hợp tỷ lệ cược (odds) nhà cái |
| **Giới tính** | Chỉ nam (ATP) |
| **Cấp giải** | ITF Futures/Challenger (phần lớn), ATP 250/500/1000, Grand Slam |
| **Số trận** | Train ≈ 91,913 · Test ≈ 32,168 (theo log chạy thật; xem `data/features/*.parquet` để có số chính xác của cả 3 tập, bao gồm Val) |
| **Khoảng thời gian** | Train < 2024, Val = 2024, Test ≥ 2025 (chia theo thời gian, không random) |

> ⚠️ **Giới hạn nghiên cứu**: phần lớn dữ liệu là giải nhỏ (ITF Futures/Challenger), không dùng dữ liệu trong trận/chấn thương/thời tiết, độ phủ odds thấp (~6%). Xem chi tiết đầy đủ tại [docs/methodology.md](docs/methodology.md) (mục 4 — Limitations & Future Work) hoặc trực tiếp trong app ở tab **Analytics → ⚠️ Giới hạn nghiên cứu**.

## Kiến trúc Pipeline

Pipeline gồm **18 bước module hoá**, đánh số theo thứ tự chạy (một số bước phân tích 14–18 độc lập với nhau, chạy sau khi đã có model):

```
src/pipelines/
├── 01_load_data.py              ← Load CSV → Parquet
├── 02_data_validation.py        ← Kiểm tra schema, duplicate
├── 03_clean_schema.py           ← Chuẩn hoá tên cột, kiểu dữ liệu
├── 04_handle_missing.py         ← Xử lý NaN (median, fillna có chủ đích)
├── 05_remove_data_leakage.py    ← Loại bỏ thông tin trong/sau trận (w_ace, score...)
├── 06_eda.py                    ← EDA toàn diện (10+ biểu đồ)
├── 07_feature_engineering.py    ← Elo rating, H2H, biến đổi đối xứng (Player A/B)
├── 08_feature_selection.py      ← Loại cột ID, xử lý categorical
├── 09_split_dataset.py          ← Time-based split (Train <2024, Val 2024, Test ≥2025)
├── 10_train_models.py           ← Train 4 model baseline
├── 11_hyperparameter_tuning.py  ← Optuna + TimeSeriesSplit CV
├── 12_evaluate_models.py        ← So sánh toàn diện + baseline strategies + calibration curve
├── 13_explain_model.py          ← SHAP values (model chính: CatBoost_tuned)
├── 14_save_model.py             ← Đóng gói model production
├── 14_statistical_significance.py ← DeLong's test + bootstrap CI so sánh các model
├── 15_data_leakage_analysis.py  ← Minh hoạ định lượng tác động của data leakage (AUC có/không leakage)
├── 16_backtest_roi.py           ← Backtest ROI (de-vig đúng chuẩn) + CI 95% + ổn định theo quý
├── 17_literature_benchmark.py   ← Đối chiếu kết quả với các nghiên cứu đã công bố
└── 18_overfitting_analysis.py   ← Gap Train/Val/Test, learning curve, validation curve, CV stability
```

Tiện ích dùng chung: `src/utils/categorical.py` — đồng nhất categorical dtype/encoding giữa train/val/test cho mọi script (tránh lỗi model gặp category chưa từng thấy lúc train, và tránh mã hoá lệch âm thầm giữa các lần chạy độc lập).

## Kết quả chính

Số liệu dưới đây phản ánh lần chạy gần nhất được ghi vào `experiments/`; xem trực tiếp các file JSON hoặc mở app (trang **Analytics**) để có số liệu mới nhất sau khi bạn tự chạy lại pipeline.

| Nguồn số liệu | File |
|---|---|
| Metrics baseline (val set) | `experiments/metrics.json` |
| Đánh giá đầy đủ trên test set (Accuracy/AUC/F1/LogLoss/Brier + baseline heuristics) | `experiments/test_evaluation.json`, `.csv` |
| Kiểm định thống kê (DeLong's test, bootstrap CI) | `experiments/statistical_significance.json` |
| Phân tích overfitting (gap, learning/validation curve, CV stability) | `experiments/overfitting_analysis.json` |
| Backtest ROI (de-vig, CI 95%, ổn định theo quý) | `experiments/backtest_roi.json` |
| Đối chiếu y văn | `experiments/literature_benchmark.json` |

**Model chính được chọn để triển khai: `CatBoost_tuned`** — chọn dựa trên AUC và gap Train/Val/Test (overfitting analysis), không chọn dựa trên ROI (tránh data snooping — xem `16_backtest_roi.py`).

### Kiểm soát chất lượng dữ liệu & phương pháp

Trong quá trình phát triển, dự án đã chủ động phát hiện và sửa nhiều vấn đề phương pháp luận, ghi chép đầy đủ tại [CHANGELOG_FIXES.md](CHANGELOG_FIXES.md), trong đó đáng chú ý:

- **2 lỗi data leakage nghiêm trọng** (cột thống kê trong trận bị sót lại, biến diff tính trước khi swap Player A/B) — đã sửa và có test tự động xác nhận (`tests/test_leakage_detection.py`, `src/pipelines/15_data_leakage_analysis.py` minh hoạ định lượng chênh lệch AUC có/không leakage).
- **Category encoding không nhất quán giữa train/val/test** (gây lỗi "category not in training set" ở XGBoost, và mã hoá lệch âm thầm ở RandomForest) — sửa tận gốc bằng `src/utils/categorical.py`, áp dụng thống nhất cho toàn bộ 6 script liên quan.
- **De-vig sai công thức** trong tính implied probability cho backtest ROI (nay khớp đúng công thức de-vig chuẩn dùng trong app).
- **Data snooping khi chọn "best model"** theo ROI đo trên chính test set — nay cố định model đầu tàu theo AUC/overfitting-gap, không theo ROI.

## Ứng dụng Demo (Streamlit)

```
project/
├── main.py                 ← Entry point, router giữa các trang, inject theme
├── pages/
│   ├── Home.py              ← Trang chủ — bento grid hero, số liệu thật
│   ├── Prediction.py        ← Dự đoán trận đấu (nhập 2 tay vợt) + SHAP + phân tích kèo cược
│   ├── Bracket.py           ← Mô phỏng nhánh đấu giải
│   ├── PlayerProfile.py     ← Hồ sơ chi tiết từng tay vợt
│   ├── Analytics.py         ← 5 tab: EDA / Đánh giá mô hình / SHAP / Thống kê mô tả / Giới hạn nghiên cứu
│   └── Dataset.py           ← Xem trực tiếp bộ dữ liệu
├── components/              ← Card, gauge, biểu đồ SHAP/radar dùng chung
├── services/                ← Load dữ liệu, inference, hồ sơ tay vợt, sinh báo cáo PDF
└── utils/                   ← Theme "Night Court" (bảng màu, typography), formatters, validators
```

Tab **Analytics → Đánh giá mô hình** trình bày đầy đủ: so sánh 4 model, ma trận nhầm lẫn, DeLong's test, calibration curve, phân tích overfitting, và backtest ROI (bảng CI 95% + ổn định theo quý) — mỗi biểu đồ đều kèm câu hỏi phân tích và nhận định, không chỉ trưng hình.

## Cài đặt & Chạy

### Yêu cầu
- Python 3.10+
- pip

### Cài đặt
```bash
cd "path/to/ATP_Prediction_Project_FIXED/proj"

# Tạo virtual environment
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate   # Linux/Mac

# Cài dependencies
pip install -r requirements.txt
```

### Chạy Pipeline
```bash
# Chạy tuần tự từ đầu (khuyến nghị lần đầu)
python src/pipelines/01_load_data.py
python src/pipelines/02_data_validation.py
python src/pipelines/03_clean_schema.py
python src/pipelines/04_handle_missing.py
python src/pipelines/05_remove_data_leakage.py
python src/pipelines/06_eda.py
python src/pipelines/07_feature_engineering.py
python src/pipelines/08_feature_selection.py
python src/pipelines/09_split_dataset.py
python src/pipelines/10_train_models.py
python src/pipelines/11_hyperparameter_tuning.py    # tốn nhiều thời gian nhất (Optuna)
python src/pipelines/12_evaluate_models.py
python src/pipelines/13_explain_model.py
python src/pipelines/14_save_model.py

# Các phân tích bổ sung (chạy sau khi đã có model ở bước 10/11)
python src/pipelines/14_statistical_significance.py
python src/pipelines/15_data_leakage_analysis.py
python src/pipelines/16_backtest_roi.py
python src/pipelines/17_literature_benchmark.py
python src/pipelines/18_overfitting_analysis.py
```

### Chạy Tests
```bash
python -m pytest tests/ -v --tb=short
```

### Chạy App Demo
```bash
python runapp.py
# hoặc
streamlit run project/main.py
```

## Cấu trúc thư mục

```
├── project/                # Ứng dụng Streamlit demo (xem chi tiết ở mục "Ứng dụng Demo" trên)
├── configs/                # config.yaml — đường dẫn data/model/reports dùng chung cho pipeline
├── data/
│   ├── raw/                # Dữ liệu gốc CSV
│   ├── interim/             # Dữ liệu trung gian (parquet)
│   ├── processed/           # Dữ liệu đã clean
│   └── features/            # Features + train/val/test splits
├── docs/
│   └── methodology.md       # Phương pháp luận chi tiết, bao gồm Limitations & Future Work
├── experiments/             # Metrics/kết quả phân tích dạng JSON/CSV (xem bảng "Kết quả chính")
├── models/
│   ├── *.joblib             # Model baseline
│   ├── tuned/                # Model đã tune (Optuna)
│   └── final/                 # Model production
├── reports/
│   ├── figures/              # Biểu đồ EDA + evaluation + overfitting + backtest
│   └── shap_plots/           # SHAP explainability
├── src/
│   ├── pipelines/            # 18 bước pipeline (xem mục "Kiến trúc Pipeline")
│   └── utils/                 # categorical.py (encoding dùng chung), logger.py, config loader
├── tests/                    # Unit test (pytest) — bao gồm test phát hiện leakage
├── runapp.py                 # Script khởi chạy app (gọi streamlit run project/main.py)
├── CHANGELOG_FIXES.md         # Nhật ký đầy đủ các lỗi đã phát hiện & sửa trong quá trình phát triển
├── requirements.txt
└── README.md
```

## Tài liệu

- [Phương pháp luận chi tiết](docs/methodology.md) — Mô tả đầy đủ pipeline, features, chiến lược chống leakage, và giới hạn nghiên cứu.
- [CHANGELOG_FIXES.md](CHANGELOG_FIXES.md) — Nhật ký toàn bộ lỗi/thiếu sót đã phát hiện và cách sửa, theo từng PHẦN đánh số.
- [Biểu đồ EDA & evaluation](reports/figures/) — Toàn bộ biểu đồ phân tích dữ liệu và đánh giá model.
- [SHAP Analysis](reports/shap_plots/) — Giải thích quyết định của model chính (CatBoost_tuned).

## Nhóm nghiên cứu

Dự án NCKH Sinh viên — Ứng dụng Machine Learning trong dự đoán kết quả thể thao.
