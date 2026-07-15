# Phương pháp luận — ATP Match Prediction using Machine Learning

## 1. Tổng quan bài toán

**Bài toán**: Cho thông tin sẵn có TRƯỚC trận đấu quần vợt (ranking, Elo rating, head-to-head record, tỷ lệ cược nhà cái, mặt sân, v.v.), dự đoán tay vợt nào sẽ thắng.

**Loại bài toán**: Binary Classification (phân loại nhị phân)
- Target = 1: Player_A thắng
- Target = 0: Player_B thắng (tức Player_A thua)

**Đặc thù**:
- Dữ liệu có tính thời gian (time-series) → phải chia train/test theo thời gian
- Dữ liệu gốc ở dạng Winner/Loser → phải biến đổi đối xứng để tránh bias

## 2. Pipeline 14 bước

### Bước 01-04: Thu thập & Tiền xử lý

| Bước | Mô tả | Input → Output |
|------|-------|----------------|
| 01 | Load CSV thô → Parquet (tăng tốc I/O) | raw CSV → 01_loaded.parquet |
| 02 | Validation schema, loại duplicate | → 02_validated.parquet |
| 03 | Lowercase tên cột, parse datetime | → 03_cleaned_schema.parquet |
| 04 | Fill NaN (median cho numeric, 'Unknown' cho categorical) | → 04_handled_missing.parquet |

### Bước 05: Loại bỏ Data Leakage

**Nguyên tắc**: Chỉ sử dụng thông tin CÓ TRƯỚC trận đấu. Loại bỏ tất cả thống kê trong/sau trận:

```
Các cột bị loại bỏ:
w_ace, w_df, w_svpt, w_1stin, w_1stwon, w_2ndwon,
l_ace, l_df, l_svpt, l_1stin, l_1stwon, l_2ndwon,
w_svgms, w_bpsaved, w_bpfaced,
l_svgms, l_bpsaved, l_bpfaced,
minutes, score
```

**Lỗi leakage đã sửa**: Bước `03_clean_schema` lowercase toàn bộ tên cột (vd: `w_1stIn` → `w_1stin`), nhưng danh sách drop ban đầu dùng tên viết hoa → không khớp → các cột leakage vẫn tồn tại. Đã sửa bằng cách thống nhất tên cột lowercase trong danh sách.

### Bước 06: Exploratory Data Analysis (EDA)

10+ biểu đồ phân tích:
1. Thống kê mô tả tổng quan
2. Phân bố giải đấu theo cấp độ (Futures 86%, Challenger, ATP...)
3. Phân bố theo mặt sân (Hard, Clay, Grass)
4. Xu hướng số trận theo năm
5. Phân bố tay thuận (R/L)
6. Tương quan rank Winner vs Loser
7. Histogram phân bố rank
8. Correlation heatmap
9. Phân bố odds nhà cái
10. Tỉ lệ upset theo surface/tourney level

### Bước 07: Feature Engineering

#### a) Elo Rating (tự tính từ lịch sử)

- Thuật toán Elo cổ điển, K-factor = 32, base rating = 1500
- **Pre-match**: Với mỗi trận, lấy Elo HIỆN TẠI (trước trận) rồi mới cập nhật
- Công thức: `E_w = 1 / (1 + 10^((R_l - R_w) / 400))`
- Cập nhật: `R_w_new = R_w + K * (1 - E_w)`

#### b) Head-to-Head (H2H)

- Đếm số lần mỗi tay vợt thắng đối thủ TRƯỚC trận hiện tại
- Chỉ dùng thông tin quá khứ → không leakage

#### c) Biến đổi đối xứng (Symmetric Transform)

**Vấn đề**: Dữ liệu gốc luôn ở dạng Winner/Loser → model học "Winner luôn thắng" (target 100% = 1).

**Giải pháp**: Random swap 50% data:
- Nếu không swap: Player_A = Winner, Player_B = Loser, Target = 1
- Nếu swap: Player_A = Loser, Player_B = Winner, Target = 0

→ Target phân bố ~50/50, model phải học từ features thực sự.

**Lỗi leakage đã sửa**: Các biến diff (`rank_diff`, `rank_points_diff`, `odds_implied_prob_diff`) được tính TRƯỚC khi swap → luôn bằng (winner - loser) bất kể target. Đã chuyển toàn bộ phép tính diff xuống SAU bước swap.

#### d) Derived Features

| Feature | Công thức | Ý nghĩa |
|---------|-----------|---------|
| rank_diff | p1_rank - p2_rank | Chênh lệch thứ hạng |
| rank_points_diff | p1_rank_points - p2_rank_points | Chênh lệch điểm ATP |
| elo_diff | p1_elo - p2_elo | Chênh lệch Elo |
| h2h_diff | p1_h2h_wins - p2_h2h_wins | Chênh lệch H2H |
| age_diff | p1_age - p2_age | Chênh lệch tuổi |
| ht_diff | p1_ht - p2_ht | Chênh lệch chiều cao |
| {odds}_implied_prob_diff | 1/p1_odds - 1/p2_odds | Chênh lệch xác suất ngụ ý |

### Bước 08: Feature Selection

Loại bỏ các cột định danh (ID, tên tay vợt, tourney_id) → chỉ giữ features có ý nghĩa dự đoán.

### Bước 09: Time-Based Split

| Tập | Khoảng thời gian | Mục đích |
|-----|-------------------|---------|
| **Train** | < 2024 | Huấn luyện model |
| **Validation** | 2024 | Chỉnh tham số, early stopping |
| **Test** | ≥ 2025 | Đánh giá cuối cùng (không được nhìn thấy khi train) |

**Tại sao không dùng random split?** Dữ liệu thể thao có tính thời gian — tay vợt thay đổi phong độ, ranking biến động. Random split sẽ để model "nhìn tương lai" để dự đoán quá khứ → kết quả lạc quan giả.

### Bước 10: Train Baseline Models

4 models với hyperparameters mặc định:
1. **CatBoost** — Native categorical support, ordered boosting
2. **LightGBM** — Leaf-wise growth, fast training
3. **XGBoost** — Level-wise growth, regularization mạnh
4. **RandomForest** — Ensemble of decision trees, non-boosting baseline

### Bước 11: Hyperparameter Tuning (Optuna + TimeSeriesSplit)

**Framework**: Optuna (Bayesian optimization)
- 50 trials mỗi model (tổng 200 trials)
- Objective: maximize ROC-AUC
- Cross-validation: **TimeSeriesSplit** (3 folds) — tôn trọng thứ tự thời gian

**Tại sao không dùng K-Fold?** K-Fold chia ngẫu nhiên → fold train có thể chứa dữ liệu tương lai so với fold val → kết quả CV lạc quan hơn thực tế.

### Bước 12: Đánh giá mô hình toàn diện

**Metrics đánh giá** (trên Test Set):
- AUC (ROC-AUC): Khả năng phân biệt winner/loser
- Accuracy: Tỉ lệ dự đoán đúng
- F1 Score: Harmonic mean Precision-Recall
- Log Loss: Chất lượng xác suất dự đoán
- Brier Score: Calibration quality

**Baseline strategies** (không ML):
1. **Always pick higher-ranked**: Chọn tay vợt có rank thấp hơn (= hạng cao hơn)
2. **Always follow odds**: Chọn tay vợt có odds nhà cái thấp hơn (= được favored)

**Phân tích lỗi**:
- Accuracy breakdown theo mặt sân
- Accuracy theo khoảng cách rank

**Calibration curve**: So sánh predicted probability vs actual win rate

### Bước 13: SHAP Explainability

Sử dụng SHAP (SHapley Additive exPlanations) để giải thích quyết định model:
- Summary plot: Feature importance ranking
- Bar plot: Mean absolute SHAP values

### Bước 14: Đóng gói Model Production

Copy model tốt nhất vào `models/final/`.

## 3. Kiểm định chất lượng (Quality Assurance)

### Unit Tests

| Test Module | Mô tả |
|------------|-------|
| `test_data_loader.py` | Kiểm tra pipeline import, data file |
| `test_feature_engineering.py` | Kiểm tra Elo/H2H logic, target balance |
| `test_leakage_detection.py` | **Kiểm tra không leakage** (|corr| < 0.9) |

### Leakage Detection tự động

Bước 07 tự động kiểm tra |correlation| > 0.9 với target và ghi cảnh báo nếu phát hiện.

## 4. Limitations & Future Work {#limitations}

### Giới hạn hiện tại
1. **Dữ liệu thiên lệch**: 86% trận đấu là Futures (giải nhỏ) → kết luận chưa chắc tổng quát cho giải lớn
2. **Chỉ nam (ATP)**: Chưa có dữ liệu WTA (nữ)
3. **Thiếu context features**: Chấn thương, phong độ gần (form), thời tiết, jet lag...
4. **Elo đơn giản**: K-factor cố định, chưa phân biệt surface-specific Elo
5. **Static features**: Chưa có rolling averages (tỉ lệ thắng 10 trận gần nhất)

### Hướng phát triển
1. Thêm Surface-specific Elo (Elo riêng cho mỗi mặt sân)
2. Rolling form features (win rate gần đây)
3. Fatigue indicators (số trận/tuần, distance travelled)
4. Mở rộng sang WTA
5. Deep learning approaches (neural network, transformer)
6. Ensemble stacking (kết hợp 4 models)
