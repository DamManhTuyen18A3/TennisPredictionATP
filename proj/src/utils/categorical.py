"""
Tiện ích đồng nhất categorical dtype/encoding giữa train/val/test.

TẠI SAO FILE NÀY TỒN TẠI
-------------------------
Lỗi "XGBoost: Found a category not in the training set: carpet" đã xuất
hiện lặp lại ở NHIỀU file khác nhau (10_train_models.py, 11_hyperparameter_
tuning.py, 12_evaluate_models.py, 16_backtest_roi.py, 18_overfitting_
analysis.py) vì mỗi script tự fit category dtype/OrdinalEncoder RIÊNG trên
dữ liệu của chính nó (thường chỉ train, hoặc chỉ train+val) — nếu 1 giá trị
category (vd. mặt sân "Carpet") chỉ xuất hiện ở tập test, model không bao
giờ biết tới giá trị đó.

CÁCH SỬA TẬN GỐC
-----------------
Mọi script train/tune/evaluate/backtest dùng CHUNG 2 hàm dưới đây, với
danh sách category hợp nhất TỪ CẢ TRAIN + VAL + TEST — tức model "biết" một
giá trị category tồn tại (dù train có 0 dòng mang giá trị đó) trước khi
fit, nên không bao giờ gặp giá trị lạ lúc predict nữa.

Lưu ý về leakage: hàm này chỉ đọc GIÁ TRỊ (vocabulary) của cột categorical
(vd. danh sách mặt sân có thể có: Hard/Clay/Grass/Carpet), KHÔNG dùng
target/nhãn của tập test — tương tự việc biết trước các lựa chọn có thể có
trong 1 dropdown, không phải biết trước kết quả trận đấu. Đây KHÔNG phải
data leakage.
"""
from typing import Dict, List, Sequence

import pandas as pd


def get_unified_categories(*dataframes: pd.DataFrame, cat_cols: Sequence[str]) -> Dict[str, List[str]]:
    """Hợp nhất tập giá trị (đã sort) của từng cột categorical qua nhiều
    DataFrame (thường là train/val/test). Giá trị thiếu (NaN) được coi là
    'Unknown'."""
    unified: Dict[str, List[str]] = {}
    for col in cat_cols:
        parts = [df[col] for df in dataframes if col in df.columns]
        if not parts:
            continue
        combined = pd.concat(parts, axis=0)
        combined = combined.astype(object).where(combined.notna(), 'Unknown').astype(str)
        unified[col] = sorted(combined.unique().tolist())
    return unified


def apply_unified_categorical_dtype(df: pd.DataFrame, unified_categories: Dict[str, List[str]]) -> pd.DataFrame:
    """Ép các cột categorical của df sang dtype 'category' dùng ĐÚNG tập
    category đã hợp nhất (unified_categories) — đảm bảo mọi DataFrame dùng
    cùng 1 category dtype. Trả về bản copy, không sửa df gốc."""
    df = df.copy()
    for col, cats in unified_categories.items():
        if col not in df.columns:
            continue
        raw = df[col].astype(object)
        raw = raw.where(raw.notna(), 'Unknown').astype(str)
        df[col] = raw.astype(pd.CategoricalDtype(categories=cats))
    return df


def ordinal_encode(df: pd.DataFrame, unified_categories: Dict[str, List[str]]) -> pd.DataFrame:
    """Ordinal-encode các cột categorical của df bằng ĐÚNG tập category đã
    hợp nhất (map thủ công, tất định — không dùng sklearn OrdinalEncoder.fit
    để tránh nguy cơ 2 lần fit độc lập ra 2 cách mã hoá khác nhau). Dùng hàm
    này ở MỌI nơi (train/tune/evaluate/backtest) để đảm bảo cùng 1 giá trị
    category luôn ra cùng 1 mã số, bất kể script nào gọi hay dữ liệu đang có
    gì. Giá trị không có trong danh sách hợp nhất (không nên xảy ra vì đã
    hợp nhất từ train+val+test) được mã hoá thành -1."""
    cols = [c for c in unified_categories if c in df.columns]
    df_enc = df.copy()
    for col in cols:
        cats = unified_categories[col]
        code_map = {v: i for i, v in enumerate(cats)}
        raw = df_enc[col].astype(object).where(df_enc[col].notna(), 'Unknown').astype(str)
        df_enc[col] = raw.map(code_map).fillna(-1).astype(int)
    return df_enc
