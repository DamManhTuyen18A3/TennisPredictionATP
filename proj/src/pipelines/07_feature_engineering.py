import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger, load_config

logger = get_logger(__name__)

def calculate_elo(df, k=32, base_rating=1500):
    """Tính Elo rating THỰC SỰ theo lịch sử thi đấu (không leakage).
    df PHẢI được sort theo tourney_date tăng dần trước khi gọi hàm này.
    Với mỗi trận, ta lấy Elo HIỆN TẠI (trước trận) của winner/loser, rồi mới cập nhật
    Elo sau khi biết kết quả. Vì vậy giá trị lưu vào mỗi dòng luôn là thông tin
    đã biết TRƯỚC trận đấu đó -> an toàn, không rò rỉ tương lai.
    """
    logger.info("Đang tính toán Elo ratings theo lịch sử (pre-match)...")
    from collections import defaultdict
    elo = defaultdict(lambda: base_rating)

    winner_elo_pre = np.empty(len(df))
    loser_elo_pre = np.empty(len(df))

    for i, (w_id, l_id) in enumerate(zip(df['winner_id'], df['loser_id'])):
        w_elo, l_elo = elo[w_id], elo[l_id]
        winner_elo_pre[i] = w_elo
        loser_elo_pre[i] = l_elo

        expected_w = 1 / (1 + 10 ** ((l_elo - w_elo) / 400))
        elo[w_id] = w_elo + k * (1 - expected_w)
        elo[l_id] = l_elo + k * (0 - (1 - expected_w))

    df['winner_elo'] = winner_elo_pre
    df['loser_elo'] = loser_elo_pre
    return df

def calculate_h2h(df):
    """Tính Head-to-Head record THỰC SỰ (số lần thắng trước đó giữa 2 tay vợt).
    Cũng chỉ dùng thông tin TRƯỚC trận hiện tại -> không leakage.
    """
    logger.info("Đang tính toán Head-to-Head theo lịch sử (pre-match)...")
    from collections import defaultdict
    h2h = defaultdict(lambda: defaultdict(int))

    winner_h2h_pre = np.empty(len(df), dtype=int)
    loser_h2h_pre = np.empty(len(df), dtype=int)

    for i, (w_id, l_id) in enumerate(zip(df['winner_id'], df['loser_id'])):
        key = (w_id, l_id) if w_id < l_id else (l_id, w_id)
        winner_h2h_pre[i] = h2h[key][w_id]
        loser_h2h_pre[i] = h2h[key][l_id]
        h2h[key][w_id] += 1

    df['h2h_winner_wins'] = winner_h2h_pre
    df['h2h_loser_wins'] = loser_h2h_pre
    return df

def feature_engineering():
    """Tạo ra các đặc trưng (features) để đưa vào mô hình dự đoán."""
    config = load_config()
    processed_dir = Path(config['data']['processed_dir'])
    features_dir = Path(config['data']['features_dir'])
    
    input_path = processed_dir / "05_clean_data.parquet"
    output_path = features_dir / "07_engineered_features.parquet"
    
    logger.info(f"Bắt đầu Feature Engineering từ: {input_path}")
    df = pd.read_parquet(input_path)
    
    # Sort data by date để đảm bảo không rò rỉ khi tính toán các chuỗi thời gian
    if 'tourney_date' in df.columns:
        df = df.sort_values('tourney_date').reset_index(drop=True)
    
    # LƯU Ý QUAN TRỌNG VỀ DATA LEAKAGE:
    # Trước đây rank_diff/rank_points_diff/age_diff/ht_diff/odds_implied_prob được tính
    # NGAY TẠI ĐÂY, tức là TRƯỚC khi dữ liệu được xoay ngẫu nhiên thành Player_A/Player_B
    # (bước swap ở dưới). Vì các cột này không có tiền tố 'winner_'/'loser_' nên vòng lặp
    # swap phía dưới không đảo dấu chúng được, dẫn đến chúng LUÔN bằng (winner - loser)
    # bất kể target là 0 hay 1 -> mô hình chỉ cần nhìn dấu của các cột này là đoán được
    # kết quả gần như tuyệt đối (đây là lý do rank_points_diff/pts_diff chiếm >80% feature
    # importance). Toàn bộ các phép tính diff/odds đã được CHUYỂN XUỐNG dưới, sau khi
    # dữ liệu đã được xoay thành p1/p2, để đảm bảo tính đúng theo target.

    # Các chỉ số nâng cao (Elo, H2H, Form, Fatigue...)
    df = calculate_elo(df)
    df = calculate_h2h(df)
    
    # 6. Biến đổi dữ liệu thành dạng đối xứng (Player A vs Player B)
    # Hiện tại data đang là Winner vs Loser. Model sẽ học cách nhận diện "Winner" luôn thắng (Mất cân bằng).
    # Ta cần xoay data lại thành Player A, Player B và Target = 1 (A thắng), 0 (A thua).
    logger.info("Chuyển đổi dữ liệu Winner/Loser thành Player_A/Player_B...")
    
    # Cách đơn giản: lấy nửa data, Player A = Winner, Player B = Loser (Target=1)
    # Nửa còn lại: Player A = Loser, Player B = Winner (Target=0)
    # Dùng numpy để random index.
    np.random.seed(42)
    swap_idx = np.random.rand(len(df)) > 0.5
    
    df_model = pd.DataFrame()
    
    # Các cột sẽ được đổi tên
    prefix_w = 'winner_'
    prefix_l = 'loser_'

    # Các cột dạng cặp winner/loser KHÔNG có tiền tố 'winner_'/'loser_' (tỷ lệ cược nhà cái,
    # h2h). Nếu không xử lý riêng, các cột này sẽ bị copy y nguyên (không đảo) ở nhánh else
    # bên dưới -> lộ luôn ai là winner. Định nghĩa map (cột_w, cột_l) -> tên_p1/p2.
    pair_cols = {
        ('b365w', 'b365l'): 'b365',
        ('psw', 'psl'): 'ps',
        ('maxw', 'maxl'): 'max_odds',
        ('avgw', 'avgl'): 'avg_odds',
        ('bfew', 'bfel'): 'bfe',
        ('h2h_winner_wins', 'h2h_loser_wins'): 'h2h_wins',
    }
    cols_handled = set()
    for (col_w, col_l), out_name in pair_cols.items():
        if col_w in df.columns and col_l in df.columns:
            df_model[f'p1_{out_name}'] = np.where(swap_idx, df[col_w], df[col_l])
            df_model[f'p2_{out_name}'] = np.where(swap_idx, df[col_l], df[col_w])
            cols_handled.update([col_w, col_l])
    
    # Khởi tạo Player_A và Player_B cho các cột có tiền tố winner_/loser_
    for col in df.columns:
        if col in cols_handled:
            continue
        if col.startswith(prefix_w):
            base_col = col.replace(prefix_w, '')
            df_model[f'p1_{base_col}'] = np.where(swap_idx, df[col], df[prefix_l + base_col])
            df_model[f'p2_{base_col}'] = np.where(swap_idx, df[prefix_l + base_col], df[col])
        elif col.startswith(prefix_l):
            pass # Đã xử lý ở trên
        else:
            # Các cột chung, không phân biệt player (tourney_date, surface, tourney_level...)
            df_model[col] = df[col]

    # Tính các đặc trưng chênh lệch (diff) SAU KHI đã swap thành p1/p2 -> đảm bảo đúng chiều
    # với target (không còn luôn bằng winner - loser nữa).
    if 'p1_rank' in df_model.columns and 'p2_rank' in df_model.columns:
        df_model['rank_diff'] = df_model['p1_rank'] - df_model['p2_rank']
    if 'p1_rank_points' in df_model.columns and 'p2_rank_points' in df_model.columns:
        df_model['rank_points_diff'] = df_model['p1_rank_points'] - df_model['p2_rank_points']
    if 'p1_age' in df_model.columns and 'p2_age' in df_model.columns:
        df_model['age_diff'] = df_model['p1_age'] - df_model['p2_age']
    if 'p1_ht' in df_model.columns and 'p2_ht' in df_model.columns:
        df_model['ht_diff'] = df_model['p1_ht'] - df_model['p2_ht']
    if 'p1_elo' in df_model.columns and 'p2_elo' in df_model.columns:
        df_model['elo_diff'] = df_model['p1_elo'] - df_model['p2_elo']
    if 'p1_h2h_wins' in df_model.columns and 'p2_h2h_wins' in df_model.columns:
        df_model['h2h_diff'] = df_model['p1_h2h_wins'] - df_model['p2_h2h_wins']
    # Xác suất ngụ ý từ tỷ lệ cược (implied probability), tính từ odds đã swap đúng chiều
    for odds_name in ['b365', 'ps', 'max_odds', 'avg_odds', 'bfe']:
        p1_col, p2_col = f'p1_{odds_name}', f'p2_{odds_name}'
        if p1_col in df_model.columns and p2_col in df_model.columns:
            df_model[f'{odds_name}_implied_prob_diff'] = (1 / df_model[p1_col]) - (1 / df_model[p2_col])

    # Tạo Target: 1 nếu p1 là Winner (tức là không swap), 0 nếu p1 là Loser (tức là đã swap)
    df_model['target'] = np.where(swap_idx, 1, 0)

    # Kiểm tra an toàn cuối cùng: cảnh báo nếu bất kỳ feature nào tương quan gần như tuyệt đối
    # với target (dấu hiệu leakage vẫn còn sót) để phát hiện sớm ở các lần chạy sau.
    numeric_cols = df_model.select_dtypes(include=[np.number]).columns.drop('target', errors='ignore')
    corrs = df_model[numeric_cols].corrwith(df_model['target']).abs().sort_values(ascending=False)
    suspicious = corrs[corrs > 0.9]
    if len(suspicious) > 0:
        logger.error(f"CẢNH BÁO LEAKAGE: các cột có |corr| > 0.9 với target: \n{suspicious}")
    
    # Lưu
    df_model.to_parquet(output_path, index=False)
    logger.info(f"Đã hoàn thành Feature Engineering. Data shape: {df_model.shape}. Target ratio: {df_model['target'].mean():.2f}")
    logger.info(f"Lưu tại: {output_path}")

if __name__ == "__main__":
    feature_engineering()
