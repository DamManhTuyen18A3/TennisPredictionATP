import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger, load_config

logger = get_logger(__name__)


def perform_eda():
    """Thực hiện Exploratory Data Analysis toàn diện và lưu biểu đồ + thống kê."""
    config = load_config()
    processed_dir = Path(config['data']['processed_dir'])
    reports_dir = Path(config['reports']['figures_dir'])
    reports_dir.mkdir(parents=True, exist_ok=True)

    input_path = processed_dir / "05_clean_data.parquet"

    logger.info(f"Bắt đầu EDA toàn diện trên dữ liệu: {input_path}")
    df = pd.read_parquet(input_path)

    # Thiết lập style
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams['figure.dpi'] = 150

    # =========================================================================
    # 0. Thống kê mô tả (Descriptive Statistics)
    # =========================================================================
    logger.info("Tạo bảng thống kê mô tả...")
    desc_stats = df.describe(include='all').T
    desc_stats.to_csv(reports_dir / 'descriptive_statistics.csv')

    # Summary table hình ảnh
    key_stats = {
        'Tổng số trận': len(df),
        'Khoảng thời gian': f"{df['tourney_date'].min().strftime('%Y-%m-%d')} → {df['tourney_date'].max().strftime('%Y-%m-%d')}" if 'tourney_date' in df.columns else 'N/A',
        'Số giải đấu': df['tourney_name'].nunique() if 'tourney_name' in df.columns else 'N/A',
        'Số tay vợt (winner)': df['winner_id'].nunique() if 'winner_id' in df.columns else 'N/A',
        'Số mặt sân': df['surface'].nunique() if 'surface' in df.columns else 'N/A',
    }
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')
    table_data = [[k, str(v)] for k, v in key_stats.items()]
    table = ax.table(cellText=table_data, colLabels=['Chỉ số', 'Giá trị'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    ax.set_title('Tổng quan dữ liệu', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(reports_dir / 'data_overview_table.png', bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 1. Phân bố giải đấu theo cấp độ (tourney_level) — giữ lại từ bản gốc
    # =========================================================================
    if 'tourney_level' in df.columns:
        logger.info("Biểu đồ 1: Tourney Level Distribution")
        level_counts = df['tourney_level'].value_counts()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Bar chart
        colors = sns.color_palette('viridis', len(level_counts))
        ax1.barh(level_counts.index, level_counts.values, color=colors)
        for i, v in enumerate(level_counts.values):
            ax1.text(v + 50, i, f'{v} ({v/len(df)*100:.1f}%)', va='center', fontsize=10)
        ax1.set_xlabel('Số trận')
        ax1.set_title('Số lượng trận theo cấp độ giải')

        # Pie chart
        ax2.pie(level_counts.values, labels=level_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=140)
        ax2.set_title('Tỉ lệ phần trăm')

        plt.suptitle('Phân bố trận đấu theo Tournament Level', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(reports_dir / 'tourney_level_distribution.png')
        plt.close()

    # =========================================================================
    # 2. Phân bố theo mặt sân (Surface)
    # =========================================================================
    if 'surface' in df.columns:
        logger.info("Biểu đồ 2: Surface Distribution")
        surface_counts = df['surface'].value_counts()

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {'Hard': '#2196F3', 'Clay': '#E65100', 'Grass': '#4CAF50',
                  'Carpet': '#9C27B0', 'Unknown': '#9E9E9E'}
        bar_colors = [colors.get(s, '#607D8B') for s in surface_counts.index]
        bars = ax.bar(surface_counts.index, surface_counts.values, color=bar_colors, edgecolor='white')
        for bar, v in zip(bars, surface_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{v}\n({v/len(df)*100:.1f}%)', ha='center', va='bottom', fontsize=10)
        ax.set_xlabel('Mặt sân')
        ax.set_ylabel('Số trận')
        ax.set_title('Phân bố trận đấu theo mặt sân', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(reports_dir / 'surface_distribution.png')
        plt.close()

    # =========================================================================
    # 3. Xu hướng theo thời gian (Yearly trend)
    # =========================================================================
    if 'tourney_date' in df.columns:
        logger.info("Biểu đồ 3: Yearly Match Trend")
        df['year'] = df['tourney_date'].dt.year
        df['month'] = df['tourney_date'].dt.to_period('M')

        yearly = df.groupby('year').size().reset_index(name='count')
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(yearly['year'], yearly['count'], color=sns.color_palette('viridis', len(yearly)),
               edgecolor='white')
        for i, row in yearly.iterrows():
            ax.text(row['year'], row['count'] + 50, str(row['count']),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_xlabel('Năm')
        ax.set_ylabel('Số trận')
        ax.set_title('Số lượng trận đấu theo năm', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(reports_dir / 'yearly_match_trend.png')
        plt.close()

    # =========================================================================
    # 4. Tay thuận (Hand) — giữ lại từ bản gốc, mở rộng
    # =========================================================================
    if 'winner_hand' in df.columns and 'loser_hand' in df.columns:
        logger.info("Biểu đồ 4: Winner/Loser Hand Distribution")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        df['winner_hand'].value_counts().plot.pie(autopct='%1.1f%%', cmap='Pastel1', ax=ax1)
        ax1.set_title('Tay thuận — Người thắng')
        ax1.set_ylabel('')
        df['loser_hand'].value_counts().plot.pie(autopct='%1.1f%%', cmap='Pastel2', ax=ax2)
        ax2.set_title('Tay thuận — Người thua')
        ax2.set_ylabel('')
        plt.suptitle('Phân bố tay thuận', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(reports_dir / 'hand_distribution.png')
        plt.close()

    # =========================================================================
    # 5. Rank Scatter — giữ lại từ bản gốc
    # =========================================================================
    if 'winner_rank' in df.columns and 'loser_rank' in df.columns:
        logger.info("Biểu đồ 5: Rank Correlation Scatter")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.scatterplot(data=df, x='winner_rank', y='loser_rank', alpha=0.15,
                        color='steelblue', s=10, ax=ax)
        ax.plot([0, 2000], [0, 2000], 'r--', alpha=0.7, label='y = x (equal rank)')
        ax.set_xlabel('Winner Rank', fontsize=12)
        ax.set_ylabel('Loser Rank', fontsize=12)
        ax.set_title('Tương quan thứ hạng: Winner vs Loser', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(reports_dir / 'rank_correlation.png')
        plt.close()

    # =========================================================================
    # 6. Rank Distribution (Histogram)
    # =========================================================================
    if 'winner_rank' in df.columns and 'loser_rank' in df.columns:
        logger.info("Biểu đồ 6: Rank Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df['winner_rank'].dropna(), bins=80, alpha=0.6, label='Winner Rank',
                color='#2196F3', density=True)
        ax.hist(df['loser_rank'].dropna(), bins=80, alpha=0.6, label='Loser Rank',
                color='#F44336', density=True)
        ax.set_xlabel('Rank')
        ax.set_ylabel('Density')
        ax.set_title('Phân bố thứ hạng — Winner vs Loser', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1500)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(reports_dir / 'rank_distribution.png')
        plt.close()

    # =========================================================================
    # 7. Correlation Heatmap (chỉ numeric features chính)
    # =========================================================================
    logger.info("Biểu đồ 7: Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    # Chọn các cột quan trọng (tránh heatmap quá lớn)
    important_cols = [c for c in [
        'winner_rank', 'loser_rank', 'winner_rank_points', 'loser_rank_points',
        'winner_age', 'loser_age', 'winner_ht', 'loser_ht',
        'b365w', 'b365l', 'avgw', 'avgl'
    ] if c in numeric_df.columns]

    if len(important_cols) >= 4:
        corr_matrix = numeric_df[important_cols].corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, ax=ax, linewidths=0.5, square=True,
                    annot_kws={'size': 9})
        ax.set_title('Ma trận tương quan — Các features chính', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(reports_dir / 'correlation_heatmap.png')
        plt.close()

    # =========================================================================
    # 8. Odds Distribution (nếu có)
    # =========================================================================
    odds_cols = [c for c in ['b365w', 'b365l', 'avgw', 'avgl'] if c in df.columns]
    if len(odds_cols) >= 2:
        logger.info("Biểu đồ 8: Odds Distribution")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        if 'b365w' in df.columns and 'b365l' in df.columns:
            axes[0].hist(df['b365w'].dropna().clip(1, 20), bins=50, alpha=0.6,
                         label='Winner Odds', color='#4CAF50')
            axes[0].hist(df['b365l'].dropna().clip(1, 20), bins=50, alpha=0.6,
                         label='Loser Odds', color='#F44336')
            axes[0].set_xlabel('Bet365 Odds')
            axes[0].set_ylabel('Count')
            axes[0].set_title('Phân bố Bet365 Odds')
            axes[0].legend()

        # Implied probability
        if 'b365w' in df.columns:
            prob_w = 1 / df['b365w'].dropna()
            prob_l = 1 / df['b365l'].dropna()
            axes[1].hist(prob_w.clip(0, 1), bins=50, alpha=0.6,
                         label='Winner Implied Prob', color='#4CAF50')
            axes[1].hist(prob_l.clip(0, 1), bins=50, alpha=0.6,
                         label='Loser Implied Prob', color='#F44336')
            axes[1].set_xlabel('Implied Probability')
            axes[1].set_ylabel('Count')
            axes[1].set_title('Phân bố Xác suất ngụ ý từ nhà cái')
            axes[1].legend()

        plt.suptitle('Phân tích Tỷ lệ cược (Odds Analysis)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(reports_dir / 'odds_distribution.png')
        plt.close()

    # =========================================================================
    # 9. Tỉ lệ Upset (tay vợt hạng thấp hơn thắng) theo surface/level
    # =========================================================================
    if 'winner_rank' in df.columns and 'loser_rank' in df.columns:
        logger.info("Biểu đồ 9: Upset Rate Analysis")
        df['is_upset'] = df['winner_rank'] > df['loser_rank']  # hạng cao hơn = số nhỏ hơn

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # By surface
        if 'surface' in df.columns:
            upset_surface = df.groupby('surface')['is_upset'].mean().sort_values()
            ax = axes[0]
            bars = ax.barh(upset_surface.index, upset_surface.values * 100,
                           color=sns.color_palette('coolwarm', len(upset_surface)))
            for bar, v in zip(bars, upset_surface.values):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        f'{v*100:.1f}%', va='center', fontsize=10)
            ax.set_xlabel('Tỉ lệ Upset (%)')
            ax.set_title('Tỉ lệ Upset theo mặt sân')
            ax.set_xlim(0, 60)

        # By tourney level
        if 'tourney_level' in df.columns:
            upset_level = df.groupby('tourney_level')['is_upset'].mean().sort_values()
            ax = axes[1]
            bars = ax.barh(upset_level.index, upset_level.values * 100,
                           color=sns.color_palette('coolwarm', len(upset_level)))
            for bar, v in zip(bars, upset_level.values):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        f'{v*100:.1f}%', va='center', fontsize=10)
            ax.set_xlabel('Tỉ lệ Upset (%)')
            ax.set_title('Tỉ lệ Upset theo cấp độ giải')
            ax.set_xlim(0, 60)

        plt.suptitle('Phân tích Upset (hạng thấp thắng hạng cao)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(reports_dir / 'upset_rate_analysis.png')
        plt.close()

        # Cleanup temp cols
        df.drop(columns=['is_upset'], inplace=True, errors='ignore')

    # =========================================================================
    # 10. Phân bố tuổi (Age Distribution)
    # =========================================================================
    if 'winner_age' in df.columns and 'loser_age' in df.columns:
        logger.info("Biểu đồ 10: Age Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df['winner_age'].dropna(), bins=40, alpha=0.6, label='Winner Age',
                color='#4CAF50', density=True)
        ax.hist(df['loser_age'].dropna(), bins=40, alpha=0.6, label='Loser Age',
                color='#FF9800', density=True)
        ax.set_xlabel('Tuổi')
        ax.set_ylabel('Density')
        ax.set_title('Phân bố tuổi — Winner vs Loser', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(reports_dir / 'age_distribution.png')
        plt.close()

    # Cleanup temp cols
    df.drop(columns=['year', 'month'], inplace=True, errors='ignore')

    logger.info(f"Đã hoàn thành EDA toàn diện. {10} biểu đồ + thống kê lưu tại: {reports_dir}")


if __name__ == "__main__":
    perform_eda()
