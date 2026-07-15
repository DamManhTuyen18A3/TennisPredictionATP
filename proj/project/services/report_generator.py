"""
Report Generator Service (TÍNH NĂNG MỚI)
============================================

Xuất báo cáo PDF tổng hợp toàn bộ phân tích (KPI dữ liệu, so sánh mô hình,
kiểm định thống kê, confusion matrix, SHAP) — dùng số liệu THẬT đọc trực
tiếp từ `experiments/` và dữ liệu đã xử lý, để người dùng có thể tải về nộp
kèm báo cáo NCKH, không cần tự chụp màn hình từng phần.
"""

import io
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image,
)

# LỖI ĐÃ SỬA: font mặc định "Helvetica" của reportlab chỉ hỗ trợ bảng mã
# Latin-1, KHÔNG có dấu tiếng Việt (ữ, ệ, ầ...) — chữ có dấu bị vỡ thành ■.
# Đăng ký font DejaVu Sans (hỗ trợ đầy đủ Unicode/tiếng Việt), đóng gói SẴN
# trong project (project/assets/fonts/) thay vì dựa vào font hệ thống — để
# báo cáo PDF ra đúng dấu trên MỌI máy (kể cả Windows không cài DejaVu Sans).
_FONTS_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"
_FONT_REGULAR = "DejaVuSans"
_FONT_BOLD = "DejaVuSans-Bold"
_FONT_ITALIC = "DejaVuSans-Oblique"

try:
    pdfmetrics.registerFont(TTFont(_FONT_REGULAR, str(_FONTS_DIR / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont(_FONT_BOLD, str(_FONTS_DIR / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont(_FONT_ITALIC, str(_FONTS_DIR / "DejaVuSans-Oblique.ttf")))
    _VN_FONT_OK = True
except Exception:
    # Fallback an toàn: nếu vì lý do nào đó không load được font (thiếu file),
    # dùng lại Helvetica để báo cáo vẫn xuất được (chỉ mất dấu) thay vì crash.
    _FONT_REGULAR, _FONT_BOLD, _FONT_ITALIC = "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"
    _VN_FONT_OK = False


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", fontSize=22, leading=26, spaceAfter=6,
        textColor=rl_colors.HexColor("#0B120F"), fontName=_FONT_BOLD,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeading", fontSize=14, leading=18, spaceBefore=16, spaceAfter=8,
        textColor=rl_colors.HexColor("#0B120F"), fontName=_FONT_BOLD,
    ))
    styles.add(ParagraphStyle(
        name="BodyVN", fontSize=10, leading=15, spaceAfter=6, fontName=_FONT_REGULAR,
    ))
    styles.add(ParagraphStyle(
        name="Insight", fontSize=9.5, leading=14, spaceAfter=8,
        backColor=rl_colors.HexColor("#F0FFEA"), borderPadding=6,
        fontName=_FONT_ITALIC,
    ))
    return styles


def _metric_table(rows, col_widths=None):
    tbl = Table(rows, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#0B120F")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), _FONT_BOLD),
        ("FONTNAME", (0, 1), (-1, -1), _FONT_REGULAR),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor("#F5F7F5")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return tbl


def generate_pdf_report(
    dataset_stats: dict,
    model_metrics: dict,
    evaluation: dict,
    significance: dict,
    confusion_matrix_data: dict,
    shap_top_features: list = None,
) -> bytes:
    """Tạo báo cáo PDF tổng hợp — trả về bytes để dùng với st.download_button.

    Toàn bộ số liệu đưa vào đều là tham số THẬT lấy từ app (không tính toán
    giả lập trong hàm này) — đảm bảo báo cáo khớp 100% với những gì hiển thị
    trên giao diện Streamlit.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=2 * cm, bottomMargin=2 * cm, leftMargin=2 * cm, rightMargin=2 * cm,
    )
    styles = _styles()
    story = []

    # --- Trang bìa ---
    story.append(Paragraph("ATP Match Prediction System", styles["ReportTitle"]))
    story.append(Paragraph("Báo cáo phân tích dữ liệu &amp; đánh giá mô hình", styles["SectionHeading"]))
    story.append(Paragraph(
        f"Nhóm 9 · DHKL18A3HN · NCKH Sinh viên<br/>"
        f"Ngày xuất báo cáo: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        styles["BodyVN"],
    ))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(
        "Toàn bộ số liệu trong báo cáo này lấy trực tiếp từ dữ liệu đã xử lý và kết quả "
        "đánh giá mô hình thật, không có số liệu minh hoạ hay ước lượng thủ công.",
        styles["Insight"],
    ))

    # --- 1. Tổng quan dữ liệu ---
    story.append(Paragraph("1. Tổng quan dữ liệu", styles["SectionHeading"]))
    story.append(_metric_table([
        ["Chỉ số", "Giá trị"],
        ["Tổng số trận đấu", f"{dataset_stats.get('matches', 0):,}"],
        ["Số tay vợt", f"{dataset_stats.get('players', 0):,}"],
        ["Số giải đấu", f"{dataset_stats.get('tournaments', 0):,}"],
        ["Khoảng thời gian", f"{dataset_stats.get('date_min', '')} → {dataset_stats.get('date_max', '')}"],
    ], col_widths=[8 * cm, 8 * cm]))

    # --- 2. So sánh mô hình ---
    story.append(Paragraph("2. So sánh hiệu năng các mô hình (đo trên tập test)", styles["SectionHeading"]))
    if model_metrics:
        rows = [["Model", "Accuracy", "AUC", "F1", "Log Loss"]]
        for name, vals in model_metrics.items():
            rows.append([
                name, f"{vals.get('accuracy', 0):.1%}", f"{vals.get('auc', 0):.4f}",
                f"{vals.get('f1', 0):.4f}", f"{vals.get('log_loss', 0):.4f}",
            ])
        story.append(_metric_table(rows, col_widths=[4.5 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm]))

    if evaluation:
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph("So sánh với chiến lược cơ sở (baseline không dùng ML):", styles["BodyVN"]))
        rows = [["Chiến lược", "Accuracy", "AUC"]]
        for name, vals in evaluation.items():
            rows.append([name, f"{vals.get('accuracy', 0):.1%}", f"{vals.get('auc', 0):.4f}"])
        story.append(_metric_table(rows, col_widths=[8 * cm, 4 * cm, 4 * cm]))

    # --- 3. Kiểm định thống kê ---
    if significance:
        story.append(Paragraph("3. Kiểm định ý nghĩa thống kê (DeLong's test)", styles["SectionHeading"]))
        story.append(Paragraph(significance.get("methodology_note", ""), styles["BodyVN"]))
        delong = significance.get("delong_pairwise_tests", {})
        ref = significance.get("reference_model", "")
        if delong:
            rows = [["So sánh", "Chênh lệch AUC", "p-value", "Kết luận"]]
            for pair, res in delong.items():
                other = pair.replace(f"{ref}_vs_", "")
                sig = "Có ý nghĩa (p<0.05)" if res["significant_at_0.05"] else "Không có ý nghĩa"
                rows.append([f"{ref} vs {other}", str(res["diff"]), str(res["p_value"]), sig])
            story.append(_metric_table(rows, col_widths=[6.5 * cm, 3.5 * cm, 3 * cm, 3 * cm]))

        ci = significance.get("bootstrap_confidence_intervals", {})
        if ci:
            story.append(Spacer(1, 0.3 * cm))
            story.append(Paragraph("Khoảng tin cậy 95% cho AUC (bootstrap 1000 lần resample):", styles["BodyVN"]))
            rows = [["Model", "AUC", "95% CI dưới", "95% CI trên"]]
            for name, vals in ci.items():
                rows.append([name, str(vals["point_estimate"]), str(vals["ci_lower_95"]), str(vals["ci_upper_95"])])
            story.append(_metric_table(rows, col_widths=[5 * cm, 3.5 * cm, 3.5 * cm, 3.5 * cm]))

    story.append(PageBreak())

    # --- 4. Confusion Matrix ---
    if confusion_matrix_data:
        story.append(Paragraph("4. Confusion Matrix chi tiết", styles["SectionHeading"]))
        conf = confusion_matrix_data
        rows = [
            ["", "Dự đoán: Thua", "Dự đoán: Thắng"],
            ["Thực tế: Thua", f"{conf['tn']:,} (True Negative)", f"{conf['fp']:,} (False Positive)"],
            ["Thực tế: Thắng", f"{conf['fn']:,} (False Negative)", f"{conf['tp']:,} (True Positive)"],
        ]
        story.append(_metric_table(rows, col_widths=[4 * cm, 6 * cm, 6 * cm]))
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph(
            f"Precision: {conf['precision']:.1%} · Recall: {conf['recall']:.1%} · "
            f"Tổng {conf['total']:,} trận trong tập test", styles["BodyVN"],
        ))

    # --- 5. SHAP Explainability ---
    if shap_top_features:
        story.append(Paragraph("5. Yếu tố ảnh hưởng nhiều nhất đến dự đoán (SHAP)", styles["SectionHeading"]))
        rows = [["Đặc trưng (feature)", "SHAP value trung bình"]]
        for feat, val in shap_top_features:
            rows.append([feat, f"{val:+.4f}"])
        story.append(_metric_table(rows, col_widths=[10 * cm, 6 * cm]))

    # --- 6. Hạn chế ---
    story.append(Paragraph("6. Hạn chế của nghiên cứu (nêu trung thực)", styles["SectionHeading"]))
    limitations = [
        "Với tay vợt lâu không thi đấu (có thể đã giải nghệ), tuổi/hạng được ngoại suy tuyến "
        "tính từ trận gần nhất trong dữ liệu — một giả định đơn giản hoá.",
        "Hơn 92% số trận không có tỷ lệ cược đầy đủ — model xử lý bằng cách hỗ trợ giá trị "
        "thiếu (missing value) tự nhiên thay vì áp đặt số liệu giả.",
        "Dữ liệu nghiêng nhiều về giải ITF/Challenger — độ chính xác có thể thấp hơn ở các "
        "trận Grand Slam nổi tiếng do ít dữ liệu huấn luyện hơn.",
    ]
    for lim in limitations:
        story.append(Paragraph(f"• {lim}", styles["BodyVN"]))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
