"""ATP Tennis Prediction Application

A professional Decision Support System for ATP tennis match predictions,
combining machine learning models with explainable AI and interactive analytics.

Features:
- Real-time match predictions with confidence scores
- SHAP-based explainability
- Comprehensive player and tournament analytics
- Historical trend analysis
- Professional dashboard interface

Author: ATP Predictor Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "ATP Predictor Team"

# LỖI ĐÃ SỬA: trước đây dùng absolute import (`from project.utils.logger
# import get_logger`) ngay trong chính __init__.py của package `project` —
# đây là self-import tuyệt đối, chỉ hoạt động nếu thư mục gốc dự án đã có
# sẵn trong sys.path. Đổi sang relative import (`from .utils...`) để việc
# import package `project` luôn hoạt động đúng bất kể sys.path, miễn là
# `project` được import như một package (không chạy trực tiếp file này).
from .utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["logger"]
