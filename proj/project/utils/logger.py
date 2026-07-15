"""
Structured Logging System

Provides centralized logging with context awareness for debugging and monitoring.
Logs are written to both console and file with appropriate levels.
"""

import logging
import sys

# Ensure Windows console/file logging can handle Unicode (emoji, etc.)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from pathlib import Path
from typing import Optional, Any, Dict
from datetime import datetime
import json


class StructuredLogger:
    """
    Structured logging with JSON and plain text support.
    
    Features:
    - Multiple log levels
    - JSON structured output
    - File and console handlers
    - Context tracking
    - Performance metrics
    """
    
    _instance = None
    _loggers: Dict[str, logging.Logger] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return

        # LỖI ĐÃ SỬA (2 lỗi):
        # 1) `Path("logs")` là đường dẫn TƯƠNG ĐỐI theo cwd lúc chạy lệnh —
        #    đổi sang tuyệt đối theo vị trí file này (giống các chỗ khác
        #    trong code), để log luôn ghi đúng vào thư mục gốc dự án
        #    (`<project_root>/logs`) bất kể lệnh được chạy từ đâu.
        # 2) `mkdir()` trước đây KHÔNG có try/except — nếu vì lý do nào đó
        #    (quyền ghi, ổ đĩa chỉ đọc...) không tạo được thư mục, lỗi này
        #    xảy ra ngay lúc `import project` (vì logger được khởi tạo ở
        #    project/__init__.py) và làm CRASH TOÀN BỘ ứng dụng chỉ vì
        #    logging. Nay bọc try/except: nếu không tạo được thư mục log,
        #    ứng dụng vẫn chạy bình thường (chỉ log ra console, không ghi
        #    file) thay vì crash.
        project_root = Path(__file__).resolve().parents[2]
        self.log_dir = project_root / "logs"
        try:
            self.log_dir.mkdir(exist_ok=True, parents=True)
        except Exception:
            self.log_dir = None
        self.context = {}
        self._initialized = True
    
    def get_logger(self, name: str, log_file: Optional[str] = None) -> logging.Logger:
        """
        Get or create a logger with the given name.
        
        Args:
            name: Logger name (typically __name__)
            log_file: Optional log file path
        
        Returns:
            Configured logger instance
        """
        if name in self._loggers:
            return self._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        # Avoid UnicodeEncodeError in Windows terminals using legacy encodings.
        # LỖI ĐÃ SỬA: trước đây gọi `.reconfigure()` không có try/except — một
        # số môi trường (stdout bị redirect/pipe, số Python cũ hơn...) không
        # hỗ trợ `.reconfigure()` và sẽ ném lỗi ngay khi tạo logger đầu tiên,
        # crash toàn bộ ứng dụng. Nay bọc try/except để bỏ qua an toàn.
        try:
            console_handler.stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter.default_msec_format = '%s'
        console_handler.setFormatter(console_formatter)
        # Encode errors: replace unsupported unicode so logger never crashes
        console_handler.terminator = "\n"

        logger.addHandler(console_handler)

        # File handler — CHỈ thêm nếu thư mục log tạo được thành công
        # (self.log_dir sẽ là None nếu __init__ không tạo được thư mục, xem
        # giải thích ở __init__ phía trên). Nếu không, ứng dụng vẫn chạy tốt,
        # chỉ log ra console thay vì ghi file.
        if self.log_dir is not None:
            try:
                if log_file is None:
                    log_file_path = self.log_dir / f"{name.replace('.', '_')}.log"
                else:
                    log_file_path = self.log_dir / log_file

                log_file_path.parent.mkdir(exist_ok=True, parents=True)
                file_handler = logging.FileHandler(log_file_path)
                file_handler.setLevel(logging.DEBUG)
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)
            except Exception:
                pass  # Vẫn tiếp tục chạy với console logging

        self._loggers[name] = logger
        return logger
    
    def set_context(self, **kwargs) -> None:
        """
        Set request context for structured logging.
        
        Args:
            **kwargs: Context key-value pairs
        """
        self.context.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear all context."""
        self.context.clear()
    
    def get_context(self) -> Dict[str, Any]:
        """Get current context."""
        return self.context.copy()


# Global logger instance
_logger_instance = StructuredLogger()


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured logger
    
    Example:
        logger = get_logger(__name__)
        logger.info("Application started")
    """
    return _logger_instance.get_logger(name)


def log_prediction(logger: logging.Logger, players: tuple, 
                   probability: float, confidence: str) -> None:
    """
    Log a prediction event.
    
    Args:
        logger: Logger instance
        players: Tuple of (player_a, player_b)
        probability: Win probability
        confidence: Confidence level
    """
    player_a, player_b = players
    logger.info(
        f"Prediction: {player_a} vs {player_b} | "
        f"Prob: {probability:.1%} | Confidence: {confidence}"
    )


def log_feature_importance(logger: logging.Logger, 
                          features: Dict[str, float], 
                          top_n: int = 5) -> None:
    """
    Log top feature importances.
    
    Args:
        logger: Logger instance
        features: Feature importance dictionary
        top_n: Number of top features to log
    """
    sorted_features = sorted(features.items(), 
                           key=lambda x: abs(x[1]), 
                           reverse=True)[:top_n]
    
    feature_str = ", ".join([f"{name}: {val:.4f}" 
                           for name, val in sorted_features])
    logger.info(f"Top features: {feature_str}")


def log_model_performance(logger: logging.Logger,
                         metrics: Dict[str, float],
                         model_name: str = "Model") -> None:
    """
    Log model performance metrics.
    
    Args:
        logger: Logger instance
        metrics: Performance metrics dictionary
        model_name: Model name
    """
    metrics_str = ", ".join([f"{name}: {value:.4f}" 
                           for name, value in metrics.items()])
    logger.info(f"{model_name} Performance: {metrics_str}")


def log_error_with_context(logger: logging.Logger,
                          error: Exception,
                          message: str = "Error occurred") -> None:
    """
    Log error with full context.
    
    Args:
        logger: Logger instance
        error: Exception object
        message: Error message
    """
    logger.error(
        f"{message}: {str(error)}",
        exc_info=True
    )


# Convenience logging functions
def log_info(name: str, message: str) -> None:
    """Log info message."""
    get_logger(name).info(message)


def log_warning(name: str, message: str) -> None:
    """Log warning message."""
    get_logger(name).warning(message)


def log_error(name: str, message: str) -> None:
    """Log error message."""
    get_logger(name).error(message)


def log_debug(name: str, message: str) -> None:
    """Log debug message."""
    get_logger(name).debug(message)


__all__ = [
    "StructuredLogger",
    "get_logger",
    "log_prediction",
    "log_feature_importance",
    "log_model_performance",
    "log_error_with_context",
    "log_info",
    "log_warning",
    "log_error",
    "log_debug",
]
