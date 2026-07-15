import logging
import os
import yaml
from pathlib import Path

# Thư mục gốc của toàn bộ dự án
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def get_project_root():
    return PROJECT_ROOT

def load_config(config_path="configs/config.yaml"):
    """Load yaml configuration file and convert paths to absolute paths."""
    full_config_path = PROJECT_ROOT / config_path
    
    with open(full_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    # Tự động nối các đường dẫn tương đối trong config thành đường dẫn tuyệt đối
    # Điều này giúp code có thể chạy ở bất kỳ máy nào, bất kỳ thư mục làm việc (cwd) nào
    for category in config:
        if isinstance(config[category], dict):
            for key, value in config[category].items():
                if key.endswith('_dir') or key == 'metrics_file':
                    config[category][key] = str(PROJECT_ROOT / value)
                    
    return config

def get_logger(name):
    """Setup and return a logger instance."""
    config = load_config()
    log_dir = Path(config["logging"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(config["logging"]["level"])
    
    # Prevent duplicate handlers
    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
        fh.setLevel(config["logging"]["level"])
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(config["logging"]["level"])
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
    return logger
