"""Tests cho pipeline load data (step 01).

LƯU Ý: Module Python không thể import trực tiếp tên bắt đầu bằng số (01_load_data).
Phải dùng importlib để import.
"""
import importlib
import sys
from pathlib import Path

import pytest

# Đảm bảo project root trong sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _import_pipeline(module_name: str):
    """Import pipeline module bắt đầu bằng số (01_, 02_, ...) bằng importlib."""
    module_path = PROJECT_ROOT / "src" / "pipelines" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDataLoader:
    """Test pipeline 01_load_data."""

    def test_load_data_module_imports(self):
        """Module 01_load_data phải import được mà không lỗi."""
        mod = _import_pipeline("01_load_data")
        assert hasattr(mod, 'load_data'), "Module thiếu hàm load_data"

    def test_load_data_function_exists(self):
        """Hàm load_data phải tồn tại và callable."""
        mod = _import_pipeline("01_load_data")
        assert callable(mod.load_data)

    def test_raw_data_file_exists(self, config):
        """File dữ liệu raw phải tồn tại trong thư mục raw_dir."""
        raw_dir = Path(config['data']['raw_dir'])
        raw_file = config['data']['raw_file']
        raw_path = raw_dir / raw_file

        # Chỉ assert nếu raw_dir tồn tại (tức đang chạy trên máy có dữ liệu)
        if raw_dir.exists():
            assert raw_path.exists(), f"Không tìm thấy file raw: {raw_path}"
            assert raw_path.stat().st_size > 0, "File raw data rỗng"


class TestDataValidation:
    """Test pipeline 02_data_validation."""

    def test_validation_module_imports(self):
        """Module 02_data_validation phải import được."""
        mod = _import_pipeline("02_data_validation")
        assert hasattr(mod, 'validate_data')


class TestCleanSchema:
    """Test pipeline 03_clean_schema."""

    def test_clean_schema_module_imports(self):
        """Module 03_clean_schema phải import được."""
        mod = _import_pipeline("03_clean_schema")
        assert hasattr(mod, 'clean_schema')
