import sys
import subprocess
from pathlib import Path


def main():
    """Khởi chạy ứng dụng Streamlit tự động."""
    print("Dang khoi dong ATP Match Predictor AI...")

    project_root = Path(__file__).resolve().parent
    app_path = "project/main.py"

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                app_path,
            ],
            check=True,
            cwd=str(project_root),
        )
    except KeyboardInterrupt:
        print("\nDa dong ung dung.")
    except Exception as e:
        print(f"Loi khoi chay: {e}")


if __name__ == "__main__":
    main()

