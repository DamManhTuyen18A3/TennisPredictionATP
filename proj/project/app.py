"""Backward-compatible entrypoint.

This file previously contained the Streamlit app. It has been renamed to
`app/main.py` to avoid Python import shadowing issues (when running
`streamlit run app/app.py`, Python can treat the file as a non-package `app`).

Run:
  python runapp.py
or
  streamlit run app/main.py

LỖI ĐÃ SỬA: cũng cần bootstrap sys.path giống hệt main.py — xem giải thích
chi tiết trong project/main.py.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from project.main import main


if __name__ == "__main__":
    main()

