from __future__ import annotations

import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
subprocess.run([sys.executable, "-m", "streamlit", "run", str(root / "app" / "streamlit_app.py")], check=True)
