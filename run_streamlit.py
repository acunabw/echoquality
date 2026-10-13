from __future__ import annotations

import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    app = project_root / "app" / "streamlit_app.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        "8501",
    ]
    raise SystemExit(subprocess.call(command, cwd=project_root))
