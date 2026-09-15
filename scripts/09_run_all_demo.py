from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run(project_root: Path, *args: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    command = [sys.executable, *args]
    print("\n>", " ".join(command))
    subprocess.run(command, cwd=project_root, env=env, check=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    run(root, "scripts/00_create_demo_data.py", "--output", "data/demo", "--patients", "54")
    run(root, "scripts/02_extract_features.py", "--manifest", "data/demo/manifest.csv")
    run(root, "scripts/03_analyze_indicators.py", "--bootstrap", "35", "--top-k", "12", "--min-frequency", "0.35")
    run(root, "scripts/04_train_quality_gate.py", "--fast")
    run(root, "scripts/05_build_good_reference.py")
    run(root, "scripts/06_optimize_processing.py")
    run(root, "scripts/07_evaluate_end_to_end.py", "--max-images", "10")
    run(root, "scripts/08_generate_process_diagram.py")
    print("\nDemo completo. Abra app/tk_demo.py desde Visual Studio o ejecute run_demo.bat")


if __name__ == "__main__":
    main()
