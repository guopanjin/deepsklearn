import subprocess
import sys
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parent
    app_path = project_root / "app.py"

    if not app_path.exists():
        raise FileNotFoundError(f"Cannot find Streamlit app: {app_path}")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        "8501"
    ]

    subprocess.run(cmd, cwd=project_root)


if __name__ == "__main__":
    main()