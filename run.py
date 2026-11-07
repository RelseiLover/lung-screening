"""Launch the existing screening workflow with either React or the legacy UI."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Lung screening local launcher")
    parser.add_argument("--legacy", action="store_true", help="Open the original Streamlit UI")
    parser.add_argument(
        "--check", action="store_true", help="Check the environment without starting"
    )
    parser.add_argument("--port", type=int, default=8502)
    args = parser.parse_args()
    checked = subprocess.run([sys.executable, str(ROOT / "scripts/check_environment.py")], cwd=ROOT)
    if checked.returncode or args.check:
        return checked.returncode
    if args.legacy:
        command = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ROOT / "backend/streamlit_app.py"),
            "--server.address",
            "127.0.0.1",
            "--server.port",
            str(args.port),
            "--server.headless",
            "true",
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "uvicorn",
            "web_api:app",
            "--app-dir",
            str(ROOT / "backend"),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
        ]
    print(f"Open http://127.0.0.1:{args.port}", flush=True)
    try:
        return subprocess.call(command, cwd=ROOT)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
