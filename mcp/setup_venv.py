#!/usr/bin/env python3
"""Create mcp/.venv and install Paperspace FS MCP dependencies."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MCP_DIR = Path(__file__).parent.resolve()
VENV_DIR = MCP_DIR / ".venv"
REQ = MCP_DIR / "requirements.txt"


def venv_python() -> Path:
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def main() -> int:
    py = venv_python()
    if not py.is_file():
        print(f"Creating venv at {VENV_DIR} ...")
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
    print(f"Installing {REQ} ...")
    subprocess.check_call([str(py), "-m", "pip", "install", "-U", "pip", "wheel"])
    subprocess.check_call([str(py), "-m", "pip", "install", "-r", str(REQ)])
    print(f"Done. MCP Python: {py}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
