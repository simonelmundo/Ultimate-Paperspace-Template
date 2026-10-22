#!/usr/bin/env python3
"""Print tools/list from paperspace-fs MCP over stdio."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
SERVER = ROOT / "paperspace_fs" / "server.py"


def main() -> int:
    env = os.environ.copy()
    env["PAPERSPACE_API_KEY"] = env.get("PAPERSPACE_API_KEY", "dc6398453243a752527f56c1e726f9")
    proc = subprocess.Popen(
        [str(PY), str(SERVER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(ROOT / "paperspace_fs"),
        env=env,
    )
    assert proc.stdin and proc.stdout
    messages = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "list-tools", "version": "0.1"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ]
    for msg in messages:
        proc.stdin.write(json.dumps(msg) + "\n")
    proc.stdin.flush()
    proc.stdin.close()

    tools = []
    # Read a few JSON lines from stdout
    for _ in range(10):
        line = proc.stdout.readline()
        if not line:
            break
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if data.get("id") == 2 and "result" in data:
            tools = [t["name"] for t in data["result"].get("tools", [])]
            break

    stderr = proc.stderr.read() if proc.stderr else ""
    proc.kill()
    print("TOOLS:", sorted(tools))
    print("HAS_RUN_COMMAND:", "paperspace_run_command" in tools)
    if stderr.strip():
        print("STDERR:", stderr[-1000:])
    return 0 if "paperspace_run_command" in tools else 1


if __name__ == "__main__":
    raise SystemExit(main())
