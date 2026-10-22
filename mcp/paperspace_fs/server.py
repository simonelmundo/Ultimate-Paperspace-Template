#!/usr/bin/env python3
"""
Paperspace notebook filesystem MCP (stdio).

Uses the Paperspace notebooks API + Jupyter Contents API (same approach as
renpy-project-manager/paperspace_storage.py) so Cursor can list/read/upload
files on the running Gradient notebook — including setup logs under /notebooks.

Usage (Cursor mcp.json):
  python mcp/paperspace_fs/server.py

Env:
  PAPERSPACE_API_KEY   required
  PAPERSPACE_NOTEBOOK_HOST_SUFFIX  default clg07azjl.paperspacegradient.com
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
import uuid
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP

try:
    from websocket import create_connection
except ImportError:  # pragma: no cover
    create_connection = None  # type: ignore[assignment]

API_KEY = os.getenv("PAPERSPACE_API_KEY", "").strip()
BASE_URL = os.getenv("PAPERSPACE_API_BASE", "https://api.paperspace.com").rstrip("/")
HOST_SUFFIX = os.getenv(
    "PAPERSPACE_NOTEBOOK_HOST_SUFFIX",
    "clg07azjl.paperspacegradient.com",
).lstrip(".")

# Reject obviously destructive commands unless allow_dangerous=True.
_DANGEROUS_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\brm\s+(-[a-zA-Z]*rf|-[a-zA-Z]*fr)\b",
        r"\brm\s+-r\b",
        r"\bmkfs\b",
        r"\bdd\s+.*\bof\s*=\s*/dev/",
        r"\b(shutdown|reboot|halt|poweroff)\b",
        r":\(\)\s*\{\s*:\|:&\s*\};:",
        r"\bchmod\s+-R\s+777\s+/",
        r">\s*/dev/sd[a-z]",
        r"\b(wipefs|shred)\b",
        r"\bcurl\b.*\|\s*(ba)?sh\b",
        r"\bwget\b.*\|\s*(ba)?sh\b",
    )
]

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07")
_DONE_MARKER = "__MCP_CMD_DONE__"

mcp = FastMCP("paperspace-fs")


def _api_headers() -> dict[str, str]:
    if not API_KEY:
        raise RuntimeError(
            "PAPERSPACE_API_KEY is not set. Add it to .cursor/mcp.json env or your environment."
        )
    return {"Authorization": f"Bearer {API_KEY}"}


def _jupyter_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"token {token}"}


def _notebook_url(notebook_id: str) -> str:
    return f"https://{notebook_id}.{HOST_SUFFIX}"


def get_running_notebook() -> dict[str, Any]:
    """Return first Running notebook: id, name, url, token, state, machineType."""
    response = requests.get(
        f"{BASE_URL}/v1/notebooks",
        headers=_api_headers(),
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    items = payload.get("items") or []
    running = next((n for n in items if n.get("state") == "Running"), None)
    if not running:
        raise RuntimeError("No running Paperspace notebook found.")
    notebook_id = running.get("id")
    token = running.get("token")
    if not notebook_id or not token:
        raise RuntimeError("Running notebook is missing id or Jupyter token.")
    return {
        "id": notebook_id,
        "name": running.get("name"),
        "state": running.get("state"),
        "machineType": running.get("machineType"),
        "url": _notebook_url(notebook_id),
        "token": token,
    }


def _resolve_session(
    notebook_url: str | None = None,
    token: str | None = None,
) -> tuple[str, str]:
    if notebook_url and token:
        return notebook_url.rstrip("/"), token
    info = get_running_notebook()
    return info["url"], info["token"]


def _contents_url(notebook_url: str, path: str) -> str:
    rel = (path or "").strip().lstrip("/")
    if rel:
        return f"{notebook_url}/api/contents/{rel}"
    return f"{notebook_url}/api/contents"


@mcp.tool()
def paperspace_notebook_info() -> dict[str, Any]:
    """Get the currently running Paperspace notebook (id, name, url, machine type). Token is omitted from the response."""
    info = get_running_notebook()
    return {
        "id": info["id"],
        "name": info["name"],
        "state": info["state"],
        "machineType": info["machineType"],
        "url": info["url"],
        "has_token": bool(info.get("token")),
    }


@mcp.tool()
def paperspace_list_directory(path: str = "") -> dict[str, Any]:
    """
    List files and directories under a notebook path via Jupyter Contents API.
    path is relative to the notebook root (often /notebooks), e.g. '' , 'logs', 'sd_comfy'.
    """
    notebook_url, token = _resolve_session()
    server_url = _contents_url(notebook_url, path)
    response = requests.get(
        server_url,
        headers=_jupyter_headers(token),
        params={"content": 1},
        timeout=60,
    )
    if response.status_code == 404:
        return {"success": False, "error": f"Path not found: {path or '/'}"}
    response.raise_for_status()
    payload = response.json()
    if payload.get("type") == "file":
        return {
            "success": True,
            "type": "file",
            "path": payload.get("path"),
            "size": payload.get("size"),
            "last_modified": payload.get("last_modified"),
        }
    entries = []
    for item in payload.get("content") or []:
        entries.append(
            {
                "name": item.get("name"),
                "path": item.get("path"),
                "type": item.get("type"),
                "size": item.get("size"),
                "last_modified": item.get("last_modified"),
            }
        )
    return {
        "success": True,
        "type": "directory",
        "path": payload.get("path") or path or "",
        "entries": entries,
        "count": len(entries),
    }


@mcp.tool()
def paperspace_read_file(
    path: str,
    max_chars: int = 80000,
    tail_lines: int = 0,
) -> dict[str, Any]:
    """
    Read a text file from the running notebook (Jupyter Contents API).
    Use for setup logs written under the notebook FS, e.g. logs/comfy-run.log.
    If tail_lines > 0, return only the last N lines (good for live log tails).
    max_chars caps the returned text to keep MCP responses small.
    """
    if not path or not path.strip():
        return {"success": False, "error": "path is required"}

    notebook_url, token = _resolve_session()
    server_url = _contents_url(notebook_url, path)
    response = requests.get(
        server_url,
        headers=_jupyter_headers(token),
        params={"content": 1},
        timeout=120,
    )
    if response.status_code == 404:
        return {"success": False, "error": f"File not found: {path}"}
    response.raise_for_status()
    payload = response.json()
    if payload.get("type") != "file":
        return {"success": False, "error": f"Not a file: {path} (type={payload.get('type')})"}

    fmt = payload.get("format") or "text"
    raw = payload.get("content")
    if raw is None:
        return {"success": False, "error": "File has no content field (may be too large for Contents API)"}

    if fmt == "base64":
        try:
            text = base64.b64decode(raw).decode("utf-8", errors="replace")
        except Exception as exc:
            return {"success": False, "error": f"Failed to decode base64 content: {exc}"}
    else:
        text = raw if isinstance(raw, str) else str(raw)

    total_chars = len(text)
    total_lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
    truncated = False

    if tail_lines and tail_lines > 0:
        lines = text.splitlines()
        if len(lines) > tail_lines:
            text = "\n".join(lines[-tail_lines:])
            truncated = True
            if text and not text.endswith("\n"):
                text += "\n"

    if max_chars and len(text) > max_chars:
        text = text[-max_chars:]
        truncated = True

    return {
        "success": True,
        "path": payload.get("path") or path,
        "size": payload.get("size"),
        "last_modified": payload.get("last_modified"),
        "total_chars": total_chars,
        "total_lines": total_lines,
        "truncated": truncated,
        "tail_lines": tail_lines or None,
        "content": text,
    }


@mcp.tool()
def paperspace_upload_file(local_path: str, remote_path: str) -> dict[str, Any]:
    """Upload a local file to the running notebook. remote_path is relative to notebook root."""
    if not os.path.isfile(local_path):
        return {"success": False, "error": f"Local file not found: {local_path}"}
    notebook_url, token = _resolve_session()
    with open(local_path, "rb") as handle:
        content_b64 = base64.b64encode(handle.read()).decode("ascii")
    response = requests.put(
        _contents_url(notebook_url, remote_path),
        headers=_jupyter_headers(token),
        json={"type": "file", "format": "base64", "content": content_b64},
        timeout=180,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        return {"success": False, "error": f"{exc}: {response.text[:500]}"}
    return {"success": True, "local_path": local_path, "remote_path": remote_path.lstrip("/")}


@mcp.tool()
def paperspace_write_text(remote_path: str, content: str) -> dict[str, Any]:
    """Create/overwrite a text file on the notebook (e.g. write a helper script under notebooks/)."""
    notebook_url, token = _resolve_session()
    content_b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
    response = requests.put(
        _contents_url(notebook_url, remote_path),
        headers=_jupyter_headers(token),
        json={"type": "file", "format": "base64", "content": content_b64},
        timeout=120,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        return {"success": False, "error": f"{exc}: {response.text[:500]}"}
    return {"success": True, "remote_path": remote_path.lstrip("/"), "bytes": len(content.encode("utf-8"))}


@mcp.tool()
def paperspace_create_directory(path: str) -> dict[str, Any]:
    """Create a directory (and parents) on the notebook filesystem."""
    notebook_url, token = _resolve_session()
    parts = [p for p in path.strip("/").split("/") if p]
    current = ""
    created = []
    for part in parts:
        current = f"{current}/{part}" if current else part
        server_url = _contents_url(notebook_url, current)
        response = requests.get(server_url, headers=_jupyter_headers(token), timeout=60)
        if response.status_code == 200:
            continue
        if response.status_code != 404:
            return {"success": False, "error": f"GET {current}: {response.status_code} {response.text[:300]}"}
        put = requests.put(
            server_url,
            headers=_jupyter_headers(token),
            json={"type": "directory"},
            timeout=60,
        )
        try:
            put.raise_for_status()
        except requests.HTTPError as exc:
            return {"success": False, "error": f"Create {current}: {exc}"}
        created.append(current)
    return {"success": True, "path": "/".join(parts), "created": created}


@mcp.tool()
def paperspace_delete_file(path: str) -> dict[str, Any]:
    """Delete a file on the notebook filesystem."""
    notebook_url, token = _resolve_session()
    response = requests.delete(
        _contents_url(notebook_url, path),
        headers=_jupyter_headers(token),
        timeout=120,
    )
    if response.status_code in (200, 204):
        return {"success": True, "path": path.lstrip("/")}
    return {
        "success": False,
        "error": f"Delete failed {response.status_code}: {response.text[:500]}",
    }


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _is_dangerous_command(command: str) -> str | None:
    for pattern in _DANGEROUS_PATTERNS:
        if pattern.search(command):
            return pattern.pattern
    return None


def _run_in_ephemeral_terminal(
    notebook_url: str,
    token: str,
    command: str,
    timeout_sec: int = 60,
) -> dict[str, Any]:
    """
    Create a NEW Jupyter terminal, run one command, then DELETE only that terminal.
    Never touches other existing terminals (e.g. the user's interactive session).
    """
    if create_connection is None:
        return {
            "success": False,
            "error": "websocket-client is not installed in the MCP venv. Run: python mcp/setup_venv.py",
        }

    headers = {**_jupyter_headers(token), "Content-Type": "application/json"}
    before = requests.get(f"{notebook_url}/api/terminals", headers=headers, timeout=30)
    before.raise_for_status()
    existing = {t.get("name") for t in (before.json() or [])}

    created = requests.post(f"{notebook_url}/api/terminals", headers=headers, timeout=60)
    created.raise_for_status()
    term_name = created.json()["name"]
    if term_name in existing:
        # Extremely unlikely; abort rather than risk killing someone else's term.
        return {
            "success": False,
            "error": f"Refusing to use terminal name that already existed: {term_name}",
        }

    ws = None
    output_chunks: list[str] = []
    try:
        ws_url = (
            notebook_url.replace("https://", "wss://").replace("http://", "ws://")
            + f"/terminals/websocket/{term_name}?token={token}"
        )
        ws = create_connection(
            ws_url,
            header=[f"Authorization: token {token}"],
            timeout=60,
        )

        def drain(seconds: float) -> str:
            chunks: list[str] = []
            end = time.time() + seconds
            assert ws is not None
            ws.settimeout(0.5)
            while time.time() < end:
                try:
                    raw = ws.recv()
                    data = json.loads(raw)
                    if isinstance(data, list) and data and data[0] == "stdout":
                        chunks.append(str(data[1]))
                except Exception:
                    pass
            return "".join(chunks)

        # Let the shell prompt settle.
        drain(2.5)

        # Unique marker so we know when the remote command finished.
        # IMPORTANT: only treat a *standalone* marker line as done — the terminal
        # echoes the typed command (which contains the marker text) before it runs.
        marker = f"{_DONE_MARKER}_{uuid.uuid4().hex[:8]}"
        remote = f"bash -lc {json.dumps(command)} ; printf '%s\\n' {marker}"
        ws.send(json.dumps(["stdin", remote + "\r"]))

        def marker_seen(text: str) -> bool:
            return any(line.strip() == marker for line in text.splitlines())

        deadline = time.time() + max(5, timeout_sec)
        while time.time() < deadline:
            chunk = drain(1.0)
            if chunk:
                output_chunks.append(chunk)
            joined = "".join(output_chunks)
            if marker_seen(joined):
                break

        raw_out = _strip_ansi("".join(output_chunks))
        timed_out = not marker_seen(raw_out)
        if not timed_out:
            # Keep output before the standalone marker line; drop the echoed input command line if present.
            kept: list[str] = []
            for line in raw_out.splitlines():
                if line.strip() == marker:
                    break
                kept.append(line)
            raw_out = "\n".join(kept)
            # Remove the shell-echo of our bash -lc invocation if present.
            cleaned: list[str] = []
            for line in raw_out.splitlines():
                if "bash -lc" in line and marker.split("_")[0] in line:
                    continue
                if line.strip().startswith("bash -lc "):
                    continue
                cleaned.append(line)
            raw_out = "\n".join(cleaned).strip()

        return {
            "success": not timed_out,
            "timed_out": timed_out,
            "terminal": term_name,
            "command": command,
            "output": raw_out[-120000:],
        }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "terminal": term_name,
            "command": command,
            "output": _strip_ansi("".join(output_chunks))[-120000:],
        }
    finally:
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        # Only delete the terminal we created.
        try:
            requests.delete(
                f"{notebook_url}/api/terminals/{term_name}",
                headers=headers,
                timeout=30,
            )
        except Exception:
            pass


@mcp.tool()
def paperspace_run_command(
    command: str,
    timeout_sec: int = 60,
    allow_dangerous: bool = False,
) -> dict[str, Any]:
    """
    Run a shell command on the running Paperspace notebook via an ephemeral Jupyter terminal.

    Safety:
    - Creates a NEW terminal, runs the command, then deletes ONLY that terminal.
    - Does not touch your existing notebook terminals.
    - Blocks destructive patterns (rm -rf, mkfs, shutdown, curl|sh, etc.) unless
      allow_dangerous=True.
    - Prefer read-only commands (ls, tail, cp logs into /notebooks/logs, etc.).

    Example (copy logs into MCP-visible path):
      mkdir -p /notebooks/logs && cp -f /tmp/model_download.log /tmp/log/*.log /notebooks/logs/ 2>/dev/null; ls -la /notebooks/logs
    """
    command = (command or "").strip()
    if not command:
        return {"success": False, "error": "command is required"}
    if len(command) > 4000:
        return {"success": False, "error": "command too long (max 4000 chars)"}

    if not allow_dangerous:
        hit = _is_dangerous_command(command)
        if hit:
            return {
                "success": False,
                "error": (
                    "Blocked potentially destructive command. "
                    "Re-run with allow_dangerous=True only if you are sure. "
                    f"Matched: {hit}"
                ),
                "command": command,
            }

    notebook_url, token = _resolve_session()
    return _run_in_ephemeral_terminal(
        notebook_url,
        token,
        command,
        timeout_sec=timeout_sec,
    )


def main() -> None:
    # FastMCP stdio — keep human noise on stderr only
    print("[paperspace-fs] starting MCP stdio server", file=sys.stderr, flush=True)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
