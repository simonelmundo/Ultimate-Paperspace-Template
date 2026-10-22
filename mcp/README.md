# Paperspace FS MCP
Isolated venv for the Paperspace filesystem MCP (pydantic v2 / mcp).

## Setup (once)
```bash
python mcp/setup_venv.py
```

## Cursor
Configured globally in `%USERPROFILE%\\.cursor\\mcp.json` as `paperspace-notebook` (available in every Cursor workspace). Reload MCP / restart Cursor after setup.

## Tools
- `paperspace_notebook_info` — running notebook id/url
- `paperspace_list_directory` — list notebook FS path
- `paperspace_read_file` — read/tail a text file (use for `logs/comfy-run.log`)
- `paperspace_upload_file` / `paperspace_write_text` / `paperspace_create_directory` / `paperspace_delete_file`
- `paperspace_run_command` — run a shell command via an **ephemeral** Jupyter terminal (creates a new term, runs, deletes **only** that term; never touches your existing terminals). Destructive patterns blocked unless `allow_dangerous=true`.

## Live logs tip
Copy `/tmp` logs into the notebook FS (MCP-visible), then read them:
```text
mkdir -p /notebooks/logs && cp -f /tmp/model_download.log /tmp/log/*.log /notebooks/logs/ 2>/dev/null; ls -la /notebooks/logs
```
Then: `paperspace_read_file(path="logs/model_download.log", tail_lines=80)`.
