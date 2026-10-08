#!/bin/bash
# One-shot: ensure llama-cpp-python in Comfy venv, wait for a GGUF if needed, serve on :7009
set -euo pipefail

export LOG_DIR="${LOG_DIR:-/tmp/log}"
export DATA_DIR="${DATA_DIR:-/tmp}"
export MODEL_DIR="${MODEL_DIR:-$DATA_DIR/stable-diffusion-models}"
export VENV_DIR="${VENV_DIR:-/tmp}"
export LLAMA_PORT="${LLAMA_PORT:-7009}"
export LLAMA_CTX="${LLAMA_CTX:-4096}"
export LLAMA_NGL="${LLAMA_NGL:-99}"
export LLAMA_CKPT_DIR="${LLAMA_CKPT_DIR:-$MODEL_DIR/llm_checkpoints}"
export LLAMA_GGUF_WAIT_SEC="${LLAMA_GGUF_WAIT_SEC:-7200}"
COMFY_PY="${VENV_DIR}/sd_comfy-env/bin/python"

mkdir -p "$LOG_DIR" "$LLAMA_CKPT_DIR"

log() { echo "[llama] $*"; }
die() { echo "[llama][ERROR] $*" >&2; exit 1; }

pick_gguf() {
  if [[ -n "${LLAMA_GGUF:-}" && -f "$LLAMA_GGUF" ]]; then
    echo "$LLAMA_GGUF"
    return 0
  fi
  local gguf
  gguf=$(ls -1t "$LLAMA_CKPT_DIR"/*.gguf 2>/dev/null | head -1 || true)
  [[ -n "$gguf" && -f "$gguf" ]] || return 1
  local sz
  sz=$(stat -c%s "$gguf" 2>/dev/null || echo 0)
  [[ "$sz" -ge 1048576 ]] || return 1
  echo "$gguf"
}

# Stop legacy / previous servers (do not pkill by script filename)
pkill -f "ollama serve" 2>/dev/null || true
if [[ -f /tmp/llama-server.pid ]]; then
  oldpid=$(cat /tmp/llama-server.pid 2>/dev/null || true)
  if [[ -n "${oldpid:-}" ]] && kill -0 "$oldpid" 2>/dev/null; then
    kill -TERM "$oldpid" 2>/dev/null || true
    sleep 1
    kill -9 "$oldpid" 2>/dev/null || true
  fi
  rm -f /tmp/llama-server.pid
fi
ps -eo pid=,args= | awk '/llama_cpp\.server|\/llama-server( |$)/ && $0 !~ /start_llama_server/ {print $1}' | while read -r p; do
  kill -TERM "$p" 2>/dev/null || true
done
sleep 1
if command -v lsof >/dev/null 2>&1; then
  pids=$(lsof -ti:"$LLAMA_PORT" 2>/dev/null || true)
  [[ -n "${pids:-}" ]] && kill -9 $pids 2>/dev/null || true
fi

[[ -x "$COMFY_PY" ]] || die "Comfy venv python missing: $COMFY_PY"

if ! "$COMFY_PY" -c "import llama_cpp, llama_cpp.server" 2>/dev/null; then
  log "Installing llama-cpp-python[server] into Comfy venv (CUDA)..."
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
  export PATH="$CUDA_HOME/bin:${PATH:-}"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  if ! "$COMFY_PY" -m pip install --no-cache-dir "llama-cpp-python[server]" \
      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124; then
    CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 \
      "$COMFY_PY" -m pip install --no-cache-dir "llama-cpp-python[server]"
  fi
  "$COMFY_PY" -c "import llama_cpp, llama_cpp.server" || die "import failed after install"
fi

# Wait for GGUF instead of exiting if the folder is empty (download may still be running)
waited=0
gguf=""
while true; do
  gguf=$(pick_gguf) || gguf=""
  if [[ -n "$gguf" ]]; then
    s1=$(stat -c%s "$gguf" 2>/dev/null || echo 0)
    sleep 3
    s2=$(stat -c%s "$gguf" 2>/dev/null || echo 0)
    if [[ "$s1" == "$s2" && "$s1" -ge 1048576 ]]; then
      break
    fi
    log "GGUF still growing ($s1 -> $s2), waiting..."
  else
    if (( waited == 0 )); then
      log "No .gguf yet in $LLAMA_CKPT_DIR - waiting up to ${LLAMA_GGUF_WAIT_SEC}s"
    elif (( waited % 60 == 0 )); then
      log "Still waiting for .gguf (${waited}s / ${LLAMA_GGUF_WAIT_SEC}s)..."
    fi
  fi
  if (( waited >= LLAMA_GGUF_WAIT_SEC )); then
    die "Timed out waiting for a .gguf in $LLAMA_CKPT_DIR"
  fi
  sleep 5
  waited=$((waited + 5))
done

alias_name=$(basename "$gguf")
alias_name="${alias_name%.gguf}"

log "Starting llama-cpp-python: $gguf (alias=$alias_name) :$LLAMA_PORT ctx=$LLAMA_CTX ngl=$LLAMA_NGL"
nohup "$COMFY_PY" -m llama_cpp.server \
  --model "$gguf" \
  --model_alias "$alias_name" \
  --host 0.0.0.0 \
  --port "$LLAMA_PORT" \
  --n_ctx "$LLAMA_CTX" \
  --n_gpu_layers "$LLAMA_NGL" \
  > "$LOG_DIR/llama-server.log" 2>&1 &
echo $! > /tmp/llama-server.pid
sleep 5

kill -0 "$(cat /tmp/llama-server.pid)" 2>/dev/null || {
  tail -n 80 "$LOG_DIR/llama-server.log" || true
  die "server failed - see $LOG_DIR/llama-server.log"
}

log "OK pid=$(cat /tmp/llama-server.pid) - waiting for /v1/models"
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18; do
  if curl -fsS -m 5 "http://127.0.0.1:${LLAMA_PORT}/v1/models"; then
    echo
    log "API ready"
    exit 0
  fi
  log "waiting for model load ($i/18)..."
  sleep 10
done
tail -n 60 "$LOG_DIR/llama-server.log" || true
die "API did not become ready"
