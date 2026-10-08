#!/bin/bash
# Manage the llama-cpp-python OpenAI server on :7009.
#
#   bash start_llama_server.sh stop     # kill the server, free VRAM, do NOT restart
#   bash start_llama_server.sh start    # launch the server (loaded lazily)
#   bash start_llama_server.sh          # default: restart (stop + start)
#
# The default "restart" shape is kept because sd_comfy/main.sh and older callers
# rely on a single invocation meaning "make sure it is running".
set -euo pipefail

ACTION="${1:-restart}"
case "$ACTION" in
  start|stop|restart) ;;
  *) echo "[llama][ERROR] unknown action '$ACTION' (use start|stop|restart)" >&2; exit 2 ;;
esac

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

# Stop any running / legacy LLM server and free port $LLAMA_PORT.
# Shared by the `stop` and `restart` actions - stop_llm_server must NOT start anything.
stop_llm_server() {
  # Stop legacy Ollama
  pkill -f "ollama serve" 2>/dev/null || true
  if [[ -f /tmp/ollama.pid ]]; then
    oldpid=$(cat /tmp/ollama.pid 2>/dev/null || true)
    if [[ -n "${oldpid:-}" ]] && kill -0 "$oldpid" 2>/dev/null; then
      kill -TERM "$oldpid" 2>/dev/null || true
      sleep 1
      kill -9 "$oldpid" 2>/dev/null || true
    fi
    rm -f /tmp/ollama.pid
  fi

  # Stop the tracked llama server
  if [[ -f /tmp/llama-server.pid ]]; then
    oldpid=$(cat /tmp/llama-server.pid 2>/dev/null || true)
    if [[ -n "${oldpid:-}" ]] && kill -0 "$oldpid" 2>/dev/null; then
      kill -TERM "$oldpid" 2>/dev/null || true
      sleep 1
      kill -9 "$oldpid" 2>/dev/null || true
    fi
    rm -f /tmp/llama-server.pid
  fi

  # Stop the background "wait for GGUF then start" watcher, otherwise it would
  # relaunch the server seconds after a `stop`.
  if [[ -f /tmp/llama-server-waiter.pid ]]; then
    wpid=$(cat /tmp/llama-server-waiter.pid 2>/dev/null || true)
    if [[ -n "${wpid:-}" ]] && kill -0 "$wpid" 2>/dev/null; then
      kill -TERM "$wpid" 2>/dev/null || true
      sleep 1
      kill -9 "$wpid" 2>/dev/null || true
    fi
    rm -f /tmp/llama-server-waiter.pid
  fi

  # Catch any untracked llama_cpp.server process
  ps -eo pid=,args= | awk '/llama_cpp\.server|\/llama-server( |$)/ && $0 !~ /start_llama_server/ {print $1}' | while read -r p; do
    kill -TERM "$p" 2>/dev/null || true
  done
  sleep 1

  # Last resort: whatever still holds the port
  if command -v lsof >/dev/null 2>&1; then
    pids=$(lsof -ti:"$LLAMA_PORT" 2>/dev/null || true)
    [[ -n "${pids:-}" ]] && kill -9 $pids 2>/dev/null || true
  fi
}

stop_llm_server

# `stop` is terminal: VRAM is freed and we must not fall through to the start path.
if [[ "$ACTION" == "stop" ]]; then
  rm -f /tmp/llama-server.pid /tmp/llama-server-waiter.pid
  # Mark that a stop was requested so the sd_comfy main.sh GGUF watcher does not
  # silently relaunch the server on the next boot of the stack.
  if command -v lsof >/dev/null 2>&1; then
    if lsof -ti:"$LLAMA_PORT" >/dev/null 2>&1; then
      die "port $LLAMA_PORT still in use after stop"
    fi
  fi
  log "stopped - port $LLAMA_PORT free, VRAM released"
  exit 0
fi

[[ -x "$COMFY_PY" ]] || die "Comfy venv python missing: $COMFY_PY"

llama_has_gpu() {
  "$COMFY_PY" -c "from llama_cpp.llama_cpp import llama_supports_gpu_offload; import sys; sys.exit(0 if llama_supports_gpu_offload() else 1)" 2>/dev/null
}

resolve_cuda_home() {
  local cand
  for cand in "${CUDA_HOME:-}" /usr/local/cuda-12.8 /usr/local/cuda-12.4 /usr/local/cuda-11.6 /usr/local/cuda; do
    [[ -n "$cand" && -x "$cand/bin/nvcc" ]] && { echo "$cand"; return 0; }
  done
  command -v nvcc >/dev/null 2>&1 && { dirname "$(dirname "$(command -v nvcc)")"; return 0; }
  return 1
}

# Pip success != GPU: CPU wheels report gpu_offload False. Verify, then force CUDA source.
if ! "$COMFY_PY" -c "import llama_cpp, llama_cpp.server" 2>/dev/null || ! llama_has_gpu; then
  log "Ensuring llama-cpp-python with GPU offload..."
  if [[ "${LLAMA_FORCE_CUDA_SOURCE:-0}" != "1" ]]; then
    "$COMFY_PY" -m pip uninstall -y llama-cpp-python llama-cpp-python-cuda 2>/dev/null || true
    # --only-binary=:all: = fail fast if no prebuilt CUDA wheel (do NOT compile a CPU sdist here)
    log "Trying prebuilt cu124 wheel only (no source fallback on this step)..."
    if "$COMFY_PY" -m pip install --no-cache-dir --force-reinstall --only-binary=:all: \
        "llama-cpp-python[server]" \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124; then
      if "$COMFY_PY" -c "import llama_cpp, llama_cpp.server" 2>/dev/null && llama_has_gpu; then
        log "llama-cpp-python GPU offload OK via wheel"
      else
        log "Wheel installed but gpu_offload False - will source-build CUDA"
        "$COMFY_PY" -m pip uninstall -y llama-cpp-python llama-cpp-python-cuda 2>/dev/null || true
      fi
    else
      log "No matching prebuilt CUDA wheel - will source-build"
    fi
  fi
  if ! "$COMFY_PY" -c "import llama_cpp, llama_cpp.server" 2>/dev/null || ! llama_has_gpu; then
    cuda_home=$(resolve_cuda_home) || die "No CUDA toolkit/nvcc for source build"
    export CUDA_HOME="$cuda_home"
    export PATH="$CUDA_HOME/bin:${PATH:-}"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=86"
    export FORCE_CMAKE=1
    log "Building llama-cpp-python from source with CUDA (CUDA_HOME=$CUDA_HOME)"
    "$COMFY_PY" -m pip uninstall -y llama-cpp-python llama-cpp-python-cuda 2>/dev/null || true
    CMAKE_ARGS="$CMAKE_ARGS" FORCE_CMAKE=1 \
      "$COMFY_PY" -m pip install --no-cache-dir --force-reinstall --no-binary=llama-cpp-python \
      "llama-cpp-python[server]" || die "CUDA source build failed"
  fi
  "$COMFY_PY" -c "import llama_cpp, llama_cpp.server" || die "import failed after install"
  llama_has_gpu || die "gpu_offload still False - refusing CPU-only LLM server"
  log "llama-cpp-python GPU offload OK"
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
