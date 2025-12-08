#!/bin/bash
set -e

# ============================================================================
# Text Generation WebUI Setup Script
# ============================================================================
# This script sets up oobabooga's text-generation-webui
# NOTE: This script REUSES the ComfyUI environment to save time and space
#       - PyTorch, CUDA, and most dependencies are already installed
#       - Only text-gen specific packages will be installed
#       - Run ComfyUI setup first (sd_comfy/main.sh) before running this
# ============================================================================

current_dir=$(dirname "$(realpath "$0")")
cd $current_dir

# Source parent .env and helper.sh first
source $current_dir/../.env
source $current_dir/../utils/helper.sh

# Ensure LOG_DIR is set (default to /tmp/log if not set)
export LOG_DIR="${LOG_DIR:-/tmp/log}"
mkdir -p "$LOG_DIR"

# Set textgen-specific variables (no need for local .env - parent .env + defaults are enough)
export TEXTGEN_PORT="${TEXTGEN_PORT:-7009}"
export GRADIO_ROOT_PATH="/textgen"
# Use the same LLM checkpoints directory as ComfyUI
# ComfyUI uses: $DATA_DIR/stable-diffusion-models/llm_checkpoints
export MODEL_DIR="${TEXTGEN_MODEL_DIR:-$DATA_DIR/stable-diffusion-models/llm_checkpoints}"
export REPO_DIR="${TEXTGEN_REPO_DIR:-$ROOT_REPO_DIR/text-generation-webui}"
export LINK_MODEL_TO="${TEXTGEN_LINK_MODEL_TO:-$REPO_DIR/models}"
export TEXTGEN_OPENAI_API_PORT="${TEXTGEN_OPENAI_API_PORT:-7013}"
export EXPOSE_PORTS="$EXPOSE_PORTS:$TEXTGEN_PORT:$TEXTGEN_OPENAI_API_PORT"
export PORT_MAPPING="$PORT_MAPPING:textgen:textgen_openai_api"
export REQUIRED_ENV="${REQUIRED_ENV:-}"

# Set up CUDA environment (reuse ComfyUI's setup if available, otherwise set it)
if [[ -z "$CUDA_HOME" ]]; then
    # If ComfyUI hasn't set it up yet, set the same CUDA environment
    export CUDA_HOME=/usr/local/cuda-12.8
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export FORCE_CUDA=1
    export CUDA_VISIBLE_DEVICES=0
    export TORCH_CUDA_ARCH_LIST="8.6"
    export TORCH_CUDNN_V8_API_ENABLED=1
fi

# Apply VRAM optimization (same as ComfyUI)
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:4096,garbage_collection_threshold:0.8"
export CUDA_LAUNCH_BLOCKING=0
export CUDA_DEVICE_MAX_CONNECTIONS=32

# Set up a trap to call the error_exit function on ERR signal
trap 'error_exit "### ERROR ###"' ERR


echo "### Setting up Text generation Webui ###"
log "Setting up Text generation Webui"
echo "Checking if setup is needed: REINSTALL_TEXTGEN=$REINSTALL_TEXTGEN, prepared file exists: $([ -f /tmp/textgen.prepared ] && echo 'yes' || echo 'no')"

# Check if ComfyUI environment exists
COMFY_VENV="$VENV_DIR/sd_comfy-env"
if [[ ! -d "$COMFY_VENV" ]]; then
    log "ERROR: ComfyUI environment not found at $COMFY_VENV"
    log "Please run ComfyUI setup first (sd_comfy/main.sh) to create the shared environment"
    error_exit "ComfyUI environment required"
fi

log "✅ Using shared ComfyUI environment at $COMFY_VENV"

if [[ "$REINSTALL_TEXTGEN" || ! -f "/tmp/textgen.prepared" ]]; then
  log "Running full setup (this may take several minutes)..."

    # Remove stale symlink to avoid pull conflicts
    rm -rf $LINK_MODEL_TO

    TARGET_REPO_DIR=$REPO_DIR \
    TARGET_REPO_BRANCH="main" \
    TARGET_REPO_URL="https://github.com/oobabooga/text-generation-webui" \
    UPDATE_REPO=$TEXTGEN_UPDATE_REPO \
    UPDATE_REPO_COMMIT=$TEXTGEN_UPDATE_REPO_COMMIT \
    prepare_repo
    
    # Verify repo was cloned successfully
    if [[ ! -d "$REPO_DIR" ]] || [[ ! -d "$REPO_DIR/.git" ]]; then
        log "ERROR: Repository not found at $REPO_DIR"
        log "prepare_repo may have failed"
        error_exit "Repository setup failed"
    fi
    
    # Use ComfyUI's environment instead of creating a new one
    log "🔗 Activating shared ComfyUI environment (reusing PyTorch, CUDA, and dependencies)..."
    source $COMFY_VENV/bin/activate
    
    log "Python version: $(python --version)"
    log "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not yet installed')"
    
    cd $REPO_DIR
    
    # text-generation-webui uses a requirements folder, not a single requirements.txt
    # Check for requirements files in the requirements folder
    if [[ -d "requirements" ]]; then
        # Look for the main requirements file (usually requirements.txt in the requirements folder)
        if [[ -f "requirements/requirements.txt" ]]; then
            REQ_FILE="requirements/requirements.txt"
        elif [[ -f "requirements/requirements_cuda.txt" ]]; then
            REQ_FILE="requirements/requirements_cuda.txt"
        else
            # Find any requirements file in the requirements folder
            REQ_FILE=$(find requirements -name "requirements*.txt" | head -1)
            if [[ -z "$REQ_FILE" ]]; then
                log "WARNING: No requirements file found in requirements folder"
                log "Repository contents:"
                ls -la "$REPO_DIR" | head -20
                log "Requirements folder contents:"
                ls -la "$REPO_DIR/requirements" 2>/dev/null || log "requirements folder not accessible"
                error_exit "No requirements file found"
            fi
        fi
        log "Found requirements file: $REQ_FILE"
    elif [[ -f "requirements.txt" ]]; then
        REQ_FILE="requirements.txt"
    else
        log "ERROR: No requirements file found in $REPO_DIR"
        log "Current directory: $(pwd)"
        log "Repository contents:"
        ls -la "$REPO_DIR" | head -20
        error_exit "No requirements file found"
    fi
    
    # Skip PyTorch installation - already in ComfyUI environment
    log "⏭️  Skipping PyTorch installation (already available in shared environment)"
    
    # Install only text-generation-webui specific requirements
    # Use smart upgrade strategy to minimize downgrades of packages from ComfyUI
    log "📦 Installing text-generation-webui specific requirements..."
    log "💡 Using 'only-if-needed' strategy to preserve ComfyUI's newer package versions"
    
    # --upgrade-strategy only-if-needed: Only upgrades if current version doesn't satisfy requirement
    # This minimizes downgrades (e.g., won't downgrade pillow 11.3.0 if textgen wants 10.4.0)
    # Note: Some packages may still be reinstalled if requirements.txt has exact version pins
    pip install --upgrade-strategy only-if-needed -r "$REQ_FILE" || log "⚠️  Some packages failed to install, continuing..."

    # Install GPTQ-for-LLaMa (optional - for GPTQ model quantization support)
    # Note: This may fail with newer PyTorch versions, but textgen will work without it
    log "📦 Installing GPTQ-for-LLaMa (optional quantization support)..."
    mkdir -p repositories
    TARGET_REPO_DIR=$REPO_DIR/repositories/GPTQ-for-LLaMa \
    TARGET_REPO_BRANCH="cuda" \
    TARGET_REPO_URL="https://github.com/qwopqwop200/GPTQ-for-LLaMa.git" \
    prepare_repo

    # prepare_repo changes to TARGET_REPO_DIR, so we're already in the right directory
    # But to be safe, explicitly cd to the full path
    cd $REPO_DIR/repositories/GPTQ-for-LLaMa
    log "⚠️  GPTQ-for-LLaMa may fail with PyTorch 2.8.0 (deprecated APIs) - this is optional"
    python setup_cuda.py install || {
        log "⚠️  GPTQ-for-LLaMa installation failed (likely PyTorch 2.8.0 incompatibility)"
        log "💡 Textgen will work without GPTQ support - GPTQ models won't be available"
        log "💡 You can use other quantization formats (GGUF, AWQ) instead"
    }

    # Install llama-cpp-python (for GGUF model support)
    # Build from source to avoid GLIBC version issues with precompiled binaries
    log "📦 Installing llama-cpp-python from source (for GGUF model support)..."
    pip uninstall -y llama-cpp-python llama-cpp-python-cuda 2>/dev/null || true
    
    # Force build from source with CUDA support
    # Use GGML_CUDA instead of deprecated LLAMA_CUBLAS for newer llama.cpp versions
    # --no-binary llama-cpp-python forces building from source instead of using precompiled wheels
    log "🔨 Building llama-cpp-python from source (this may take 5-10 minutes)..."
    CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --no-binary llama-cpp-python || {
        log "⚠️  llama-cpp-python installation failed (optional, continuing...)"
        log "💡 GGUF models won't be available, but other formats will work"
        log "💡 You can use Transformers/AWQ models instead of GGUF"
    }

    # Install deepspeed if needed for advanced optimizations
    log "📦 Installing deepspeed (optional optimization)..."
    pip install deepspeed || log "⚠️  DeepSpeed installation failed (optional, continuing...)"

    # Skip xformers - already installed in ComfyUI environment
    log "⏭️  Skipping xformers installation (already available in shared environment)"
    
    touch /tmp/textgen.prepared
    log "Setup completed successfully. Marker file created: /tmp/textgen.prepared"
else
    log "Setup already completed, skipping installation steps"
    log "🔗 Activating shared ComfyUI environment..."
    source $COMFY_VENV/bin/activate
fi
log "Finished Preparing Environment for Text generation Webui"


# Skip model download - users can download models manually through the UI
log "⏭️  Skipping automatic model download - download models manually through the textgen UI"

# Prepare model directory and create symlink so textgen can find models
mkdir -p "$MODEL_DIR"
rm -rf "$LINK_MODEL_TO"
ln -s "$MODEL_DIR" "$LINK_MODEL_TO"
log "📁 Model directory: $MODEL_DIR"
log "🔗 Symlinked to: $LINK_MODEL_TO"

if env | grep -q "PAPERSPACE"; then
  sed -i "s/server_port=shared.args.listen_port, inbrowser=shared.args.auto_launch, auth=auth)/server_port=shared.args.listen_port, inbrowser=shared.args.auto_launch, auth=auth, root_path='\\/textgen')/g" $REPO_DIR/server.py
fi


if [[ -z "$INSTALL_ONLY" ]]; then
  echo "### Starting Text generation Webui ###"
  log "Starting Text generation Webui"
  
  # Kill any existing textgen processes before starting
  echo "🛑 Stopping any existing textgen processes..."
  log "Checking for existing textgen processes..."
  
  # Function to kill textgen processes
  kill_existing_textgen() {
    local killed_any=false
    
    # Method 1: Kill using PID file if it exists
    if [[ -f "/tmp/textgen.pid" ]]; then
      local pid=$(cat /tmp/textgen.pid 2>/dev/null)
      if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        log "Killing process from PID file: $pid"
        # Kill the process and all its children
        pkill -P "$pid" 2>/dev/null || true
        kill -TERM "$pid" 2>/dev/null || true
        sleep 1
        # Force kill if still running
        kill -9 "$pid" 2>/dev/null || true
        killed_any=true
      fi
      # Remove stale PID file
      rm -f /tmp/textgen.pid
    fi
    
    # Method 2: Kill all Python processes running textgen on the port
    local textgen_pids=$(pgrep -f "python.*server\.py.*--listen-port.*${TEXTGEN_PORT:-7009}" 2>/dev/null || true)
    if [[ -n "$textgen_pids" ]]; then
      log "Found textgen Python processes: $textgen_pids"
      for pid in $textgen_pids; do
        if kill -0 "$pid" 2>/dev/null; then
          log "Killing textgen Python process: $pid"
          # Kill the process and its parent (service_loop)
          local parent_pid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
          kill -TERM "$pid" 2>/dev/null || true
          if [[ -n "$parent_pid" ]] && kill -0 "$parent_pid" 2>/dev/null; then
            kill -TERM "$parent_pid" 2>/dev/null || true
          fi
          killed_any=true
        fi
      done
      sleep 1
      # Force kill any remaining
      for pid in $textgen_pids; do
        if kill -0 "$pid" 2>/dev/null; then
          log "Force killing textgen Python process: $pid"
          kill -9 "$pid" 2>/dev/null || true
        fi
      done
    fi
    
    # Method 2.5: Kill any service_loop processes running textgen commands
    local service_loop_pids=$(pgrep -f "service_loop.*server\.py" 2>/dev/null || true)
    if [[ -n "$service_loop_pids" ]]; then
      log "Found service_loop processes running textgen: $service_loop_pids"
      for pid in $service_loop_pids; do
        if kill -0 "$pid" 2>/dev/null; then
          log "Killing service_loop process: $pid"
          pkill -P "$pid" 2>/dev/null || true
          kill -TERM "$pid" 2>/dev/null || true
          killed_any=true
        fi
      done
      sleep 1
      for pid in $service_loop_pids; do
        if kill -0 "$pid" 2>/dev/null; then
          kill -9 "$pid" 2>/dev/null || true
        fi
      done
    fi
    
    # Method 3: Kill processes using the port (fallback)
    if command -v lsof &>/dev/null; then
      local port_pids=$(lsof -ti:${TEXTGEN_PORT:-7009} 2>/dev/null || true)
      if [[ -n "$port_pids" ]]; then
        log "Found processes using port ${TEXTGEN_PORT:-7009}: $port_pids"
        for pid in $port_pids; do
          if kill -0 "$pid" 2>/dev/null; then
            log "Killing process using port: $pid"
            kill -TERM "$pid" 2>/dev/null || true
            killed_any=true
          fi
        done
        sleep 1
        # Force kill any remaining
        for pid in $port_pids; do
          if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
          fi
        done
      fi
    fi
    
    if [[ "$killed_any" == "true" ]]; then
      log "✅ Existing textgen processes stopped"
      sleep 2  # Give processes time to fully terminate
    else
      log "No existing textgen processes found"
    fi
  }
  
  # Execute the cleanup
  kill_existing_textgen
  
  cd $REPO_DIR
  
  # Rotate textgen log file instead of deleting it
  if [[ -f "$LOG_DIR/textgen.log" ]]; then
    # Create timestamp for old log
    timestamp=$(date +"%Y%m%d_%H%M%S")
    mv "$LOG_DIR/textgen.log" "$LOG_DIR/textgen_${timestamp}.log"
    echo "Previous textgen log archived as: textgen_${timestamp}.log"
    
    # Keep only the last 5 rotated logs to save space
    ls -t "$LOG_DIR"/textgen_*.log 2>/dev/null | tail -n +6 | xargs -r rm
  fi
  
  # Default arguments for text-generation-webui
  # --listen: Make server accessible on network (not just localhost)
  # --listen-port: Port to listen on (default: 7009)
  default_args="--listen --listen-port $TEXTGEN_PORT"
  
  log "🚀 Starting textgen with command: python server.py $default_args"
  PYTHONUNBUFFERED=1 service_loop "python server.py $default_args" > $LOG_DIR/textgen.log 2>&1 &
  echo $! > /tmp/textgen.pid
  log "Text generation Webui service started in background (PID: $(cat /tmp/textgen.pid))"
  # Give the service a moment to start before continuing
  sleep 2

  # undo the change for git pull to work
  if env | grep -q "PAPERSPACE"; then
    sed -i "s/server_port=shared.args.listen_port, inbrowser=shared.args.auto_launch, auth=auth, root_path='\\/textgen')/server_port=shared.args.listen_port, inbrowser=shared.args.auto_launch, auth=auth)/g" $REPO_DIR/server.py
  fi
fi


send_to_discord "Text generation Webui Started"

if env | grep -q "PAPERSPACE"; then
  send_to_discord "Link: https://$PAPERSPACE_FQDN/textgen/"
fi


if [[ -n "${CF_TOKEN}" ]]; then
  if [[ "$RUN_SCRIPT" != *"textgen"* ]]; then
    export RUN_SCRIPT="$RUN_SCRIPT,textgen"
  fi
  bash $current_dir/../cloudflare_reload.sh
fi

echo "### Done ###"
