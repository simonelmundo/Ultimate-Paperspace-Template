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

# Set textgen-specific variables (no need for local .env - parent .env + defaults are enough)
export TEXTGEN_PORT="${TEXTGEN_PORT:-7009}"
export GRADIO_ROOT_PATH="/textgen"
export MODEL_DIR="${TEXTGEN_MODEL_DIR:-$DATA_DIR/llm-models}"
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
    log "📦 Installing text-generation-webui specific requirements..."
    pip install -r "$REQ_FILE" || log "⚠️  Some packages failed to install, continuing..."

    mkdir -p repositories
    TARGET_REPO_DIR=$REPO_DIR/repositories/GPTQ-for-LLaMa \
    TARGET_REPO_BRANCH="cuda" \
    TARGET_REPO_URL="https://github.com/qwopqwop200/GPTQ-for-LLaMa.git" \
    prepare_repo

    # prepare_repo changes to TARGET_REPO_DIR, so we're already in the right directory
    # But to be safe, explicitly cd to the full path
    cd $REPO_DIR/repositories/GPTQ-for-LLaMa
    python setup_cuda.py install

    pip uninstall -y llama-cpp-python
    CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir

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


if [[ -z "$SKIP_MODEL_DOWNLOAD" ]]; then
  echo "### Downloading Model for Text generation Webui ###"
  log "Downloading Model for Text generation Webui"
  # Prepare model dir and link it under the models folder inside the repo
  mkdir -p $MODEL_DIR
  rm -rf $LINK_MODEL_TO
  ln -s $MODEL_DIR $LINK_MODEL_TO
  if [[ ! -f $MODEL_DIR/config.yaml ]]; then 
      current_dir_save=$(pwd) 
      cd $REPO_DIR
      commit=$(git rev-parse HEAD)
      wget -q https://raw.githubusercontent.com/oobabooga/text-generation-webui/$commit/models/config.yaml -P $MODEL_DIR
      cd $current_dir_save
  fi


  llm_model_download
  log "Finished Downloading Models for Text generation Webui"
else
  log "Skipping Model Download for Text generation Webui"
fi

if env | grep -q "PAPERSPACE"; then
  sed -i "s/server_port=shared.args.listen_port, inbrowser=shared.args.auto_launch, auth=auth)/server_port=shared.args.listen_port, inbrowser=shared.args.auto_launch, auth=auth, root_path='\\/textgen')/g" $REPO_DIR/server.py
fi


if [[ -z "$INSTALL_ONLY" ]]; then
  echo "### Starting Text generation Webui ###"
  log "Starting Text generation Webui"
  cd $REPO_DIR
  share_args="--chat --listen-port $TEXTGEN_PORT --xformers ${EXTRA_TEXTGEN_ARGS}"
  if [ -v TEXTGEN_ENABLE_OPENAI_API ] && [ ! -z "$TEXTGEN_ENABLE_OPENAI_API" ];then
    loader_arg=""
    if echo "$TEXTGEN_OPENAI_MODEL" | grep -q "GPTQ"; then
      loader_arg="--loader exllama"
    fi
    if echo "$TEXTGEN_OPENAI_MODEL" | grep -q "LongChat"; then
      loader_arg+=" --max_seq_len 8192 --compress_pos_emb 4"
    fi
    PYTHONUNBUFFERED=1 OPENEDAI_PORT=7013 service_loop "python server.py --model $TEXTGEN_OPENAI_MODEL $loader_arg --extensions openai $share_args" > $LOG_DIR/textgen.log 2>&1 &
  else
    PYTHONUNBUFFERED=1 service_loop "python server.py  $share_args" > $LOG_DIR/textgen.log 2>&1 &
  fi
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
