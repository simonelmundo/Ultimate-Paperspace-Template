#!/bin/bash
set -e

current_dir=$(dirname "$(realpath "$0")")
cd $current_dir
source .env

# Upgrade Git to latest version
echo "Upgrading Git to latest version..."
add-apt-repository -y ppa:git-core/ppa
apt-get update
apt-get install -y git

# Ensure LOG_DIR is set and create it if it doesn't exist
LOG_DIR="/tmp/log"
mkdir -p "$LOG_DIR" || { echo "Failed to create log directory: $LOG_DIR"; exit 1; }

# Define log file paths
MAIN_LOG="$LOG_DIR/main_operations.log"
RUN_LOG="$LOG_DIR/run.log"

# Ensure log files are created and writable
touch "$MAIN_LOG" "$RUN_LOG" || { echo "Failed to create log files"; exit 1; }

# Function to add timestamps to log lines
add_timestamp() {
    while IFS= read -r line; do
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        echo "[$timestamp] $line"
    done
}

# Export the function so it's available in subshells
export -f add_timestamp

# Redirect all output to both logs with timestamps
exec > >(add_timestamp | tee -a "$MAIN_LOG" "$RUN_LOG") 2>&1

# Ensure all subprocesses inherit the redirection
export BASH_ENV="$LOG_DIR/bash_env"
cat << 'EOF' > "$BASH_ENV"
add_timestamp() {
    while IFS= read -r line; do
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        echo "[$timestamp] $line"
    done
}
exec > >(add_timestamp | tee -a "$MAIN_LOG" "$RUN_LOG") 2>&1
EOF

# Function to log errors (already includes timestamp)
log_error() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] ERROR: $1" | tee -a "$MAIN_LOG" "$RUN_LOG" >&2
}

# Trap errors
trap 'log_error "Script exited with error"; exit 1' ERR

# Now all output will be logged to both files
echo "Starting main.sh operations at $(date)"

setup_environment() {
    # Check system CUDA version
    local cuda_version=$(nvcc --version | grep 'release' | awk '{print $6}' || echo "unknown")
    echo "System CUDA Version: $cuda_version"
    
    # Add version-specific installation logic
    if [[ "$cuda_version" != "11.6" ]]; then
        echo "Warning: Mismatched CUDA version - Expected 11.6, found $cuda_version"
    fi
    
    # Keep CUDA-related vars for other operations
    export CUDA_HOME=/usr/local/cuda-11.6
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export FORCE_CUDA=1
    export CUDA_VISIBLE_DEVICES=0
    export PYOPENGL_PLATFORM="osmesa"  
    export WINDOW_BACKEND="headless"
}

# Function to fix CUDA and PyTorch versions
fix_torch_versions() {
    echo "Checking and fixing PyTorch/CUDA versions..."
    
    # Clean up existing installations
    echo "Removing existing PyTorch installations..."
    pip uninstall -y torch torchvision torchaudio xformers || true
    pip cache purge || true
    
    # Install specific versions for CUDA 11.6/12.x compatibility
    echo "Installing PyTorch ecosystem with CUDA 11.6/12.x compatibility..."
    local torch_index="https://download.pytorch.org/whl/cu121"
    
    # Install PyTorch core packages first
    pip install \
        torch==2.4.1+cu121 \
        torchvision==0.19.1+cu121 \
        torchaudio==2.4.1+cu121 \
        --extra-index-url "$torch_index" || {
            echo "Warning: PyTorch core packages installation had issues, but continuing..."
        }
    
    # Install xformers with CUDA 12.1 compatibility
    echo "Installing xformers..."
    pip install xformers==0.0.25.post1 --no-deps || {
        echo "Warning: xformers installation had issues, but continuing..."
    }
    
    # Verify installations with comprehensive checks
    echo "Verifying installations..."
    python3 -c "
try:
    import torch
    import torchvision
    import torchaudio
    
    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda if cuda_available else 'N/A'
    
    print(f'PyTorch: {torch.__version__}')
    print(f'TorchVision: {torchvision.__version__}') 
    print(f'TorchAudio: {torchaudio.__version__}')
    print(f'CUDA Available: {cuda_available}')
    print(f'CUDA Version: {cuda_version}')
    
    if not cuda_available:
        print('Warning: CUDA not available after installation')
        
    if torch.__version__ != '2.4.1+cu121':
        print('Warning: Unexpected PyTorch version installed')
        
except ImportError as e:
    print(f'Warning: Missing package - {str(e)}')
except Exception as e:
    print(f'Warning: Verification had issues: {str(e)}')
" || echo "Warning: Verification script had issues, but continuing..."
    
    echo "PyTorch ecosystem installation completed"
    return 0
}

echo "### Setting up Stable Diffusion Comfy ###"
log "Setting up Stable Diffusion Comfy"
if [[ "$REINSTALL_SD_COMFY" || ! -f "/tmp/sd_comfy.prepared" ]]; then
    # Verify NVIDIA and setup environment first
    setup_environment

    TARGET_REPO_URL="https://github.com/comfyanonymous/ComfyUI.git" \
    TARGET_REPO_DIR=$REPO_DIR \
    UPDATE_REPO=$SD_COMFY_UPDATE_REPO \
    UPDATE_REPO_COMMIT=$SD_COMFY_UPDATE_REPO_COMMIT \

    # Ensure requirements.txt is not blocking the update
    cd $REPO_DIR
    if [[ -n "$(git status --porcelain requirements.txt)" ]]; then
        echo "Local changes detected in requirements.txt. Discarding changes..."
        git checkout -- requirements.txt
    fi

    prepare_repo 

    symlinks=(
      "$REPO_DIR/output:$IMAGE_OUTPUTS_DIR/stable-diffusion-comfy"
      "$MODEL_DIR:$WORKING_DIR/models"
      "$MODEL_DIR/sd:$LINK_MODEL_TO"
      "$MODEL_DIR/lora:$LINK_LORA_TO"
      "$MODEL_DIR/vae:$LINK_VAE_TO"
      "$MODEL_DIR/upscaler:$LINK_UPSCALER_TO"
      "$MODEL_DIR/controlnet:$LINK_CONTROLNET_TO"
      "$MODEL_DIR/embedding:$LINK_EMBEDDING_TO"
      "$MODEL_DIR/llm_checkpoints:$LINK_LLM_TO"
        )
    prepare_link "${symlinks[@]}"
    rm -rf $VENV_DIR/sd_comfy-env
    python3.10 -m venv $VENV_DIR/sd_comfy-env
    source $VENV_DIR/sd_comfy-env/bin/activate

    # Install system dependencies
    echo "Installing system dependencies..."
    apt-get update && apt-get install -y \
        libatlas-base-dev \
        libblas-dev \
        liblapack-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libbz2-dev \
        python2-dev \
        libopenblas-dev \
        cmake \
        build-essential \
        libx11-dev \
        libxrandr-dev \
        libxinerama-dev \
        libxcursor-dev \
        libxi-dev \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        libglew-dev \
        libglfw3-dev \
        mesa-common-dev || {
        echo "Warning: Some packages failed to install"
    }

    # Update pip and dependencies
    pip install pip==24.0
    pip install --upgrade wheel setuptools

    # Install numpy with version constraint
    pip install "numpy>=1.26.0,<2.3.0"

    fix_torch_versions

    # Install and setup DepthFlow
    echo "Setting up DepthFlow..."
    export DEPTHFLOW_SUPPRESS_ROOT_WARNING=1

    # Clean and install DepthFlow
    cd /storage/stable-diffusion-comfy/custom_nodes
    rm -rf DepthFlow  # Remove existing installation

    # Ensure wheel cache directory exists
    WHEEL_CACHE_DIR="/storage/wheel_cache"
    mkdir -p "$WHEEL_CACHE_DIR"

    # Install DepthFlow with wheel caching
    if FORCE_CUDA=1 pip install --cache-dir="$WHEEL_CACHE_DIR" "git+https://github.com/BrokenSource/DepthFlow.git@v0.8.0"; then
        # Create symlink and verify
        if python3 -c "import DepthFlow; print(f'DepthFlow {DepthFlow.__version__} installed')" 2>/dev/null; then
            ln -sf "/tmp/sd_comfy-env/lib/python3.10/site-packages/DepthFlow" DepthFlow
        else
            echo "DepthFlow installation verification failed, but continuing..."
        fi
    else
        echo "DepthFlow installation failed, but continuing..."
    fi

    # Function to safely process requirements file with persistent storage
    process_requirements() {
        local req_file="$1"
        local indent="${2:-}"  # Indentation for nested requirements
        
        # Clean the path
        req_file="$(echo "$req_file" | tr -d ' ')"
        
        # Check if file exists
        if [ ! -f "$req_file" ]; then
            echo "${indent}Skipping: File not found - $req_file"
            return 0
        fi
        
        # Create a persistent pip cache directory in storage
        export PIP_CACHE_DIR="$ROOT_REPO_DIR/.pip_cache"
        mkdir -p "$PIP_CACHE_DIR"
        
        echo "${indent}Processing: $req_file"
        while IFS= read -r requirement || [ -n "$requirement" ]; do
            # Skip empty lines and comments
            if [[ -z "$requirement" || "$requirement" =~ ^[[:space:]]*# ]]; then
                continue
            fi
            
            # Clean the requirement string
            requirement="$(echo "$requirement" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
            
            # Skip local directory references
            if [[ "$requirement" =~ ^file:///storage/stable-diffusion-comfy ]]; then
                echo "${indent}  Skipping local reference: $requirement"
                continue
            fi
            
            # If requirement is a -r reference, process that file
            if [[ "$requirement" =~ ^-r ]]; then
                local included_file="${requirement#-r}"
                included_file="$(echo "$included_file" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
                process_requirements "$included_file" "${indent}  "
            else
                echo "${indent}  Installing: $requirement"
                if ! pip install --cache-dir="$PIP_CACHE_DIR" "$requirement"; then
                    echo "${indent}  Warning: Failed to install $requirement"
                fi
            fi
        done < "$req_file"
    }

    # Install TensorFlow with persistent cache
    export PIP_CACHE_DIR="$ROOT_REPO_DIR/.pip_cache"
    mkdir -p "$PIP_CACHE_DIR"
    pip install --cache-dir="$PIP_CACHE_DIR" "tensorflow>=2.8.0,<2.19.0"

    # Process main requirements file
    process_requirements "/notebooks/sd_comfy/additional_requirements.txt"
    fix_torch_versions
    touch /tmp/sd_comfy.prepared
else
    # Just ensure PyTorch versions are correct
    fix_torch_versions
    setup_environment
    source $VENV_DIR/sd_comfy-env/bin/activate
    
fi

log "Finished Preparing Environment for Stable Diffusion Comfy"

if [[ -z "$SKIP_MODEL_DOWNLOAD" ]]; then
  echo "### Downloading Model for Stable Diffusion Comfy ###"
  log "Downloading Model for Stable Diffusion Comfy"
  bash $current_dir/../utils/sd_model_download/main.sh
  log "Finished Downloading Models for Stable Diffusion Comfy"
else
  log "Skipping Model Download for Stable Diffusion Comfy"
fi


if [[ -z "$INSTALL_ONLY" ]]; then
  echo "### Starting Stable Diffusion Comfy ###"
  log "Starting Stable Diffusion Comfy"
  cd "$REPO_DIR"
  
  # Delete the previous log file if it exists
  if [[ -f "$LOG_DIR/sd_comfy.log" ]]; then
    rm "$LOG_DIR/sd_comfy.log"
  fi
  
  PYTHONUNBUFFERED=1 service_loop "python main.py --dont-print-server --highvram --port $SD_COMFY_PORT ${EXTRA_SD_COMFY_ARGS}" > $LOG_DIR/sd_comfy.log 2>&1 &
  echo $! > /tmp/sd_comfy.pid
fi

send_to_discord "Stable Diffusion Comfy Started"

if env | grep -q "PAPERSPACE"; then
  send_to_discord "Link: https://$PAPERSPACE_FQDN/sd-comfy/"
fi


if [[ -n "${CF_TOKEN}" ]]; then
  if [[ "$RUN_SCRIPT" != *"sd_comfy"* ]]; then
    export RUN_SCRIPT="$RUN_SCRIPT,sd_comfy"
  fi
  bash $current_dir/../cloudflare_reload.sh
fi
