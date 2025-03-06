#!/bin/bash
set -e

#######################################
# STEP 1: INITIAL SETUP AND LOGGING
#######################################
# Initialize script environment
current_dir=$(dirname "$(realpath "$0")")
cd "$current_dir" || { echo "Failed to change directory"; exit 1; }
source .env || { echo "Failed to source .env"; exit 1; }

# Configure logging system
LOG_DIR="/tmp/log"
MAIN_LOG="$LOG_DIR/main_operations.log"
RUN_LOG="$LOG_DIR/run.log"

# Setup logging infrastructure
setup_logging() {
    mkdir -p "$LOG_DIR" || { echo "Failed to create log directory: $LOG_DIR"; exit 1; }
    touch "$MAIN_LOG" "$RUN_LOG" || { echo "Failed to create log files"; exit 1; }
    
    # Timestamp function for consistent logging
    add_timestamp() {
        while IFS= read -r line; do
            printf "[%(%Y-%m-%d %H:%M:%S)T] %s\n" -1 "$line"
        done
    }
    export -f add_timestamp

    # Redirect all output to logs with timestamps
    exec > >(add_timestamp | tee -a "$MAIN_LOG" "$RUN_LOG") 2>&1

    # Ensure subprocesses inherit logging configuration
    export BASH_ENV="$LOG_DIR/bash_env"
    cat << 'EOF' > "$BASH_ENV"
add_timestamp() {
    while IFS= read -r line; do
        printf "[%(%Y-%m-%d %H:%M:%S)T] %s\n" -1 "$line"
    done
}
exec > >(add_timestamp | tee -a "$MAIN_LOG" "$RUN_LOG") 2>&1
EOF
}

# Error handling and logging
log_error() {
    printf "[%(%Y-%m-%d %H:%M:%S)T] ERROR: %s\n" -1 "$1" | tee -a "$MAIN_LOG" "$RUN_LOG" >&2
}
trap 'log_error "Script exited with error"; exit 1' ERR

# Initialize logging system
setup_logging
echo "Starting main.sh operations at $(date)"
#######################################
# STEP 2: CUDA AND ENVIRONMENT SETUP
#######################################

# Common environment variables for CUDA
setup_cuda_env() {
    export CUDA_HOME=/usr/local/cuda-12.1
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export FORCE_CUDA=1
    export CUDA_VISIBLE_DEVICES=0
    export PYOPENGL_PLATFORM="osmesa"
    export WINDOW_BACKEND="headless"
}

install_cuda_12() {
    echo "Installing CUDA 12.1..."
    
    # Clean up existing CUDA 11.x if present
    if dpkg -l | grep -q "cuda-11"; then
        echo "Removing existing CUDA 11.x installations..."
        apt-get --purge remove -y cuda-11-* || echo "No CUDA 11.x found to remove"
    fi
    
    # Install CUDA 12.1
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt-get update -qq
    apt-get install -y cuda-toolkit-12-1
    rm -f cuda-keyring_1.1-1_all.deb
    
    # Configure environment
    setup_cuda_env
    
    # Make environment persistent
    cat > /etc/profile.d/cuda12.sh << 'EOL'
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export FORCE_CUDA=1
EOL
    chmod +x /etc/profile.d/cuda12.sh
    
    # Verify installation
    echo "Verifying CUDA 12.1 installation..."
    nvcc --version || { echo "CUDA installation verification failed"; return 1; }
}

setup_environment() {
    # Check current CUDA version
    local cuda_version=$(nvcc --version 2>/dev/null | grep 'release' | awk '{print $6}' || echo "unknown")
    echo "System CUDA Version: $cuda_version"
    
    # Install CUDA 12.1 if needed
    if [[ "$cuda_version" != "12.1" ]]; then
        install_cuda_12
    else
        echo "CUDA 12.1 already installed."
        setup_cuda_env
    fi
}

#######################################
# STEP 3: PYTORCH VERSION MANAGEMENT
#######################################
# Define package versions and URLs as constants
readonly TORCH_VERSION="2.4.1+cu121"
readonly TORCHVISION_VERSION="0.19.1+cu121"
readonly TORCHAUDIO_VERSION="2.4.1+cu121"
readonly XFORMERS_VERSION="0.0.28.post1"
readonly TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"

# Function to clean up existing installations
clean_torch_installations() {
    echo "Removing existing PyTorch installations..."
    pip uninstall -y torch torchvision torchaudio xformers || true
    pip cache purge || true
}

# Function to install PyTorch core packages
install_torch_core() {
    echo "Installing PyTorch core packages..."
    pip install \
        torch==${TORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        torchaudio==${TORCHAUDIO_VERSION} \
        --extra-index-url "${TORCH_INDEX_URL}" || {
            echo "Warning: PyTorch core packages installation had issues, but continuing..."
        }
}

# Function to install xformers
install_xformers() {
    echo "Installing xformers..."
    pip install xformers==${XFORMERS_VERSION} || {
        echo "Warning: xformers installation had issues, but continuing..."
    }
}

# Function to verify installations
verify_installations() {
    echo "Verifying installations..."
    python3 -c "
import torch
import torchvision
import torchaudio
import xformers

def print_version(package, version):
    print(f'{package.__name__.capitalize()}: {version}')

try:
    print_version(torch, torch.__version__)
    print_version(torchvision, torchvision.__version__)
    print_version(torchaudio, torchaudio.__version__)
    print_version(xformers, xformers.__version__)
    
    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda if cuda_available else 'N/A'
    
    print(f'CUDA Available: {cuda_available}')
    print(f'CUDA Version: {cuda_version}')
    
    if not cuda_available:
        print('Warning: CUDA not available after installation')
        
    if torch.__version__ != '${TORCH_VERSION}':
        print('Warning: Unexpected PyTorch version installed')
        
except ImportError as e:
    print(f'Warning: Missing package - {str(e)}')
except Exception as e:
    print(f'Warning: Verification script had issues: {str(e)}')
" || echo "Warning: Verification script had issues, but continuing..."
}

# Main function to fix torch versions
fix_torch_versions() {
    echo "Checking and fixing PyTorch/CUDA versions..."
    
    clean_torch_installations
    install_torch_core
    install_xformers
    verify_installations
    
    echo "PyTorch ecosystem installation completed"
    return 0
}

echo "### Setting up Stable Diffusion Comfy ###"
log "Setting up Stable Diffusion Comfy"
#######################################
# STEP 4: STABLE DIFFUSION SETUP
#######################################
if [[ "$REINSTALL_SD_COMFY" || ! -f "/tmp/sd_comfy.prepared" ]]; then
    # Initialize environment
    export PIP_QUIET=1
    setup_environment

    # Repository configuration
    export TARGET_REPO_URL="https://github.com/comfyanonymous/ComfyUI.git" \
           TARGET_REPO_DIR=$REPO_DIR \
           UPDATE_REPO=$SD_COMFY_UPDATE_REPO \
           UPDATE_REPO_COMMIT=$SD_COMFY_UPDATE_REPO_COMMIT

    # Prepare repository
    cd $REPO_DIR
    [[ -n "$(git status --porcelain requirements.txt)" ]] && {
        echo "Local changes detected in requirements.txt. Discarding changes..."
        git checkout -- requirements.txt
    }
    prepare_repo

    # Create directory symlinks
    prepare_link "$REPO_DIR/output:$IMAGE_OUTPUTS_DIR/stable-diffusion-comfy" \
                 "$MODEL_DIR:$WORKING_DIR/models" \
                 "$MODEL_DIR/sd:$LINK_MODEL_TO" \
                 "$MODEL_DIR/lora:$LINK_LORA_TO" \
                 "$MODEL_DIR/vae:$LINK_VAE_TO" \
                 "$MODEL_DIR/upscaler:$LINK_UPSCALER_TO" \
                 "$MODEL_DIR/controlnet:$LINK_CONTROLNET_TO" \
                 "$MODEL_DIR/embedding:$LINK_EMBEDDING_TO" \
                 "$MODEL_DIR/llm_checkpoints:$LINK_LLM_TO"

    # Virtual environment setup
    rm -rf $VENV_DIR/sd_comfy-env
    python3.10 -m venv $VENV_DIR/sd_comfy-env
    source $VENV_DIR/sd_comfy-env/bin/activate

    # System dependencies
    echo "Installing system dependencies..."
    apt-get update && apt-get install -y \
        libatlas-base-dev libblas-dev liblapack-dev \
        libjpeg-dev libpng-dev libtiff-dev libbz2-dev \
        python2-dev libopenblas-dev cmake build-essential \
        libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev \
        libxi-dev libgl1-mesa-dev libglu1-mesa-dev \
        libglew-dev libglfw3-dev mesa-common-dev || {
        echo "Warning: Some packages failed to install"
    }

    # Python environment setup
    pip install pip==24.0
    pip install --upgrade wheel setuptools
    pip install "numpy>=1.26.0,<2.3.0"
    fix_torch_versions

    # DepthFlow installation
    echo "Setting up DepthFlow..."
    export DEPTHFLOW_SUPPRESS_ROOT_WARNING=1
    WHEEL_CACHE_DIR="/storage/wheel_cache"
    mkdir -p "$WHEEL_CACHE_DIR"
    
    cd /storage/stable-diffusion-comfy/custom_nodes
    rm -rf DepthFlow
    if FORCE_CUDA=1 pip install --cache-dir="$WHEEL_CACHE_DIR" "git+https://github.com/BrokenSource/DepthFlow.git@v0.8.0"; then
        python3 -c "import DepthFlow; print(f'DepthFlow {DepthFlow.__version__} installed')" 2>/dev/null && \
            ln -sf "/tmp/sd_comfy-env/lib/python3.10/site-packages/DepthFlow" DepthFlow || \
            echo "DepthFlow installation verification failed, but continuing..."
    else
        echo "DepthFlow installation failed, but continuing..."
    fi

    # Process requirements files
    process_requirements() {
        local req_file="$1"
        local indent="${2:-}"
        
        req_file="$(echo "$req_file" | tr -d ' ')"
        [[ ! -f "$req_file" ]] && {
            echo "${indent}Skipping: File not found - $req_file"
            return 0
        }

        export PIP_CACHE_DIR="$ROOT_REPO_DIR/.pip_cache"
        mkdir -p "$PIP_CACHE_DIR"
        
        echo "${indent}Processing: $req_file"
        while IFS= read -r requirement || [ -n "$requirement" ]; do
            [[ -z "$requirement" || "$requirement" =~ ^[[:space:]]*# ]] && continue
            requirement="$(echo "$requirement" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
            
            if [[ "$requirement" =~ ^-r ]]; then
                local included_file="${requirement#-r}"
                included_file="$(echo "$included_file" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
                process_requirements "$included_file" "${indent}  "
            elif [[ ! "$requirement" =~ ^file:///storage/stable-diffusion-comfy ]]; then
                echo "${indent}  Installing: $requirement"
                pip install --quiet --cache-dir="$PIP_CACHE_DIR" "$requirement" || \
                    echo "${indent}  Warning: Failed to install $requirement"
            fi
        done < "$req_file"
    }
    process_requirements "/notebooks/sd_comfy/additional_requirements.txt"

    # TensorFlow installation
    pip install --cache-dir="$PIP_CACHE_DIR" "tensorflow>=2.8.0,<2.19.0"

    # SageAttention installation
    echo "Installing SageAttention for wanVideo support..."
    export CUDA_HOME=/usr/local/cuda-12.1
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export FORCE_CUDA=1
    export TORCH_CUDA_ARCH_LIST="8.6"
    export MAX_JOBS=$(nproc)
    export USE_NINJA=1

    # Create persistent cache directories
    export TORCH_EXTENSIONS_DIR="/storage/.torch_extensions"
    export SAGEATTENTION_CACHE_DIR="/storage/.sageattention_cache"
    mkdir -p "$TORCH_EXTENSIONS_DIR" "$SAGEATTENTION_CACHE_DIR"

    # Enable ccache with larger cache size
    command -v ccache &> /dev/null && {
        export CMAKE_C_COMPILER_LAUNCHER=ccache
        export CMAKE_CXX_COMPILER_LAUNCHER=ccache
        ccache --max-size=3G
        ccache -z  # Clear stats
    }

    # Check if SageAttention is already installed and cached
    SAGE_VERSION="2.1.1"  # Update this to match the version you need
    SAGE_CACHE_MARKER="$SAGEATTENTION_CACHE_DIR/sage_${SAGE_VERSION}_$(uname -m)_cuda$(nvcc --version | grep release | awk '{print $6}' | cut -c2-).installed"

    if [ -f "$SAGE_CACHE_MARKER" ]; then
        echo "Using cached SageAttention build..."
        # Just link the existing installation
        mkdir -p "/storage/stable-diffusion-comfy/custom_nodes/sage_attention"
        ln -sf "$VENV_DIR/sd_comfy-env/lib/python3.10/site-packages/sage_attention" \
            "/storage/stable-diffusion-comfy/custom_nodes/sage_attention"
    else
        # Install dependencies first
        pip install --cache-dir="$PIP_CACHE_DIR" \
            "ninja>=1.11.0" \
            "triton>=3.0.0" \
            "accelerate>=1.1.1" \
            "diffusers>=0.31.0" \
            "transformers>=4.39.3"

        cd "$ROOT_REPO_DIR"
        [ -d "SageAttention" ] || git clone https://github.com/thu-ml/SageAttention.git
        cd SageAttention
        
        # Optimize build with parallel compilation and caching
        python setup.py build_ext --inplace -j$(nproc)
        python setup.py install
        
        # Create cache marker
        touch "$SAGE_CACHE_MARKER"
        
        cd "$ROOT_REPO_DIR"
        
        mkdir -p "/storage/stable-diffusion-comfy/custom_nodes/sage_attention"
        ln -sf "$VENV_DIR/sd_comfy-env/lib/python3.10/site-packages/sage_attention" \
            "/storage/stable-diffusion-comfy/custom_nodes/sage_attention"
        
        echo "Successfully installed SageAttention"
        
        # Show ccache stats
        command -v ccache &> /dev/null && ccache -s
    fi

    fix_torch_versions
    touch /tmp/sd_comfy.prepared
else
    # Just ensure PyTorch versions are correct
    fix_torch_versions
    setup_environment
    source $VENV_DIR/sd_comfy-env/bin/activate
fi

log "Finished Preparing Environment for Stable Diffusion Comfy"

#######################################
# STEP 5: MODEL DOWNLOAD
#######################################
if [[ -z "$SKIP_MODEL_DOWNLOAD" ]]; then
  echo "### Downloading Model for Stable Diffusion Comfy ###"
  log "Downloading Model for Stable Diffusion Comfy"
  bash $current_dir/../utils/sd_model_download/main.sh
  log "Finished Downloading Models for Stable Diffusion Comfy"
else
  log "Skipping Model Download for Stable Diffusion Comfy"
fi

#######################################
# STEP 6: START STABLE DIFFUSION
#######################################
if [[ -z "$INSTALL_ONLY" ]]; then
  echo "### Starting Stable Diffusion Comfy ###"
  log "Starting Stable Diffusion Comfy"
  cd "$REPO_DIR"
  
  if [[ -f "$LOG_DIR/sd_comfy.log" ]]; then
    rm "$LOG_DIR/sd_comfy.log"
  fi
  
  PYTHONUNBUFFERED=1 service_loop "python main.py --dont-print-server --highvram --port $SD_COMFY_PORT ${EXTRA_SD_COMFY_ARGS}" > $LOG_DIR/sd_comfy.log 2>&1 &
  echo $! > /tmp/sd_comfy.pid
fi

#######################################
# STEP 7: FINAL NOTIFICATIONS
#######################################
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
