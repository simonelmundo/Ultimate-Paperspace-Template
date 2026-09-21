#!/bin/bash
set -e

echo ""
echo "=================================================="
echo "        STABLE DIFFUSION COMFY SETUP SCRIPT"
echo "=================================================="
echo ""

#######################################
# STEP 1: INITIAL SETUP AND LOGGING
#######################################
# Initialize script environment
echo "Initializing script environment..."
echo "Script path (\$0): $0"
current_dir=$(dirname "$(realpath "$0")")
echo "Resolved script directory: $current_dir"

cd "$current_dir" || { echo "Failed to change directory to '$current_dir'"; exit 1; }
echo "Successfully changed working directory to: $(pwd)"

if [ ! -f ".env" ]; then
    echo "ERROR: '.env' file not found in script directory ($(pwd))."
    echo "Please ensure the .env file is located alongside the main.sh script."
    exit 1
fi
source .env || { echo "Failed to source .env"; exit 1; }

# Source helper functions (for prepare_link, etc.)
if [[ -f "$current_dir/../utils/helper.sh" ]]; then
    source "$current_dir/../utils/helper.sh"
elif [[ -f "/notebooks/utils/helper.sh" ]]; then
    source "/notebooks/utils/helper.sh"
fi

# Venv lives in /tmp (recreated each start).
# Existing caches (.pip_cache, .sageattention_cache, .sam2_cache, .wheel_cache, …) are
# LEFT ALONE — never trimmed/evicted, and they do NOT count toward the budget.
#
# BOOT_CACHE_EXTRA_DIR: small EXTRA area for SAM2 wheels + optional pip crumbs.
# Grandfathered caches (.pip_cache, .wheel_cache, .sageattention_cache, .sam2_cache, …)
# are never trimmed by this script.
VENV_DIR=${VENV_DIR:-/tmp}
export VENV_DIR
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/storage/.pip_cache}
export WHEEL_CACHE_DIR="${WHEEL_CACHE_DIR:-/storage/.wheel_cache}"
export BOOT_CACHE_EXTRA_DIR="${BOOT_CACHE_EXTRA_DIR:-/storage/.comfy_boot_extra}"
export BOOT_CACHE_MAX_GB="${BOOT_CACHE_MAX_GB:-3}"  # soft target for EXTRA only
# Skip slow path by default; set UPDATE_CUSTOM_NODES=1 / INSTALL_LORA_ON_START=1 to opt in.
export SKIP_CUSTOM_NODE_UPDATE="${SKIP_CUSTOM_NODE_UPDATE:-1}"
export INSTALL_LORA_ON_START="${INSTALL_LORA_ON_START:-0}"
mkdir -p "$VENV_DIR" "$PIP_CACHE_DIR" "$WHEEL_CACHE_DIR" "$BOOT_CACHE_EXTRA_DIR"

#######################################
# EXTRA BOOT CACHE (soft ~BOOT_CACHE_MAX_GB target)
#######################################
# Grandfathered (never counted, never deleted by this script):
#   /storage/.pip_cache .wheel_cache .sageattention_cache .sam2_cache .torch_extensions

boot_cache_extra_bytes() {
    if [[ ! -d "$BOOT_CACHE_EXTRA_DIR" ]]; then
        echo 0
        return
    fi
    du -sb "$BOOT_CACHE_EXTRA_DIR" 2>/dev/null | awk '{print $1}'
}

boot_cache_max_bytes() {
    echo $(( BOOT_CACHE_MAX_GB * 1024 * 1024 * 1024 ))
}

# True if sageattention + sam2 wheels are on disk (no nvcc/toolkit needed to rebuild).
boot_extension_wheels_ready() {
    local arch="$(uname -m)"
    local sage_wheel sam_wheel
    sage_wheel=$(find "$WHEEL_CACHE_DIR" /storage/.sageattention_cache -type f -name "sageattention-*-cp310-*-linux_${arch}.whl" 2>/dev/null | head -1)
    sam_wheel=$(find "$BOOT_CACHE_EXTRA_DIR/wheels" "$WHEEL_CACHE_DIR" /storage/.sam2_cache -type f \( -name 'sam_2-*.whl' -o -name 'SAM_2-*.whl' \) 2>/dev/null | head -1)
    [[ -n "$sage_wheel" && -f "$sage_wheel" && -n "$sam_wheel" && -f "$sam_wheel" ]]
}

# Soft budget for EXTRA (SAM2 wheels / pip crumbs). Never touches grandfathered caches.
enforce_boot_cache_budget() {
    local max now mb
    max=$(boot_cache_max_bytes)
    now=$(boot_cache_extra_bytes)
    mb=$(( (now + 1024 * 1024 - 1) / 1024 / 1024 ))
    local msg="📦 Extra boot cache: ${mb}MB (soft target ${BOOT_CACHE_MAX_GB}GB; existing caches excluded)"
    if declare -F log >/dev/null 2>&1; then log "$msg"; else echo "$msg"; fi

    if (( now <= max )); then
        return 0
    fi

    local warn="⚠️ Extra cache ${mb}MB > soft ${BOOT_CACHE_MAX_GB}GB — clearing EXTRA/pip only"
    if declare -F log >/dev/null 2>&1; then log "$warn"; else echo "$warn"; fi
    rm -rf "$BOOT_CACHE_EXTRA_DIR/pip" 2>/dev/null || true
    now=$(boot_cache_extra_bytes)
    mb=$(( now / 1024 / 1024 ))
    msg="   after pip-extra cleanup: ${mb}MB"
    if declare -F log >/dev/null 2>&1; then log "$msg"; else echo "$msg"; fi
}

# Configure logging system
LOG_DIR="/tmp/log"
MAIN_LOG="$LOG_DIR/main_operations.log"
RUN_LOG="$LOG_DIR/run.log"

# Setup logging infrastructure
setup_logging() {
    mkdir -p "$LOG_DIR" || { echo "Failed to create log directory: $LOG_DIR"; exit 1; }
    touch "$MAIN_LOG" "$RUN_LOG" || { echo "Failed to create log files"; exit 1; }
    
    # For now, we will not redirect all output to avoid issues with set -e
    # and process substitution. Functions will explicitly log where needed.
}

# Error handling and logging
log_error() {
    printf "[%(%Y-%m-%d %H:%M:%S)T] ERROR: %s\n" -1 "$1" | tee -a "$MAIN_LOG" "$RUN_LOG" >&2
}
# Use a gentler ERR trap that logs but doesn't exit
trap 'log_error "Command failed, but continuing..."' ERR

# Function to temporarily disable ERR trap
disable_err_trap() {
    trap - ERR
}

# Function to re-enable ERR trap
enable_err_trap() {
    trap 'log_error "Command failed, but continuing..."' ERR
}

# Simple log function that just echoes the message
log() {
    echo "$1"
}

# Apply managed ComfyUI symlinks from sd_comfy/comfy_symlinks.txt (relative_path -> target).
# Safe to re-run; replaces existing path/symlink at each link location.
apply_comfy_symlinks() {
    local manifest="${1:-$current_dir/comfy_symlinks.txt}"
    local repo_root="${REPO_DIR:-/storage/stable-diffusion-comfy}"
    local line rel target dest parent applied=0 skipped=0

    if [[ ! -f "$manifest" ]]; then
        log_error "Symlink manifest not found: $manifest"
        return 1
    fi
    if [[ ! -d "$repo_root" ]]; then
        log_error "ComfyUI repo not found: $repo_root"
        return 1
    fi

    echo "🔗 Applying ComfyUI symlinks from $(basename "$manifest")..."
    while IFS= read -r line || [[ -n "$line" ]]; do
        # skip blanks and comments
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        if [[ ! "$line" =~ ^(.+)[[:space:]]-\>[[:space:]](.+)$ ]]; then
            echo "⚠️  Skipping malformed symlink line: $line"
            ((skipped++)) || true
            continue
        fi
        rel="${BASH_REMATCH[1]}"
        target="${BASH_REMATCH[2]}"
        # trim whitespace
        rel="${rel#"${rel%%[![:space:]]*}"}"
        rel="${rel%"${rel##*[![:space:]]}"}"
        target="${target#"${target%%[![:space:]]*}"}"
        target="${target%"${target##*[![:space:]]}"}"
        dest="$repo_root/$rel"
        parent="$(dirname "$dest")"
        mkdir -p "$parent"
        rm -rf "$dest"
        ln -s "$target" "$dest"
        echo "   $rel -> $target"
        ((applied++)) || true
    done < "$manifest"
    echo "✅ Applied $applied symlinks ($skipped skipped)"
}

# Initialize logging system
setup_logging
echo "Starting main.sh operations at $(date)"

#######################################
# STEP 0: PYTHON 3.10 COMPREHENSIVE SETUP
#######################################
echo ""
echo "=================================================="
echo "        STEP 0: PYTHON 3.10 COMPREHENSIVE SETUP"
echo "=================================================="
echo ""

# Set LD_LIBRARY_PATH for Python shared library (needed when compiled with --enable-shared)
export LD_LIBRARY_PATH="/storage/python_versions/python3.10/lib:${LD_LIBRARY_PATH:-}"

# Update ldconfig to make Python shared library available system-wide (permanent fix)
if [ -f "/storage/python_versions/python3.10/lib/libpython3.10.so.1.0" ]; then
    if [ ! -f "/etc/ld.so.conf.d/python3.10.conf" ]; then
        echo "/storage/python_versions/python3.10/lib" | sudo tee /etc/ld.so.conf.d/python3.10.conf > /dev/null 2>&1
        sudo ldconfig > /dev/null 2>&1 || true
    fi
fi

# Check Python 3.10 is working and set it as default
PYTHON_EXECUTABLE="/storage/python_versions/python3.10/bin/python3.10"
# Check all critical modules required for PyTorch, torchvision, and ComfyUI
if [ -x "$PYTHON_EXECUTABLE" ] && "$PYTHON_EXECUTABLE" -c "import _bz2, _lzma, _ssl, ssl, _sqlite3, sqlite3, _ctypes, _hashlib, _json, _multiprocessing, _pickle, _socket, _struct, _uuid, zlib" 2>/dev/null; then
    log "✅ Python 3.10 is ready with all critical modules"
    
    # Create symlinks to use Python 3.10 as default
    log "🔗 Setting Python 3.10 as default..."
    ln -sf "$PYTHON_EXECUTABLE" /usr/local/bin/python3.10
    ln -sf "$PYTHON_EXECUTABLE" /usr/local/bin/python3
    
    # Update PATH to prioritize our Python 3.10
    export PATH="/storage/python_versions/python3.10/bin:$PATH"
    
    log "✅ Python 3.10 is now the default Python"
    log "📍 Python version: $($PYTHON_EXECUTABLE --version)"
    
    # Verify _lzma specifically (critical for torchvision)
    if "$PYTHON_EXECUTABLE" -c "import _lzma" 2>/dev/null; then
        log "✅ _lzma module verified (required for torchvision)"
    else
        log_error "⚠️  _lzma module missing - torchvision may fail"
        log_error "   Python 3.10 may need to be recompiled with liblzma-dev"
    fi
else
    log_error "❌ Python 3.10 not working or missing critical modules"
    log_error "   Missing modules may include: _bz2, _lzma, _ssl, _sqlite3, _ctypes, etc."
    log_error "   Python 3.10 may need to be recompiled with all required development libraries"
    log_error "   Required packages: liblzma-dev, libbz2-dev, libssl-dev, libsqlite3-dev, libffi-dev, zlib1g-dev"
    exit 1
fi

#######################################
# STEP 1: INITIAL SETUP AND LOGGING
#######################################
echo ""
echo "=================================================="
echo "           STEP 1: INITIAL SETUP AND LOGGING"
echo "=================================================="
echo ""
log "Script initialized successfully"
log "Working directory: $(pwd)"
log "Environment file sourced"

#######################################
# STEP 2: CREATE MODEL SYMLINKS
#######################################
echo ""
echo "=================================================="
echo "           STEP 2: CREATE MODEL SYMLINKS"
echo "=================================================="
echo ""

# Create symlinks for model directories (full list lives in comfy_symlinks.txt).
# Re-applied again after ComfyUI git pull so updates cannot leave placeholders behind.
echo "Creating model directory symlinks from manifest..."
apply_comfy_symlinks "$current_dir/comfy_symlinks.txt" || log_error "⚠️ Model symlink apply had issues (continuing)"

echo "✅ Model symlinks created successfully"

# Create system directory symlinks (output, model directories, etc.)
echo "Creating system directory symlinks..."
prepare_link "$REPO_DIR/output:$IMAGE_OUTPUTS_DIR/stable-diffusion-comfy" \
             "$MODEL_DIR:$WORKING_DIR/models" \
             "$MODEL_DIR/sd:$LINK_MODEL_TO" \
             "$MODEL_DIR/lora:$LINK_LORA_TO" \
             "$MODEL_DIR/vae:$LINK_VAE_TO" \
             "$MODEL_DIR/upscaler:$LINK_UPSCALER_TO" \
             "$MODEL_DIR/controlnet:$LINK_CONTROLNET_TO" \
             "$MODEL_DIR/embedding:$LINK_EMBEDDING_TO" \
             "$MODEL_DIR/llm_checkpoints:$LINK_LLM_TO"

echo "✅ System directory symlinks created successfully"

#######################################
# STEP 3: DOWNLOAD MODELS (BACKGROUND)
#######################################
if [[ -z "$SKIP_MODEL_DOWNLOAD" ]]; then
  echo ""
  echo "=================================================="
  echo "        STEP 3: DOWNLOAD MODELS (BACKGROUND)"
  echo "=================================================="
  echo ""
  echo "### Downloading Models for Stable Diffusion Comfy in Background ###"
  
  # Install dependencies upfront to avoid blocking CUDA installation
  log "📦 Installing model download dependencies (aria2 + Python modules)..."
  
  # Install aria2 (apt-get - must happen before backgrounding to avoid dpkg conflicts)
  if ! dpkg -s aria2 >/dev/null 2>&1; then
    apt-get install -qq aria2 -y > /dev/null 2>&1 || log_error "Failed to install aria2"
  fi
  
  # Install Python modules for model download script (pip - quick)
  # huggingface_hub + hf_transfer: fast/stable HF downloads (aria2c -x16 causes 403s on HF CDN)
  MODULES=("requests" "gdown" "bs4" "python-dotenv")
  for module in "${MODULES[@]}"; do
    if ! pip show $module >/dev/null 2>&1; then
      pip install --quiet --no-cache-dir $module 2>/dev/null || log_error "Failed to install $module"
    fi
  done
  # Force hub>=0.23 even if an older hub is already installed (local_dir / hf_transfer)
  pip install --quiet --no-cache-dir "huggingface_hub>=0.23.0,<1.0" "hf_transfer" 2>/dev/null \
    || log_error "Failed to install huggingface_hub/hf_transfer"
  
  log "✅ Model download dependencies ready"
  log "Starting Model Download for Stable Diffusion Comfy in background..."
  log "💡 Models will download in background while the rest of the setup continues!"
  log "💡 You can start using ComfyUI as soon as it starts, even if models are still downloading!"
  log "💡 Hugging Face: huggingface_hub/hf_transfer (aria2c -x4 fallback); other hosts: aria2c -x16"
  
  # Start model download in background (HF via hub; other hosts via aria2)
  bash $current_dir/../utils/sd_model_download/main.sh > /tmp/model_download.log 2>&1 &
  download_pid=$!
  echo "$download_pid" > /tmp/model_download.pid
  log "📋 Model download started with PID: $download_pid in background"
  log "📋 Check download progress with: tail -f /tmp/model_download.log"
  log "📋 Stop download with: kill \$(cat /tmp/model_download.pid)"
else
  log "Skipping Model Download for Stable Diffusion Comfy"
fi

#######################################
# STEP 4: CUDA AND ENVIRONMENT SETUP
#######################################
echo ""
echo "=================================================="
echo "        STEP 4: CUDA AND ENVIRONMENT SETUP"
echo "=================================================="
echo ""

# Common environment variables for CUDA
setup_cuda_env() {
    export CUDA_HOME=/usr/local/cuda-12.8
    # Prepend CUDA bin and lib paths to ensure they are found first
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export FORCE_CUDA=1
    export CUDA_VISIBLE_DEVICES=0
    export PYOPENGL_PLATFORM="osmesa"
    export WINDOW_BACKEND="headless"
    
    # A4000 optimization: Target Ampere architecture specifically (same as A6000)
    export TORCH_CUDA_ARCH_LIST="8.6"
    
    # Adjust VRAM usage for A4000 (16GB) - More conservative allocation
    export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:4096,garbage_collection_threshold:0.8"
    
    # Aggressive CUDA performance settings (likely still okay)
    export CUDA_LAUNCH_BLOCKING=0
    export CUDA_DEVICE_MAX_CONNECTIONS=32
    export NCCL_P2P_LEVEL=NVL # Relevant if using NVLink
    
    # A4000-specific optimization (CuDNN V8 API should be fine)
    export TORCH_CUDNN_V8_API_ENABLED=1
    
    echo "CUDA Environment Variables Set:"
    echo "  PATH (start): $CUDA_HOME/bin:..."
    echo "  LD_LIBRARY_PATH (start): $CUDA_HOME/lib64:..."
    log "✅ CUDA environment variables configured"
}

# Execute CUDA setup
log "🔧 Setting up CUDA environment..."
setup_cuda_env
log "✅ CUDA environment setup completed"


install_cuda_12() {
    echo "Installing CUDA 12.8 and essential build tools..."
    local APT_INSTALL_LOG="$LOG_DIR/apt_cuda_install.log"
    
    # Clean up any old marker files (markers don't work for /usr/local/ which doesn't persist)
    rm -f /storage/.cuda_12.8_installed /storage/.cuda_12.6_installed /storage/.cuda_12.1_installed
    rm -f /storage/.system_deps_installed  # Also clean up system deps marker (not reliable)

    # Check if CUDA 12.8 is actually installed (verify binary, not marker)
    setup_cuda_env
    hash -r
    if command -v nvcc &>/dev/null && [[ "$(nvcc --version 2>&1 | grep 'release' | awk '{print $6}' | sed 's/^V//')" == "12.8"* ]]; then
        echo "✅ CUDA 12.8 already installed and verified (found at $(which nvcc))."
        return 0
    else
        echo "CUDA 12.8 not found or wrong version. Installing..."
    fi
    
    # Clean up existing CUDA 11.x if present
    if dpkg -l | grep -q "cuda-11"; then
        echo "Removing existing CUDA 11.x installations..."
        apt-get remove --purge -y 'cuda-11-*' 'cuda-repo-ubuntu*-11-*' 'nvidia-cuda-toolkit' || echo "No CUDA 11.x found or removal failed."
        apt-get autoremove -y
    fi

    # Install CUDA repository key
    echo "Adding CUDA repository key..."
    wget -qO /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i /tmp/cuda-keyring.deb
    rm -f /tmp/cuda-keyring.deb

    echo "Running apt-get update..."
    if ! apt-get update >> "$APT_INSTALL_LOG" 2>&1; then
        log_error "apt-get update failed. Check $APT_INSTALL_LOG for details."
        cat "$APT_INSTALL_LOG"
        return 1
    fi

    # List of CUDA packages to install
    local CUDA_PACKAGES=(
        "cuda-cudart-12-8"
        "cuda-cudart-dev-12-8"
        "cuda-nvcc-12-8"
        "cuda-cupti-12-8"
        "cuda-cupti-dev-12-8"
        "libcublas-12-8"
        "libcublas-dev-12-8"
        "libcufft-12-8"
        "libcufft-dev-12-8"
        "libcurand-12-8"
        "libcurand-dev-12-8"
        "libcusolver-12-8"
        "libcusolver-dev-12-8"
        "libcusparse-12-8"
        "libcusparse-dev-12-8"
        "libnpp-12-8"
        "libnpp-dev-12-8"
    )
    
    echo "Installing CUDA packages..."
    apt-get install -y \
        build-essential \
        python3-dev \
        libatlas-base-dev \
        libblas-dev \
        liblapack-dev \
        libjpeg-dev \
        libpng-dev \
        libgl1- \
        "${CUDA_PACKAGES[@]}" >> "$APT_INSTALL_LOG" 2>&1
    
    local apt_exit_code=$?
    
    if [ $apt_exit_code -ne 0 ]; then
        log_error "CUDA installation failed. Exit code: $apt_exit_code"
        cat "$APT_INSTALL_LOG"
        return 1
    fi

    # Configure environment immediately after install
    setup_cuda_env
    hash -r

    # Make environment persistent
    cat > /etc/profile.d/cuda12.sh << 'EOL'
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export FORCE_CUDA=1
EOL
    chmod +x /etc/profile.d/cuda12.sh

    # Verify installation (no marker file needed - CUDA in /usr/local/ doesn't persist on Paperspace)
    echo "Verifying CUDA 12.8 installation..."
    if command -v nvcc &>/dev/null; then
        local installed_version
        installed_version=$(nvcc --version 2>&1 | grep 'release' | awk '{print $6}' | sed 's/^V//')
        if [[ "$installed_version" == "12.8"* ]]; then
            echo "✅ CUDA 12.8 installation verified successfully (Version: $installed_version)."
            echo "   Note: CUDA is installed in /usr/local/ and will need reinstallation after reboot on Paperspace"
            return 0
        else
            log_error "CUDA 12.8 installation verification failed. Found nvcc, but version is '$installed_version'."
            log_error "Which nvcc: $(which nvcc)"
            log_error "PATH: $PATH"
            return 1
        fi
    else
        log_error "CUDA 12.8 installation verification failed. NVCC command not found after installation attempt."
        log_error "Check /usr/local/cuda-12.8/bin exists and contains nvcc."
        ls -l /usr/local/cuda-12.8/bin/nvcc || true
        return 1
    fi
}

setup_environment() {
    echo "Attempting to set up CUDA 12.8 environment..."
    # Set the desired environment variables FIRST
    setup_cuda_env

    # Clear the shell's command hash to ensure PATH changes are recognized
    hash -r
    echo "Command hash cleared."

    # Prefer the real 12.8 binary if it exists even when an older nvcc is earlier on PATH
    if [[ -x /usr/local/cuda-12.8/bin/nvcc ]]; then
        export PATH="/usr/local/cuda-12.8/bin:$PATH"
        hash -r
    fi

    # Fast path: prebuilt sage/sam2 wheels mean we do not need the CUDA toolkit this boot.
    # Torch runtime CUDA comes from the pip wheel. Saves ~3–4 min on Paperspace cold boots.
    if [[ "${FORCE_CUDA_INSTALL:-0}" != "1" ]] && boot_extension_wheels_ready; then
        log "✅ Prebuilt extension wheels present — skipping CUDA 12.8 toolkit install (set FORCE_CUDA_INSTALL=1 to override)"
        if command -v nvcc &>/dev/null; then
            echo "NVCC present: $(nvcc --version 2>&1 | grep release || true)"
        else
            echo "NVCC not on PATH (OK for inference when wheels are cached)"
        fi
        return 0
    fi

    # Now check if nvcc is available in the configured PATH
    if command -v nvcc &>/dev/null; then
        # If nvcc is found, check its version
        local cuda_version
        # Pipe stderr to stdout for grep, handle potential errors finding version string
        cuda_version=$(nvcc --version 2>&1 | grep 'release' | awk '{print $6}' || echo "unknown")
        # Remove potential leading 'V' if present
        cuda_version=${cuda_version#V}

        echo "Detected CUDA Version (after setting env and clearing hash): $cuda_version"

        # Verify if the detected version is the target 12.8
        if [[ "$cuda_version" == "12.8"* ]]; then
            echo "CUDA 12.8 environment appears correctly configured."
            log "✅ CUDA 12.8 already configured correctly"
            # Environment is already set by setup_cuda_env above
        else
            echo "Found nvcc, but version is '$cuda_version', not 12.8. Attempting installation/reconfiguration..."
            log "⚠️ CUDA version mismatch: $cuda_version (expected 12.8)"
            log "🔧 Installing CUDA 12.8..."
            install_cuda_12
            # Re-clear hash after potential installation changes PATH again
            hash -r
        fi
    else
        # If nvcc is NOT found even after setting the PATH and clearing hash
        echo "NVCC not found after setting environment variables and clearing hash. Installing CUDA 12.8..."
        log "⚠️ NVCC not found, installing CUDA 12.8..."
        install_cuda_12
        # Re-clear hash after potential installation changes PATH again
        hash -r
    fi
}

# Define package versions and URLs as constants (updated to latest stable)
readonly TORCH_VERSION="2.8.0+cu128"
readonly TORCHVISION_VERSION="0.23.0+cu128" 
readonly TORCHAUDIO_VERSION="2.8.0+cu128"
readonly XFORMERS_VERSION="0.0.32.post2"
readonly TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
# Exact diffusers: newer releases remove is_k_diffusion_available and break See-through / fluxtrainer imports.
# ComfyUI-Manager often runs pip after boot — keep this pin and re-run repair if imports break again.
readonly DIFFUSERS_PIN="==0.37.1"
# transformers 4.56+ breaks WAS Node Suite (Blip / apply_chunking_to_forward); stay on 4.54.x.
readonly TRANSFORMERS_PIN=">=4.54.0,<4.55.0"
# diffusers>=0.37 may pull huggingface-hub 1.x; Comfy/transformers require hub <1.0 — re-pin after diffusers.
readonly HF_HUB_TRANSFORMERS_PIN=">=0.34.0,<1.0"
# OpenCV 4.13+ requires numpy>=2 (breaks accelerate/scipy/mediapipe in this venv); stay on 4.11 for cv2.ximgproc + numpy 1.x.
readonly OPENCV_CONTRIB_PIN="==4.11.0.86"
readonly NUMPY_COMFY_PIN=">=1.26.4,<2.0.0"
readonly ACCELERATE_PIN=">=1.9.0"
# PyTorch 2.8 ships Triton 3.4 without triton.ops; older bitsandbytes breaks diffusers quantizer imports.
readonly BITSANDBYTES_TRITON3_PIN=">=0.45.1"
# setuptools 82+ removed pkg_resources; SUPIR / pytorch_lightning still import it.
readonly SETUPTOOLS_PIN=">=69,<82"

# Function to install critical packages that are commonly needed by custom nodes
install_critical_packages() {
    log "📦 Installing critical packages for custom nodes..."
    
    local critical_packages=(
        "blend_modes" "deepdiff" "rembg" "webcolors" "ultralytics" "inflect" "soxr" "groundingdino-py" 
        "insightface" "opencv-contrib-python${OPENCV_CONTRIB_PIN}" "facexlib" "onnxruntime" "timm" 
        "segment-anything" "scikit-image" "piexif" "transformers${TRANSFORMERS_PIN}" "scikit-learn"
        "scipy>=1.11.4" "numpy${NUMPY_COMFY_PIN}" "dill" "matplotlib" "oss2" "gguf" "diffusers${DIFFUSERS_PIN}" 
        "huggingface_hub${HF_HUB_TRANSFORMERS_PIN}" "pytorch_lightning" "sounddevice" "av>=12.0.0,<14.0.0" "accelerate${ACCELERATE_PIN}" "pyOpenSSL"
        "setuptools${SETUPTOOLS_PIN}" "comfy-env" "bitsandbytes${BITSANDBYTES_TRITON3_PIN}"
        "decord" "pandas"
    )
    
    # Create Python script to check all packages at once (much faster)
    cat > /tmp/check_packages.py << 'CHECKEOF'
import sys
import importlib.util

# Package name mapping for imports that differ from package names
PACKAGE_MAPPING = {
    'opencv-python': 'cv2',
    'opencv-python-headless': 'cv2',
    'opencv-contrib-python': 'cv2',
    'scikit-image': 'skimage',
    'scikit-learn': 'sklearn',
    'Pillow': 'PIL',
    'pillow': 'PIL',
    'pyOpenSSL': 'OpenSSL',
    'setuptools': 'pkg_resources',
    'comfy-env': 'comfy_env',
}

def normalize_package_name(pkg):
    """Extract base package name and normalize"""
    # Remove version specifiers
    base = pkg.split('>=')[0].split('==')[0].split('<')[0].split('>')[0].split('!=')[0].strip()
    return PACKAGE_MAPPING.get(base, base.replace('-', '_'))

def is_installed(pkg):
    """Check if package is importable"""
    try:
        module_name = normalize_package_name(pkg)
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False

# Read packages from command line arguments
packages = sys.argv[1:]
missing = [pkg for pkg in packages if not is_installed(pkg)]

# Print missing packages (one per line)
for pkg in missing:
    print(pkg)
CHECKEOF
    
    log "🔍 Checking which packages are already installed..."
    local missing_packages
    missing_packages=$(python /tmp/check_packages.py "${critical_packages[@]}" 2>/dev/null)
    
    if [[ -z "$missing_packages" ]]; then
        log "✅ All critical packages already installed (0 to install)"
        rm -f /tmp/check_packages.py
        return 0
    fi
    
    # Count missing packages
    local missing_count=$(echo "$missing_packages" | wc -l)
    log "📊 Found $missing_count packages to install"
    
    # Convert to array for pip
    local missing_array=()
    while IFS= read -r pkg; do
        [[ -n "$pkg" ]] && missing_array+=("$pkg")
    done <<< "$missing_packages"
    
    # PIP_CACHE_DIR is set to /storage/.pip_cache (persistent across notebook restarts).
    # No --no-cache-dir here so pip reuses downloaded wheels on every cold run.
    log "📦 Installing $missing_count missing packages in batch..."
    local start_time=$(date +%s)

    if pip install --quiet --disable-pip-version-check "${missing_array[@]}" 2>/dev/null; then
        local end_time=$(date +%s)
        log "✅ Installed $missing_count packages in $((end_time - start_time))s (wheels cached in $PIP_CACHE_DIR)"
        rm -f /tmp/check_packages.py
        return 0
    else
        log_error "❌ Batch install failed, falling back to individual installs..."
        local installed_count=0 failed_count=0
        for pkg in "${missing_array[@]}"; do
            if pip install --quiet --disable-pip-version-check "$pkg" 2>/dev/null; then
                ((installed_count++))
            else
                log_error "❌ Failed: $pkg"
                ((failed_count++))
            fi
        done
        log "📊 Individual install: $installed_count ok, $failed_count failed"
        rm -f /tmp/check_packages.py
        return $failed_count
    fi
}

# Force diffusers pin + setuptools + single OpenCV (contrib) after batch installs / Comfy requirements.
# Re-sync huggingface-hub / numpy / transformers / accelerate after diffusers (its deps can break Comfy core).
ensure_comfy_custom_node_pip_stack() {
    log "🔧 Pinning custom-node pip stack (diffusers / transformers / numpy / opencv / bitsandbytes)..."
    log "💡 If ComfyUI-Manager upgrades these later, re-run the script to restore pins."
    disable_err_trap

    # opencv-contrib-python conflicts with opencv-python and opencv-python-headless (all expose cv2).
    # Uninstall all variants, then wipe leftover cv2 namespace stubs (pip uninstall often leaves an empty
    # cv2/ tree that shadows the real binary and breaks INTER_CUBIC / guidedFilter / Impact / WAS).
    pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python 2>/dev/null || true
    local site_packages
    site_packages="$(python -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
    if [[ -n "$site_packages" ]]; then
        rm -rf "${site_packages}/cv2" \
               "${site_packages}"/opencv_*.dist-info \
               "${site_packages}"/opencv_*.libs \
               "${site_packages}"/cv2*.so 2>/dev/null || true
    fi

    # Single pip call: all pins resolved together so there are no redundant network fetches or
    # duplicate installs. Cached wheels are allowed (drop --no-cache-dir) to speed up repeat runs.
    if pip install --disable-pip-version-check \
        "setuptools${SETUPTOOLS_PIN}" \
        "wheel" \
        "diffusers${DIFFUSERS_PIN}" \
        "transformers${TRANSFORMERS_PIN}" \
        "huggingface_hub${HF_HUB_TRANSFORMERS_PIN}" \
        "numpy${NUMPY_COMFY_PIN}" \
        "accelerate${ACCELERATE_PIN}" \
        "tokenizers>=0.20,<0.23" \
        "opencv-contrib-python${OPENCV_CONTRIB_PIN}" \
        "bitsandbytes${BITSANDBYTES_TRITON3_PIN}" \
        "python-multipart>=0.0.18" \
        "comfy-env" \
        "submitit" \
        "scikit-learn"; then
        log "✅ Custom-node pip stack pinned (diffusers ${DIFFUSERS_PIN}, transformers ${TRANSFORMERS_PIN}, opencv-contrib ${OPENCV_CONTRIB_PIN}, numpy ${NUMPY_COMFY_PIN})"
    else
        log_error "⚠️ Pip stack pin had issues — some custom nodes may misbehave"
    fi

    # If opencv still looks like a namespace stub, force a clean contrib reinstall.
    if ! python -c "import cv2; from cv2.ximgproc import guidedFilter; assert hasattr(cv2, 'INTER_CUBIC') and hasattr(cv2, 'CV_8U')" 2>/dev/null; then
        log_error "⚠️ OpenCV broken after pin — wiping cv2 stubs and force-reinstalling opencv-contrib${OPENCV_CONTRIB_PIN}"
        if [[ -n "$site_packages" ]]; then
            rm -rf "${site_packages}/cv2" \
                   "${site_packages}"/opencv_*.dist-info \
                   "${site_packages}"/opencv_*.libs \
                   "${site_packages}"/cv2*.so 2>/dev/null || true
        fi
        pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python 2>/dev/null || true
        pip install --disable-pip-version-check --force-reinstall \
            "numpy${NUMPY_COMFY_PIN}" \
            "opencv-contrib-python${OPENCV_CONTRIB_PIN}" || log_error "⚠️ OpenCV force-reinstall failed"
    else
        log "✅ OpenCV contrib OK (ximgproc.guidedFilter + INTER_CUBIC)"
    fi

    # Verify setuptools / pkg_resources (SUPIR / pytorch_lightning need this at import time).
    if ! python -c "import pkg_resources; assert hasattr(pkg_resources, 'declare_namespace')" 2>/dev/null; then
        log_error "⚠️ pkg_resources missing — forcing setuptools${SETUPTOOLS_PIN} for SUPIR/Lightning"
        pip install --disable-pip-version-check --force-reinstall "setuptools${SETUPTOOLS_PIN}" || true
    else
        log "✅ setuptools / pkg_resources OK for Lightning / SUPIR"
    fi

    enable_err_trap
}

# When /tmp/sd_comfy.prepared exists, venv may still have old diffusers; repair only if checks fail.
ensure_comfy_custom_node_pip_stack_if_needed() {
    disable_err_trap
    if python <<'PYCHK'
import sys
try:
    import pkg_resources  # noqa: F401 — pytorch_lightning / SUPIR
except Exception:
    sys.exit(11)
try:
    import diffusers
    parts = [int(x) for x in diffusers.__version__.split(".")[:2] if x.isdigit()]
    while len(parts) < 2:
        parts.append(0)
    if tuple(parts) != (0, 37):
        sys.exit(12)
except Exception:
    sys.exit(12)
try:
    from diffusers.utils.import_utils import is_k_diffusion_available  # noqa: F401
except Exception:
    sys.exit(22)
try:
    import diffusers.models.unets  # noqa: F401
except Exception:
    sys.exit(13)
try:
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler  # noqa: F401
except Exception:
    sys.exit(14)
try:
    from diffusers import FluxTransformer2DModel  # noqa: F401
except Exception:
    sys.exit(15)
try:
    from cv2.ximgproc import guidedFilter  # noqa: F401
except Exception:
    sys.exit(16)
try:
    import comfy_env  # noqa: F401
except Exception:
    sys.exit(17)
try:
    from importlib.metadata import version as pkg_version
    _hub = pkg_version("huggingface-hub")
    _hub_major = int(_hub.split(".")[0])
    if _hub_major >= 1:
        sys.exit(18)
except Exception:
    sys.exit(18)
try:
    import transformers
    _tp = transformers.__version__.split(".")
    _tmaj = int(_tp[0])
    _tmin = int(_tp[1]) if len(_tp) > 1 else 0
    if not (_tmaj == 4 and 54 <= _tmin < 55):
        sys.exit(19)
except Exception:
    sys.exit(19)
try:
    import bitsandbytes as _bnb
    _bv = _bnb.__version__.split(".")
    _bmaj = int(_bv[0])
    _bmin = int(_bv[1]) if len(_bv) > 1 and _bv[1].isdigit() else 0
    if _bmaj == 0 and _bmin < 45:
        sys.exit(20)
except Exception:
    sys.exit(20)
sys.exit(0)
PYCHK
    then
        enable_err_trap
        log "✅ Custom-node pip stack OK (diffusers 0.37.x, transformers 4.54.x, unets, Flux, OpenCV ximgproc, comfy_env)"
        return 0
    fi
    enable_err_trap
    log "🔧 Custom-node pip stack incomplete — running repair (11–22: deps, hub, transformers 4.54.x, diffusers 0.37.x, k-diffusion symbol, bitsandbytes, …)..."
    ensure_comfy_custom_node_pip_stack
}

# SAM2 Installation Process (with wheel caching like SageAttention)
# Wheel is ~0.5MB — always prefer cache; never rebuild if a good wheel exists (~2 min saved).

promote_sam2_wheel() {
    local wheel="${1:-}"
    mkdir -p "$BOOT_CACHE_EXTRA_DIR/wheels" "$WHEEL_CACHE_DIR"
    if [[ -z "$wheel" || ! -f "$wheel" ]]; then
        wheel=$(find "$WHEEL_CACHE_DIR" /storage/.sam2_cache "$BOOT_CACHE_EXTRA_DIR/wheels" -type f \( -name 'sam_2-*.whl' -o -name 'SAM_2-*.whl' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -f2- -d' ')
    fi
    [[ -n "$wheel" && -f "$wheel" ]] || return 1
    # Durable copies for next cold boot (does not delete/alter source tree contents beyond copy)
    cp -f "$wheel" "$BOOT_CACHE_EXTRA_DIR/wheels/$(basename "$wheel")" 2>/dev/null || true
    cp -f "$wheel" "$WHEEL_CACHE_DIR/$(basename "$wheel")" 2>/dev/null || true
    log "💾 SAM2 wheel cached: $(basename "$wheel") ($(du -h "$wheel" | awk '{print $1}'))"
    return 0
}

find_sam2_cached_wheel() {
    find "$BOOT_CACHE_EXTRA_DIR/wheels" "$WHEEL_CACHE_DIR" /storage/.sam2_cache -type f \( -name 'sam_2-*.whl' -o -name 'SAM_2-*.whl' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -f2- -d' '
}

install_sam2_optimized() {
    log "Verifying SAM2 installation..."
    
    # Setup environment
    setup_cuda_env
    
    # First, just try to import it. If it works, we're done.
    # Must not run from the SAM2 source tree — that directory contains a sam2/ package
    # and would false-positive even when the venv does not have the wheel installed.
    if (cd /tmp && python -c "from sam2.build_sam import build_sam2") &>/dev/null; then
        log "✅ SAM2 is already installed and importable."
        promote_sam2_wheel || true
        return 0
    fi

    # Fast path: install from cached wheel (avoids ~2 min source rebuild every boot)
    if install_sam2_from_cached_wheel; then
        if (cd /tmp && python -c "from sam2.build_sam import build_sam2") &>/dev/null; then
            log "✅ SAM2 installed from cached wheel."
            promote_sam2_wheel || true
            enforce_boot_cache_budget || true
            return 0
        fi
        log_error "Cached SAM2 wheel installed but import failed — will rebuild"
    fi

    log "SAM2 not found. Proceeding with installation..."
    
    # Proceed with full installation from source
    install_sam2_dependencies
    if clone_or_update_sam2_repo; then
         build_and_install_sam2 || return 1
    else
         log_error "Failed to clone or update SAM2 repository. Skipping build."
         return 1
    fi

    # Final check after building from source (from a directory that is not the repo)
    if (cd /tmp && python -c "from sam2.build_sam import build_sam2") &>/dev/null; then
        log "✅ SAM2 successfully built and installed."
        promote_sam2_wheel || true
        enforce_boot_cache_budget || true
        return 0
    else
        log_error "❌ SAM2 installation verification failed."
        return 1
    fi
}

install_sam2_from_cached_wheel() {
    local wheel
    wheel=$(find_sam2_cached_wheel)
    if [[ -z "$wheel" || ! -f "$wheel" ]]; then
        log "🔍 No cached SAM2 wheel found"
        return 1
    fi
    log "⚡ Installing SAM2 from cached wheel: $wheel"
    if pip install --force-reinstall --no-deps --disable-pip-version-check "$wheel"; then
        promote_sam2_wheel "$wheel" || true
        return 0
    fi
    log_error "Failed to install cached SAM2 wheel"
    return 1
}

install_sam2_dependencies() {
    log "Installing SAM2 dependencies (do not touch torch/torchvision; those stay pinned to cu128)..."
    pip install --no-cache-dir --disable-pip-version-check \
        "hydra-core>=1.3.2" "iopath>=0.1.10" "omegaconf" \
        "pillow" "numpy" "scipy" "matplotlib" "scikit-image" \
        "ninja>=1.11.0" "packaging"
}

clone_or_update_sam2_repo() {
    local sam2_cache_base="/storage/.sam2_cache"
    local sam2_build_dir="$sam2_cache_base/src"
    
    # Get CUDA version for cache directory
    local cuda_version
    if command -v nvcc &>/dev/null; then
        cuda_version=$(nvcc --version | grep 'release' | awk '{print $6}' | sed 's/V//' | sed 's/\.//g')
    else
        cuda_version="128"  # Default to CUDA 12.8
    fi
    
    export SAM2_CACHE_DIR="${sam2_cache_base}/v2_cuda${cuda_version}"
    sam2_build_dir="$SAM2_CACHE_DIR/src"
    mkdir -p "$sam2_build_dir"
    
    if [ ! -d "$sam2_build_dir/.git" ]; then
        log "Cloning SAM2 repository into $sam2_build_dir..."
        git clone --depth 1 https://github.com/facebookresearch/sam2.git "$sam2_build_dir" || {
            log_error "Failed to clone SAM2 repository."
            return 1
        }
    else
        log "Updating SAM2 repository in $sam2_build_dir..."
        (cd "$sam2_build_dir" && git fetch && git pull) || {
            log "Failed to update SAM2 repository, using existing code."
        }
    fi
    cd "$sam2_build_dir" || return 1
    log "Current SAM2 commit: $(git rev-parse HEAD)"
    return 0
}

build_and_install_sam2() {
    local sam2_build_dir="$SAM2_CACHE_DIR/src"
    if [[ ! -d "$sam2_build_dir" ]] || ! cd "$sam2_build_dir"; then
         log_error "SAM2 source directory $sam2_build_dir not found or cannot cd into it."
         return 1
    fi

    local venv_python="$VENV_DIR/sd_comfy-env/bin/python"
    if [[ ! -x "$venv_python" ]]; then
        log_error "Virtual environment Python not found or not executable at $venv_python"
        return 1
    fi

    # If a wheel already exists in dist/ (or elsewhere), install it — do NOT wipe and rebuild.
    local existing_wheel
    existing_wheel=$(find "$sam2_build_dir/dist" -maxdepth 1 -type f -name '*.whl' -print -quit 2>/dev/null)
    if [[ -z "$existing_wheel" ]]; then
        existing_wheel=$(find_sam2_cached_wheel)
    fi
    if [[ -n "$existing_wheel" && -f "$existing_wheel" ]]; then
        log "⚡ Reusing existing SAM2 wheel (skip rebuild): $existing_wheel"
        promote_sam2_wheel "$existing_wheel" || true
        if pip install --force-reinstall --no-deps --disable-pip-version-check "$existing_wheel"; then
            log "✅ SAM2 wheel installed successfully (cached)"
            return 0
        fi
        log_error "Cached wheel install failed — falling back to rebuild"
    fi

    log "Building SAM2 wheel in $(pwd)..."
    log "--- Verifying Environment BEFORE Build ---"
    log "CUDA_HOME=$CUDA_HOME"
    log "PATH=$PATH"
    log "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    log "NVCC Version: $(nvcc --version 2>/dev/null || echo 'NVCC not found')"
    log "Python Version: $(python --version || echo 'python not found')"
    log "-----------------------------------------"

    # Preserve any existing wheel before cleaning build dirs
    local preserved=""
    if [[ -n "$existing_wheel" && -f "$existing_wheel" ]]; then
        preserved="$BOOT_CACHE_EXTRA_DIR/wheels/$(basename "$existing_wheel")"
        mkdir -p "$BOOT_CACHE_EXTRA_DIR/wheels"
        cp -f "$existing_wheel" "$preserved" 2>/dev/null || true
    fi
    
    rm -rf build *.egg-info
    # Keep dist/*.whl if present; only remove non-wheel junk
    if [[ -d dist ]]; then
        find dist -type f ! -name '*.whl' -delete 2>/dev/null || true
    fi

    # Install with optimizations
    export MAX_JOBS=$(nproc)  # Use all available cores
    export USE_NINJA=1        # Use Ninja for faster builds
    
    # SAM2 may need CUDA environment variables for building extensions
    setup_cuda_env

    log "Running build command: $venv_python setup.py bdist_wheel"
    if ! "$venv_python" setup.py bdist_wheel; then
        log_error "❌ SAM2 wheel build command failed"
        log_error "This may be due to missing dependencies or CUDA compilation issues"
        log_error "SAM2 can still work without CUDA extensions, but some features may be limited"
        # Last resort: try preserved wheel
        if [[ -n "$preserved" && -f "$preserved" ]]; then
            log "Falling back to preserved SAM2 wheel"
            pip install --force-reinstall --no-deps --disable-pip-version-check "$preserved" && return 0
        fi
        return 1
    fi

    local built_wheel
    # Wheel is named SAM_2 / sam_2 (underscore), not sam2 — match any built wheel.
    built_wheel=$(find "$sam2_build_dir/dist" -name "*.whl" -print -quit)

    if [[ -n "$built_wheel" ]]; then
        log "Found built wheel: $built_wheel"
        log "Installing newly built wheel: $built_wheel"
        # --no-deps: SAM2 metadata can pull a generic torch wheel and overwrite cu128.
        if pip install --force-reinstall --no-deps --disable-pip-version-check "$built_wheel"; then
            log "✅ SAM2 wheel installed successfully"
            promote_sam2_wheel "$built_wheel" || true
            return 0
        else
            log_error "❌ Failed to install SAM2 wheel"
            return 1
        fi
    else
        log_error "❌ Failed to build SAM2 wheel - no wheel file found in dist/"
        log_error "Check build logs above for compilation errors"
        return 1
    fi
}

# Function to fix common custom node import errors
fix_custom_node_import_errors() {
    log "🔧 Checking for missing custom node dependencies..."
    
    # Use a single Python script to check all imports at once (much faster)
    local missing_packages=$(python -c "
import sys
missing = []
try:
    import rembg
except ImportError:
    missing.append('rembg')
try:
    import onnxruntime
except ImportError:
    missing.append('onnxruntime')
try:
    import cv2
except ImportError:
    missing.append('opencv-python')
try:
    import trimesh
except ImportError:
    missing.append('trimesh')
try:
    import pkg_resources
except ImportError:
    missing.append('setuptools')
try:
    import decord
except ImportError:
    missing.append('decord')
try:
    import pandas
except ImportError:
    missing.append('pandas')
print(' '.join(missing))
" 2>/dev/null || echo "")
    
    if [[ -n "$missing_packages" ]]; then
        log "📦 Installing missing packages: $missing_packages"
        pip install --quiet --no-cache-dir $missing_packages 2>/dev/null || log_error "Some packages failed to install"
    fi
    
    log "✅ Custom node dependencies check completed"
}

# Function removed - redundant with install_xformers()


# Function to check xformers status without fixing
check_xformers_status() {
    log "🔍 Checking xformers status..."
    
    if python -c "import xformers" 2>/dev/null; then
        local xformers_version=$(python -c "import xformers; print(xformers.__version__)" 2>/dev/null)
        log "✅ xformers $xformers_version is working correctly"
            return 0
        else
        log "❌ xformers is not working or not installed"
        return 1
    fi
}

# Function to check if PyTorch versions match requirements (simplified)
check_torch_versions() {
    log "🔍 Checking PyTorch ecosystem versions..."
    
    # Check if packages are installed and working
    local torch_working=false
    local torchvision_working=false
    local torchaudio_working=false
    local xformers_working=false
    local cuda_working=false
    
    # Test PyTorch
    if python -c "import torch; print(torch.__version__)" 2>/dev/null; then
        torch_working=true
        # Check CUDA
        if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            cuda_working=true
        fi
    fi
    
    # Test other packages
    python -c "import torchvision" 2>/dev/null && torchvision_working=true
    python -c "import torchaudio" 2>/dev/null && torchaudio_working=true
    python -c "import xformers" 2>/dev/null && xformers_working=true
    
    # Simple decision logic
    if [[ "$torch_working" == "true" && "$cuda_working" == "true" ]]; then
        if [[ "$torchvision_working" == "true" && "$torchaudio_working" == "true" ]]; then
            log "✅ PyTorch ecosystem is working correctly"
            return 0  # No reinstallation needed
        else
            log "⚠️ Core PyTorch working, but some packages missing"
            return 2  # Install missing packages only
        fi
    else
        log "❌ PyTorch ecosystem has issues, needs reinstallation"
        return 1  # Full reinstallation needed
    fi
}

# Function to install only missing PyTorch packages (simplified)
install_missing_torch_packages() {
    log "📦 Installing missing PyTorch packages..."
    
    local missing_packages=()
    
    # Check what's actually missing
    python -c "import torchvision" 2>/dev/null || missing_packages+=("torchvision==${TORCHVISION_VERSION}")
    python -c "import torchaudio" 2>/dev/null || missing_packages+=("torchaudio==${TORCHAUDIO_VERSION}")
    
    if [[ ${#missing_packages[@]} -eq 0 ]]; then
        log "✅ No missing packages to install"
        return 0
    fi
    
    log "📦 Installing missing packages: ${missing_packages[*]}"
    
    # Install missing packages with correct CUDA version
    if pip install --no-cache-dir --ignore-installed "${missing_packages[@]}" --extra-index-url "${TORCH_INDEX_URL}"; then
        log "✅ Successfully installed missing packages: ${missing_packages[*]}"
        return 0
    else
        log_error "❌ Failed to install missing packages"
        return 1
    fi
}

# Function to clean up existing installations
clean_torch_installations() {
    echo "Performing deep cleanup of PyTorch installations..."
    echo "This will uninstall: torch, torchvision, torchaudio, xformers (if present)"
    
    # First, try normal uninstall
    pip uninstall -y torch torchvision torchaudio xformers || true
    
    # Deep cleanup: Remove corrupted packages manually
    echo "Performing deep cleanup of potentially corrupted packages..."
    local site_packages_dir=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "/tmp/sd_comfy-env/lib/python3.10/site-packages")
    
    if [[ -d "$site_packages_dir" ]]; then
        echo "Removing corrupted PyTorch package directories from: $site_packages_dir"
        # Remove torch-related directories
        rm -rf "$site_packages_dir"/torch* "$site_packages_dir"/xformers* "$site_packages_dir"/*torch* || true
        # Remove invalid distribution markers
        rm -rf "$site_packages_dir"/-orch* || true
        echo "Manual package directory cleanup completed."
    fi
    
    # Do NOT purge existing /storage/.pip_cache — grandfathered and outside the 3GB extra budget.
    echo "Keeping existing pip cache at $PIP_CACHE_DIR (untouched; extra cache is $BOOT_CACHE_EXTRA_DIR)"
    
    # Clear Python cache to prevent import issues
    echo "Clearing Python bytecode cache..."
    find "$site_packages_dir" -name "*.pyc" -delete 2>/dev/null || true
    find "$site_packages_dir" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    
    echo "Deep cleanup completed."
}

# Function to install PyTorch core packages
install_torch_core() {
    echo "Installing PyTorch core packages (torch, torchvision, torchaudio)..."
    # New torch wheels go into EXTRA pip cache only (≤3GB). Existing /storage/.pip_cache untouched.
    mkdir -p "$BOOT_CACHE_EXTRA_DIR/pip"
    local install_cmd="env PIP_CACHE_DIR=$BOOT_CACHE_EXTRA_DIR/pip pip install --ignore-installed torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --extra-index-url ${TORCH_INDEX_URL}"
    
    log "Running core install command: $install_cmd"
    
    if $install_cmd; then
        log "PyTorch core packages installation command finished successfully."
        # Quick verification
        python -c "import torch; print(f'Core install OK: Torch {torch.__version__} imported successfully.')" || return 1
        enforce_boot_cache_budget || true
        return 0
    else
        local status=$?
        log_error "PyTorch core packages installation failed with status $status."
        return $status
    fi
}

    # Function to install xformers (simplified - using your proven method)
    install_xformers() {
    log "📦 Installing xformers..."
    
    mkdir -p "$BOOT_CACHE_EXTRA_DIR/pip"
    # xformers downloads land in EXTRA cache only
    env PIP_CACHE_DIR="$BOOT_CACHE_EXTRA_DIR/pip" pip install --disable-pip-version-check --no-deps --quiet \
        xformers==${XFORMERS_VERSION} --extra-index-url "https://download.pytorch.org/whl/cu128" 2>/dev/null || \
    env PIP_CACHE_DIR="$BOOT_CACHE_EXTRA_DIR/pip" pip install --disable-pip-version-check --force-reinstall --quiet \
        xformers --index-url "https://download.pytorch.org/whl/cu128" 2>/dev/null || \
    env PIP_CACHE_DIR="$BOOT_CACHE_EXTRA_DIR/pip" pip install --disable-pip-version-check --force-reinstall --quiet \
        xformers 2>/dev/null || \
    log_error "⚠️ All xformers installation strategies failed, continuing without"
    enforce_boot_cache_budget || true
    
    # Verify installation
    if python -c "import xformers; print(f'✅ xformers {xformers.__version__} installed successfully')" 2>/dev/null; then
        log "✅ xformers installation completed"
            return 0
        else
        log_error "❌ xformers installation failed"
                return 1
    fi
}

# Function removed - redundant with verify_installations()

# Function to verify installations (simplified)
verify_installations() {
    log "🔍 Verifying PyTorch ecosystem installations..."
    
    # Check PyTorch packages
    local torch_version=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "not_installed")
    local torchvision_version=$(python -c "import torchvision; print(torchvision.__version__)" 2>/dev/null || echo "not_installed")
    local torchaudio_version=$(python -c "import torchaudio; print(torchaudio.__version__)" 2>/dev/null || echo "not_installed")
    local xformers_version=$(python -c "import xformers; print(xformers.__version__)" 2>/dev/null || echo "not_installed")
    
    log "📦 Installed versions:"
    log "  - torch: $torch_version"
    log "  - torchvision: $torchvision_version"
    log "  - torchaudio: $torchaudio_version"
    log "  - xformers: $xformers_version"
    
    # Check CUDA availability
    local cuda_available=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
    log "🔧 CUDA Available: $cuda_available"
    
    # Simple success/failure check
    if [[ "$torch_version" != "not_installed" && "$cuda_available" == "True" ]]; then
        log "✅ PyTorch ecosystem verification successful"
        return 0
    else
        log_error "❌ PyTorch ecosystem verification failed"
        return 1
    fi
}

# Main function to fix torch versions
fix_torch_versions() {
    echo "Checking PyTorch/CUDA versions..."
    
    # Check what needs to be done
    log "🔍 Checking if PyTorch packages are actually installed..."
    local check_result
    set +e  # Temporarily disable set -e to allow non-zero returns
    check_torch_versions
    check_result=$?
    set -e  # Re-enable set -e
    case $check_result in
        0)
            log "✅ PyTorch ecosystem already working, skipping reinstallation"
            verify_installations
            ;;
        1)
            log "🔧 PyTorch ecosystem needs installation (packages not found)..."
            clean_torch_installations

            # Install core first, then xformers
            if ! install_torch_core; then
                log_error "PyTorch core installation failed. Aborting."
                return 1
            fi
            
            # Install xformers as part of PyTorch ecosystem setup
            log "📦 Installing xformers as part of PyTorch ecosystem..."
            install_xformers || log_error "xformers installation failed (continuing)"

            # Final verification of PyTorch ecosystem
            log "🔍 Final verification of PyTorch ecosystem..."
            verify_installations
            
            # Create a marker to indicate recent successful installation
            touch "/tmp/pytorch_ecosystem_fresh_install"
            ;;
        2)
            log "⚠️ Core PyTorch working, installing only missing packages..."
            
            # Install missing packages without full reinstallation
            if install_missing_torch_packages; then
                log "✅ Successfully installed missing packages"
                verify_installations
            else
                log "❌ Failed to install missing packages, falling back to full reinstallation"
                # Fall back to case 1 logic
                clean_torch_installations

                if ! install_torch_core; then
                    log_error "PyTorch core installation failed. Aborting."
                    return 1
                fi
                
                if ! install_xformers; then
                    log_error "xformers installation failed. Continuing, but there may be issues."
                fi
                
                verify_installations
                touch "/tmp/pytorch_ecosystem_fresh_install"
            fi
            ;;
        *)
            log_error "Unexpected return code from check_torch_versions: $check_result"
            return 1
            ;;
    esac
    
    log "✅ PyTorch ecosystem setup completed"
    return 0
}

echo "### Setting up Stable Diffusion Comfy ###"
log "Setting up Stable Diffusion Comfy"
#######################################
# STEP 5: STABLE DIFFUSION SETUP
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
    
       # Ensure we're on a branch before updating
    if [[ -d ".git" ]]; then
        # Check if we're in detached HEAD state
        if git symbolic-ref -q HEAD >/dev/null; then
            echo "On branch $(git branch --show-current)"
        else
            echo "Detected detached HEAD state, checking out main branch..."
            git checkout main || git checkout master || {
                echo "Creating and checking out main branch..."
                git checkout -b main
            }
        fi
    fi 
    
    # Check and update ComfyUI to latest version before installation
    echo ""
    echo "=================================================="
    echo "           CHECKING COMFYUI UPDATES"
    echo "=================================================="
    echo ""
    
    if [ -d ".git" ]; then
        echo "📋 Checking ComfyUI version information..."
        
        # Get current commit hash and branch
        current_commit=$(git rev-parse HEAD 2>/dev/null || echo "Unknown")
        current_branch=$(git branch --show-current 2>/dev/null || echo "Unknown")
        current_date=$(git log -1 --format="%cd" --date=short 2>/dev/null || echo "Unknown")
        
        echo "📍 Current ComfyUI Status:"
        echo "   Branch: $current_branch"
        echo "   Commit: $current_commit"
        echo "   Date: $current_date"
        
        # Under set -e, failing git fetch/rev-parse must not run as a top-level command (ERR trap logs but shell still exits).
        echo ""
        echo "🔄 Checking for updates..."
        # Fetch only the checked-out branch when possible — avoids unrelated ref errors on the
        # remote (e.g. both branch "dev" and "dev/..." which break a full git fetch).
        fetch_ok=0
        if [ -n "$current_branch" ] && [ "$current_branch" != "Unknown" ]; then
            git fetch origin "$current_branch" 2>&1 && fetch_ok=1
        else
            git fetch origin 2>&1 && fetch_ok=1
        fi
        if [ "$fetch_ok" -ne 1 ]; then
            echo "⚠️  git fetch failed (network, auth, rate limit, conflicting remote refs, or repo issue). Remote comparison may be incomplete."
        fi
        
        # Compare local vs remote
        local_commit=$(git rev-parse HEAD 2>/dev/null) || local_commit=""
        remote_commit=$(git rev-parse "origin/${current_branch}" 2>/dev/null) || remote_commit=""
        
        if [ -z "$local_commit" ] || [ -z "$remote_commit" ]; then
            echo "⚠️  Could not compare local vs remote (missing refs after fetch or unknown branch)."
        elif [ "$local_commit" = "$remote_commit" ]; then
            echo "✅ ComfyUI is up to date with the latest version!"
        else
            echo "⚠️  ComfyUI has updates available!"
            echo "   Local:  $local_commit"
            echo "   Remote: $remote_commit"
            echo ""
            echo "🔄 Updating ComfyUI to latest version..."
            
            # Autostash any local dirt (patches, symlink placeholder deletions, etc.)
            # so pull is not blocked by files like comfy/text_encoders/llama.py.
            if git pull --autostash origin "$current_branch"; then
                echo "✅ ComfyUI successfully updated to latest version!"
                
                # Update custom nodes as well
                echo "🔄 Updating custom nodes..."
                if [ -d "custom_nodes" ]; then
                    # Temporarily disable set -e and ERR trap to allow custom node update failures without script exit
                    set +e
                    disable_err_trap
                    
                    updated_nodes=0
                    failed_nodes=0
                    
                    for git_dir in custom_nodes/*/.git; do
                        if [[ -d "$git_dir" ]]; then
                            node_dir="${git_dir%/.git}"
                            node_name=$(basename "$node_dir")
                            
                            echo "📁 Updating custom node: $node_name"
                            if cd "$node_dir"; then
                                if git fetch --all &>/dev/null && git reset --hard origin/HEAD &>/dev/null; then
                                    echo "✅ Updated: $node_name"
                                    ((updated_nodes++))
                                else
                                    echo "⚠️  Failed to update: $node_name"
                                    ((failed_nodes++))
                                fi
                                cd - > /dev/null
                            fi
                        fi
                    done
                    
                    # Re-enable set -e and ERR trap
                    set -e
                    enable_err_trap
                    
                    echo "📊 Custom nodes update summary: $updated_nodes successful, $failed_nodes failed"
                fi
                
                # Update ComfyUI Manager specifically if it exists
                if [ -d "custom_nodes/comfyui-manager" ]; then
                    echo "🔧 Updating ComfyUI Manager..."
                    # Temporarily disable set -e and ERR trap to allow ComfyUI Manager update failures without script exit
                    set +e
                    disable_err_trap
                    
                    cd "custom_nodes/comfyui-manager"
                    if git fetch --all &>/dev/null && git reset --hard origin/HEAD &>/dev/null; then
                        echo "✅ ComfyUI Manager updated successfully"
                    else
                        echo "⚠️  ComfyUI Manager update had issues"
                    fi
                    cd - > /dev/null
                    
                    # Re-enable set -e and ERR trap
                    set -e
                    enable_err_trap
                fi
                
                echo "🔄 ComfyUI and custom nodes updated successfully!"
                
            else
                echo "❌ Failed to update ComfyUI. Please check the repository status."
            fi
        fi
        
        # Always re-apply managed model/custom_node symlinks after update attempt
        # (pull may restore placeholder dirs under models/).
        apply_comfy_symlinks "$current_dir/comfy_symlinks.txt" || log_error "⚠️ Post-pull symlink apply had issues (continuing)"
        if [[ -f /notebooks/logs/patch_minimax_flash_decode.py ]]; then
            echo "🔧 Re-applying MiniMax flash decode patch (if needed)..."
            python3 /notebooks/logs/patch_minimax_flash_decode.py >/dev/null 2>&1 || true
        fi
        
        # Show recent commits
        echo ""
        echo "📝 Recent commits:"
        git log --oneline -5 2>/dev/null | sed 's/^/   /' || echo "   Unable to show recent commits"
        
    else
        echo "⚠️  ComfyUI repository not found or not a git repository"
    fi 
    

    # Virtual environment setup using storage Python 3.10 (ephemeral under /tmp)
    mkdir -p "$VENV_DIR"
    echo "Creating fresh virtual environment at $VENV_DIR/sd_comfy-env"
    rm -rf "$VENV_DIR/sd_comfy-env"
    "$PYTHON_EXECUTABLE" -m venv "$VENV_DIR/sd_comfy-env" || { log_error "Failed to create virtual environment"; exit 1; }
    
    # Activate the virtual environment
    source "$VENV_DIR/sd_comfy-env/bin/activate" || { log_error "Failed to activate virtual environment"; exit 1; }
    echo "Virtual environment activated: $VENV_DIR/sd_comfy-env"

    # System dependencies (apt-get is smart enough to skip installed packages)
    echo "Checking/installing system dependencies..."
    apt-get update -qq && apt-get install -y \
        libatlas-base-dev libblas-dev liblapack-dev \
        libjpeg-dev libpng-dev \
        python3-dev build-essential \
        libgl1-mesa-dev \
        espeak-ng \
        ffmpeg \
        pigz > /dev/null 2>&1 || {
        echo "Warning: Some packages failed to install"
    }
    echo "✅ System dependencies check completed (including pigz for fast CUDA caching)"

    # Python environment setup
    pip install pip==24.0
    pip install --upgrade wheel "setuptools${SETUPTOOLS_PIN}"
    pip install "numpy${NUMPY_COMFY_PIN}"



    # ========================================
    # DEFINE ALL FUNCTIONS BEFORE EXECUTION
    # ========================================

    # Emergency PyTorch Recovery Function
    emergency_pytorch_recovery() {
        echo "🚨 EMERGENCY: Detected corrupted PyTorch installation. Performing full recovery..."
        log_error "PyTorch ecosystem is corrupted. Starting emergency recovery procedure."
        
        # Perform aggressive cleanup
        clean_torch_installations
        
        # Reinstall PyTorch ecosystem from scratch
        echo "Reinstalling PyTorch ecosystem from scratch..."
        if install_torch_core; then
            echo "✅ PyTorch core recovery successful"
        else
            log_error "❌ PyTorch core recovery failed. Cannot proceed with SageAttention."
            return 1
        fi
        
        # Verify recovery
        local torch_check
        torch_check=$(python -c "import torch; print(f'Recovery check: torch {torch.__version__} working')" 2>&1)
        local torch_status=$?
        
        if [[ $torch_status -eq 0 ]]; then
            echo "✅ PyTorch recovery verified: $torch_check"
            return 0
        else
            log_error "❌ PyTorch recovery verification failed: $torch_check"
            return 1
        fi
    }

    # SageAttention Installation Process
    install_sageattention() {
        # Initialize environment
        echo "Verifying SageAttention installation..."
        setup_environment
        create_directories
        setup_ccache
        
        # CRITICAL: Check if PyTorch is working before proceeding
        echo "Checking PyTorch ecosystem health before SageAttention installation..."
        local torch_health_check
        torch_health_check=$(python -c "import torch; print(f'PyTorch {torch.__version__} working')" 2>&1)
        local torch_health_status=$?
        
        if [[ $torch_health_status -ne 0 ]]; then
            log_error "PyTorch ecosystem is broken. Error: $torch_health_check"
            if emergency_pytorch_recovery; then
                echo "✅ Emergency PyTorch recovery completed. Proceeding with SageAttention..."
            else
                log_error "❌ Emergency PyTorch recovery failed. Skipping SageAttention installation."
                return 1
            fi
        else
            echo "✅ PyTorch ecosystem health check passed: $torch_health_check"
        fi
        
        # First, just try to import it. If it works, we're done.
        if python -c "import sageattention" &>/dev/null; then
            log "✅ SageAttention is already installed and importable."
            return 0
        fi

        log "SageAttention not found. Proceeding with installation..."

        # Now, check for a compatible cached wheel.
        if check_and_install_cached_wheel; then
            log "✅ Successfully installed SageAttention from cached wheel."
            # Final verification
            if python -c "import sageattention" &>/dev/null; then
                 log "✅ SageAttention import confirmed after wheel installation."
                 return 0
            else
                 log_error "Installed from wheel, but import still fails. This likely means the cached wheel is incompatible."
                 # Fall through to build
            fi
        fi

        log "No suitable cached wheel found or installation from wheel failed. Proceeding with full build."
        
        # Proceed with full installation from source
        install_dependencies
        if clone_or_update_repo; then
             build_and_install # This function will cache the wheel on success
        else
             log_error "Failed to clone or update SageAttention repository. Skipping build."
             return 1 # Cannot proceed
        fi

        # Final check after building from source
        log "Performing final SageAttention verification..."
        pushd /tmp > /dev/null # Change to neutral directory to avoid import conflicts
        local final_import_output
        local final_import_status
        final_import_output=$(python -c "import sageattention; print(f'✅ SageAttention {sageattention.__version__} successfully built and installed from source.')" 2>&1)
        final_import_status=$?
        popd > /dev/null # Return to original directory
        
        if [[ $final_import_status -eq 0 ]]; then
            log "$final_import_output"
            return 0
        else
            log_error "❌ Final SageAttention verification failed."
            log_error "Import error output:"
            log_error "$final_import_output"
            # Don't fail completely since previous verification passed
            log_error "Previous verification passed, so SageAttention may still be functional."
            return 0  # Return success to continue script
        fi
    }

    # SageAttention Helper Functions
    setup_environment() {
        export CUDA_HOME=/usr/local/cuda-12.8
        export PATH=$CUDA_HOME/bin:$PATH
        export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
        export FORCE_CUDA=1
        export TORCH_CUDA_ARCH_LIST="8.6"
        export MAX_JOBS=$(nproc)
        export USE_NINJA=1
        echo "SageAttention Environment Setup:"
        echo "  CUDA_HOME=$CUDA_HOME"
        echo "  NVCC Check: $(nvcc --version || echo 'NVCC not found')"
        echo "  Python Check: $(python --version || echo 'python not found')"
    }

    create_directories() {
        export TORCH_EXTENSIONS_DIR="/storage/.torch_extensions"
        local sage_cache_base="/storage/.sageattention_cache"
        
        # Get CUDA version more reliably
        local cuda_version
        if command -v nvcc &>/dev/null; then
            cuda_version=$(nvcc --version | grep 'release' | awk '{print $6}' | sed 's/V//' | sed 's/\.//g')
        else
            cuda_version="128"  # Default to CUDA 12.8
        fi
        
        export SAGEATTENTION_CACHE_DIR="${sage_cache_base}/v2_cuda${cuda_version}"
        export WHEEL_CACHE_DIR="/storage/.wheel_cache"
        
        mkdir -p "$TORCH_EXTENSIONS_DIR" "$SAGEATTENTION_CACHE_DIR" "$WHEEL_CACHE_DIR"
        
        echo "Created/Ensured directories:"
        echo "  Torch Extensions: $TORCH_EXTENSIONS_DIR"
        echo "  SageAttention Cache: $SAGEATTENTION_CACHE_DIR"
        echo "  Wheel Cache: $WHEEL_CACHE_DIR"
        echo "  CUDA Version: $cuda_version"
        
        # Debug: Show what's in the wheel cache
        echo "  Wheel Cache Contents:"
        if [[ -d "$WHEEL_CACHE_DIR" ]]; then
            find "$WHEEL_CACHE_DIR" -name "sageattention*.whl" -type f 2>/dev/null | head -5 | sed 's/^/    /'
        else
            echo "    Wheel cache directory not found"
        fi
    }

    setup_ccache() {
        if command -v ccache &> /dev/null; then
            export CMAKE_C_COMPILER_LAUNCHER=ccache
            export CMAKE_CXX_COMPILER_LAUNCHER=ccache
            ccache --max-size=3G
            ccache -z
        fi
    }

    check_and_install_cached_wheel() {
        local arch=$(uname -m)
        local python_executable="$VENV_DIR/sd_comfy-env/bin/python"

        local py_version_short
        py_version_short=$("$python_executable" -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')" 2>/dev/null)
        if [[ -z "$py_version_short" ]]; then
            log_error "Could not determine Python version for wheel search."
            return 1
        fi
        local python_version_tag="cp${py_version_short}"

        log "🔍 Checking for cached SageAttention wheel..."
        log "  Python version: $python_version_tag"
        log "  Architecture: $arch"
        log "  Wheel cache dir: $WHEEL_CACHE_DIR"

        # Debug: Show all SageAttention wheels in cache
        log "  All SageAttention wheels in cache:"
        find "$WHEEL_CACHE_DIR" -name "sageattention*.whl" -type f 2>/dev/null | sed 's/^/    /' || log "    No wheels found"

        # Look for ANY SageAttention wheel in the wheel cache (version-agnostic)
        local sage_wheel
        sage_wheel=$(find "$WHEEL_CACHE_DIR" -maxdepth 1 -type f -name "sageattention-*-${python_version_tag}-*-linux_${arch}.whl" -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d' ')

        if [[ ! -f "$sage_wheel" ]]; then
            log "❌ No suitable cached wheel found in $WHEEL_CACHE_DIR for Python ${python_version_tag}."
            log "  Search pattern: sageattention-*-${python_version_tag}-*-linux_${arch}.whl"
            return 1
        fi

        log "Found cached wheel: $(basename "$sage_wheel"). Attempting installation..."
        
        # Check if SageAttention is already working
        if python -c "import sageattention" 2>/dev/null; then
            log "✅ SageAttention is already working, skipping cached wheel installation"
            return 0
        fi
        
        if "$python_executable" -m pip install --force-reinstall --no-cache-dir --disable-pip-version-check "$sage_wheel"; then
            log "Installation of cached wheel succeeded."
            return 0
        else
            log_error "Installation of cached wheel $(basename "$sage_wheel") failed."
            log_error "This wheel may be corrupt or incompatible. Deleting it."
            rm -f "$sage_wheel"
            return 1
        fi
    }

    install_dependencies() {
        log "Installing SageAttention dependencies..."
        pip install --no-cache-dir --disable-pip-version-check \
            "ninja>=1.11.0" \
            "packaging"
    }

    clone_or_update_repo() {
        local sage_build_dir="$SAGEATTENTION_CACHE_DIR/src"
        if [ ! -d "$sage_build_dir/.git" ]; then
            log "Cloning SageAttention repository into $sage_build_dir..."
            git clone https://github.com/thu-ml/SageAttention.git "$sage_build_dir" || {
                log_error "Failed to clone SageAttention repository."
                return 1
            }
        else
            log "Updating SageAttention repository in $sage_build_dir..."
            (cd "$sage_build_dir" && git fetch && git pull) || {
                log_warning "Failed to update SageAttention repository, using existing code."
            }
        fi
        cd "$sage_build_dir" || return 1
        log "Current SageAttention commit: $(git rev-parse HEAD)"
        return 0
    }

    build_and_install() {
        local sage_build_dir="$SAGEATTENTION_CACHE_DIR/src"
        if [[ ! -d "$sage_build_dir" ]] || ! cd "$sage_build_dir"; then
             log_error "SageAttention source directory $sage_build_dir not found or cannot cd into it."
             return 1
        fi

        log "Building SageAttention wheel in $(pwd)..."
        rm -rf build dist *.egg-info

        local venv_python="$VENV_DIR/sd_comfy-env/bin/python"
        if [[ ! -x "$venv_python" ]]; then
            log_error "Virtual environment Python not found or not executable at $venv_python"
            return 1
        fi

        log "Running build command: $venv_python setup.py bdist_wheel"
        "$venv_python" setup.py bdist_wheel

        local built_wheel
        built_wheel=$(find "$sage_build_dir/dist" -name "sageattention*.whl" -print -quit)

        if [[ -n "$built_wheel" ]]; then
            log "Found built wheel: $built_wheel"
            cp "$built_wheel" "$WHEEL_CACHE_DIR/"
            log "Cached built wheel to $WHEEL_CACHE_DIR/$(basename "$built_wheel")"

            log "Installing newly built wheel: $built_wheel"
            if pip install --force-reinstall --no-cache-dir --disable-pip-version-check "$built_wheel"; then
                log "✅ SageAttention wheel installed successfully"
                return 0
            else
                log_error "❌ Failed to install SageAttention wheel"
                return 1
            fi
        else
            log_error "❌ Failed to build SageAttention wheel"
            return 1
                 fi
     }

    # ========================================
    # EXECUTE INSTALLATION STEPS
    # ========================================

    # --- STEP 4: SETUP PYTORCH ECOSYSTEM ---
    echo ""
    echo "=================================================="
    echo "         STEP 4: SETUP PYTORCH ECOSYSTEM"
    echo "=================================================="
    echo ""
    fix_torch_versions
    fix_torch_status=$?
    if [[ $fix_torch_status -ne 0 ]]; then
        log_error "PyTorch ecosystem setup failed (Status: $fix_torch_status). Cannot proceed."
        exit 1
    else
        echo "✅ PyTorch ecosystem setup completed successfully."
    fi

    # --- STEP 5: INSTALL CUSTOM NODE DEPENDENCIES ---
    echo ""
    echo "=================================================="
    echo "       STEP 5: INSTALL CUSTOM NODE DEPENDENCIES"
    echo "=================================================="
    echo ""
    
    # Now that PyTorch is ready, install custom node dependencies
    log "🔧 Installing custom node dependencies (PyTorch ecosystem is now ready)..."
    fix_custom_node_import_errors || log_error "Some custom node import fixes failed (continuing)"
    
    log "✅ Custom node dependencies completed"

    # Define handle_successful_installation function before using it
    handle_successful_installation() {
        # This function ensures the SageAttention module path can be found
        local sage_module_path
        log "Attempting to determine SageAttention module path..."
        pushd /tmp > /dev/null # Change to neutral directory
        sage_module_path=$(python -c "import sageattention, os; print(os.path.dirname(sageattention.__file__))" 2>&1)
        local path_status=$?
        popd > /dev/null # Return to original directory

        if [[ $path_status -eq 0 && -n "$sage_module_path" && -d "$sage_module_path" ]]; then
             log "✅ SageAttention setup complete. Module path found: $sage_module_path"
             return 0
        else
             log_error "⚠️ SageAttention installed and imports, but failed to determine module path via Python."
             log_error "Python output: $sage_module_path"
             return 1 # Indicate partial failure
        fi
    }

    # --- STEP 6: UPDATE CUSTOM NODES ---
    echo ""
    echo "=================================================="
    echo "            STEP 6: UPDATE CUSTOM NODES"
    echo "=================================================="
    echo ""
    
    # Function to update all custom nodes from Git repositories
    update_custom_nodes() {
        local nodes_dir="$REPO_DIR/custom_nodes"
        [[ ! -d "$nodes_dir" ]] && return 0
        
        log "🔄 Updating all custom nodes from Git repositories..."
        local updated_nodes=0
        local failed_nodes=0
        
        for git_dir in "$nodes_dir"/*/.git; do
            if [[ -d "$git_dir" ]]; then
                local node_dir="${git_dir%/.git}"
                local node_name=$(basename "$node_dir")
                
                log "📁 Updating Git node: $node_name"
                
                if cd "$node_dir"; then
                    if git fetch --all &>/dev/null && git reset --hard origin/HEAD &>/dev/null; then
                        log "✅ Git update successful for $node_name"
                        ((updated_nodes++))
                    else
                        log_error "❌ Git update failed for $node_name"
                        ((failed_nodes++))
                    fi
                    cd - > /dev/null
                else
                    log_error "❌ Failed to access directory for $node_name"
                    ((failed_nodes++))
                fi
            fi
        done
        
        # Summary
        log "📊 Custom node update summary: $updated_nodes successful, $failed_nodes failed"
        [[ $failed_nodes -gt 0 ]] && log_error "⚠️ Some custom nodes had issues - check logs above"
        
        log "✅ Custom node Git updates complete!"
        return 0
    }
    
    # Execute custom node updates (skipped by default — ~1 min; set UPDATE_CUSTOM_NODES=1 or SKIP_CUSTOM_NODE_UPDATE=0)
    if [[ "${UPDATE_CUSTOM_NODES:-0}" == "1" || "${SKIP_CUSTOM_NODE_UPDATE}" == "0" ]]; then
        set +e
        disable_err_trap
        
        update_custom_nodes || log_error "Custom nodes update had issues (continuing)"
        custom_nodes_status=$?
        
        # Re-enable set -e and ERR trap
        set -e
        enable_err_trap
        
        if [[ $custom_nodes_status -eq 0 ]]; then
            echo "✅ Custom nodes update completed successfully."
        else
            log_error "⚠️ Custom nodes update had issues (Status: $custom_nodes_status)"
            log_error "Some custom nodes may not be up-to-date"
        fi
    else
        log "⏭️ Skipping custom-node git updates (default). Set UPDATE_CUSTOM_NODES=1 to enable."
        custom_nodes_status=0
    fi

    # --- STEP 7: INSTALL SAGEATTENTION OPTIMIZATION ---
    echo ""
    echo "=================================================="
    echo "      STEP 7: INSTALL SAGEATTENTION OPTIMIZATION"
    echo "=================================================="
    echo ""
    
    # Temporarily disable set -e and ERR trap to allow SageAttention failure without script exit
    set +e
    disable_err_trap
    
    install_sageattention
    sageattention_status=$?
    
    # Re-enable set -e and ERR trap
    enable_err_trap
    
    if [[ $sageattention_status -eq 0 ]]; then
        echo "✅ SageAttention installation completed successfully."
        # Complete SageAttention setup verification
        set +e
        disable_err_trap
        handle_successful_installation
        sage_setup_status=$?
        set -e
        enable_err_trap
        
        if [[ $sage_setup_status -eq 0 ]]; then
            echo "✅ SageAttention setup verification completed successfully."
        else
            log_error "⚠️ SageAttention setup verification had issues (Status: $sage_setup_status)"
            log_error "SageAttention may not work properly"
        fi
    else
        log_error "⚠️ SageAttention installation had issues (Status: $sageattention_status)"
        log_error "ComfyUI will continue without SageAttention optimizations"
    fi

    # Custom node dependencies already handled in Step 5

    # --- STEP 6: INSTALL PYTHON DEPENDENCIES ---
    echo ""
    echo "=================================================="
    echo "        STEP 6: INSTALL PYTHON DEPENDENCIES"
    echo "=================================================="
    echo ""
    
    # Note: TensorFlow installation moved to background after ComfyUI starts (Step 11.5)
    
    # Optimized requirements processing with dependency caching
    process_requirements() {
        local req_file="$1"
        local indent="${2:-}"
        local cache_dir="/storage/.pip_cache"
        local combined_reqs="/tmp/combined_requirements.txt"
        local verify_script="/tmp/verify_imports.py"
        
        # Clean input file path
        req_file="$(echo "$req_file" | tr -d ' ')"
        [[ ! -f "$req_file" ]] && {
            echo "${indent}Skipping: File not found - $req_file"
            return 0
        }
        
        # Temporarily disable set -e for this function to prevent termination on individual failures
        set +e

        echo "${indent}Processing: $req_file"
        
        # Set up cache directory
        mkdir -p "$cache_dir"
        
        # Suppress pip upgrade notices by setting environment variable
        export PIP_DISABLE_PIP_VERSION_CHECK=1
        
        # Create a single combined requirements file
        echo -n > "$combined_reqs"
        
        # Collect all requirements recursively
        function collect_reqs() {
            local file="$1"
            local ind="$2"
            
            [[ ! -f "$file" ]] && return 0
            
            # Add requirements from this file (with error handling)
            if ! grep -v "^-r\|^#\|^$" "$file" >> "$combined_reqs" 2>/dev/null; then
                echo "${ind}Warning: Failed to read requirements from $file"
                return 0
            fi
            
            # Process included requirements files
            grep "^-r" "$file" 2>/dev/null | sed 's/^-r\s*//' | while read -r included_file; do
                # Resolve relative paths
                if [[ "$included_file" != /* ]]; then
                    included_file="$(dirname "$file")/$included_file"
                fi
                
                if [[ -f "$included_file" ]]; then
                    echo "${ind}Including: $included_file"
                    collect_reqs "$included_file" "$ind  "
                else
                    echo "${ind}Warning: Included file not found - $included_file"
                fi
            done
        }
        
        collect_reqs "$req_file" "$indent"
        
        # Deduplicate and normalize requirements
        echo "${indent}Deduplicating and resolving conflicts..."
        
        # Create a Python script to handle version conflicts
        cat > "/tmp/resolve_conflicts.py" << 'EOF'
import re
import sys
from collections import defaultdict

def parse_requirement(req):
    # Extract package name and version specifier
    match = re.match(r'^([a-zA-Z0-9_\-\.]+)(.*)$', req)
    if not match:
        return req, ""
    
    name, version_spec = match.groups()
    return name.lower(), version_spec

# Read requirements
with open(sys.argv[1], 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith(('git+', 'http'))]

# Group by package name
package_versions = defaultdict(list)
for req in requirements:
    name, version_spec = parse_requirement(req)
    if name and version_spec:
        package_versions[name].append(version_spec)

# Resolve conflicts by using the most permissive version
resolved = []
for req in requirements:
    name, version_spec = parse_requirement(req)
    
    # Skip git/http requirements
    if req.startswith(('git+', 'http')):
        resolved.append(req)
        continue
        
    # If this package has multiple version specs, use the most permissive one
    if name in package_versions and len(package_versions[name]) > 1:
        # For simplicity, we'll use the shortest version spec as a heuristic
        # This isn't perfect but helps with common cases
        if version_spec == min(package_versions[name], key=len):
            resolved.append(req)
    else:
        resolved.append(req)

# Write resolved requirements
with open(sys.argv[2], 'w') as f:
    for req in sorted(set(resolved)):
        f.write(f"{req}\n")
EOF

        # Run the conflict resolution script
        if ! python "/tmp/resolve_conflicts.py" "$combined_reqs" "/tmp/resolved_requirements.txt" 2>/dev/null; then
            echo "${indent}Warning: Conflict resolution failed for $req_file, using original requirements"
            # Keep original file if conflict resolution fails
        else
            mv "/tmp/resolved_requirements.txt" "$combined_reqs"
        fi
        
        # Create verification script
        cat > "$verify_script" << 'EOF'
import sys
import importlib.util
import re

def normalize_package_name(name):
    # Extract base package name (remove version specifiers, etc.)
    base_name = re.sub(r'[<>=!~;].*$', '', name).strip()
    
    # Handle special cases
    mapping = {
        'opencv-contrib-python': 'cv2',
        'opencv-contrib-python-headless': 'cv2',
        'opencv-python': 'cv2',
        'opencv-python-headless': 'cv2',
        'scikit-image': 'skimage',
        'scikit-learn': 'sklearn',
        'scikit_image': 'skimage',
        'scikit_learn': 'sklearn',
        'pytorch': 'torch',
        'pillow': 'PIL',
        'Pillow': 'PIL',
    }
    
    return mapping.get(base_name, base_name)

def is_package_importable(package_name):
    try:
        module_name = normalize_package_name(package_name)
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except (ImportError, ValueError, AttributeError):
        return False

# Get list of packages to check
with open(sys.argv[1], 'r') as f:
    packages = [line.strip() for line in f if line.strip() and not line.startswith(('git+', 'http'))]

# Check which packages are missing
missing_packages = []
for pkg in packages:
    if not is_package_importable(pkg):
        missing_packages.append(pkg)

# Write missing packages to output file
with open(sys.argv[2], 'w') as f:
    for pkg in missing_packages:
        f.write(f"{pkg}\n")
EOF
        
        # Verify which packages are actually missing
        echo "${indent}Verifying package imports..."
        if ! python "$verify_script" "$combined_reqs" "/tmp/missing_packages.txt" 2>/dev/null; then
            echo "${indent}Warning: Package verification failed for $req_file, skipping verification"
            touch "/tmp/missing_packages.txt"  # Create empty file to continue
        fi
        
        # Install packages in smaller batches to avoid dependency conflicts
        if [[ -s "/tmp/missing_packages.txt" ]]; then
            echo "${indent}Installing missing packages in batches..."
            
            # Split into smaller batches of 10 packages each
            split -l 10 "/tmp/missing_packages.txt" "/tmp/pkg_batch_"
            
            # Install each batch separately
            for batch in /tmp/pkg_batch_*; do
                echo "${indent}Installing batch $(basename "$batch")..."
                # Add timeout of 60 seconds (1 minute) to pip batch installation
                if ! timeout 60s pip install --no-cache-dir --disable-pip-version-check -r "$batch" 2>/dev/null; then
                    echo "${indent}Batch installation failed or timed out after 1 minute, falling back to individual installation..."
                    while read -r pkg; do
                        echo "${indent}  Installing: $pkg"
                        pip install --no-cache-dir --disable-pip-version-check "$pkg" 2>/dev/null || echo "${indent}  Failed to install: $pkg (continuing)"
                    done < "$batch"
                fi
            done
        else
            echo "${indent}All requirements already satisfied"
        fi
        
        # Handle GitHub repositories separately
        echo "${indent}Installing GitHub repositories..."
        grep -E "git\+https?://" "$combined_reqs" | while read -r repo; do
            echo "${indent}  Installing: $repo"
            pip install --no-cache-dir --disable-pip-version-check "$repo" 2>/dev/null || echo "${indent}  Failed to install: $repo (continuing)"
        done
        
        # Clean up
        rm -f "$combined_reqs" "$verify_script" "/tmp/missing_packages.txt" "/tmp/resolve_conflicts.py" /tmp/pkg_batch_*
        
        # Re-enable set -e at the end of the function
        set -e
        echo "${indent}Completed processing: $req_file"
    }

        # Call the function with the requirements file - with error handling
    echo "Processing main ComfyUI requirements..."
    if ! process_requirements "$REPO_DIR/requirements.txt"; then
        log_error "⚠️ Failed to process main ComfyUI requirements, but continuing..."
    fi
    
    echo "Processing additional requirements..."
    if ! process_requirements "/notebooks/sd_comfy/additional_requirements.txt"; then
        log_error "⚠️ Failed to process additional requirements, but continuing..."
    fi

    # Note: All SageAttention helper functions are defined earlier in the script to avoid duplication

    # --- STEP 7: INSTALL CRITICAL PACKAGES ---
    echo ""
    echo "=================================================="
    echo "         STEP 7: INSTALL CRITICAL PACKAGES"
    echo "=================================================="
    echo ""
    
    # Execute critical packages installation
    log "🔧 Installing critical packages (commonly needed by custom nodes)..."
    install_critical_packages || log_error "Some critical packages failed to install (continuing)"
    critical_packages_status=$?
    if [[ $critical_packages_status -eq 0 ]]; then
        echo "✅ Critical packages installation completed successfully."
    else
        log_error "⚠️ Critical packages installation had issues (Status: $critical_packages_status)"
        log_error "Some custom nodes may not work properly"
    fi

    # Skip expensive force-reinstalls when pins already satisfy custom-node checks.
    ensure_comfy_custom_node_pip_stack_if_needed || log_error "⚠️ Custom-node pip stack check/repair had issues (continuing)"

    # --- STEP 8: VERIFY INSTALLATIONS ---
    echo ""
    echo "=================================================="
    echo "            STEP 8: VERIFY INSTALLATIONS"
    echo "=================================================="
    echo ""

    # Installation Success Handling (function already defined above)

    # Symlink Creation (Optional - Keep definition but commented out call in handle_successful_installation)
    create_compatibility_symlink() {
        local module_path=$1
        SITE_PACKAGES_DIR=$(python -c "import site; print(site.getsitepackages()[0])")
        if [ -d "$SITE_PACKAGES_DIR" ]; then
            cd "$SITE_PACKAGES_DIR"
            [ ! -d "sage_attention" ] && ln -sf "$module_path" "sage_attention"
            echo "Created compatibility symlink in $SITE_PACKAGES_DIR"
        else
            echo "Warning: Could not find site-packages directory for compatibility symlink"
        fi
    }

    # Dependency Installation
    install_dependencies() {
        log "Installing SageAttention dependencies..."
        pip install --no-cache-dir --disable-pip-version-check \
            "ninja>=1.11.0" \
            "packaging" # Added packaging as it's often needed by setup.py
    }

    # Repository Management
    clone_or_update_repo() {
        local sage_build_dir="$SAGEATTENTION_CACHE_DIR/src"
        if [ ! -d "$sage_build_dir/.git" ]; then
            log "Cloning SageAttention repository into $sage_build_dir..."
            git clone https://github.com/thu-ml/SageAttention.git "$sage_build_dir" || {
                log_error "Failed to clone SageAttention repository."
                return 1 # Indicate failure
            }
        else
            log "Updating SageAttention repository in $sage_build_dir..."
            (cd "$sage_build_dir" && git fetch && git pull) || {
                log_warning "Failed to update SageAttention repository, using existing code."
                # Continue even if pull fails
            }
        fi
        cd "$sage_build_dir" || return 1 # Ensure we are in the correct directory
        log "Current SageAttention commit: $(git rev-parse HEAD)"
        return 0
    }

    # Build and Installation
    build_and_install() {
        local sage_build_dir="$SAGEATTENTION_CACHE_DIR/src"
        if [[ ! -d "$sage_build_dir" ]] || ! cd "$sage_build_dir"; then
             log_error "SageAttention source directory $sage_build_dir not found or cannot cd into it."
             return 1
        fi

        log "Building SageAttention wheel in $(pwd)..."
        log "--- Verifying Environment BEFORE Build ---"
        log "CUDA_HOME=$CUDA_HOME"
        log "PATH=$PATH"
        log "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
        log "NVCC Version: $(nvcc --version || echo 'NVCC not found')"
        log "Python Version: $(python --version || echo 'python not found')"
        log "PIP Version: $(pip --version || echo 'pip not found')"
        log "-----------------------------------------"

        # Clean previous build artifacts
        rm -rf build dist *.egg-info

        # Always use standard build process first for simplicity
        standard_build # This calls 'python setup.py bdist_wheel'

        # Check if a wheel was built
        local built_wheel
        built_wheel=$(find "$sage_build_dir/dist" -name "sageattention*.whl" -print -quit)

        if [[ -n "$built_wheel" ]]; then
            # Get the CUDA version again *after* build for marker file
            local cuda_version_built_with
            cuda_version_built_with=$(nvcc --version | grep release | awk '{print $6}' | cut -c2- || echo "unknown")
            handle_built_wheel "$built_wheel" "$cuda_version_built_with"
        else
            log "❌ Failed to build SageAttention wheel. No wheel file found in dist/. Check build logs above."
            # Allow script to continue based on original logic
        fi
    }

    # Optimized Build Process (Keep definition but don't call initially)
    optimized_build() {
        log "Attempting optimized build with setup_optimized.py..."
        # Ensure the custom setup file exists
        if [[ ! -f "setup_optimized.py" ]]; then
            log_error "setup_optimized.py not found in $(pwd). Cannot perform optimized build."
            return 1
        fi
        # Use default python, ensure it's the correct one
        # Remove filtering to see all output
        python setup_optimized.py bdist_wheel
        if [[ $? -ne 0 ]]; then log_error "Optimized build command failed."; return 1; fi
        return 0
    }

    # Standard Build Process
    standard_build() {
        log "Using standard build process (setup.py)..."
        if [[ ! -f "setup.py" ]]; then
            log_error "setup.py not found in $(pwd). Cannot perform standard build."
            return 1
        fi
        # --- Explicitly use the venv Python ---
        local venv_python="$VENV_DIR/sd_comfy-env/bin/python"
        if [[ ! -x "$venv_python" ]]; then
            log_error "Virtual environment Python not found or not executable at $venv_python"
            return 1
        fi
        log "Using Python executable: $venv_python"
        # Log sys.path right before build
        log "Checking sys.path for $venv_python before build..."
        "$venv_python" -c "import sys; import pprint; print('--- sys.path ---'); pprint.pprint(sys.path); print('--- end sys.path ---')" || log_error "Failed to check sys.path"

        # Run the build command with the explicit Python path
        log "Running build command: $venv_python setup.py bdist_wheel"
        # Remove filtering to see all output
        "$venv_python" setup.py bdist_wheel
        local build_status=$?
        if [[ $build_status -ne 0 ]]; then 
            log_error "Standard build command failed with status $build_status."
            return 1 # Indicate failure
        fi
        log "Standard build command finished successfully."
        return 0
    }

    # Built Wheel Handling
    handle_built_wheel() {
        local wheel_path="$1"
        local cuda_version_built_with="$2" # Expecting version like 12.8
        log "Found built wheel: $wheel_path"
        
        # Ensure WHEEL_CACHE_DIR is set and exists
        if [[ -z "$WHEEL_CACHE_DIR" ]]; then
            log_error "WHEEL_CACHE_DIR is not set in handle_built_wheel!"
            export WHEEL_CACHE_DIR="/storage/.wheel_cache"
            mkdir -p "$WHEEL_CACHE_DIR"
        fi
        
        # Just copy the wheel to the cache directory without renaming.
        cp "$wheel_path" "$WHEEL_CACHE_DIR/"
        log "Cached built wheel to $WHEEL_CACHE_DIR/$(basename "$wheel_path")"

        # Attempt to install the built wheel
        log "Installing newly built wheel: $wheel_path"
        # Use --force-reinstall to ensure clean install over any previous attempts
        if pip install --force-reinstall --no-cache-dir --disable-pip-version-check "$wheel_path"; then
            log "Verifying installation via import..."
            pushd /tmp > /dev/null # Change to neutral directory
            local import_output
            local import_status
            import_output=$(python -c "import sageattention; print('SageAttention imported successfully')" 2>&1)
            import_status=$?
            popd > /dev/null # Return to original directory

            if [ $import_status -eq 0 ]; then
                log "✅ Import verified after installing built wheel."
                handle_successful_installation
            else
                log_error "❌ SageAttention installed from built wheel but failed import check."
                log_error "Python import error output:"
                log_error "-----------------------------------------"
                echo "$import_output" | while IFS= read -r line; do log_error "$line"; done
                log_error "-----------------------------------------"
                log_warning "Continuing script, but SageAttention might not work."
            fi
        else
            log_error "❌ Failed to install SageAttention wheel from $wheel_path. Continuing script..."
        fi
    }

    # Execute installation
    
    # Note: Requirements processing already completed in STEP 5 above
    # Note: SageAttention is installed earlier in the process

    # Final checks and marker file
    touch /tmp/sd_comfy.prepared
    echo "Stable Diffusion Comfy setup complete."
else
    echo "Stable Diffusion Comfy already prepared. Skipping setup."
    
    # Check ComfyUI version and update status even when skipping installation
    echo ""
    echo "=================================================="
    echo "           CHECKING COMFYUI STATUS"
    echo "=================================================="
    echo ""
    
    # Activate venv even if skipping setup
    if [ -f "$VENV_DIR/sd_comfy-env/bin/activate" ]; then
        source "$VENV_DIR/sd_comfy-env/bin/activate"
    else
        log_error "Virtual environment not found at $VENV_DIR/sd_comfy-env"
        exit 1
    fi

    promote_sam2_wheel || true

    ensure_comfy_custom_node_pip_stack_if_needed || log_error "⚠️ Custom-node pip stack check/repair had issues (continuing)"
        
        # Check current ComfyUI version
        if [ -d "$REPO_DIR/.git" ]; then
            cd "$REPO_DIR"
            echo "📋 Checking ComfyUI version information..."
            
            # Get current commit hash and branch
            current_commit=$(git rev-parse HEAD 2>/dev/null || echo "Unknown")
            current_branch=$(git branch --show-current 2>/dev/null || echo "Unknown")
            
            # Get current commit date
            current_date=$(git log -1 --format="%cd" --date=short 2>/dev/null || echo "Unknown")
            
            echo "📍 Current ComfyUI Status:"
            echo "   Branch: $current_branch"
            echo "   Commit: $current_commit"
            echo "   Date: $current_date"
            
            # Check if there are updates available (see install path: avoid bare git under set -e)
            echo ""
            echo "🔄 Checking for updates..."
            fetch_ok=0
            if [ -n "$current_branch" ] && [ "$current_branch" != "Unknown" ]; then
                git fetch origin "$current_branch" 2>&1 && fetch_ok=1
            else
                git fetch origin 2>&1 && fetch_ok=1
            fi
            if [ "$fetch_ok" -ne 1 ]; then
                echo "⚠️  git fetch failed (network, auth, rate limit, conflicting remote refs, or repo issue). Remote comparison may be incomplete."
            fi
            
            # Compare local vs remote
            local_commit=$(git rev-parse HEAD 2>/dev/null) || local_commit=""
            remote_commit=$(git rev-parse "origin/${current_branch}" 2>/dev/null) || remote_commit=""
            
            if [ -z "$local_commit" ] || [ -z "$remote_commit" ]; then
                echo "⚠️  Could not compare local vs remote (missing refs after fetch or unknown branch)."
            elif [ "$local_commit" = "$remote_commit" ]; then
                echo "✅ ComfyUI is up to date with the latest version!"
            else
                echo "⚠️  ComfyUI has updates available!"
                echo "   Local:  $local_commit"
                echo "   Remote: $remote_commit"
                echo ""
                echo "🔄 Updating ComfyUI (git pull --autostash)..."
                if git pull --autostash origin "$current_branch"; then
                    echo "✅ ComfyUI successfully updated to latest version!"
                else
                    echo "❌ Failed to update ComfyUI. Please check the repository status."
                fi
            fi

            # Re-apply managed symlinks after any update attempt
            apply_comfy_symlinks "$current_dir/comfy_symlinks.txt" || log_error "⚠️ Post-pull symlink apply had issues (continuing)"
            if [[ -f /notebooks/logs/patch_minimax_flash_decode.py ]]; then
                python3 /notebooks/logs/patch_minimax_flash_decode.py >/dev/null 2>&1 || true
            fi
            
            # Show recent commits
            echo ""
            echo "📝 Recent commits:"
            git log --oneline -5 2>/dev/null | sed 's/^/   /' || echo "   Unable to show recent commits"
            
        else
            echo "⚠️  ComfyUI repository not found or not a git repository"
        fi
fi

log "Finished Preparing Environment for Stable Diffusion Comfy"
enforce_boot_cache_budget || true
touch /tmp/sd_comfy.prepared

echo ""
echo "=================================================="
echo "           ENVIRONMENT SETUP COMPLETE!"
echo "=================================================="
echo ""

#######################################
# STEP 9: START COMFYUI
#######################################
if [[ -z "$INSTALL_ONLY" ]]; then
  echo ""
  echo "=================================================="
  echo "             STEP 9: START COMFYUI"
  echo "=================================================="
  echo ""

  # Last-chance deps (decord for comfyui-rmbg SAM3, etc.) after all pip stack repair above
  if [ -f "$VENV_DIR/sd_comfy-env/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/sd_comfy-env/bin/activate"
    fix_custom_node_import_errors || log_error "Some custom node import fixes failed (continuing)"
  fi

  # Impact Pack probes sam2 at import time — must be installed BEFORE ComfyUI starts
  # (background install after start leaves SAM2 permanently unavailable until restart).
  echo ""
  echo "=================================================="
  echo "        STEP 8.5: INSTALL SAM2 (BEFORE COMFYUI)"
  echo "=================================================="
  echo ""
  set +e
  disable_err_trap
  install_sam2_optimized
  sam2_prestart_status=$?
  set -e
  enable_err_trap
  if [[ $sam2_prestart_status -eq 0 ]]; then
    log "✅ SAM2 ready for Impact Pack before ComfyUI start"
  else
    log_error "⚠️ SAM2 install failed before ComfyUI start (FaceDetailer SAM2 will be unavailable)"
  fi
  
  # Kill any existing ComfyUI processes before starting
  echo "🛑 Stopping any existing ComfyUI processes..."
  log "Checking for existing ComfyUI processes..."
  
  # Function to kill ComfyUI processes
  kill_existing_comfyui() {
    local killed_any=false
    
    # Method 1: Kill using PID file if it exists
    if [[ -f "/tmp/sd_comfy.pid" ]]; then
      local pid=$(cat /tmp/sd_comfy.pid 2>/dev/null)
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
      rm -f /tmp/sd_comfy.pid
    fi
    
    # Method 2: Kill all Python processes running ComfyUI on the port
    local comfyui_pids=$(pgrep -f "python.*main\.py.*--port.*${SD_COMFY_PORT:-7005}" 2>/dev/null || true)
    if [[ -n "$comfyui_pids" ]]; then
      log "Found ComfyUI Python processes: $comfyui_pids"
      for pid in $comfyui_pids; do
        if kill -0 "$pid" 2>/dev/null; then
          log "Killing ComfyUI Python process: $pid"
          kill -TERM "$pid" 2>/dev/null || true
          killed_any=true
        fi
      done
      sleep 1
      # Force kill any remaining
      for pid in $comfyui_pids; do
        if kill -0 "$pid" 2>/dev/null; then
          log "Force killing ComfyUI Python process: $pid"
          kill -9 "$pid" 2>/dev/null || true
        fi
      done
    fi
    
    # Method 3: Kill processes using the port (fallback)
    if command -v lsof &>/dev/null; then
      local port_pids=$(lsof -ti:${SD_COMFY_PORT:-7005} 2>/dev/null || true)
      if [[ -n "$port_pids" ]]; then
        log "Found processes using port ${SD_COMFY_PORT:-7005}: $port_pids"
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
      log "✅ Existing ComfyUI processes stopped"
      sleep 2  # Give processes time to fully terminate
    else
      log "No existing ComfyUI processes found"
    fi
  }
  
  # Execute the cleanup
  kill_existing_comfyui
  
  echo "### Starting Stable Diffusion Comfy ###"
  log "Starting Stable Diffusion Comfy"
  cd "$REPO_DIR"
  
  # Rotate ComfyUI log file instead of deleting it
  if [[ -f "$LOG_DIR/sd_comfy.log" ]]; then
    # Create timestamp for old log
    timestamp=$(date +"%Y%m%d_%H%M%S")
    mv "$LOG_DIR/sd_comfy.log" "$LOG_DIR/sd_comfy_${timestamp}.log"
    echo "Previous ComfyUI log archived as: sd_comfy_${timestamp}.log"
    
    # Keep only the last 5 rotated logs to save space
    ls -t "$LOG_DIR"/sd_comfy_*.log 2>/dev/null | tail -n +6 | xargs -r rm
  fi
  
  # A4000-specific VRAM optimization settings (16GB)
  export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:4096,garbage_collection_threshold:0.8"
  
  # --- ENSURE CORRECT TORCH VERSIONS AT RUNTIME ---
  # Skip redundant check if we just completed a fresh installation.
  if [[ -f "/tmp/pytorch_ecosystem_fresh_install" ]]; then
      echo "Skipping PyTorch version check - fresh installation completed successfully"
      rm -f "/tmp/pytorch_ecosystem_fresh_install"  # Clean up marker
  else
      echo "Verifying PyTorch ecosystem versions before launch..."
      fix_torch_versions # This will now just check unless versions are wrong
      fix_torch_status=$? 

      if [[ $fix_torch_status -ne 0 ]]; then
          log_error "fix_torch_versions function failed during pre-launch check with status $fix_torch_status."
          exit 1
      fi
  fi

  # Launch ComfyUI with A4000-optimized parameters using SageAttention
  echo "NOTE: A pip dependency warning regarding xformers and torch versions may appear below."
  echo "This is expected with the current package versions and can be safely ignored."
  
  # Debug: Check if custom nodes should be disabled (set DISABLE_CUSTOM_NODES=1 to disable)
  CUSTOM_NODES_FLAG=""
  if [[ -n "${DISABLE_CUSTOM_NODES}" ]]; then
    CUSTOM_NODES_FLAG="--disable-all-custom-nodes"
    echo "⚠️  Custom nodes DISABLED for debugging"
  fi
  
  
  # Frontend: pin to ComfyUI core/PyPI stable (currently 1.53.6).
  # Avoid @latest / GitHub-only lines (e.g. 1.55.x) — those are daily builds.
  # Override with COMFY_FRONTEND_VERSION=... or USE_LEGACY_FRONTEND=1 if needed.
  COMFY_FRONTEND_VERSION="${COMFY_FRONTEND_VERSION:-1.53.6}"
  FRONTEND_FLAG="--front-end-version Comfy-Org/ComfyUI_frontend@${COMFY_FRONTEND_VERSION}"
  echo "📦 Using frontend version: ${COMFY_FRONTEND_VERSION}"
  
  if [[ -n "${USE_LEGACY_FRONTEND}" ]]; then
    FRONTEND_FLAG="--front-end-version Comfy-Org/ComfyUI_legacy_frontend@latest"
    echo "⚠️  Overriding to LEGACY frontend"
  fi
  
  # MiniMax Music 3: comfy_kitchen.flash_attention_decode needs a newer NVIDIA
  # driver than Paperspace Free (550 / CUDA 12.4) provides for torch cu128.
  # Forces Comfy's existing plain-KV fallback (Comfy-Org/ComfyUI#15605/#15607).
  # Set COMFY_DISABLE_KITCHEN_FLASH_DECODE=0 to re-enable on machines with driver >= ~570.
  export COMFY_DISABLE_KITCHEN_FLASH_DECODE="${COMFY_DISABLE_KITCHEN_FLASH_DECODE:-1}"
  if [[ "${COMFY_DISABLE_KITCHEN_FLASH_DECODE}" != "0" ]]; then
    echo "⚠️  Kitchen flash_attention_decode DISABLED (MiniMax Music3 / old driver workaround)"
    # Re-apply after ComfyUI git pull overwrites llama.py
    if [[ -f /notebooks/logs/patch_minimax_flash_decode.py ]]; then
      python3 /notebooks/logs/patch_minimax_flash_decode.py >/dev/null 2>&1 || true
    fi
  fi

  COMFYUI_CMD="python main.py \
    --port $SD_COMFY_PORT \
    --dont-print-server \
    --bf16-vae \
    --cache-lru 5 \
    --reserve-vram 0.5 \
    --enable-compress-response-body \
    --cuda-malloc \
    --preview-method latent2rgb \
    --disable-api-nodes \
    --disable-cuda-graphs \
    $CUSTOM_NODES_FLAG \
    $FRONTEND_FLAG"
  PYTHONUNBUFFERED=1 service_loop "$COMFYUI_CMD" > $LOG_DIR/sd_comfy.log 2>&1 &
  echo $! > /tmp/sd_comfy.pid
  
  # Wait a moment for ComfyUI to start
  sleep 3
  log "✅ ComfyUI started successfully! You can now access it at http://localhost:$SD_COMFY_PORT"

  # Start deferred entry scripts (default: image_browser) alongside ComfyUI — after setup, non-blocking.
  if [[ "${RUN_SCRIPT:-}" == *"image_browser"* ]] && [[ "${DEFER_SCRIPTS:-image_browser}" == *"image_browser"* ]]; then
    ib_dir="$current_dir/../image_browser"
    if [[ -d "$ib_dir" ]]; then
      mkdir -p "$LOG_DIR"
      log "🖼️ Starting image_browser alongside ComfyUI (setup+launch in background)..."
      nohup bash -c "cd \"$ib_dir\" && bash control.sh reload" >> "$LOG_DIR/image_browser_entry.log" 2>&1 &
      echo $! > /tmp/image_browser_entry.pid
      log "📋 image_browser log: tail -f $LOG_DIR/image_browser_entry.log"
    else
      log "⚠️ image_browser folder not found at $ib_dir — skipping"
    fi
  fi
  
  #######################################
  # STEP 9.1: START OLLAMA (AFTER COMFYUI) - DISABLED
  #######################################
  # Ollama install/start commented out (e.g. requires zstd; enable if needed)
  # echo ""
  # echo "=================================================="
  # echo "        STEP 9.1: START OLLAMA (AFTER COMFYUI)"
  # echo "=================================================="
  # echo ""
  # check_cuda_for_ollama() { ... }
  # check_cuda_for_ollama
  # if ! command -v ollama &> /dev/null; then curl -fsSL https://ollama.com/install.sh | sh; fi
  # ollama serve > $LOG_DIR/ollama.log 2>&1 & ...

  #######################################
  # STEP 9.4: INSTALL LORA EASY TRAINING SCRIPTS
#######################################
echo ""
echo "=================================================="
echo "        STEP 9.4: INSTALL LORA TRAINING"
echo "=================================================="
echo ""

install_lora_training() {
    echo "=================================================="
    echo "   STEP 9.4: INSTALL LORA EASY TRAINING SCRIPTS"
    echo "=================================================="
    
    local lora_training_dir="/tmp/lora-training"
    local comfy_venv_dir="$VENV_DIR/sd_comfy-env"
    local log_file="$LOG_DIR/lora_training.log"
    local config_dir="/storage/lora_training/config"
    local datasets_dir="/storage/lora_training/datasets"

    # 1. CLONE REPOSITORY
    log "1. Checking Repository..."
    if [[ ! -d "$lora_training_dir" ]]; then
        log "   Cloning https://github.com/derrian-distro/LoRA_Easy_Training_Scripts.git..."
        git clone --recurse-submodules https://github.com/derrian-distro/LoRA_Easy_Training_Scripts.git "$lora_training_dir"
    else
        log "   Updating existing repository..."
        cd "$lora_training_dir" && git pull && git submodule update --init --recursive && cd - > /dev/null
    fi

    # 1b. kohya sd-scripts on main (minimal): needed for Anima LoRA (anima_train_network.py, networks/lora_anima.py)
    local sd_scripts_dir="$lora_training_dir/backend/sd_scripts"
    if [[ -d "$sd_scripts_dir" ]]; then
        log "   Syncing sd-scripts to origin/main..."
        ( cd "$sd_scripts_dir" && git fetch origin && git checkout main && git pull ) >>"$log_file" 2>&1 || log "   ⚠️ sd-scripts sync warning — see $log_file"
        for f in anima_train_network.py networks/lora_anima.py docs/anima_train_network.md; do
            if [[ -e "$sd_scripts_dir/$f" ]]; then
                log "   ✅ sd-scripts $f"
            else
                log "   ⚠️ sd-scripts missing $f — Anima training TOML needs newer kohya-ss/sd-scripts (checkout main + pull in submodule)"
            fi
        done
    fi

    # 2. INSTALL PYTHON DEPENDENCIES
    log "2. Installing Python Dependencies (Comfy env: $comfy_venv_dir)..."
    source "$comfy_venv_dir/bin/activate"
    
    # Upgrade pip/setuptools/wheel to avoid "egg_info" / metadata-generation-failed (setuptools 58+ removed egg_info)
    # Keep setuptools <82 so pkg_resources remains available for SUPIR / Lightning.
    log "   Upgrading pip, setuptools, wheel..."
    pip install --no-cache-dir -q --upgrade pip "setuptools${SETUPTOOLS_PIN}" wheel 2>/dev/null || true
    
    # Install sd-scripts dependencies (backend/sd_scripts submodule)
    if [[ -d "$lora_training_dir/backend/sd_scripts" ]]; then
        cd "$lora_training_dir/backend/sd_scripts"
        if [[ -f "requirements.txt" ]]; then
            # Filter to avoid breaking ComfyUI (same strategy as AI Toolkit)
            # Keep Comfy's versions of: torch, torchvision, torchaudio, xformers, numpy, scipy, opencv, pillow
            grep -vE "^torch[^a-z]|^torchvision|^torchaudio|^xformers|^numpy|^scipy|^opencv-python|^Pillow|^setuptools|^python-multipart" "requirements.txt" > "/tmp/lora_backend_reqs.txt"
            log "   Installing sd-scripts requirements (filtered)..."
            if ! pip install --no-cache-dir -q -r "/tmp/lora_backend_reqs.txt" 2>/dev/null; then
                log "   Retrying with legacy resolver..."
                pip install --no-cache-dir -q -r "/tmp/lora_backend_reqs.txt" --use-deprecated=legacy-resolver 2>&1 | grep -v "already satisfied" || true
            fi
        fi
        cd - > /dev/null
    else
        log "   ⚠️ backend/sd_scripts directory not found, skipping sd-scripts requirements"
    fi
    
    # LyCORIS: use sd-scripts built-in LoRA (network_module='networks.lora')
    log "   LyCORIS support: using sd-scripts built-in (network_module='networks.lora')"
    
    # Optional optimizers (same env as ComfyUI: cu128). bitsandbytes needs BNB_CUDA_VERSION=122 at install and in train-lora (wheels only up to 12.2).
    log "   Installing optional optimizers (bitsandbytes, lion-pytorch, prodigyopt)..."
    BNB_CUDA_VERSION=122 pip install --no-cache-dir -q bitsandbytes lion-pytorch prodigyopt 2>>"$log_file" || true

    # 3. CREATE DIRECTORIES
    log "3. Creating directories..."
    mkdir -p "$config_dir" "$datasets_dir"
    mkdir -p "/tmp/stable-diffusion-models/lora"  # Output directory for trained LoRAs
    
    # 4. CONFIGURE ACCELERATE
    log "4. Configuring Accelerate..."
    mkdir -p ~/.cache/huggingface/accelerate
    cat > ~/.cache/huggingface/accelerate/default_config.yaml << EOF
compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: 'NO'
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
    
    log "   ✅ Accelerate configured"

    # 5. CREATE TRAINING SCRIPT WRAPPER
    log "5. Creating training script wrapper..."
    cat > "$lora_training_dir/train_from_config.sh" << 'EOF'
#!/bin/bash
# LoRA Training Script Wrapper
# Usage: ./train_from_config.sh <config.toml>

if [[ -z "$1" ]]; then
    echo "Usage: $0 <config.toml>"
    exit 1
fi

CONFIG_PATH="$1"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

# Activate environment (same as ComfyUI: sd_comfy-env, cu128)
source "${VENV_DIR:-/tmp}/sd_comfy-env/bin/activate"
export BNB_CUDA_VERSION=122

# Run training (sd_scripts is in backend/sd_scripts subdirectory)
cd /tmp/lora-training/backend/sd_scripts
accelerate launch --config_file ~/.cache/huggingface/accelerate/default_config.yaml \
    train_network.py \
    --config_file="$CONFIG_PATH" \
    --lowram
EOF
    chmod +x "$lora_training_dir/train_from_config.sh"
    
    # Create symlink for easy access
    ln -sf "$lora_training_dir/train_from_config.sh" /usr/local/bin/train-lora 2>/dev/null || true

    log "✅ LoRA Easy Training Scripts backend installed successfully"
    touch /tmp/lora_training.prepared
    
    # Success summary
    echo ""
    echo "=================================================="
    echo "   LORA TRAINING BACKEND INSTALL SUMMARY"
    echo "=================================================="
    echo "   Python env:      Comfy venv ($comfy_venv_dir)"
    echo "   Repo:            $lora_training_dir"
    echo "   Config dir:      $config_dir"
    echo "   Datasets dir:    $datasets_dir"
    echo "   Output dir:      /tmp/stable-diffusion-models/lora"
    echo "   Training script: train-lora <config.toml>"
    echo "=================================================="
    echo ""
    echo "   Manual Training Commands:"
    echo "   # Create test config at: $config_dir/test.toml"
    echo "   # Then run:"
    echo "   train-lora $config_dir/test.toml"
    echo ""
    echo "   # Or full command:"
    echo "   source ${VENV_DIR:-/tmp}/sd_comfy-env/bin/activate"
    echo "   cd $lora_training_dir/backend/sd_scripts"
    echo "   accelerate launch train_network.py --config_file=/path/to/config.toml"
    echo "=================================================="
    echo ""
}

# Execute LoRA training setup off the critical path by default (~30s+).
# Set INSTALL_LORA_ON_START=1 to install synchronously before "Comfy started".
if [[ "${INSTALL_LORA_ON_START}" == "1" ]]; then
  if [[ ! -f "/tmp/lora_training.prepared" ]] || [[ -n "$REINSTALL_LORA_TRAINING" ]]; then
      install_lora_training
  else
      log "✅ LoRA Easy Training Scripts already installed (backend only; use train-lora <config.toml>)"
  fi
else
  log "⏭️ LoRA training install deferred to background (INSTALL_LORA_ON_START=0)"
  (
    if [[ ! -f "/tmp/lora_training.prepared" ]] || [[ -n "$REINSTALL_LORA_TRAINING" ]]; then
      install_lora_training >"$LOG_DIR/lora_training_bg.log" 2>&1
    fi
  ) &
fi

  #######################################
  # STEP 9.2: INSTALL TENSORFLOW (BACKGROUND)
  #######################################
  echo ""
  echo "=================================================="
  echo "   STEP 9.2: INSTALL TENSORFLOW (BACKGROUND)"
  echo "=================================================="
  echo ""
  echo "📦 Installing TensorFlow in background (using /tmp venv)..."
  log "Starting TensorFlow installation in background..."
  
  # Create a background script to install TensorFlow in /tmp
  cat > /tmp/install_tensorflow.sh << 'TENSORFLOW_SCRIPT'
#!/bin/bash
set +e  # Don't exit on errors

# Create temporary venv for TensorFlow
if [ ! -d "/tmp/tensorflow-env" ]; then
  echo "Creating temporary TensorFlow environment in /tmp..."
  /storage/python_versions/python3.10/bin/python3.10 -m venv /tmp/tensorflow-env || exit 1
fi

# Activate and install
source /tmp/tensorflow-env/bin/activate || exit 1
echo "Installing TensorFlow in /tmp environment..."
pip install --quiet --no-cache-dir "tensorflow>=2.8.0,<2.19.0" > /tmp/tensorflow_install.log 2>&1

if [ $? -eq 0 ]; then
  echo "✅ TensorFlow installed successfully in /tmp/tensorflow-env" >> /tmp/tensorflow_install.log
else
  echo "⚠️ TensorFlow installation failed (optional dependency)" >> /tmp/tensorflow_install.log
fi

# Create activation helper script
cat > /tmp/activate_tensorflow.sh << 'HELPER'
#!/bin/bash
# Helper script to activate TensorFlow environment
source /tmp/tensorflow-env/bin/activate
echo "TensorFlow environment activated from /tmp"
echo "Python: $(which python)"
HELPER
chmod +x /tmp/activate_tensorflow.sh

TENSORFLOW_SCRIPT

  # Make script executable and run in background
  chmod +x /tmp/install_tensorflow.sh
  /tmp/install_tensorflow.sh > /tmp/tensorflow_bg.log 2>&1 &
  echo $! > /tmp/tensorflow_install.pid
  
  log "📋 TensorFlow installation started in background (PID: $(cat /tmp/tensorflow_install.pid))"
  log "📋 Check installation progress: tail -f /tmp/tensorflow_install.log"
  log "💡 TensorFlow will be available in /tmp/tensorflow-env (activate with: source /tmp/activate_tensorflow.sh)"
  
  #######################################
  # STEP 9.3: SAM2 (already done before ComfyUI start)
  #######################################
  echo ""
  echo "=================================================="
  echo "        STEP 9.3: SAM2 STATUS"
  echo "=================================================="
  echo ""
  if (cd /tmp && python -c "from sam2.build_sam import build_sam2") &>/dev/null; then
    log "✅ SAM2 already installed (installed before ComfyUI start)"
  else
    log "📦 SAM2 missing — installing in background (restart ComfyUI after it finishes for Impact Pack)"
    (
      set +e
      install_sam2_optimized > /tmp/sam2_install.log 2>&1
      if [ $? -eq 0 ]; then
        echo "✅ SAM2 installed successfully — restart ComfyUI to enable Impact Pack SAM2" >> /tmp/sam2_install.log
      else
        echo "⚠️ SAM2 installation failed (optional dependency)" >> /tmp/sam2_install.log
      fi
    ) &
    echo $! > /tmp/sam2_install.pid
    log "📋 SAM2 installation started in background (PID: $(cat /tmp/sam2_install.pid))"
  fi
fi

#######################################
# STEP 10: FINAL SETUP COMPLETION
#######################################
echo ""
echo "=================================================="
echo "           STEP 10: FINAL SETUP COMPLETION"
echo "=================================================="
echo ""
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

#######################################
# STEP 11: START KEEPALIVE PROCESS
#######################################
echo ""
echo "=================================================="
echo "        STEP 11: START KEEPALIVE PROCESS"
echo "=================================================="
echo ""

# Kill any existing keepalive process
if [[ -f "/tmp/keepalive.pid" ]]; then
  pid=$(cat /tmp/keepalive.pid 2>/dev/null)
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    log "🛑 Stopping existing keepalive process (PID: $pid)..."
    kill -TERM "$pid" 2>/dev/null || true
    sleep 1
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f /tmp/keepalive.pid
fi

# Start keepalive process in background with nohup to ensure it survives script exit
log "🔄 Starting Paperspace keepalive process..."
nohup bash -c "while true; do touch /tmp/.keepalive_tmp && rm -f /tmp/.keepalive_tmp; sleep 30; done" > /tmp/keepalive.log 2>&1 &
KEEPALIVE_PID=$!
echo $KEEPALIVE_PID > /tmp/keepalive.pid
disown $KEEPALIVE_PID 2>/dev/null || true

# Verify it started
sleep 1
if kill -0 $KEEPALIVE_PID 2>/dev/null; then
  log "✅ Keepalive process started (PID: $KEEPALIVE_PID)"
  log "💡 This process will prevent Paperspace notebook from shutting down due to inactivity"
  log "📋 Keepalive logs: tail -f /tmp/keepalive.log"
else
  log_error "❌ Failed to start keepalive process"
fi

echo ""
echo "=================================================="
echo "           SCRIPT EXECUTION COMPLETE!"
echo "=================================================="
echo ""
echo ""

