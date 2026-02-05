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

# Default to persistent locations when not provided in .env
VENV_DIR=${VENV_DIR:-/storage/.venvs}
if [[ "$VENV_DIR" != "/storage"* ]]; then
    VENV_DIR="/storage/.venvs"
fi
export VENV_DIR
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/storage/.pip_cache}
mkdir -p "$VENV_DIR" "$PIP_CACHE_DIR"

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

# Create symlinks for model directories
echo "Creating model directory symlinks..."
cd "/storage/stable-diffusion-comfy/models/" && rm -rf diffusion_models && ln -s /tmp/stable-diffusion-models/sd diffusion_models
cd "/storage/stable-diffusion-comfy/models/" && rm -rf text_encoders && ln -s /tmp/stable-diffusion-models/lora text_encoders
cd "/storage/stable-diffusion-comfy/models/" && rm -rf sams && ln -s /tmp/stable-diffusion-models/upscaler sams
cd "/storage/stable-diffusion-comfy/models/" && rm -rf clip_vision && ln -s /tmp/stable-diffusion-models/upscaler clip_vision
cd "/storage/stable-diffusion-comfy/custom_nodes/comfyui_controlnet_aux/ckpts" && rm -rf lllyasviel && ln -s /tmp/stable-diffusion-models/controlnet lllyasviel
cd "/storage/stable-diffusion-comfy/models/" && rm -rf ipadapter && ln -s /tmp/stable-diffusion-models/upscaler ipadapter
cd "/storage/stable-diffusion-comfy/models/" && rm -rf clip_vision && ln -s /tmp/stable-diffusion-models/upscaler clip_vision
cd "/storage/stable-diffusion-comfy/models/" && rm -rf inpaint && ln -s /tmp/stable-diffusion-models/vae inpaint
cd "/storage/stable-diffusion-comfy/models/" && rm -rf RMBG && ln -s /tmp/stable-diffusion-models/controlnet RMBG

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
  MODULES=("requests" "gdown" "bs4" "python-dotenv")
  for module in "${MODULES[@]}"; do
    if ! pip show $module >/dev/null 2>&1; then
      pip install --quiet --no-cache-dir $module 2>/dev/null || log_error "Failed to install $module"
    fi
  done
  
  log "✅ Model download dependencies ready"
  log "Starting Model Download for Stable Diffusion Comfy in background..."
  log "💡 Models will download in background while the rest of the setup continues!"
  log "💡 You can start using ComfyUI as soon as it starts, even if models are still downloading!"
  
  # Start model download in background (now it's 99% just aria2 downloads)
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

# Function to install critical packages that are commonly needed by custom nodes
install_critical_packages() {
    log "📦 Installing critical packages for custom nodes..."
    
    local critical_packages=(
        "blend_modes" "deepdiff" "rembg" "webcolors" "ultralytics" "inflect" "soxr" "groundingdino-py" 
        "insightface" "opencv-python" "opencv-contrib-python" "facexlib" "onnxruntime" "timm" 
        "segment-anything" "scikit-image" "piexif" "transformers" "opencv-python-headless" 
        "scipy>=1.11.4" "numpy" "dill" "matplotlib" "oss2" "gguf" "diffusers" 
        "huggingface_hub>=0.34.0" "pytorch_lightning" "sounddevice" "av>=12.0.0,<14.0.0" "accelerate" "pyOpenSSL"
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
    
    # Setup wheel cache for faster future installs
    local WHEEL_CACHE="/storage/.critical_packages_wheels"
    mkdir -p "$WHEEL_CACHE"
    
    # Install all missing packages in ONE batch command (much faster!)
    log "📦 Installing all missing packages in batch (faster than one-by-one)..."
    local start_time=$(date +%s)
    
    # Batch install with pip (parallel downloads, single dependency resolution)
    # Try to use cached wheels first, then download if needed
    if pip install --quiet --find-links="$WHEEL_CACHE" --no-cache-dir "${missing_array[@]}" 2>/dev/null; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log "✅ Successfully installed $missing_count packages in ${duration} seconds"
        if [[ $missing_count -gt 0 ]]; then
            local avg=$((duration / missing_count))
            log "📊 Average: ${avg} seconds per package (batch mode)"
        fi
        
        # Cache wheels for future instant installs
        log "💾 Caching wheels to storage for future instant installs..."
        pip download --no-deps --dest "$WHEEL_CACHE" "${missing_array[@]}" 2>/dev/null || true
        
        rm -f /tmp/check_packages.py
        return 0
    else
        log_error "❌ Batch installation failed, falling back to individual installation..."
        
        # Fallback: Install individually (slower but more reliable)
        local installed_count=0
        local failed_count=0
        
        for pkg in "${missing_array[@]}"; do
            log "📦 Installing: $pkg"
            if pip install --quiet --no-cache-dir "$pkg" 2>/dev/null; then
                log "✅ Successfully installed: $pkg"
                # Cache this wheel too
                pip download --no-deps --dest "$WHEEL_CACHE" "$pkg" 2>/dev/null || true
                ((installed_count++))
            else
                log_error "❌ Failed to install: $pkg"
                ((failed_count++))
            fi
        done
        
        log "📊 Individual install: $installed_count installed, $failed_count failed"
        rm -f /tmp/check_packages.py
        return $failed_count
    fi
}

# SAM2 Installation Process (with wheel caching like SageAttention)
install_sam2_optimized() {
    log "Verifying SAM2 installation..."
    
    # Setup environment
    setup_cuda_env
    
    # First, just try to import it. If it works, we're done.
    if python -c "import sam2" &>/dev/null; then
        log "✅ SAM2 is already installed and importable."
        return 0
    fi

    log "SAM2 not found. Proceeding with installation..."
    
    # Proceed with full installation from source
    install_sam2_dependencies
    if clone_or_update_sam2_repo; then
         build_and_install_sam2
    else
         log_error "Failed to clone or update SAM2 repository. Skipping build."
         return 1
    fi

    # Final check after building from source
    if python -c "import sam2" &>/dev/null; then
        log "✅ SAM2 successfully built and installed."
        return 0
    else
        log_error "❌ SAM2 installation verification failed."
        return 1
    fi
}

install_sam2_dependencies() {
    log "Installing SAM2 dependencies..."
    pip install --no-cache-dir --disable-pip-version-check \
        "torch>=1.9.0" "torchvision>=0.10.0" "opencv-python" \
        "pillow" "numpy" "scipy" "matplotlib" "scikit-image" \
        "timm" "transformers" "huggingface_hub" \
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

    log "Building SAM2 wheel in $(pwd)..."
    log "--- Verifying Environment BEFORE Build ---"
    log "CUDA_HOME=$CUDA_HOME"
    log "PATH=$PATH"
    log "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    log "NVCC Version: $(nvcc --version 2>/dev/null || echo 'NVCC not found')"
    log "Python Version: $(python --version || echo 'python not found')"
    log "-----------------------------------------"
    
    rm -rf build dist *.egg-info

    local venv_python="$VENV_DIR/sd_comfy-env/bin/python"
    if [[ ! -x "$venv_python" ]]; then
        log_error "Virtual environment Python not found or not executable at $venv_python"
        return 1
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
        return 1
    fi

    local built_wheel
    built_wheel=$(find "$sam2_build_dir/dist" -name "sam2*.whl" -print -quit)

    if [[ -n "$built_wheel" ]]; then
        log "Found built wheel: $built_wheel"
        log "Installing newly built wheel: $built_wheel"
        if pip install --force-reinstall --no-cache-dir --disable-pip-version-check "$built_wheel"; then
            log "✅ SAM2 wheel installed successfully"
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
    
    echo "Clearing pip cache..."
    pip cache purge || true
    
    # Clear Python cache to prevent import issues
    echo "Clearing Python bytecode cache..."
    find "$site_packages_dir" -name "*.pyc" -delete 2>/dev/null || true
    find "$site_packages_dir" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    
    echo "Deep cleanup completed."
}

# Function to install PyTorch core packages
install_torch_core() {
    echo "Installing PyTorch core packages (torch, torchvision, torchaudio)..."
    # Use --no-cache-dir and --ignore-installed for maximum safety against conflicts.
    local install_cmd="pip install --no-cache-dir --ignore-installed torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --extra-index-url ${TORCH_INDEX_URL}"
    
    log "Running core install command: $install_cmd"
    
    if $install_cmd; then
        log "PyTorch core packages installation command finished successfully."
        # Quick verification
        python -c "import torch; print(f'Core install OK: Torch {torch.__version__} imported successfully.')" || return 1
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
    
    # Use your proven installation method from main.sh
    pip install --no-cache-dir --disable-pip-version-check --no-deps --quiet \
        xformers==${XFORMERS_VERSION} --extra-index-url "https://download.pytorch.org/whl/cu128" 2>/dev/null || \
    pip install --no-cache-dir --disable-pip-version-check --force-reinstall --quiet \
        xformers --index-url "https://download.pytorch.org/whl/cu128" 2>/dev/null || \
    pip install --no-cache-dir --disable-pip-version-check --force-reinstall --quiet \
        xformers 2>/dev/null || \
    log_error "⚠️ All xformers installation strategies failed, continuing without"
    
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
            # Clean everything first
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
        
        # Check if there are updates available
        echo ""
        echo "🔄 Checking for updates..."
        git fetch origin 2>/dev/null
        
        # Compare local vs remote
        local_commit=$(git rev-parse HEAD 2>/dev/null)
        remote_commit=$(git rev-parse origin/$current_branch 2>/dev/null)
        
        if [ "$local_commit" = "$remote_commit" ]; then
            echo "✅ ComfyUI is up to date with the latest version!"
        else
            echo "⚠️  ComfyUI has updates available!"
            echo "   Local:  $local_commit"
            echo "   Remote: $remote_commit"
            echo ""
            echo "🔄 Updating ComfyUI to latest version..."
            
            # Perform the update
            if git pull origin $current_branch; then
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
        
        # Show recent commits
        echo ""
        echo "📝 Recent commits:"
        git log --oneline -5 2>/dev/null | sed 's/^/   /' || echo "   Unable to show recent commits"
        
    else
        echo "⚠️  ComfyUI repository not found or not a git repository"
    fi 
    

    # Virtual environment setup using storage Python 3.10
    # Ensure VENV_DIR is set to persistent location
    if [[ "$VENV_DIR" != "/storage"* ]]; then
        VENV_DIR="/storage/.venvs"
        export VENV_DIR
    fi
    
    mkdir -p "$VENV_DIR"
    
    if [ ! -d "$VENV_DIR/sd_comfy-env" ]; then
        echo "Creating virtual environment at $VENV_DIR/sd_comfy-env"
        "$PYTHON_EXECUTABLE" -m venv "$VENV_DIR/sd_comfy-env" || { log_error "Failed to create virtual environment"; exit 1; }
    fi
    
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
        pigz > /dev/null 2>&1 || {
        echo "Warning: Some packages failed to install"
    }
    echo "✅ System dependencies check completed (including pigz for fast CUDA caching)"

    # Python environment setup
    pip install pip==24.0
    pip install --upgrade wheel setuptools
    pip install "numpy>=1.26.0,<2.3.0"



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

    # Nunchaku Installation Process (Simple Wheel Install)
    install_nunchaku() {
        log "🔧 Installing Nunchaku quantization library..."
        
        # Check if already installed and working (single Python call to check and get version)
        local nunchaku_check=$(python -c "import nunchaku; print(nunchaku.__version__)" 2>/dev/null)
        if [[ -n "$nunchaku_check" ]]; then
            log "✅ Nunchaku $nunchaku_check already installed and working"
            return 0
        fi
        
        # Check PyTorch compatibility (only if not already installed)
        if ! python -c "import torch; v=torch.__version__.split('+')[0]; major,minor=map(int,v.split('.')[:2]); exit(0 if major>2 or (major==2 and minor>=5) else 1)" 2>/dev/null; then
            log_error "❌ Nunchaku requires PyTorch >=2.5"
            return 1
        fi
        
        # Install Nunchaku wheel directly from URL
        log "🔄 Installing Nunchaku wheel from GitHub releases..."
        if pip install --no-cache-dir --no-deps --force-reinstall https://github.com/nunchaku-tech/nunchaku/releases/download/v1.0.2/nunchaku-1.0.2+torch2.8-cp310-cp310-linux_x86_64.whl; then
            log "✅ Nunchaku wheel installed successfully"
            # Quick verification
            if python -c "import nunchaku" 2>/dev/null; then
                log "✅ Nunchaku installation verified"
                return 0
            else
                log_error "❌ Nunchaku import failed after wheel installation"
                return 1
            fi
        else
            log_error "❌ Failed to install Nunchaku wheel"
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
    
    # Execute custom node updates
    # Temporarily disable set -e and ERR trap to allow custom node update failures without script exit
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

    # --- STEP 7: INSTALL NUNCHAKU QUANTIZATION ---
    echo ""
    echo "=================================================="
    echo "       STEP 7: INSTALL NUNCHAKU QUANTIZATION"
    echo "=================================================="
    echo ""
    
    # Temporarily disable set -e and ERR trap to allow Nunchaku failure without script exit
    set +e
    disable_err_trap
    
    install_nunchaku
    nunchaku_status=$?
    
    # Re-enable set -e and ERR trap
    set -e
    enable_err_trap
    
    if [[ $nunchaku_status -eq 0 ]]; then
        echo "✅ Nunchaku installation completed successfully."
    else
        log_error "⚠️ Nunchaku installation had issues (Status: $nunchaku_status)"
        log_error "ComfyUI will continue without Nunchaku quantization support"
    fi

    # --- STEP 8: INSTALL SAGEATTENTION OPTIMIZATION ---
    echo ""
    echo "=================================================="
    echo "      STEP 8: INSTALL SAGEATTENTION OPTIMIZATION"
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
    # Note: SageAttention and Nunchaku are now installed earlier in the process

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
    if [[ "$VENV_DIR" != "/storage"* ]]; then
        VENV_DIR="/storage/.venvs"
        export VENV_DIR
    fi
    
    if [ -f "$VENV_DIR/sd_comfy-env/bin/activate" ]; then
        source "$VENV_DIR/sd_comfy-env/bin/activate"
    else
        log_error "Virtual environment not found at $VENV_DIR/sd_comfy-env"
        exit 1
    fi
        
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
            
            # Check if there are updates available
            echo ""
            echo "🔄 Checking for updates..."
            git fetch origin 2>/dev/null
            
            # Compare local vs remote
            local_commit=$(git rev-parse HEAD 2>/dev/null)
            remote_commit=$(git rev-parse origin/$current_branch 2>/dev/null)
            
            if [ "$local_commit" = "$remote_commit" ]; then
                echo "✅ ComfyUI is up to date with the latest version!"
            else
                echo "⚠️  ComfyUI has updates available!"
                echo "   Local:  $local_commit"
                echo "   Remote: $remote_commit"
                echo ""
                echo "💡 ComfyUI updates are now handled at the beginning of the installation process"
                echo "   Run the script again to get the latest updates"
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
  # Skip redundant check if we just completed a fresh installation
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
  
  
  # Frontend version - hardcoded to 1.25.10 for reverse proxy compatibility
  # Override with USE_LEGACY_FRONTEND=1 if needed
  FRONTEND_FLAG="--front-end-version Comfy-Org/ComfyUI_frontend@1.25.10"
  echo "📦 Using frontend version: 1.25.10"
  
  if [[ -n "${USE_LEGACY_FRONTEND}" ]]; then
    FRONTEND_FLAG="--front-end-version Comfy-Org/ComfyUI_legacy_frontend@latest"
    echo "⚠️  Overriding to LEGACY frontend"
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
    $CUSTOM_NODES_FLAG \
    $FRONTEND_FLAG"
  PYTHONUNBUFFERED=1 service_loop "$COMFYUI_CMD" > $LOG_DIR/sd_comfy.log 2>&1 &
  echo $! > /tmp/sd_comfy.pid
  
  # Wait a moment for ComfyUI to start
  sleep 3
  log "✅ ComfyUI started successfully! You can now access it at http://localhost:$SD_COMFY_PORT"
  
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

    # 2. INSTALL PYTHON DEPENDENCIES
    log "2. Installing Python Dependencies (Comfy env: $comfy_venv_dir)..."
    source "$comfy_venv_dir/bin/activate"
    
    # Install sd-scripts dependencies (backend/sd_scripts submodule)
    if [[ -d "$lora_training_dir/backend/sd_scripts" ]]; then
        cd "$lora_training_dir/backend/sd_scripts"
        if [[ -f "requirements.txt" ]]; then
            # Filter to avoid breaking ComfyUI (same strategy as AI Toolkit)
            # Keep Comfy's versions of: torch, torchvision, torchaudio, xformers, numpy, scipy, opencv, pillow
            grep -vE "^torch[^a-z]|^torchvision|^torchaudio|^xformers|^numpy|^scipy|^opencv-python|^Pillow|^setuptools|^python-multipart" "requirements.txt" > "/tmp/lora_backend_reqs.txt"
            log "   Installing sd-scripts requirements (filtered)..."
            pip install --no-cache-dir -q -r "/tmp/lora_backend_reqs.txt" 2>&1 | grep -v "already satisfied" || true
        fi
        cd - > /dev/null
    else
        log "   ⚠️ backend/sd_scripts directory not found, skipping sd-scripts requirements"
    fi
    
    # Install LyCORIS (main repo includes it, no separate install needed for sd-scripts)
    # Note: sd-scripts supports network_module="lycoris.kohya" without needing lycoris-lora package
    log "   LyCORIS support: using sd-scripts built-in (network_module='networks.lora')"
    
    # Install bitsandbytes for 8bit/4bit optimizers (Prodigy, AdamW8bit)
    log "   Installing bitsandbytes (for Prodigy optimizer)..."
    pip install --no-cache-dir -q bitsandbytes 2>&1 | grep -v "already satisfied" || log "   ⚠️ bitsandbytes install failed (optional)"
    
    # Install Lion optimizer (optional)
    log "   Installing lion-pytorch (optional optimizer)..."
    pip install --no-cache-dir -q lion-pytorch 2>&1 | grep -v "already satisfied" || true
    
    # Install Prodigy optimizer
    log "   Installing prodigyopt (for Prodigy optimizer)..."
    pip install --no-cache-dir -q prodigyopt 2>&1 | grep -v "already satisfied" || log "   ⚠️ prodigyopt install failed"

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

# Activate environment
source /storage/.venvs/sd_comfy-env/bin/activate

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
    echo "   source /storage/.venvs/sd_comfy-env/bin/activate"
    echo "   cd $lora_training_dir/backend/sd_scripts"
    echo "   accelerate launch train_network.py --config_file=/path/to/config.toml"
    echo "=================================================="
    echo ""

    # 6. CREATE SIMPLE WEB UI FOR CONFIG MANAGEMENT (OPTIONAL - CONTINUES EVEN IF FAILS)
    log "6. Creating Web UI for LoRA Training (optional)..."
    mkdir -p "$lora_training_dir/web_ui"
    
    # Create Flask app for web UI
    cat > "$lora_training_dir/web_ui/app.py" << 'WEBUI_EOF'
#!/usr/bin/env python3
"""Simple web UI for LoRA Easy Training Scripts"""
from flask import Flask, render_template_string, request, jsonify, send_file
import os
import subprocess
import glob
import json
from datetime import datetime
import toml

app = Flask(__name__)

CONFIG_DIR = "/storage/lora_training/config"
OUTPUT_DIR = "/tmp/stable-diffusion-models/lora"
TRAINING_LOG = "/storage/logs/lora_training.log"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>LoRA Easy Training Scripts</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #1a1a1a; color: #e0e0e0; padding: 20px; }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { color: #4CAF50; margin-bottom: 30px; }
        .section { background: #2a2a2a; border-radius: 8px; padding: 20px; margin-bottom: 20px; }
        .section h2 { color: #64B5F6; margin-bottom: 15px; }
        .config-list { display: grid; gap: 10px; }
        .config-item { background: #333; padding: 15px; border-radius: 5px; display: flex; justify-content: space-between; align-items: center; }
        .config-name { font-weight: 500; color: #FFF; }
        .config-meta { font-size: 0.85em; color: #999; }
        button { background: #4CAF50; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-size: 14px; }
        button:hover { background: #45a049; }
        button.danger { background: #f44336; }
        button.danger:hover { background: #da190b; }
        button:disabled { background: #555; cursor: not-allowed; }
        .log-viewer { background: #1a1a1a; border: 1px solid #444; border-radius: 5px; padding: 15px; max-height: 400px; overflow-y: auto; font-family: 'Courier New', monospace; font-size: 12px; }
        .lora-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 15px; }
        .lora-item { background: #333; padding: 15px; border-radius: 5px; }
        .lora-name { font-weight: 500; color: #FFF; margin-bottom: 5px; word-break: break-all; }
        .lora-size { font-size: 0.85em; color: #999; }
        .status { padding: 5px 10px; border-radius: 3px; font-size: 0.85em; display: inline-block; margin-top: 10px; }
        .status.running { background: #FF9800; }
        .status.idle { background: #4CAF50; }
        .status.error { background: #f44336; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 LoRA Easy Training Scripts</h1>
        
        <div class="section">
            <h2>Training Status</h2>
            <div id="status" class="status idle">Idle</div>
            <div style="margin-top: 15px;">
                <button onclick="stopTraining()" id="stopBtn" disabled>Stop Training</button>
                <button onclick="refreshStatus()" style="margin-left: 10px;">Refresh</button>
            </div>
        </div>
        
        <div class="section">
            <h2>Available Configs</h2>
            <div class="config-list" id="configList">
                <p style="color: #999;">Loading configs...</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Training Log</h2>
            <div class="log-viewer" id="logViewer">No logs yet...</div>
            <button onclick="refreshLog()" style="margin-top: 10px;">Refresh Log</button>
        </div>
        
        <div class="section">
            <h2>Trained LoRAs</h2>
            <div class="lora-grid" id="loraGrid">
                <p style="color: #999;">Loading LoRAs...</p>
            </div>
        </div>
    </div>
    
    <script>
        let currentTrainingPid = null;
        
        async function loadConfigs() {
            const response = await fetch('/api/configs');
            const configs = await response.json();
            const list = document.getElementById('configList');
            
            if (configs.length === 0) {
                list.innerHTML = '<p style="color: #999;">No configs found. Upload TOML configs from your app.</p>';
                return;
            }
            
            list.innerHTML = configs.map(config => `
                <div class="config-item">
                    <div>
                        <div class="config-name">${config.name}</div>
                        <div class="config-meta">Modified: ${config.modified}</div>
                    </div>
                    <button onclick="startTraining('${config.path}')">Train</button>
                </div>
            `).join('');
        }
        
        async function startTraining(configPath) {
            if (!confirm('Start training with this config?')) return;
            
            const response = await fetch('/api/train', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({config: configPath})
            });
            const result = await response.json();
            
            if (result.success) {
                alert('Training started! Check the log below for progress.');
                currentTrainingPid = result.pid;
                document.getElementById('stopBtn').disabled = false;
                refreshStatus();
            } else {
                alert('Failed to start training: ' + result.message);
            }
        }
        
        async function stopTraining() {
            if (!confirm('Stop current training?')) return;
            
            const response = await fetch('/api/stop', {method: 'POST'});
            const result = await response.json();
            alert(result.message);
            document.getElementById('stopBtn').disabled = true;
            refreshStatus();
        }
        
        async function refreshStatus() {
            const response = await fetch('/api/status');
            const status = await response.json();
            const statusEl = document.getElementById('status');
            
            statusEl.className = 'status ' + status.status;
            statusEl.textContent = status.message;
            document.getElementById('stopBtn').disabled = status.status !== 'running';
            currentTrainingPid = status.pid;
        }
        
        async function refreshLog() {
            const response = await fetch('/api/log');
            const data = await response.json();
            document.getElementById('logViewer').textContent = data.log || 'No logs yet...';
        }
        
        async function loadLoras() {
            const response = await fetch('/api/loras');
            const loras = await response.json();
            const grid = document.getElementById('loraGrid');
            
            if (loras.length === 0) {
                grid.innerHTML = '<p style="color: #999;">No trained LoRAs yet.</p>';
                return;
            }
            
            grid.innerHTML = loras.map(lora => `
                <div class="lora-item">
                    <div class="lora-name">${lora.name}</div>
                    <div class="lora-size">${lora.size}</div>
                    <button onclick="window.location.href='/api/download/${encodeURIComponent(lora.name)}'">Download</button>
                </div>
            `).join('');
        }
        
        // Auto-refresh every 5 seconds
        setInterval(() => {
            refreshStatus();
            if (currentTrainingPid) refreshLog();
        }, 5000);
        
        // Initial load
        loadConfigs();
        loadLoras();
        refreshStatus();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/configs')
def get_configs():
    configs = []
    for config_path in glob.glob(f"{CONFIG_DIR}/*.toml"):
        stat = os.stat(config_path)
        configs.append({
            'name': os.path.basename(config_path),
            'path': config_path,
            'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        })
    return jsonify(configs)

@app.route('/api/train', methods=['POST'])
def start_training():
    data = request.json
    config_path = data.get('config')
    
    if not config_path or not os.path.exists(config_path):
        return jsonify({'success': False, 'message': 'Config not found'})
    
    try:
        # Start training in background
        # Use stdbuf to unbuffer output for real-time logging
        cmd = [
            '/usr/local/bin/train-lora',
            config_path
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=open(TRAINING_LOG, 'w'),
            stderr=subprocess.STDOUT
        )
        
        # Save PID
        with open('/tmp/lora_training.pid', 'w') as f:
            f.write(str(process.pid))
        
        return jsonify({'success': True, 'pid': process.pid})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/stop', methods=['POST'])
def stop_training():
    try:
        if os.path.exists('/tmp/lora_training.pid'):
            with open('/tmp/lora_training.pid', 'r') as f:
                pid = int(f.read().strip())
            
            # Kill the wrapper script
            os.kill(pid, 15)  # SIGTERM
            
            # Also try to kill accelerate/python processes started by it
            subprocess.run(['pkill', '-f', 'train_network.py'])
            
            os.remove('/tmp/lora_training.pid')
            return jsonify({'success': True, 'message': 'Training stopped'})
        return jsonify({'success': False, 'message': 'No training running'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/status')
def get_status():
    if os.path.exists('/tmp/lora_training.pid'):
        with open('/tmp/lora_training.pid', 'r') as f:
            try:
                pid = int(f.read().strip())
                # Check if process is still running
                os.kill(pid, 0)
                return jsonify({'status': 'running', 'message': f'Training (PID: {pid})', 'pid': pid})
            except (OSError, ValueError):
                # Process dead or file corrupted
                if os.path.exists('/tmp/lora_training.pid'):
                    os.remove('/tmp/lora_training.pid')
    
    return jsonify({'status': 'idle', 'message': 'No training running', 'pid': None})

@app.route('/api/log')
def get_log():
    if os.path.exists(TRAINING_LOG):
        with open(TRAINING_LOG, 'r') as f:
            # Get last 1000 lines
            lines = f.readlines()
            log = ''.join(lines[-1000:])
            return jsonify({'log': log})
    return jsonify({'log': ''})

@app.route('/api/loras')
def get_loras():
    loras = []
    if os.path.exists(OUTPUT_DIR):
        for lora_path in glob.glob(f"{OUTPUT_DIR}/*.safetensors"):
            stat = os.stat(lora_path)
            size_mb = stat.st_size / (1024 * 1024)
            loras.append({
                'name': os.path.basename(lora_path),
                'size': f'{size_mb:.2f} MB',
                'path': lora_path
            })
    return jsonify(loras)

@app.route('/api/download/<path:filename>')
def download_lora(filename):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(TRAINING_LOG), exist_ok=True)
    app.run(host='0.0.0.0', port=8675, debug=False)
WEBUI_EOF

    chmod +x "$lora_training_dir/web_ui/app.py"
    
    # Install Flask (optional - don't fail if it doesn't work)
    log "   Installing Flask..."
    pip install --no-cache-dir -q flask toml 2>&1 | grep -v "already satisfied" || log "   ⚠️ Flask/toml install failed (UI will not work)"
    
    # 7. START WEB UI (OPTIONAL - CONTINUES EVEN IF FAILS)
    log "7. Starting Web UI (optional)..."
    start_lora_training_ui || log "   ⚠️ Web UI failed to start (backend still works via train-lora command)"
}

start_lora_training_ui() {
    local lora_training_dir="/tmp/lora-training"
    local log_file="$LOG_DIR/lora_training_ui.log"
    
    # Check if Flask is installed
    if ! python3 -c "import flask" 2>/dev/null; then
        log "   ⚠️ Flask not installed, skipping UI startup"
        return 1
    fi
    
    # Stop existing UI
    if [[ -f /tmp/lora_training_ui.pid ]]; then
        read -r ui_pid < /tmp/lora_training_ui.pid 2>/dev/null
        [[ -n "$ui_pid" ]] && kill -9 "$ui_pid" 2>/dev/null || true
        rm -f /tmp/lora_training_ui.pid
    fi
    
    # Clear port 8675
    fuser -k 8675/tcp 2>/dev/null || true
    sleep 2
    
    # Check if web_ui directory exists
    if [[ ! -f "$lora_training_dir/web_ui/app.py" ]]; then
        log "   ⚠️ Web UI files not found, skipping"
        return 1
    fi
    
    # Activate venv and start UI
    source "$VENV_DIR/sd_comfy-env/bin/activate"
    cd "$lora_training_dir/web_ui"
    
    nohup python3 app.py > "$log_file" 2>&1 &
    local ui_pid=$!
    echo "$ui_pid" > /tmp/lora_training_ui.pid
    
    sleep 3
    
    if kill -0 $ui_pid 2>/dev/null; then
        log "✅ LoRA Training UI started (PID: $ui_pid)"
        
        echo ""
        echo "=================================================="
        echo "   LORA TRAINING WEB UI"
        echo "=================================================="
        echo "   URL:             http://localhost:8675"
        echo "   PID:             $ui_pid"
        echo "   Log:             $log_file"
        echo "=================================================="
        echo ""
        echo "   Access via browser to:"
        echo "   - View and launch training configs"
        echo "   - Monitor training progress"
        echo "   - Download trained LoRAs"
        echo "=================================================="
        echo ""
        return 0
    else
        log "⚠️ LoRA Training UI failed to start (check $log_file)"
        log "   Backend still works via: train-lora <config.toml>"
        return 1
    fi
}

# Execute
if [[ ! -f "/tmp/lora_training.prepared" ]] || [[ -n "$REINSTALL_LORA_TRAINING" ]]; then
    install_lora_training
else
    log "✅ LoRA Easy Training Scripts already installed. Restarting UI..."
    start_lora_training_ui
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
  # STEP 9.3: INSTALL SAM2 (BACKGROUND)
  #######################################
  echo ""
  echo "=================================================="
  echo "        STEP 9.3: INSTALL SAM2 (BACKGROUND)"
  echo "=================================================="
  echo ""
  echo "📦 Installing SAM2 in background..."
  log "Starting SAM2 installation in background..."
  
  # Run SAM2 installation in background (functions are already defined)
  (
    set +e
    install_sam2_optimized > /tmp/sam2_install.log 2>&1
    if [ $? -eq 0 ]; then
      echo "✅ SAM2 installed successfully" >> /tmp/sam2_install.log
    else
      echo "⚠️ SAM2 installation failed (optional dependency)" >> /tmp/sam2_install.log
    fi
  ) &
  echo $! > /tmp/sam2_install.pid
  
  log "📋 SAM2 installation started in background (PID: $(cat /tmp/sam2_install.pid))"
  log "📋 Check installation progress: tail -f /tmp/sam2_install.log"
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

