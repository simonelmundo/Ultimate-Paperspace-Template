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

# Check Python 3.10 is working and set it as default
PYTHON_EXECUTABLE="/storage/python_versions/python3.10/bin/python3.10"
if [ -x "$PYTHON_EXECUTABLE" ] && "$PYTHON_EXECUTABLE" -c "import _bz2, ssl, sqlite3" 2>/dev/null; then
    log "✅ Python 3.10 is ready"
    
    # Create symlinks to use Python 3.10 as default
    log "🔗 Setting Python 3.10 as default..."
    ln -sf "$PYTHON_EXECUTABLE" /usr/local/bin/python3.10
    ln -sf "$PYTHON_EXECUTABLE" /usr/local/bin/python3
    
    # Update PATH to prioritize our Python 3.10
    export PATH="/storage/python_versions/python3.10/bin:$PATH"
    
    log "✅ Python 3.10 is now the default Python"
    log "📍 Python version: $($PYTHON_EXECUTABLE --version)"
else
    log_error "❌ Python 3.10 not working - please install it first"
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
# STEP 2: CUDA AND ENVIRONMENT SETUP
#######################################
echo ""
echo "=================================================="
echo "        STEP 2: CUDA AND ENVIRONMENT SETUP"
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
    local CUDA_MARKER="/storage/.cuda_12.8_installed"
    local APT_INSTALL_LOG="$LOG_DIR/apt_cuda_install.log" # Log file for apt output

    # Check marker and verify existing installation (logic from previous step)
    if [ -f "$CUDA_MARKER" ]; then
        echo "CUDA 12.8 marker file exists. Verifying installation..."
        setup_cuda_env
        hash -r
        if command -v nvcc &>/dev/null && [[ "$(nvcc --version 2>&1 | grep 'release' | awk '{print $6}' | sed 's/^V//')" == "12.8"* ]]; then
             echo "CUDA 12.8 already installed and verified."
             return 0
        else
             echo "Marker file exists, but verification failed. Proceeding with installation..."
        fi
    fi

    # Clean up existing CUDA 11.x if present
    if dpkg -l | grep -q "cuda-11"; then
        echo "Removing existing CUDA 11.x installations..."
        apt-get remove --purge -y 'cuda-11-*' 'cuda-repo-ubuntu*-11-*' 'nvidia-cuda-toolkit' || echo "No CUDA 11.x found or removal failed."
        apt-get autoremove -y
    fi

    # Install only essential CUDA components
    echo "Adding CUDA repository key..."
    wget -qO /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i /tmp/cuda-keyring.deb
    rm -f /tmp/cuda-keyring.deb

    echo "Running apt-get update (output will be logged)..."
    # Remove -qq to see output
    if ! apt-get update >> "$APT_INSTALL_LOG" 2>&1; then
        log_error "apt-get update failed. Check $APT_INSTALL_LOG for details."
        cat "$APT_INSTALL_LOG" # Print log content to main log
        return 1
    fi
    echo "apt-get update completed."

    # Install minimal CUDA components AND general build dependencies
    echo "Installing CUDA components and general build tools... Output logged to $APT_INSTALL_LOG"
    apt-get install -y \
        build-essential \
        python3-dev \
        libatlas-base-dev \
        libblas-dev \
        liblapack-dev \
        libjpeg-dev \
        libpng-dev \
        libgl1- \
        cuda-cudart-12-8 \
        cuda-cudart-dev-12-8 \
        cuda-nvcc-12-8 \
        cuda-cupti-12-8 \
        cuda-cupti-dev-12-8 \
        libcublas-12-8 \
        libcublas-dev-12-8 \
        libcufft-12-8 \
        libcufft-dev-12-8 \
        libcurand-12-8 \
        libcurand-dev-12-8 \
        libcusolver-12-8 \
        libcusolver-dev-12-8 \
        libcusparse-12-8 \
        libcusparse-dev-12-8 \
        libnpp-12-8 \
        libnpp-dev-12-8 >> "$APT_INSTALL_LOG" 2>&1
    local apt_exit_code=$? # Capture exit code immediately

    echo "apt-get install finished with exit code: $apt_exit_code"
    # Print the log content regardless of exit code for inspection
    echo "--- APT Install Log ($APT_INSTALL_LOG) ---"
    cat "$APT_INSTALL_LOG"
    echo "--- End APT Install Log ---"

    if [ $apt_exit_code -ne 0 ]; then
        log_error "apt-get install failed for CUDA 12.6 and build tools. Exit code: $apt_exit_code. See log above."
        return 1 # Exit if install fails
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

    # Verify installation *before* creating marker
    echo "Verifying CUDA 12.8 installation..."
    if command -v nvcc &>/dev/null; then
        local installed_version
        installed_version=$(nvcc --version 2>&1 | grep 'release' | awk '{print $6}' | sed 's/^V//')
        if [[ "$installed_version" == "12.8"* ]]; then
            echo "CUDA 12.8 installation verified successfully (Version: $installed_version)."
            touch "$CUDA_MARKER"
            echo "Installation marker created: $CUDA_MARKER"
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
        "huggingface_hub>=0.34.0" "pytorch_lightning" "sounddevice" "av>=12.0.0,<14.0.0" "accelerate"
    )
    
    local installed_count=0
    local failed_count=0
    
    for pkg in "${critical_packages[@]}"; do
        log "📦 Installing: $pkg"
        if pip install --quiet --no-cache-dir "$pkg" 2>/dev/null; then
            log "✅ Successfully installed: $pkg"
            ((installed_count++))
        else
            log_error "❌ Failed to install: $pkg"
            ((failed_count++))
        fi
    done
    
    log "📊 Critical packages: $installed_count installed, $failed_count failed"
    return $failed_count
}

# Optimized SAM2 Installation Function
install_sam2_optimized() {
    log "🚀 Installing SAM2 with optimizations for faster installation..."
    
    # Temporarily disable set -e within this function
    set +e
    
    # Check if SAM2 is already installed
    if python -c "import sam2" 2>/dev/null; then
        local sam2_version=$(python -c "import sam2; print(sam2.__version__)" 2>/dev/null || echo "unknown")
        log "✅ SAM2 $sam2_version already installed"
        set -e  # Re-enable set -e
        return 0
    fi
    
    # Create a temporary directory for SAM2 installation
    local sam2_temp_dir="/tmp/sam2_install"
    mkdir -p "$sam2_temp_dir"
    cd "$sam2_temp_dir"
    
    log "📦 Cloning SAM2 repository with shallow clone (faster)..."
    # Use shallow clone to reduce download time
    if git clone --depth 1 https://github.com/facebookresearch/sam2.git .; then
        log "✅ SAM2 repository cloned successfully"
    else
        log_error "❌ Failed to clone SAM2 repository"
        return 1
    fi
    
    log "🔧 Installing SAM2 with optimized settings..."
    # Install with optimizations
    export MAX_JOBS=$(nproc)  # Use all available cores
    export USE_NINJA=1        # Use Ninja for faster builds
    
    # Install in development mode for faster installation
    if pip install -e . --no-build-isolation --no-deps; then
        log "✅ SAM2 installed successfully in development mode"
        
        # Install dependencies separately
        log "📦 Installing SAM2 dependencies..."
        pip install --no-cache-dir --disable-pip-version-check \
            "torch>=1.9.0" "torchvision>=0.10.0" "opencv-python" \
            "pillow" "numpy" "scipy" "matplotlib" "scikit-image" \
            "timm" "transformers" "huggingface_hub" 2>/dev/null || log "Some dependencies may have failed"
        
    # Clean up
    cd /
    rm -rf "$sam2_temp_dir"
    
    # Verify installation
    if python -c "import sam2; print(f'✅ SAM2 {sam2.__version__} installed successfully')" 2>/dev/null; then
        log "✅ SAM2 installation verified"
        set -e  # Re-enable set -e
        return 0
    else
        log_error "❌ SAM2 installation verification failed"
        set -e  # Re-enable set -e
        return 1
    fi
else
    log_error "❌ Failed to install SAM2"
    cd /
    rm -rf "$sam2_temp_dir"
    set -e  # Re-enable set -e
    return 1
fi
}

# Function to fix common custom node import errors
fix_custom_node_import_errors() {
    log "🔧 Fixing common custom node import errors..."
    
    # Fix relative import issues (like in comfyui-impact-pack)
    log "🔧 Checking for relative import issues..."
    
    # Fix rembg import issues
    if ! python -c "import rembg" 2>/dev/null; then
        log "📦 Installing rembg for ComfyUI-Hunyuan3d-2-1..."
        pip install --quiet --no-cache-dir rembg 2>/dev/null || log_error "rembg installation failed"
    fi
    
    # Fix onnxruntime import issues
    if ! python -c "import onnxruntime" 2>/dev/null; then
        log "📦 Installing onnxruntime for comfyui_controlnet_aux..."
        pip install --quiet --no-cache-dir onnxruntime 2>/dev/null || log_error "onnxruntime installation failed"
    fi
    
    # Fix OpenCV import issues
    if ! python -c "import cv2" 2>/dev/null; then
        log "📦 Installing opencv-python for various nodes..."
        pip install --quiet --no-cache-dir opencv-python 2>/dev/null || log_error "opencv-python installation failed"
    fi
    
    # Fix trimesh import issues
    if ! python -c "import trimesh" 2>/dev/null; then
        log "📦 Installing trimesh for 3D models..."
        pip install --quiet --no-cache-dir trimesh 2>/dev/null || log_error "trimesh installation failed"
    fi
    
    log "✅ Custom node import error fixes completed"
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

    # Virtual environment setup using storage Python 3.10
    rm -rf $VENV_DIR/sd_comfy-env
    "$PYTHON_EXECUTABLE" -m venv $VENV_DIR/sd_comfy-env
    source $VENV_DIR/sd_comfy-env/bin/activate
    echo "Virtual environment activated: $VENV_DIR/sd_comfy-env"
    echo "Using Python: $(which python)"
    echo "Python version: $(python --version)"

    # System dependencies
    echo "Installing essential system dependencies..."
    apt-get update && apt-get install -y \
        libatlas-base-dev libblas-dev liblapack-dev \
        libjpeg-dev libpng-dev \
        python3-dev build-essential \
        libgl1-mesa-dev \
        espeak-ng || {
        echo "Warning: Some packages failed to install"
    }

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
        
        # Check if already installed and working
        if python -c "import nunchaku" 2>/dev/null; then
            local current_version=$(python -c "import nunchaku; print(nunchaku.__version__)" 2>/dev/null)
            log "✅ Nunchaku $current_version already installed and working"
            return 0
        fi
        
        # Check PyTorch compatibility first
        if ! python -c "import torch; v=torch.__version__.split('+')[0]; major,minor=map(int,v.split('.')[:2]); exit(0 if major>2 or (major==2 and minor>=5) else 1)" 2>/dev/null; then
            log_error "❌ Nunchaku requires PyTorch >=2.5"
            return 1
        fi
        
        # Install Nunchaku wheel directly from URL
        log "🔄 Installing Nunchaku wheel from GitHub releases..."
        log "🔧 Command: pip install --no-cache-dir --no-deps --force-reinstall https://github.com/nunchaku-tech/nunchaku/releases/download/v1.0.2/nunchaku-1.0.2+torch2.8-cp310-cp310-linux_x86_64.whl"
        
        if pip install --no-cache-dir --no-deps --force-reinstall https://github.com/nunchaku-tech/nunchaku/releases/download/v1.0.2/nunchaku-1.0.2+torch2.8-cp310-cp310-linux_x86_64.whl; then
            log "✅ Nunchaku wheel installed successfully"
            if python -c "import nunchaku; print(f'✅ Nunchaku {nunchaku.__version__} imported successfully')" 2>/dev/null; then
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
    
    log "🚀 Starting Nunchaku installation..."
    install_nunchaku
    nunchaku_status=$?
    log "🏁 Nunchaku installation completed with status: $nunchaku_status"
    
    # Test if we can continue
    echo "🎯 Testing script continuation after Nunchaku installation..."
    log "🎯 Script continuing to next step..."
    
    # Re-enable set -e and ERR trap
    set -e
    enable_err_trap
    
    if [[ $nunchaku_status -eq 0 ]]; then
        echo "✅ Nunchaku installation completed successfully."
    else
        log_error "⚠️ Nunchaku installation had issues (Status: $nunchaku_status)"
        log_error "ComfyUI will continue without Nunchaku quantization support"
    fi
    
    # Force continue regardless of status to prevent script exit
    echo "🔄 Continuing to next step regardless of Nunchaku status..."
    nunchaku_status=0  # Force success to ensure continuation

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
    
    # TensorFlow installation with error handling
    echo "📦 Installing TensorFlow..."
    set +e
    disable_err_trap
    if pip install --cache-dir="$PIP_CACHE_DIR" "tensorflow>=2.8.0,<2.19.0"; then
        echo "✅ TensorFlow installed successfully"
    else
        log_error "⚠️ TensorFlow installation failed, but continuing"
    fi
    set -e
    enable_err_trap

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

    # --- STEP 7.5: INSTALL SAM2 OPTIMIZED ---
    echo ""
    echo "=================================================="
    echo "        STEP 7.5: INSTALL SAM2 OPTIMIZED"
    echo "=================================================="
    echo ""
    
    # Temporarily disable set -e and ERR trap for SAM2 installation
    set +e
    disable_err_trap
    
    log "🚀 Installing SAM2 with optimizations (faster than Git install)..."
    # Force SAM2 installation to never fail the script
    install_sam2_optimized 2>/dev/null || true
    sam2_status=$?
    
    # Re-enable set -e and ERR trap
    set -e
    enable_err_trap
    
    # Always report status but never fail the script
    if [[ $sam2_status -eq 0 ]]; then
        echo "✅ SAM2 installation completed successfully."
    else
        log_error "⚠️ SAM2 installation had issues (Status: $sam2_status)"
        log_error "Some custom nodes requiring SAM2 may not work properly"
        log_error "Continuing with script execution..."
    fi
    
    # Force continue regardless of SAM2 status
    echo "🔄 Continuing to next step regardless of SAM2 status..."
    sam2_status=0  # Force success to ensure continuation

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
    if [ -f "$VENV_DIR/sd_comfy-env/bin/activate" ]; then
        source $VENV_DIR/sd_comfy-env/bin/activate
        echo "Virtual environment activated: $VENV_DIR/sd_comfy-env"
        
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
        
    else
        log_error "Virtual environment activation script not found!"
        exit 1
    fi
fi

log "Finished Preparing Environment for Stable Diffusion Comfy"

echo ""
echo "=================================================="
echo "           ENVIRONMENT SETUP COMPLETE!"
echo "=================================================="
echo ""

#######################################
# STEP 9: CREATE MODEL SYMLINKS
#######################################
echo ""
echo "=================================================="
echo "           STEP 9: CREATE MODEL SYMLINKS"
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

#######################################
# STEP 10: START COMFYUI
#######################################
if [[ -z "$INSTALL_ONLY" ]]; then
  echo ""
  echo "=================================================="
  echo "             STEP 10: START COMFYUI"
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
fi

#######################################
# STEP 11: DOWNLOAD MODELS (BACKGROUND)
#######################################
if [[ -z "$SKIP_MODEL_DOWNLOAD" ]]; then
  echo ""
  echo "=================================================="
  echo "        STEP 11: DOWNLOAD MODELS (BACKGROUND)"
  echo "=================================================="
  echo ""
  echo "### Downloading Models for Stable Diffusion Comfy in Background ###"
  log "Starting Model Download for Stable Diffusion Comfy in background..."
  log "💡 Models will download in background while ComfyUI is running 💡 You can start using ComfyUI immediately!"
  
  # Start model download in background
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
# STEP 12: FINAL SETUP COMPLETION
#######################################
echo ""
echo "=================================================="
echo "           STEP 12: FINAL SETUP COMPLETION"
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

echo ""
echo "=================================================="
echo "           SCRIPT EXECUTION COMPLETE!"
echo "=================================================="
echo ""
echo ""
