#!/bin/bash
set -e

# === 1. INITIALIZATION AND CONFIGURATION ===
current_dir=$(dirname "$(realpath "$0")")
cd $current_dir
source .env

# Error collection array
declare -a ERROR_LOG=()

# Error handling function
log_error() {
    local error_msg="$1"
    local error_context="$2"
    ERROR_LOG+=("[$error_context] $error_msg")
    echo "⚠️ Error in $error_context: $error_msg"
}

# Replace error exit with error logging
trap 'log_error "$(tail -n 1 /tmp/sd_comfy_error.log 2>/dev/null || echo "Unknown error")" "${BASH_SOURCE[0]}:${BASH_LINENO[0]}"' ERR

# Safe execution wrapper
safe_exec() {
    local context="$1"
    shift
    if ! "$@" 2>/tmp/sd_comfy_error.log; then
        log_error "$(cat /tmp/sd_comfy_error.log)" "$context"
        return 0
    fi
}

# Directory Structure
PERSISTENT_VENV_DIR="/storage/environments/sd_comfy-env"
TEMP_DIR="/tmp/sd_comfy"
MODEL_CACHE_DIR="/tmp/sd_comfy/model_cache"
RUNTIME_DIR="/tmp/sd_comfy/runtime"

# Initialize tracking arrays
declare -a failed_installations=()
declare -a skipped_steps=()

# === 2. UTILITY FUNCTIONS ===
check_core_packages() {
    python3 -c "
import sys
required = {'torch', 'DepthFlow', 'xformers', 'shaderflow'}
installed = {pkg.key for pkg in __import__('pkg_resources').working_set}
missing = required - installed
if missing:
    print('Missing packages:', missing)
    sys.exit(1)
sys.exit(0)
" 2>/dev/null || {
        log_error "Core package check failed" "Package Verification"
        return 0
    }
}

try_install() {
    local package="$1"
    echo "Installing: $package"
    if ! pip install $package --quiet 2>/tmp/pip_error.log | grep -v "Requirement already satisfied"; then
        failed_installations+=("$package: $(cat /tmp/pip_error.log)")
        echo "⚠️ Failed to install: $package"
        return 0
    fi
}

process_requirements() {
    local req_file="$1"
    if [ ! -f "$req_file" ]; then
        echo "No $req_file found, skipping..."
        return 0
    fi
    echo "Processing requirements from: $req_file"
    while IFS= read -r requirement || [ -n "$requirement" ]; do
        if [[ ! -z "$requirement" && ! "$requirement" =~ ^# ]]; then
            if [[ "$requirement" =~ ^file:///storage/stable-diffusion-comfy ]]; then
                echo "Skipping local reference: $requirement"
                continue
            fi
            try_install "$requirement"
        fi
    done < "$req_file"
}

setup_nvidia_environment() {
    export CUDA_HOME=/usr/local/cuda-11.6
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export NVIDIA_VISIBLE_DEVICES="all"
    export NVIDIA_DRIVER_CAPABILITIES="all"
    export WINDOW_BACKEND="headless"
    export PYOPENGL_PLATFORM="egl"
    export __GLX_VENDOR_LIBRARY_NAME="nvidia"
    export __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    export FORCE_CUDA=1
    export CUDA_VISIBLE_DEVICES=0
    export DEPTHFLOW_SUPPRESS_ROOT_WARNING=1
    export CMAKE_ARGS="-DUSE_X11=ON"
}

# === 3. MAIN INSTALLATION FLOW ===
echo "### Setting up Stable Diffusion Comfy ###"
log "Setting up Stable Diffusion Comfy"

# Create directories
safe_exec "Directory Setup" mkdir -p "$PERSISTENT_VENV_DIR" "$TEMP_DIR" "$MODEL_CACHE_DIR" "$RUNTIME_DIR"

if [[ "$REINSTALL_SD_COMFY" || ! -f "$PERSISTENT_VENV_DIR/sd_comfy.prepared" ]]; then
    # Repository setup
    safe_exec "Repository Setup" \
    TARGET_REPO_URL="https://github.com/comfyanonymous/ComfyUI.git" \
    TARGET_REPO_DIR=$REPO_DIR \
    UPDATE_REPO=$SD_COMFY_UPDATE_REPO \
    UPDATE_REPO_COMMIT=$SD_COMFY_UPDATE_REPO_COMMIT \
    prepare_repo

    # Symlinks setup
    symlinks=(
        "$REPO_DIR/output:$IMAGE_OUTPUTS_DIR/stable-diffusion-comfy"
        "$MODEL_CACHE_DIR:$WORKING_DIR/models"
        "$MODEL_CACHE_DIR/sd:$LINK_MODEL_TO"
        "$MODEL_CACHE_DIR/lora:$LINK_LORA_TO"
        "$MODEL_CACHE_DIR/vae:$LINK_VAE_TO"
        "$MODEL_CACHE_DIR/upscaler:$LINK_UPSCALER_TO"
        "$MODEL_CACHE_DIR/controlnet:$LINK_CONTROLNET_TO"
        "$MODEL_CACHE_DIR/embedding:$LINK_EMBEDDING_TO"
        "$MODEL_CACHE_DIR/llm_checkpoints:$LINK_LLM_TO"
    )
    safe_exec "Symlink Setup" prepare_link "${symlinks[@]}"

    # Virtual environment setup
    if [ ! -d "$PERSISTENT_VENV_DIR" ]; then
        safe_exec "VEnv Setup" python3.10 -m venv "$PERSISTENT_VENV_DIR"
    fi
    source "$PERSISTENT_VENV_DIR/bin/activate" || log_error "Failed to activate virtual environment" "VEnv Activation"

    # System dependencies
    echo "Installing system dependencies..."
    safe_exec "System Dependencies" apt-get update
    safe_exec "System Dependencies" apt-get install -y \
        libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
        libatlas-base-dev libblas-dev liblapack-dev \
        libjpeg-dev libpng-dev libtiff-dev libbz2-dev \
        libgl1-mesa-dev python2-dev libopenblas-dev \
        cmake build-essential \
        nvidia-driver-525 libnvidia-gl-525

    # Setup NVIDIA environment
    setup_nvidia_environment

    # Install core packages if needed
    if [[ "$REINSTALL_SD_COMFY" == "true" ]] || ! check_core_packages; then
        safe_exec "Base Packages" try_install "pip==24.0"
        safe_exec "Base Packages" try_install "--upgrade wheel setuptools"
        safe_exec "Base Packages" try_install "numpy>=1.26.0,<2.3.0"

        # PyTorch ecosystem
        echo "Installing PyTorch with CUDA support..."
        safe_exec "PyTorch" pip install --quiet torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 \
            --index-url https://download.pytorch.org/whl/cu116

        # Core dependencies
        try_install "PyOpenGL PyOpenGL_accelerate"
        cd $REPO_DIR
        try_install "xformers"

        # DepthFlow setup
        echo "Installing DepthFlow..."
        pip uninstall -y depthflow shaderflow 2>/dev/null || true
        try_install "shaderflow"
        echo "cuda" > /tmp/depthflow_answer
        FORCE_CUDA=1 cat /tmp/depthflow_answer | try_install "depthflow[shaderflow]==0.8.0.dev0"
        rm -f /tmp/depthflow_answer

        # Setup DepthFlow symlink
        cd /storage/stable-diffusion-comfy/custom_nodes
        if [ -d "DepthFlow" ] || [ -L "DepthFlow" ]; then
            rm -rf DepthFlow
        fi
        if python3 -c "import DepthFlow" 2>/dev/null; then
            ln -sf "$PERSISTENT_VENV_DIR/lib/python3.10/site-packages/DepthFlow" DepthFlow
        fi

        # Additional packages
        try_install "tensorflow>=2.8.0,<2.19.0"
        try_install "imgui-bundle --no-cache-dir"

        # Process requirement files
        process_requirements "requirements.txt"
        process_requirements "/storage/stable-diffusion-comfy/custom_nodes/ComfyUI-Depthflow-Nodes/requirements.txt"
        process_requirements "/notebooks/sd_comfy/additional_requirements.txt"

        # Report failures
        if [ ${#failed_installations[@]} -ne 0 ]; then
            echo "### Failed installations ###"
            printf '%s\n' "${failed_installations[@]}"
        else
            echo "✅ All packages installed successfully!"
        fi
    else
        echo "✅ Core packages already installed, skipping installation"
    fi

    touch "$PERSISTENT_VENV_DIR/sd_comfy.prepared"
else
    source "$PERSISTENT_VENV_DIR/bin/activate"
fi

log "Finished Preparing Environment for Stable Diffusion Comfy"

# === 4. MODEL DOWNLOAD ===
if [[ -z "$SKIP_MODEL_DOWNLOAD" ]]; then
    echo "### Downloading Model for Stable Diffusion Comfy ###"
    log "Downloading Model for Stable Diffusion Comfy"
    bash $current_dir/../utils/sd_model_download/main.sh
    log "Finished Downloading Models for Stable Diffusion Comfy"
else
    log "Skipping Model Download for Stable Diffusion Comfy"
fi

# === 5. SERVICE STARTUP ===
if [[ -z "$INSTALL_ONLY" ]]; then
    safe_exec "Service Startup" cd "$REPO_DIR"
    safe_exec "Service Startup" \
        PYTHONUNBUFFERED=1 service_loop "python main.py --dont-print-server --highvram --port $SD_COMFY_PORT ${EXTRA_SD_COMFY_ARGS}" \
        > "$RUNTIME_DIR/sd_comfy.log" 2>&1 &
    echo $! > "$RUNTIME_DIR/sd_comfy.pid"
fi

# === 6. NOTIFICATIONS AND CLEANUP ===
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
# =============================================================================
# DepthFlow Environment Verification
# Comprehensive diagnostics for DepthFlow GPU acceleration
# =============================================================================

echo "
╔════════════════════════════════════════╗
║   DepthFlow Environment Verification   ║
╚════════════════════════════════════════╝"

# -----------------------------------------------------------------------------
# 1. NVIDIA Driver and System Check
# -----------------------------------------------------------------------------
echo -e "\n[1/7] 🔍 Checking NVIDIA Driver and System..."

# Function to safely execute commands
safe_exec() {
    "$@" 2>/dev/null || true
}

# Function to extract version numbers
get_version_number() {
    echo "$1" | grep -oP '\d+\.\d+(\.\d+)?' 2>/dev/null || echo "unknown"
}

check_nvidia_versions() {
    echo "Checking NVIDIA component versions..."
    echo "----------------------------------------"
    
    # Get NVIDIA driver version from nvidia-smi
    NVIDIA_SMI_VERSION=$(safe_exec nvidia-smi --query-gpu=driver_version --format=csv,noheader)
    echo "NVIDIA Driver (nvidia-smi): ${NVIDIA_SMI_VERSION:-unknown}"
    
    # Get installed NVIDIA package versions
    NVIDIA_DRIVER_PKG=$(safe_exec dpkg -l | grep nvidia-driver | awk '{print $2 " " $3}')
    NVIDIA_LIB_VERSION=$(safe_exec ldconfig -p | grep libnvidia-ml.so | head -n1 | grep -oP '\d+\.\d+\.\d+')
    echo "NVIDIA Driver Package : ${NVIDIA_DRIVER_PKG:-unknown}"
    echo "NVIDIA Library       : ${NVIDIA_LIB_VERSION:-unknown}"
    
    # Check EGL/OpenGL libraries
    echo -e "\nEGL/OpenGL Libraries:"
    for lib in libEGL_nvidia.so libGLX_nvidia.so libGLESv2_nvidia.so; do
        LIB_VERSION=$(safe_exec ldconfig -p | grep $lib | head -n1 | grep -oP '\d+\.\d+\.\d+')
        echo "$lib: ${LIB_VERSION:-not found}"
    done
    
    # Compare versions if available
    if [[ -n "$NVIDIA_SMI_VERSION" && -n "$NVIDIA_LIB_VERSION" ]]; then
        SMI_VER=$(get_version_number "$NVIDIA_SMI_VERSION")
        LIB_VER=$(get_version_number "$NVIDIA_LIB_VERSION")
        
        if [[ "$SMI_VER" != "$LIB_VER" && "$SMI_VER" != "unknown" && "$LIB_VER" != "unknown" ]]; then
            echo -e "\n⚠️  Version Mismatch Detected:"
            echo "• Driver version (nvidia-smi): $SMI_VER"
            echo "• Library version: $LIB_VER"
            echo -e "\nPossible causes:"
            echo "1. Multiple driver versions installed"
            echo "2. Incomplete driver installation"
            echo "3. System needs reboot after update"
            
            echo -e "\nInstalled NVIDIA packages:"
            safe_exec dpkg -l | grep -E "nvidia-|cuda" | awk '{printf "%-20s: %s\n", $2, $3}'
        fi
    fi
}

# Main check logic
if safe_exec nvidia-smi > /tmp/nvidia_check; then
    echo "Driver Information:"
    echo "----------------------------------------"
    cat /tmp/nvidia_check 2>/dev/null || echo "No driver information available"
    echo "----------------------------------------"
    
    echo "GPU Memory Status (Per Device):"
    echo "----------------------------------------"
    safe_exec nvidia-smi --query-gpu=index,gpu_name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu --format=csv,noheader | while IFS="," read -r id name total used free temp util; do
        echo "GPU $id: $name"
        echo "  Memory  : $used/$total (Free: $free)"
        echo "  Temp    : $temp"
        echo "  Load    : $util"
    done
    
    # Add detailed version checks
    check_nvidia_versions
    
    echo "✅ NVIDIA drivers are functional"
else
    echo "⚠️  Warning: NVIDIA driver issue detected"
    echo "Error: $(cat /tmp/nvidia_check 2>/dev/null || echo 'No error information available')"
    
    # Try to get more information about the failure
    echo -e "\nTroubleshooting Information:"
    echo "----------------------------------------"
    echo "1. Checking driver status:"
    safe_exec systemctl status nvidia-* || echo "No NVIDIA services found"
    
    echo -e "\n2. Checking loaded NVIDIA modules:"
    safe_exec lsmod | grep nvidia || echo "No NVIDIA modules loaded"
    
    echo -e "\n3. Checking NVIDIA packages:"
    safe_exec dpkg -l | grep -E "nvidia-|cuda" || echo "No NVIDIA packages found"
    
    echo -e "\n4. Checking GPU devices:"
    safe_exec lspci | grep -i nvidia || echo "No NVIDIA devices found"
fi

# Clean up
rm -f /tmp/nvidia_check 2>/dev/null || true

# -----------------------------------------------------------------------------
# 2. CUDA Version and Compatibility Check
# -----------------------------------------------------------------------------
echo -e "\n[2/7] 🔍 Checking CUDA Environment..."

# Check System CUDA
if CUDA_PATH=$(which nvcc 2>/dev/null); then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo "System CUDA Version: $CUDA_VERSION"
    
    # Check PyTorch CUDA
    echo -e "\nPyTorch CUDA Compatibility:"
    echo "----------------------------------------"
    python3 -c "
import torch
import re

def normalize_version(version):
    # Extract just major.minor version (e.g., '11.6' from '11.6.124')
    match = re.match(r'(\d+\.\d+)', str(version))
    return match.group(1) if match else version

pytorch_cuda = normalize_version(torch.version.cuda)
system_cuda = normalize_version('$CUDA_VERSION')

print(f'PyTorch Version : {torch.__version__}')
print(f'PyTorch CUDA   : {pytorch_cuda}')
print(f'System CUDA    : {system_cuda} (full: $CUDA_VERSION)')

if pytorch_cuda != system_cuda:
    print(f'\n⚠️  Version Mismatch:')
    print(f'• System CUDA ({system_cuda}) does not match PyTorch CUDA ({pytorch_cuda})')
    print(f'• This may cause GPU acceleration issues')
    print(f'\nSuggested fixes:')
    print(f'1. Install PyTorch for CUDA {system_cuda}')
    print(f'2. Or upgrade system CUDA to {pytorch_cuda}')
else:
    print(f'\n✅ CUDA versions are compatible')
"
else
    echo "⚠️  CUDA not found on system"
fi

# -----------------------------------------------------------------------------
# 3. OpenGL/EGL Libraries Check
# -----------------------------------------------------------------------------
echo -e "\n[3/7] 🔍 Checking Graphics Libraries..."
echo "NVIDIA OpenGL/EGL Libraries:"
echo "----------------------------------------"
# Only check specific NVIDIA libraries
for lib in "libGL_nvidia" "libEGL_nvidia" "libGLX_nvidia" "libGLESv1_CM_nvidia" "libGLESv2_nvidia"; do
    find /usr/lib/x86_64-linux-gnu -name "${lib}*.so*" 2>/dev/null | while read -r file; do
        if [ -L "$file" ]; then
            target=$(readlink -f "$file")
            echo "  $file -> $target"
        else
            echo "  $file"
        fi
    done
done
echo "----------------------------------------"

if [ -f "/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0" ]; then
    echo "✅ NVIDIA EGL libraries found"
    echo -e "\nOpenGL Capabilities:"
    echo "----------------------------------------"
    PYOPENGL_PLATFORM=egl python3 -c "
import OpenGL.GL as gl
try:
    print(f'Vendor   : {gl.glGetString(gl.GL_VENDOR).decode()}')
    print(f'Renderer : {gl.glGetString(gl.GL_RENDERER).decode()}')
    print(f'Version  : {gl.glGetString(gl.GL_VERSION).decode()}')
except Exception as e:
    print(f'⚠️  OpenGL query failed: {e}')" 2>/dev/null
else
    echo "⚠️  NVIDIA EGL libraries missing"
fi

# -----------------------------------------------------------------------------
# 4. Environment Configuration Check
# -----------------------------------------------------------------------------
echo -e "\n[4/7] 🔍 Checking Environment Configuration..."
echo "System Environment Variables:"
echo "----------------------------------------"
declare -A env_vars=(
    ["NVIDIA_VISIBLE_DEVICES"]="GPU visibility"
    ["NVIDIA_DRIVER_CAPABILITIES"]="Driver capabilities"
    ["PYOPENGL_PLATFORM"]="OpenGL platform"
    ["WINDOW_BACKEND"]="Window backend"
    ["__GLX_VENDOR_LIBRARY_NAME"]="GLX vendor"
    ["CUDA_VISIBLE_DEVICES"]="CUDA devices"
    ["CUDA_HOME"]="CUDA installation"
    ["LD_LIBRARY_PATH"]="Library path"
)

for var in "${!env_vars[@]}"; do
    value=${!var:-"Not Set"}
    desc=${env_vars[$var]}
    printf "%-25s: %s\n" "$var" "$value"
    if [ "$value" = "Not Set" ]; then
        printf "%-25s  ℹ️  %s\n" " " "$desc required"
    fi
done

# -----------------------------------------------------------------------------
# 5. Graphics Configuration Check
# -----------------------------------------------------------------------------
echo -e "\n[5/7] 🔍 Checking Graphics Configuration..."
echo "EGL/OpenGL Configuration:"
echo "----------------------------------------"
if [ -d "/usr/share/glvnd/egl_vendor.d/" ]; then
    find /usr/share/glvnd/egl_vendor.d/ -type f -exec echo "==> {}" \; -exec cat {} \; 2>/dev/null
    echo "✅ EGL vendor configurations found"
else
    echo "⚠️  EGL vendor directory missing"
fi

# -----------------------------------------------------------------------------
# 6. DepthFlow Installation Check
# -----------------------------------------------------------------------------
echo -e "\n[6/7] 🔍 Checking DepthFlow Setup..."
export DEPTHFLOW_SUPPRESS_ROOT_WARNING=1

echo "DepthFlow Dependencies and GPU Test:"
echo "----------------------------------------"
python3 -c """
import importlib
import sys

def check_package(package_name):
    try:
        module = importlib.import_module(package_name)
        return True
    except:
        return False

# Basic package check
packages = [
    'DepthFlow',
    'torch',
    'numpy',
    'moderngl',
    'shaderflow'
]

print('Package Status:')
for pkg_name in packages:
    try:
        if check_package(pkg_name):
            print(f'✅ {pkg_name:<12} Installed')
        else:
            print(f'⚠️  {pkg_name:<12} Not installed')
    except:
        print(f'⚠️  {pkg_name:<12} Check failed')

# Test GPU functionality
print('\nGPU Test:')
try:
    import torch
    import DepthFlow
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f'✅ GPU Found: {torch.cuda.get_device_name(0)}')
        
        # Test DepthFlow GPU
        try:
            model = DepthFlow.load_model('small')  # or 'base' or 'large'
            model.to('cuda')
            print('✅ DepthFlow loaded on GPU')
            
            # Optional: Test inference
            try:
                import numpy as np
                test_input = np.zeros((1, 3, 384, 384))  # Example input
                with torch.no_grad():
                    _ = model(torch.from_numpy(test_input).cuda().float())
                print('✅ DepthFlow GPU inference test passed')
            except:
                print('⚠️  Could not test inference')
        except:
            print('⚠️  Could not load DepthFlow model on GPU')
    else:
        print('⚠️  No GPU available')
except Exception as e:
    print(f'⚠️  GPU test failed: {str(e)}')

# Check compatibility
print('\nCompatibility Check:')
try:
    import torch
    import numpy as np
    
    torch_version = torch.__version__.split('+')[0]
    numpy_version = np.__version__
    cuda_version = torch.version.cuda if torch.cuda.is_available() else 'N/A'
    
    print(f'PyTorch : {torch_version}')
    print(f'CUDA    : {cuda_version}')
    print(f'NumPy   : {numpy_version}')
    
    # Known good configurations
    if torch_version.startswith('1.13'):
        print('✅ PyTorch version compatible with DepthFlow')
    else:
        print('⚠️  Untested PyTorch version')
except:
    print('⚠️  Could not check version compatibility')
"""

# -----------------------------------------------------------------------------
# 7. Compatibility Matrix Check
# -----------------------------------------------------------------------------
echo -e "\n[7/7] 🔍 Checking Compatibility Matrix..."
echo "Component Version Matrix:"
echo "----------------------------------------"
python3 -c """
import platform
import sys

def get_version(package_name):
    try:
        if package_name == 'torch':
            import torch
            return {
                'version': torch.__version__,
                'cuda': torch.version.cuda if torch.cuda.is_available() else 'N/A',
                'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
            }
        elif package_name == 'system':
            return {
                'os': platform.system(),
                'release': platform.release(),
                'python': platform.python_version()
            }
        else:
            return {'version': 'unknown'}
    except:
        return {'version': 'error checking'}

# Get component versions
system_info = get_version('system')
torch_info = get_version('torch')

# Print matrix
try:
    print(f'System      : {system_info["os"]} {system_info["release"]}')
    print(f'Python      : {system_info["python"]}')
    print(f'PyTorch     : {torch_info["version"]}')
    print(f'CUDA        : {torch_info["cuda"]}')
    print(f'GPU         : {torch_info["gpu"]}')
except:
    print('⚠️  Error displaying version matrix')
"""

# =============================================================================
# Final Summary and Recommendations
# =============================================================================
echo -e "\n╔════════════════════════════════╗"
echo "║      Environment Summary      ║"
echo "╚════════════════════════════════╝"

# Component Status Check
echo "Component Status:"
echo "------------------------------------------------"
nvidia_driver=$(nvidia-smi > /dev/null 2>&1 && echo "✅" || echo "⚠️ ")
cuda_toolkit=$(nvcc --version > /dev/null 2>&1 && echo "✅" || echo "⚠️ ")
egl_libs=$([ -f "/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0" ] && echo "✅" || echo "⚠️ ")
env_vars=$([ ! -z "$PYOPENGL_PLATFORM" ] && echo "✅" || echo "⚠️ ")
egl_config=$([ -f "/usr/share/glvnd/egl_vendor.d/10_nvidia.json" ] && echo "✅" || echo "⚠️ ")
depthflow=$(DEPTHFLOW_SUPPRESS_ROOT_WARNING=1 python3 -c "import DepthFlow" 2>/dev/null && echo "✅" || echo "⚠️ ")

printf "%-20s : %s\n" "NVIDIA Drivers" "$nvidia_driver"
printf "%-20s : %s\n" "CUDA Toolkit" "$cuda_toolkit"
printf "%-20s : %s\n" "OpenGL/EGL Libs" "$egl_libs"
printf "%-20s : %s\n" "Environment Vars" "$env_vars"
printf "%-20s : %s\n" "EGL Configuration" "$egl_config"
printf "%-20s : %s\n" "DepthFlow" "$depthflow"

# Detailed Recommendations
echo -e "\nDetailed Recommendations:"
echo "------------------------------------------------"

if [ "$nvidia_driver" = "⚠️ " ]; then
    echo "🔧 NVIDIA Driver Issues:"
    echo "  • Verify driver installation: nvidia-smi"
    echo "  • Check driver/CUDA compatibility"
    echo "  • Consider running: sudo nvidia-smi -pm 1"
fi

if [ "$env_vars" = "⚠️ " ]; then
    echo "🔧 Environment Configuration:"
    echo "  • Add to your environment:"
    echo "    export PYOPENGL_PLATFORM=egl"
    echo "    export __GLX_VENDOR_LIBRARY_NAME=nvidia"
    echo "    export NVIDIA_DRIVER_CAPABILITIES=all"
fi

if [ "$depthflow" = "⚠️ " ]; then
    echo "🔧 DepthFlow Setup:"
    echo "  • Verify installation: pip show depthflow"
    echo "  • Check dependencies: pip install -r requirements.txt"
    echo "  • Ensure CUDA toolkit matches PyTorch version"
fi

echo -e "\nNote: ⚠️  warnings indicate potential issues that need attention"
echo "### Verification Complete ###"
# === 4. ERROR REPORTING ===
if [ ${#ERROR_LOG[@]} -ne 0 ]; then
    echo -e "\n=== 📝 Error Summary ==="
    echo "The following errors occurred but did not stop execution:"
    printf '%s\n' "${ERROR_LOG[@]}"
fi

if [ ${#failed_installations[@]} -ne 0 ]; then
    echo -e "\n=== 📝 Failed Installations ==="
    printf '%s\n' "${failed_installations[@]}"
fi

if [ ${#skipped_steps[@]} -ne 0 ]; then
    echo -e "\n=== 📝 Skipped Steps ==="
    printf '%s\n' "${skipped_steps[@]}"
fi
rm -f /tmp/sd_comfy_error.log /tmp/pip_error.log
