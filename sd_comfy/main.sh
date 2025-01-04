#!/bin/bash
set -e

current_dir=$(dirname "$(realpath "$0")")
cd $current_dir
source .env
verify_nvidia() {
    local needs_install=0
    
    # Check NVIDIA driver
    if ! nvidia-smi &>/dev/null; then
        echo "NVIDIA drivers not found"
        needs_install=1
    fi
    
    # Check GL libraries
    if ! ldconfig -p | grep -q "libEGL_nvidia.so.0"; then
        echo "NVIDIA GL libraries not found"
        needs_install=1
    fi
    
    if [ $needs_install -eq 1 ]; then
        echo "Installing NVIDIA drivers and libraries..."
        apt-get update
        apt-get install -y nvidia-driver-535 libnvidia-gl-535 libnvidia-egl-wayland1
    fi
    
    return 0
}
# 2. Set environment variables globally
setup_environment() {
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
}

# Set up a trap to call the error_exit function on ERR signal
trap 'error_exit "### ERROR ###"' ERR


echo "### Setting up Stable Diffusion Comfy ###"
log "Setting up Stable Diffusion Comfy"
if [[ "$REINSTALL_SD_COMFY" || ! -f "/tmp/sd_comfy.prepared" ]]; then
    # Verify NVIDIA and setup environment first
    verify_nvidia || exit 1
    setup_environment

    TARGET_REPO_URL="https://github.com/comfyanonymous/ComfyUI.git" \
    TARGET_REPO_DIR=$REPO_DIR \
    UPDATE_REPO=$SD_COMFY_UPDATE_REPO \
    UPDATE_REPO_COMMIT=$SD_COMFY_UPDATE_REPO_COMMIT \
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

    # Install required system packages
    echo "Installing system dependencies..."
    apt-get update && apt-get install -y \
        libx11-dev \
        libxrandr-dev \
        libxinerama-dev \
        libxcursor-dev \
        libxi-dev \
        libatlas-base-dev \
        libblas-dev \
        liblapack-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libbz2-dev \
        libgl1-mesa-dev \
        python2-dev \
        libopenblas-dev \
        cmake \
        build-essential \
        || true
    

# Set up NVIDIA EGL configuration
echo "Setting up NVIDIA EGL configuration..."
if [ ! -f "/usr/share/glvnd/egl_vendor.d/10_nvidia.json" ]; then
    mkdir -p /usr/share/glvnd/egl_vendor.d
    cat > /usr/share/glvnd/egl_vendor.d/10_nvidia.json << 'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libnvidia-egl-wayland.so.1"
    }
}
EOF
    echo "Created EGL configuration file"
else
    echo "EGL configuration already exists"
fi
    # Initialize array for failed installations
    failed_installations=()

    # Function to attempt installation and track failures
    try_install() {
        local package="$1"
        echo "Installing: $package"
        if ! pip install $package 2>&1 | tee /tmp/pip_install.log; then
            failed_installations+=("$package")
            echo "Failed to install: $package"
            return 0
        fi
        return 0
    }

    # Function to safely process requirements file
    process_requirements() {
        local req_file="$1"
        echo "Processing requirements from: $req_file"
        while IFS= read -r requirement || [ -n "$requirement" ]; do
            if [[ ! -z "$requirement" && ! "$requirement" =~ ^# ]]; then
                # Skip local directory references
                if [[ "$requirement" =~ ^file:///storage/stable-diffusion-comfy ]]; then
                    echo "Skipping local directory reference: $requirement"
                    continue
                fi
                try_install "$requirement"
            fi
        done < "$req_file"
    }

    # Function to verify DepthFlow installation
    verify_depthflow() {
        echo "Verifying DepthFlow installation..."
        if python3 -c "import DepthFlow.Motion; import DepthFlow.Resources" 2>/dev/null; then
            echo "DepthFlow modules verified successfully"
            return 0
        else
            echo "DepthFlow verification failed, attempting reinstallation..."
            pip uninstall -y depthflow 2>/dev/null || true
            try_install "depthflow --no-cache-dir"
            
            # Verify again after reinstall
            if python3 -c "import DepthFlow.Motion; import DepthFlow.Resources" 2>/dev/null; then
                echo "DepthFlow reinstallation successful"
                return 0
            else
                echo "DepthFlow installation failed. Adding to failed installations."
                failed_installations+=("depthflow (modules missing: Motion, Resources)")
                return 1
            fi
        fi
    }

    # Install base requirements first
    try_install "pip==24.0"
    try_install "--upgrade wheel setuptools"
    try_install "numpy>=1.26.0,<2.3.0"

    # Clean and install PyTorch ecosystem
    echo "Installing PyTorch ecosystem..."
    pip uninstall -y torch torchvision torchaudio xformers shaderflow depthflow
    pip cache purge

    # Install in correct order with specific versions
    pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
    pip install xformers==0.0.16  # Specific version compatible with torch 1.13.1
    pip install shaderflow

    # Verify PyTorch installation
    python3 -c "import torch; assert torch.__version__.startswith('1.13.1')" || {
        echo "PyTorch installation verification failed"
        failed_installations+=("pytorch-verification")
    }

    # Install OpenGL dependencies
    try_install "PyOpenGL"
    try_install "PyOpenGL_accelerate"

    # Setup DepthFlow
    echo "Installing and configuring DepthFlow..."
export DEPTHFLOW_SUPPRESS_ROOT_WARNING=1

    # Clean up any existing installations first
    pip uninstall -y depthflow shaderflow 2>/dev/null || true
    cd /storage/stable-diffusion-comfy/custom_nodes
    if [ -d "DepthFlow" ]; then
        rm -rf DepthFlow
    elif [ -L "DepthFlow" ]; then
        rm -f DepthFlow
    fi

    # Install dependencies first
    try_install "shaderflow"

    # Create answer file for DepthFlow's CUDA prompt
    echo "cuda" > /tmp/depthflow_answer

    # Install DepthFlow with shaderflow extras
    FORCE_CUDA=1 cat /tmp/depthflow_answer | try_install "depthflow[shaderflow]==0.8.0.dev0"
    rm /tmp/depthflow_answer

    # Create clean symlink for DepthFlow
    if python3 -c "import DepthFlow" 2>/dev/null; then
        echo "Setting up DepthFlow symlink..."
        cd /storage/stable-diffusion-comfy/custom_nodes
        ln -sf "/tmp/sd_comfy-env/lib/python3.10/site-packages/DepthFlow" DepthFlow
        echo "DepthFlow symlink created successfully"
    else
        echo "Warning: DepthFlow installation not found"
    fi

    # Handle tensorflow version compatibility
    if ! pip install "tensorflow==2.6.2" 2>/dev/null; then
        echo "Attempting to install compatible tensorflow version..."
        try_install "tensorflow>=2.8.0,<2.19.0"
    fi
    
    # Handle imgui-bundle with specific build requirements
    export CMAKE_ARGS="-DUSE_X11=ON"
    try_install "imgui-bundle --no-cache-dir"
    
    # Process requirements files
    if [ -f "requirements.txt" ]; then
        process_requirements "requirements.txt"
    else
        echo "No requirements.txt found, skipping..."
    fi
    
    # Install custom nodes requirements
    echo "Installing DepthFlow node requirements..."
    if [ -f "/storage/stable-diffusion-comfy/custom_nodes/ComfyUI-Depthflow-Nodes/requirements.txt" ]; then
        process_requirements "/storage/stable-diffusion-comfy/custom_nodes/ComfyUI-Depthflow-Nodes/requirements.txt"
    else
        echo "No DepthFlow requirements.txt found, skipping..."
    fi
    
    process_requirements "/notebooks/sd_comfy/additional_requirements.txt"

    # Display failed installations if any
    if [ ${#failed_installations[@]} -ne 0 ]; then
        echo "### The following installations failed ###"
        printf '%s\n' "${failed_installations[@]}"
        echo "### End of failed installations ###"
    else
        echo "All packages installed successfully!"
    fi

    
    touch /tmp/sd_comfy.prepared
else
    
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
if nvidia-smi > /tmp/nvidia_check 2>&1; then
    echo "Driver Information:"
    echo "----------------------------------------"
    cat /tmp/nvidia_check
    echo "----------------------------------------"
    
    echo "GPU Memory Status (Per Device):"
    echo "----------------------------------------"
    nvidia-smi --query-gpu=index,gpu_name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu --format=csv,noheader | while IFS="," read -r id name total used free temp util; do
        echo "GPU $id: $name"
        echo "  Memory  : $used/$total (Free: $free)"
        echo "  Temp    : $temp"
        echo "  Load    : $util"
    done
    
    echo -e "\nDriver/Library Versions:"
    echo "----------------------------------------"
    dpkg -l | grep -E "nvidia-driver|nvidia-utils|cuda" | awk '{printf "%-20s: %s\n", $2, $3}'
    
    echo "✅ NVIDIA drivers are working"
else
    echo "⚠️  Warning: NVIDIA driver issue detected"
    echo "Error: $(cat /tmp/nvidia_check)"
fi

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
