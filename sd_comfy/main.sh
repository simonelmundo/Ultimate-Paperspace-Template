#!/bin/bash
set -e

current_dir=$(dirname "$(realpath "$0")")
cd $current_dir
source .env
setup_environment() {
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
    
    # Uninstall all torch-related packages
    echo "Removing existing PyTorch installations..."
    pip uninstall -y torch torchvision torchaudio xformers || true
    pip cache purge || true
    
    # Install correct versions for CUDA 11.6
    echo "Installing PyTorch ecosystem with CUDA 11.6..."
    pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116 || {
        echo "Warning: PyTorch installation had issues, but continuing..."
    }
    pip install xformers==0.0.16 || {
        echo "Warning: xformers installation had issues, but continuing..."
    }
    
    # Verify installations
    echo "Verifying installations..."
    python3 -c "
try:
    import torch
    import torchvision
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'TorchVision version: {torchvision.__version__}')
    print(f'CUDA version: {torch.version.cuda}')
except Exception as e:
    print(f'Warning: Verification had issues: {str(e)}')
" || echo "Warning: Verification script had issues, but continuing..."
    
    echo "PyTorch ecosystem installation completed"
    return 0  # Always return success
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
        mesa-common-dev

    # Verify installation
    if [ $? -ne 0 ]; then
        echo "Warning: Some packages failed to install"
        # But continue anyway since we want the script to keep running
    fi

    # Clean and install PyTorch ecosystem
    echo "Installing PyTorch ecosystem..."
    pip uninstall -y torch torchvision torchaudio xformers depthflow
    pip cache purge

    # Install in correct order with specific versions
    pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
    pip install xformers==0.0.16  # Specific version compatible with torch 1.13.1

 

    # Update pip first
    pip install pip==24.0

    # Update wheel and setuptools separately
    pip install --upgrade wheel
    pip install --upgrade setuptools

    # Continue with other installations
    pip install "numpy>=1.26.0,<2.3.0"

    # Clean and install PyTorch ecosystem
    echo "Installing PyTorch ecosystem..."
    pip uninstall -y torch torchvision torchaudio xformers shaderflow depthflow
    pip cache purge

    # Install in correct order with specific versions
    pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
    pip install xformers==0.0.16  # Specific version compatible with torch 1.13.1
    pip install shaderflow

    # Install and setup DepthFlow
    echo "Setting up DepthFlow..."
    export DEPTHFLOW_SUPPRESS_ROOT_WARNING=1

    # Clean and install DepthFlow
    cd /storage/stable-diffusion-comfy/custom_nodes
    rm -rf DepthFlow  # Remove existing installation
    FORCE_CUDA=1 pip install --no-cache-dir "depthflow==0.8.0.dev0"

    # Create symlink and verify
    if python3 -c "import DepthFlow; print(f'DepthFlow {DepthFlow.__version__} installed')" 2>/dev/null; then
        ln -sf "/tmp/sd_comfy-env/lib/python3.10/site-packages/DepthFlow" DepthFlow
    else
        echo "DepthFlow installation failed"
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
echo -e "\n[1/3] 🔍 Checking NVIDIA Driver and System..."
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
echo -e "\n[2/3] 🔍 Checking CUDA Environment..."

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
# 6. DepthFlow Installation Check
# -----------------------------------------------------------------------------
echo -e "\n[3/3] 🔍 Checking DepthFlow Setup..."
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


# Run the fix at the end
echo "Running final version check and fixes..."
fix_torch_versions || true  # Continue even if function has issues
