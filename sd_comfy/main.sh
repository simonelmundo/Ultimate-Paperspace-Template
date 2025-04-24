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
    export CUDA_VISIBLE_DEVICES=0 # Redundant, already set above
    
    # Set environment variables to moderate VRAM usage for 16GB
    export COMFY_MAX_LOADED_MODELS=5 # Reduced from 100
    export COMFY_MAX_IMAGE_CACHE_SIZE=8 # Reduced from 32
}

install_cuda_12() {
    echo "Installing CUDA 12.1 (minimal installation)..."
    
    # Create marker file to avoid reinstallation
    CUDA_MARKER="/storage/.cuda_12.1_installed"
    if [ -f "$CUDA_MARKER" ]; then
        echo "CUDA 12.1 already installed (marker file exists)."
        setup_cuda_env
        return 0
    fi
    
    # Clean up existing CUDA 11.x if present
    if dpkg -l | grep -q "cuda-11"; then
        echo "Removing existing CUDA 11.x installations..."
        apt-get --purge remove -y cuda-11-* || echo "No CUDA 11.x found to remove"
    fi
    
    # Install only essential CUDA components
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt-get update -qq
    
    # Install only the minimal required packages instead of the full toolkit
    echo "Installing minimal CUDA components..."
    apt-get install -y --no-install-recommends \
        cuda-cudart-12-1 \
        cuda-nvcc-12-1 \
        cuda-cupti-12-1 \
        libcublas-12-1 \
        libcufft-12-1 \
        libcurand-12-1 \
        libcusolver-12-1 \
        libcusparse-12-1 \
        libnpp-12-1
    
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
    
    # Create marker file
    touch "$CUDA_MARKER"
    
    # Verify installation
    echo "Verifying CUDA 12.1 installation..."
    nvcc --version || { echo "CUDA installation verification failed"; return 1; }
}

setup_environment() {
    # Check if NVCC is available
    if command -v nvcc &>/dev/null; then
        local cuda_version=$(nvcc --version | grep 'release' | awk '{print $6}' || echo "unknown")
        echo "System CUDA Version: $cuda_version"
        
        # Install CUDA 12.1 if needed
        if [[ "$cuda_version" != "12.1" ]]; then
            install_cuda_12
        else
            echo "CUDA 12.1 already installed."
            setup_cuda_env
        fi
    else
        echo "NVCC not found, installing CUDA 12.1..."
        install_cuda_12
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

# Function to check if PyTorch versions match requirements
check_torch_versions() {
    echo "Checking PyTorch versions..."
    
    # Check if packages are installed with correct versions
    local TORCH_INSTALLED=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "not_installed")
    local TORCHVISION_INSTALLED=$(python3 -c "import torchvision; print(torchvision.__version__)" 2>/dev/null || echo "not_installed")
    local TORCHAUDIO_INSTALLED=$(python3 -c "import torchaudio; print(torchaudio.__version__)" 2>/dev/null || echo "not_installed")
    local XFORMERS_INSTALLED=$(python3 -c "import xformers; print(xformers.__version__)" 2>/dev/null || echo "not_installed")
    
    # Extract base versions without CUDA suffix for comparison
    local TORCH_BASE_VERSION=$(echo "${TORCH_VERSION}" | cut -d'+' -f1)
    local TORCHVISION_BASE_VERSION=$(echo "${TORCHVISION_VERSION}" | cut -d'+' -f1)
    local TORCHAUDIO_BASE_VERSION=$(echo "${TORCHAUDIO_VERSION}" | cut -d'+' -f1)
    
    # Extract installed base versions
    local TORCH_INSTALLED_BASE=$(echo "${TORCH_INSTALLED}" | cut -d'+' -f1)
    local TORCHVISION_INSTALLED_BASE=$(echo "${TORCHVISION_INSTALLED}" | cut -d'+' -f1)
    local TORCHAUDIO_INSTALLED_BASE=$(echo "${TORCHAUDIO_INSTALLED}" | cut -d'+' -f1)
    
    # Check CUDA availability
    local CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
    
    echo "Current versions:"
    echo "- torch: ${TORCH_INSTALLED} (required: ${TORCH_VERSION})"
    echo "- torchvision: ${TORCHVISION_INSTALLED} (required: ${TORCHVISION_VERSION})"
    echo "- torchaudio: ${TORCHAUDIO_INSTALLED} (required: ${TORCHAUDIO_VERSION})"
    echo "- xformers: ${XFORMERS_INSTALLED} (required: ${XFORMERS_VERSION})"
    echo "- CUDA available: ${CUDA_AVAILABLE}"
    
    # Check if any package needs reinstallation
    if [[ "${TORCH_INSTALLED_BASE}" != "${TORCH_BASE_VERSION}" || 
          "${TORCHVISION_INSTALLED_BASE}" != "${TORCHVISION_BASE_VERSION}" || 
          "${TORCHAUDIO_INSTALLED_BASE}" != "${TORCHAUDIO_BASE_VERSION}" || 
          "${XFORMERS_INSTALLED}" != "${XFORMERS_VERSION}" || 
          "${CUDA_AVAILABLE}" != "True" ]]; then
        echo "PyTorch ecosystem needs reinstallation"
        return 1  # Needs reinstallation
    else
        echo "PyTorch ecosystem is already at the correct versions"
        return 0  # No reinstallation needed
    fi
}

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
        
    if torch.__version__.split('+')[0] != '${TORCH_VERSION}'.split('+')[0]:
        print('Warning: Unexpected PyTorch version installed')
        
except ImportError as e:
    print(f'Warning: Missing package - {str(e)}')
except Exception as e:
    print(f'Warning: Verification script had issues: {str(e)}')
" || echo "Warning: Verification script had issues, but continuing..."
}

# Main function to fix torch versions
fix_torch_versions() {
    echo "Checking PyTorch/CUDA versions..."
    
    # Only reinstall if versions don't match
    if ! check_torch_versions; then
        echo "Installing required PyTorch versions..."
        clean_torch_installations
        install_torch_core
        install_xformers
        verify_installations
    else
        echo "PyTorch ecosystem already at correct versions, skipping reinstallation"
    fi
    
    echo "PyTorch ecosystem setup completed"
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
    echo "Installing essential system dependencies..."
    apt-get update && apt-get install -y \
        libatlas-base-dev libblas-dev liblapack-dev \
        libjpeg-dev libpng-dev \
        python3-dev build-essential \
        libgl1-mesa-dev || {
        echo "Warning: Some packages failed to install"
    }

    # Python environment setup
    pip install pip==24.0
    pip install --upgrade wheel setuptools
    pip install "numpy>=1.26.0,<2.3.0"

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
            
            # Add requirements from this file
            grep -v "^-r\|^#\|^$" "$file" >> "$combined_reqs"
            
            # Process included requirements files
            grep "^-r" "$file" | sed 's/^-r\s*//' | while read -r included_file; do
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
        python "/tmp/resolve_conflicts.py" "$combined_reqs" "/tmp/resolved_requirements.txt"
        mv "/tmp/resolved_requirements.txt" "$combined_reqs"
        
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
        python "$verify_script" "$combined_reqs" "/tmp/missing_packages.txt"
        
        # Install packages in smaller batches to avoid dependency conflicts
        if [[ -s "/tmp/missing_packages.txt" ]]; then
            echo "${indent}Installing missing packages in batches..."
            
            # Split into smaller batches of 10 packages each
            split -l 10 "/tmp/missing_packages.txt" "/tmp/pkg_batch_"
            
            # Install each batch separately
            for batch in /tmp/pkg_batch_*; do
                echo "${indent}Installing batch $(basename "$batch")..."
                if ! pip install --no-cache-dir --disable-pip-version-check -r "$batch" 2>/dev/null; then
                    echo "${indent}Batch installation failed, falling back to individual installation..."
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
    }

    # Call the function with the requirements file
    process_requirements "/notebooks/sd_comfy/additional_requirements.txt"

    # TensorFlow installation
    pip install --cache-dir="$PIP_CACHE_DIR" "tensorflow>=2.8.0,<2.19.0"
    # SageAttention Installation Process
    install_sageattention() {
        # Initialize environment
        echo "Starting SageAttention installation for HunyuanVideo support..."
        setup_environment
        create_directories
        setup_ccache
        
        # Check for cached installation
        if check_and_install_cached_wheel; then
            return 0
        fi
        
        # Proceed with full installation
        install_dependencies
        clone_or_update_repo
        build_and_install
    }

    # Environment Setup
    setup_environment() {
        export CUDA_HOME=/usr/local/cuda-12.1
        export PATH=$CUDA_HOME/bin:$PATH
        export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
        export FORCE_CUDA=1
        export TORCH_CUDA_ARCH_LIST="8.6"
        export MAX_JOBS=$(nproc)
        export USE_NINJA=1
        export TORCH_NVCC_FLAGS="-Xptxas --disable-warnings"
        export NVCC_FLAGS="-Xptxas --disable-warnings"
    }

    # Directory Management
    create_directories() {
        export TORCH_EXTENSIONS_DIR="/storage/.torch_extensions"
        export SAGEATTENTION_CACHE_DIR="/storage/.sageattention_cache"
        export WHEEL_CACHE_DIR="/storage/.wheel_cache"
        mkdir -p "$TORCH_EXTENSIONS_DIR" "$SAGEATTENTION_CACHE_DIR" "$WHEEL_CACHE_DIR"
        echo "Created cache directories"
    }

    # Ccache Configuration
    setup_ccache() {
        if command -v ccache &> /dev/null; then
            export CMAKE_C_COMPILER_LAUNCHER=ccache
            export CMAKE_CXX_COMPILER_LAUNCHER=ccache
            ccache --max-size=3G
            ccache -z
        fi
    }

    # Cached Wheel Handling
    check_and_install_cached_wheel() {
        SAGE_VERSION="2.1.1"
        CUDA_VERSION=$(nvcc --version | grep release | awk '{print $6}' | cut -c2-)
        ARCH=$(uname -m)
        SAGE_CACHE_MARKER="$SAGEATTENTION_CACHE_DIR/sage_${SAGE_VERSION}_${ARCH}_cuda${CUDA_VERSION}.installed"
        SAGE_WHEEL="$WHEEL_CACHE_DIR/sageattention-${SAGE_VERSION}-py3.10-${ARCH}-linux-gnu.whl"

        if [ -f "$SAGE_WHEEL" ]; then
            echo "Found cached wheel at $SAGE_WHEEL"
            if pip install --no-index --find-links="$WHEEL_CACHE_DIR" sageattention==$SAGE_VERSION; then
                handle_successful_installation
                return 0
            fi
        fi
        return 1
    }

    # Installation Success Handling
    handle_successful_installation() {
        SAGE_MODULE_PATH=$(python -c "import sageattention; print(sageattention.__path__[0])")
        create_compatibility_symlink "$SAGE_MODULE_PATH"
        touch "$SAGE_CACHE_MARKER"
        echo "✅ SageAttention setup complete using cached wheel"
    }

    # Symlink Creation
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
        echo "Installing SageAttention dependencies..."
        pip install --cache-dir="$PIP_CACHE_DIR" \
            "ninja>=1.11.0" \
            "triton>=3.0.0" \
            "accelerate>=1.1.1" \
            "diffusers>=0.31.0" \
            "transformers>=4.39.3"
    }

    # Repository Management
    clone_or_update_repo() {
        SAGE_BUILD_DIR="/storage/SageAttention"
        if [ -d "$SAGE_BUILD_DIR" ]; then
            cd "$SAGE_BUILD_DIR"
            git pull
        else
            git clone https://github.com/thu-ml/SageAttention.git "$SAGE_BUILD_DIR"
            cd "$SAGE_BUILD_DIR"
        fi
    }

    # Build and Installation
    build_and_install() {
        echo "Building SageAttention wheel..."
        # Determine build type (optimized or standard)
        if [[ -f "csrc/fused/fused_attention.cu" ]]; then
            optimized_build
        else
            standard_build
        fi

        # Check if a wheel was built
        BUILT_WHEEL=$(find dist -name "*.whl" | head -1)
        if [[ -n "$BUILT_WHEEL" ]]; then
            handle_built_wheel "$BUILT_WHEEL"
        else
            # Log failure instead of exiting
            log "❌ Failed to build SageAttention wheel. No wheel file found in dist/. Continuing script..."
            # Do not exit, allow the script to continue
        fi
    }

    # Optimized Build Process
    optimized_build() {
        echo "Using custom optimized build..."
        cat > setup_optimized.py << 'EOF'
import os
import sys
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Optimize build process
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
os.environ['MAX_JOBS'] = os.environ.get('MAX_JOBS', str(os.cpu_count()))
os.environ['NVCC_FLAGS'] = os.environ.get('NVCC_FLAGS', '') + ' -Xfatbin -compress-all'

cuda_sources = ['csrc/fused/fused_attention.cu']
cpp_sources = ['csrc/fused/pybind.cpp']

setup(
    name='sageattention',
    version='2.1.1',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='sageattention._fused',
            sources=cpp_sources + cuda_sources,
            extra_compile_args={
                'cxx': ['-O3', '-fopenmp', '-lgomp', '-std=c++17', '-DENABLE_BF16', '-D_GLIBCXX_USE_CXX11_ABI=0'],
                'nvcc': [
                    '-O3', '--use_fast_math', '-std=c++17',
                    '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '-U__CUDA_NO_HALF2_OPERATORS__', '-DENABLE_BF16',
                    '--threads', '4', '-D_GLIBCXX_USE_CXX11_ABI=0'
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
EOF
        echo "Starting optimized build with progress reporting..."
        python setup_optimized.py bdist_wheel 2>&1 | grep -v "ptxas info\|bytes stack frame\|bytes spill" | grep --line-buffered "" | sed 's/^/[BUILD] /'
    }

    # Standard Build Process
    standard_build() {
        echo "Using standard build process..."
        python setup.py bdist_wheel 2>&1 | grep -v "ptxas info\|bytes stack frame\|bytes spill" | grep --line-buffered "" | sed 's/^/[BUILD] /'
    }

    # Built Wheel Handling
    handle_built_wheel() {
        local wheel_path=$1
        cp "$wheel_path" "$WHEEL_CACHE_DIR/"
        log "Cached wheel at $WHEEL_CACHE_DIR/$(basename $wheel_path)"

        # Attempt to install the built wheel
        if pip install "$wheel_path"; then
            # Verify installation via import
            if python -c "import sageattention; print('SageAttention installed successfully')" &>/dev/null; then
                handle_successful_installation
            else
                # Log failure instead of exiting
                log "❌ SageAttention installed but failed import check. Continuing script..."
                # Do not exit, allow the script to continue
            fi
        else
            # Log failure instead of exiting
            log "❌ Failed to install SageAttention wheel from $wheel_path. Continuing script..."
            # Do not exit, allow the script to continue
        fi
    }

    # Execute installation (will now continue even on failure)
    install_sageattention

    # Fix torch versions (will run regardless of SageAttention status)
    fix_torch_versions
    touch /tmp/sd_comfy.prepared
    log "Completed SageAttention installation attempt and environment preparation"
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
  
  # A4000-specific VRAM optimization settings (16GB)
  export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:4096,garbage_collection_threshold:0.8"
  
  # Launch ComfyUI with A4000-optimized parameters using SageAttention
  PYTHONUNBUFFERED=1 service_loop "python main.py \
    --dont-print-server \
    --port $SD_COMFY_PORT \
    --cuda-malloc \
    --use-sage-attention \
    --preview-method auto \
    --bf16-vae \
    --fp16-unet \
    --cache-lru 5 \
    --reserve-vram 0.5 \
    --fast \
    --enable-compress-response-body \
    ${EXTRA_SD_COMFY_ARGS}" > $LOG_DIR/sd_comfy.log 2>&1 &
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
