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
    export CUDA_HOME=/usr/local/cuda-12.6
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
    export CUDA_VISIBLE_DEVICES=0 # Redundant, already set above
    
    echo "CUDA Environment Variables Set:"
    echo "  CUDA_HOME=$CUDA_HOME"
    echo "  PATH (start): $CUDA_HOME/bin:..."
    echo "  LD_LIBRARY_PATH (start): $CUDA_HOME/lib64:..."
}

install_cuda_12() {
    echo "Installing CUDA 12.6 and essential build tools..."
    local CUDA_MARKER="/storage/.cuda_12.6_installed"
    local APT_INSTALL_LOG="$LOG_DIR/apt_cuda_install.log" # Log file for apt output

    # Check marker and verify existing installation (logic from previous step)
    if [ -f "$CUDA_MARKER" ]; then
        echo "CUDA 12.6 marker file exists. Verifying installation..."
        setup_cuda_env
        hash -r
        if command -v nvcc &>/dev/null && [[ "$(nvcc --version 2>&1 | grep 'release' | awk '{print $6}' | sed 's/^V//')" == "12.6"* ]]; then
             echo "CUDA 12.6 already installed and verified."
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
        libgl1-mesa-dev \
        cuda-cudart-12-6 \
        cuda-cudart-dev-12-6 \
        cuda-nvcc-12-6 \
        cuda-cupti-12-6 \
        cuda-cupti-dev-12-6 \
        libcublas-12-6 \
        libcublas-dev-12-6 \
        libcufft-12-6 \
        libcufft-dev-12-6 \
        libcurand-12-6 \
        libcurand-dev-12-6 \
        libcusolver-12-6 \
        libcusolver-dev-12-6 \
        libcusparse-12-6 \
        libcusparse-dev-12-6 \
        libnpp-12-6 \
        libnpp-dev-12-6 >> "$APT_INSTALL_LOG" 2>&1
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
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export FORCE_CUDA=1
EOL
    chmod +x /etc/profile.d/cuda12.sh

    # Verify installation *before* creating marker
    echo "Verifying CUDA 12.6 installation..."
    if command -v nvcc &>/dev/null; then
        local installed_version
        installed_version=$(nvcc --version 2>&1 | grep 'release' | awk '{print $6}' | sed 's/^V//')
        if [[ "$installed_version" == "12.6"* ]]; then
            echo "CUDA 12.6 installation verified successfully (Version: $installed_version)."
            touch "$CUDA_MARKER"
            echo "Installation marker created: $CUDA_MARKER"
            return 0
        else
            log_error "CUDA 12.6 installation verification failed. Found nvcc, but version is '$installed_version'."
            log_error "Which nvcc: $(which nvcc)"
            log_error "PATH: $PATH"
            return 1
        fi
    else
        log_error "CUDA 12.6 installation verification failed. NVCC command not found after installation attempt."
        log_error "Check /usr/local/cuda-12.6/bin exists and contains nvcc."
        ls -l /usr/local/cuda-12.6/bin/nvcc || true
        return 1
    fi
}

setup_environment() {
    echo "Attempting to set up CUDA 12.6 environment..."
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

        # Verify if the detected version is the target 12.6
        if [[ "$cuda_version" == "12.6"* ]]; then
            echo "CUDA 12.6 environment appears correctly configured."
            # Environment is already set by setup_cuda_env above
        else
            echo "Found nvcc, but version is '$cuda_version', not 12.6. Attempting installation/reconfiguration..."
            install_cuda_12
            # Re-clear hash after potential installation changes PATH again
            hash -r
        fi
    else
        # If nvcc is NOT found even after setting the PATH and clearing hash
        echo "NVCC not found after setting environment variables and clearing hash. Installing CUDA 12.6..."
        install_cuda_12
        # Re-clear hash after potential installation changes PATH again
        hash -r
    fi
}

#######################################
# STEP 3: PYTORCH VERSION MANAGEMENT
#######################################
# Define package versions and URLs as constants
readonly TORCH_VERSION="2.7.1+cu126"
readonly TORCHVISION_VERSION="0.22.1+cu126"
readonly TORCHAUDIO_VERSION="2.7.1+cu126"
readonly XFORMERS_VERSION="0.0.28.post1"
readonly TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"

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
    
    # Check if core PyTorch packages need reinstallation (these are critical)
    local core_packages_need_reinstall=false
    if [[ "${TORCH_INSTALLED_BASE}" != "${TORCH_BASE_VERSION}" || 
          "${TORCHVISION_INSTALLED_BASE}" != "${TORCHVISION_BASE_VERSION}" || 
          "${TORCHAUDIO_INSTALLED_BASE}" != "${TORCHAUDIO_BASE_VERSION}" || 
          "${CUDA_AVAILABLE}" != "True" ]]; then
        core_packages_need_reinstall=true
    fi
    
    # Check if xformers needs installation (this is optional but recommended)
    local xformers_needs_install=false
    if [[ "${XFORMERS_INSTALLED}" == "not_installed" ]]; then
        xformers_needs_install=true
    fi
    
    # Determine if reinstallation is needed
    if [[ "$core_packages_need_reinstall" == "true" ]]; then
        echo "PyTorch core packages need reinstallation"
        return 1  # Needs reinstallation
    elif [[ "$xformers_needs_install" == "true" ]]; then
        echo "PyTorch core packages are correct, but xformers is missing"
        echo "Will install xformers separately without reinstalling core packages"
        return 2  # Needs xformers installation only
    else
        echo "PyTorch ecosystem already at the correct versions"
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
    local install_cmd="pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --extra-index-url ${TORCH_INDEX_URL}"
    log "Running install command: $install_cmd"
    # Execute and capture output/status
    local output
    output=$($install_cmd 2>&1)
    local status=$?
    log "Pip install output:\n$output"
    if [[ $status -ne 0 ]]; then
        log_error "PyTorch core packages installation failed with status $status."
        # Decide if we should exit here or just warn
        # return 1 # Option to fail fast
    else
        log "PyTorch core packages installation command finished."
        # Add immediate import check
        log "Verifying torch import immediately after core install..."
        python -c "import torch; print(f'Torch {torch.__version__} imported successfully from {torch.__file__}')" || log_error "Failed to import torch immediately after core install."
    fi
    return $status # Return original status
}

# Function to install xformers
install_xformers() {
    echo "Installing xformers..."
    local install_cmd="pip install xformers==${XFORMERS_VERSION}"
    log "Running install command: $install_cmd"
    # Execute and capture output/status
    local output
    output=$($install_cmd 2>&1)
    local status=$?
    log "Pip install output:\n$output"
     if [[ $status -ne 0 ]]; then
        log_error "xformers installation failed with status $status."
        # return 1 # Option to fail fast
    else
        log "xformers installation command finished."
        # Add immediate import check
        log "Verifying xformers import immediately after install..."
        python -c "import xformers; print(f'xformers {xformers.__version__} imported successfully from {xformers.__file__}')" || log_error "Failed to import xformers immediately after install."
    fi
    return $status # Return original status
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
    
    # Check what needs to be done
    local check_result
    check_torch_versions
    check_result=$?
    
    case $check_result in
        0)
            echo "PyTorch ecosystem already at correct versions, skipping reinstallation"
            # Even if skipping, run a quick verification
            log "Running verification even though versions seem correct..."
            verify_installations
            ;;
        1)
            echo "Installing required PyTorch versions..."
            clean_torch_installations
            # Use || true for now to prevent exit on warning/failure within install functions
            install_torch_core || true 
            install_xformers || true
            verify_installations # This already prints versions
            ;;
        2)
            echo "PyTorch core packages are correct, installing xformers only..."
            # Install xformers without touching core packages
            install_xformers || true
            verify_installations # This already prints versions
            ;;
        *)
            echo "Unexpected return code from check_torch_versions: $check_result"
            log_error "PyTorch version check failed with unexpected status"
            return 1
            ;;
    esac
    
    log "PyTorch ecosystem setup completed"
    return 0 # Always return 0 for now, rely on logs for errors
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
    echo "Virtual environment activated: $VENV_DIR/sd_comfy-env"

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

    # --- FIX TORCH VERSIONS *BEFORE* SAGEATTENTION ---
    echo "Ensuring correct PyTorch ecosystem versions during initial setup..."
    fix_torch_versions # <<< ADDED THIS CALL
    fix_torch_status=$? # Capture exit status
    if [[ $fix_torch_status -ne 0 ]]; then
        # Log error, but maybe don't exit yet, let Sage build fail clearly
        log_error "fix_torch_versions function reported issues during initial setup (Status: $fix_torch_status). Check logs above."
        # exit 1 # Option to exit immediately
    else
        log "Initial PyTorch ecosystem setup/check completed."
    fi
    # --- END TORCH FIX ---

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
    }

    # Call the function with the requirements file

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
        local current_cuda_version
        current_cuda_version=$(nvcc --version | grep release | awk '{print $6}' | cut -c2- || echo "unknown")
        echo "Verifying cache against detected CUDA version: $current_cuda_version"
        if check_and_install_cached_wheel "$current_cuda_version"; then
            log "SageAttention already installed and cached for CUDA $current_cuda_version."
            return 0
        fi
        log "No suitable cached wheel found or installation failed. Proceeding with full build."
        
        # Proceed with full installation
        install_dependencies
        if clone_or_update_repo; then
             build_and_install
        else
             log_error "Failed to clone or update SageAttention repository. Skipping build."
             # Allow script to continue based on original logic
        fi
    }

    # Environment Setup
    setup_environment() {
        export CUDA_HOME=/usr/local/cuda-12.6
        export PATH=$CUDA_HOME/bin:$PATH
        export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
        export FORCE_CUDA=1
        export TORCH_CUDA_ARCH_LIST="8.6"
        export MAX_JOBS=$(nproc)
        export USE_NINJA=1
        echo "SageAttention Environment Setup:"
        echo "  CUDA_HOME=$CUDA_HOME"
        echo "  NVCC Check: $(nvcc --version || echo 'NVCC not found')"
        echo "  Python Check: $(python --version || echo 'python not found')" # Ensure this is python3.10
    }

    # Directory Management
    create_directories() {
        export TORCH_EXTENSIONS_DIR="/storage/.torch_extensions"
        local sage_cache_base="/storage/.sageattention_cache"
        local cuda_ver_for_path=$(nvcc --version | grep release | awk '{print $6}' | cut -c2- | sed 's/\.//g' || echo "unknowncuda")
        export SAGEATTENTION_CACHE_DIR="${sage_cache_base}/v2_cuda${cuda_ver_for_path}"
        export WHEEL_CACHE_DIR="/storage/.wheel_cache"
        mkdir -p "$TORCH_EXTENSIONS_DIR" "$SAGEATTENTION_CACHE_DIR" "$WHEEL_CACHE_DIR"
        echo "Created/Ensured directories:"
        echo "  Torch Extensions: $TORCH_EXTENSIONS_DIR"
        echo "  SageAttention Cache: $SAGEATTENTION_CACHE_DIR"
        echo "  Wheel Cache: $WHEEL_CACHE_DIR"
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
        local cuda_version_detected="$1"
        local sage_version="2.1.1"
        local arch=$(uname -m)
        local sage_cache_marker="$SAGEATTENTION_CACHE_DIR/sage_${sage_version}_${arch}_cuda${cuda_version_detected}.installed"
        local python_executable
        if [[ -n "$VENV_DIR" && -x "$VENV_DIR/sd_comfy-env/bin/python" ]]; then
            python_executable="$VENV_DIR/sd_comfy-env/bin/python"
        else
            log_error "[DEBUG] VENV Python executable not found or VENV_DIR not set! Falling back to 'python'."
            python_executable="python"
        fi
        local sage_wheel

        # Check marker file first
        if [ -f "$sage_cache_marker" ]; then
            log "Found installation marker: $sage_cache_marker"
            log "Verifying import from existing installation using: $python_executable"

            # --- BEGIN ENHANCED PYTHON DEBUG COMMAND ---
            local python_debug_script=$(cat <<EOF
import sys, os, site, pprint, traceback

print(f"--- Python Debug Info ---")
print(f"sys.executable: {sys.executable}")
print(f"sys.version: {sys.version}")
print(f"sys.prefix: {sys.prefix}")
print(f"sys.path:")
pprint.pprint(sys.path)
print(f"site.getsitepackages(): {site.getsitepackages()}")

# Check environment variables from Python's perspective
print(f"os.environ['PATH']: {os.environ.get('PATH', 'Not Set')}")
print(f"os.environ['LD_LIBRARY_PATH']: {os.environ.get('LD_LIBRARY_PATH', 'Not Set')}")
print(f"os.environ['PYTHONPATH']: {os.environ.get('PYTHONPATH', 'Not Set')}")
print(f"os.getcwd(): {os.getcwd()}") # Print current working directory from Python's perspective

# List contents of site-packages
site_packages_dirs = site.getsitepackages()
if not site_packages_dirs: # Handle cases like system python where getsitepackages might be empty
    site_packages_dirs = [os.path.join(sys.prefix, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages')]

for sp_dir in site_packages_dirs:
    print(f"--- Contents of site-packages dir '{sp_dir}': ---")
    try:
        if os.path.isdir(sp_dir):
            contents = os.listdir(sp_dir)
            pprint.pprint(sorted(contents)) # Sort for easier reading
        else:
            print(f"  Warning: Path '{sp_dir}' is not a directory.")
    except FileNotFoundError:
        print(f"  Error: Directory '{sp_dir}' not found.")
    except Exception as e:
        print(f"  Error listing contents of '{sp_dir}': {e}")
    print(f"--- End Contents of '{sp_dir}' ---")


print(f"--- Attempting Import ---")
import_successful = False
try:
    import sageattention
    print(f"Successfully imported sageattention")
    print(f"Location: {getattr(sageattention, '__file__', 'N/A')}")
    print(f"Version: {getattr(sageattention, '__version__', 'N/A')}")
    import_successful = True
except ImportError as e:
    print(f"ImportError occurred: {e}")
    print(f"Error type: {type(e)}")
    # traceback.print_exc() # Uncomment for full traceback if needed
except ModuleNotFoundError as e:
    print(f"ModuleNotFoundError occurred: {e}")
    print(f"Error type: {type(e)}")
    # traceback.print_exc() # Uncomment for full traceback if needed
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    print(f"Error type: {type(e)}")
    traceback.print_exc()

print(f"--- End Python Debug Info ---")
# Exit with 0 if import was successful, 1 otherwise
sys.exit(0 if import_successful else 1)
EOF
            )
            # --- END ENHANCED PYTHON DEBUG COMMAND ---


            # Execute the detailed Python debug script
            log "[DEBUG] --- Executing Python Debug Script ---"
            local import_output
            local import_status
            # Change to a neutral directory before running the check
            pushd /tmp > /dev/null
            # Run the script using the determined python executable
            import_output=$("$python_executable" -c "$python_debug_script" 2>&1)
            import_status=$?
            popd > /dev/null
            log "[DEBUG] --- Python Debug Script Finished (Exit Status: $import_status) ---"

            # Log the full output from the Python script
            log "[DEBUG] --- Python Script Output ---"
            echo "$import_output" | while IFS= read -r line; do log "[DEBUG] Python: $line"; done
            log "[DEBUG] --- End Python Script Output ---"


            # Check the exit status from the Python script
            if [ $import_status -eq 0 ]; then
               log "✅ SageAttention import verified from existing installation (via Python script)."
               return 0 # Success - marker valid, installation works
            else
               log_error "⚠️ Found marker but import failed (Python script exit status: $import_status). Removing marker and proceeding."
               # The detailed error is already logged above within the Python script output
               rm -f "$sage_cache_marker" # Remove invalid marker
               # Continue to check for wheel below
            fi
        fi

        # If marker not found OR import failed, check for wheel

        # Determine Python version tag (e.g., cp310) for wheel searching
        local py_version_short # e.g., 310
        py_version_short=$("$python_executable" -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')" 2>/dev/null)
        local python_version_tag="cp${py_version_short}" # e.g., cp310

        if [[ -z "$py_version_short" ]]; then
            log_error "Could not determine Python version for wheel search. Proceeding to build."
        else
            log "Searching for cached wheel with Python tag: ${python_version_tag}, sage_version: ${sage_version}, arch: ${arch} in ${WHEEL_CACHE_DIR}"
            # PEP 425: {python tag}-{abi tag}-{platform tag}. e.g., sageattention-2.1.1-cp310-cp310-linux_x86_64.whl
            # Search for a wheel matching the current Python version tag and architecture.
            sage_wheel=$(find "$WHEEL_CACHE_DIR" -maxdepth 1 -type f -name "sageattention-${sage_version}-${python_version_tag}-*-linux_${arch}.whl" -print -quit)

            if [[ -z "$sage_wheel" ]]; then
                log "Specific Python tag search failed. Trying broader search for ${python_version_tag}..."
                # Broader search: sageattention-VERSION*{python_version_tag}*{arch}.whl
                sage_wheel=$(find "$WHEEL_CACHE_DIR" -maxdepth 1 -type f -name "sageattention-${sage_version}*${python_version_tag}*${arch}*.whl" -print -quit)
            fi
        fi
        
        if [[ -f "$sage_wheel" ]]; then
            log "Found compatible cached wheel: $sage_wheel (CUDA version $cuda_version_detected context)"
            # Use --force-reinstall in case a broken version exists
            # Use the venv's pip explicitly
            if "$python_executable" -m pip install --force-reinstall --no-cache-dir --disable-pip-version-check "$sage_wheel"; then
                log "Verifying installation from cached wheel via import..."
                pushd /tmp > /dev/null # Change to neutral directory
                local wheel_import_output
                local wheel_import_status
                # Use the venv's python explicitly
                wheel_import_output=$("$python_executable" -c "import sageattention; print('SageAttention imported successfully from cached wheel.')" 2>&1)
                wheel_import_status=$?
                popd > /dev/null # Return to original directory

                if [ $wheel_import_status -eq 0 ]; then
                    log "$wheel_import_output" # Print success message from python
                    # Installation from wheel succeeded, create marker etc.
                    handle_successful_installation "$cuda_version_detected"
                    log "✅ Cached wheel installed and import verified."
                    return 0 # Success - installed from cache
                else
                    log_error "❌ Installed cached wheel but import verification failed."
                    log_error "Python import error output:"
                    log_error "-----------------------------------------"
                    echo "$wheel_import_output" | while IFS= read -r line; do log_error "$line"; done
                    log_error "-----------------------------------------"
                    # Do not return success, proceed to build
                fi
            else
                log_error "⚠️ Failed to install cached wheel: $sage_wheel. Proceeding to build."
            fi
        else
             log "No suitable installation marker or compatible cached wheel found in $WHEEL_CACHE_DIR for Python ${python_version_tag} / CUDA ${cuda_version_detected} context."
        fi
        log "Proceeding to build SageAttention."
        return 1 # Failure - need to build
    }

    # Installation Success Handling
    handle_successful_installation() {
        local cuda_version_installed="$1" # Expecting version like 12.6
        local sage_version="2.1.1"
        local arch=$(uname -m)
        # Ensure SAGEATTENTION_CACHE_DIR is set
        if [[ -z "$SAGEATTENTION_CACHE_DIR" ]]; then
            log_error "SAGEATTENTION_CACHE_DIR is not set in handle_successful_installation!"
            return 1
        fi
        local sage_cache_marker="$SAGEATTENTION_CACHE_DIR/sage_${sage_version}_${arch}_cuda${cuda_version_installed}.installed"

        local sage_module_path
        log "Attempting to determine SageAttention module path..."
        pushd /tmp > /dev/null # Change to neutral directory
        sage_module_path=$(python -c "import sageattention, os; print(os.path.dirname(sageattention.__file__))" 2>&1)
        local path_status=$?
        popd > /dev/null # Return to original directory


        if [[ $path_status -eq 0 && -n "$sage_module_path" && -d "$sage_module_path" ]]; then
             log "SageAttention module path found: $sage_module_path"
             # Symlink creation might not be strictly necessary if ComfyUI uses sageattention directly
             # create_compatibility_symlink "$sage_module_path"
             touch "$sage_cache_marker"
             log "✅ SageAttention setup complete. Installation marker created: $sage_cache_marker"
             return 0
        else
             log_error "⚠️ SageAttention installed and imports, but failed to determine module path via Python."
             log_error "Python output: $sage_module_path"
             log_error "Marker file will NOT be created. Build may re-occur on next run."
             # Marker file is NOT created if we can't verify path
             return 1 # Indicate partial failure
        fi
    }

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
        local cuda_version_built_with="$2" # Expecting version like 12.6
        log "Found built wheel: $wheel_path"
        # Ensure WHEEL_CACHE_DIR is set
        if [[ -z "$WHEEL_CACHE_DIR" ]]; then
            log_error "WHEEL_CACHE_DIR is not set in handle_built_wheel!"
            # Attempt to create it based on expected structure
            export WHEEL_CACHE_DIR="/storage/.wheel_cache"
            mkdir -p "$WHEEL_CACHE_DIR"
            log_warning "Attempted to create WHEEL_CACHE_DIR: $WHEEL_CACHE_DIR"
        fi
        mkdir -p "$WHEEL_CACHE_DIR" # Ensure it exists
        cp "$wheel_path" "$WHEEL_CACHE_DIR/"
        log "Cached wheel to $WHEEL_CACHE_DIR/$(basename "$wheel_path")"

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
                # Pass CUDA version used for build to create correct marker
                handle_successful_installation "$cuda_version_built_with"
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
    
    process_requirements "$REPO_DIR/requirements.txt"
    process_requirements "/notebooks/sd_comfy/additional_requirements.txt"
    install_sageattention

    # --- NUNCHAKU INSTALLATION ---
    install_nunchaku() {
        echo "Installing Nunchaku for enhanced machine learning capabilities..."
        log "Starting Nunchaku installation process"
        
        # Setup Nunchaku cache directories
        local nunchaku_cache_base="/storage/.nunchaku_cache"
        local nunchaku_version="0.2.0"
        local python_version
        python_version=$(python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')" 2>/dev/null)
        local arch=$(uname -m)
        local torch_version
        torch_version=$(python -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "unknown")
        
        # Create cache directory structure
        export NUNCHAKU_CACHE_DIR="${nunchaku_cache_base}/v${nunchaku_version}_cp${python_version}_torch${torch_version}_${arch}"
        mkdir -p "$NUNCHAKU_CACHE_DIR" "$WHEEL_CACHE_DIR"
        
        log "Nunchaku cache directory: $NUNCHAKU_CACHE_DIR"
        log "Wheel cache directory: $WHEEL_CACHE_DIR"
        
        # Check if Nunchaku is already installed and working
        local nunchaku_check_script="
import sys
try:
    import nunchaku
    print(f'Nunchaku {nunchaku.__version__} already installed and working')
    sys.exit(0)
except ImportError:
    print('Nunchaku not found, proceeding with installation')
    sys.exit(1)
except Exception as e:
    print(f'Nunchaku import error: {e}')
    sys.exit(1)
"
        
        if python -c "$nunchaku_check_script" 2>/dev/null; then
            log "Nunchaku already installed and working, skipping installation"
            return 0
        fi
        
        # Check PyTorch version to ensure compatibility
        local torch_version_check="
import torch
version = torch.__version__.split('+')[0]
major, minor = map(int, version.split('.')[:2])
if major > 2 or (major == 2 and minor >= 5):
    print(f'PyTorch {version} meets Nunchaku requirements (>=2.5)')
    sys.exit(0)
else:
    print(f'PyTorch {version} below Nunchaku requirements (>=2.5)')
    sys.exit(1)
"
        
        if ! python -c "$torch_version_check" 2>/dev/null; then
            log_error "PyTorch version incompatible with Nunchaku. Nunchaku requires PyTorch >=2.5"
            log_error "Current PyTorch version may need upgrade. Skipping Nunchaku installation."
            return 1
        fi
        
        if [[ -z "$python_version" ]]; then
            log_error "Could not determine Python version for Nunchaku wheel selection"
            return 1
        fi
        
        log "Detected Python version: $python_version"
        log "Detected PyTorch version: $torch_version"
        
        # Check for cached wheel first
        local nunchaku_wheel_name="nunchaku-${nunchaku_version}+torch2.6-cp${python_version}-cp${python_version}-linux_${arch}.whl"
        local cached_wheel="$WHEEL_CACHE_DIR/$nunchaku_wheel_name"
        local nunchaku_marker="$NUNCHAKU_CACHE_DIR/nunchaku_${nunchaku_version}_cp${python_version}_torch${torch_version}_${arch}.installed"
        
        # Check if we have a cached installation marker
        if [ -f "$nunchaku_marker" ]; then
            log "Found Nunchaku installation marker: $nunchaku_marker"
            log "Verifying cached installation..."
            
            local marker_verification_script="
import sys
try:
    import nunchaku
    print(f'✅ Cached Nunchaku {nunchaku.__version__} installation verified')
    print(f'Location: {nunchaku.__file__}')
    sys.exit(0)
except ImportError as e:
    print(f'❌ Cached Nunchaku import failed: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ Cached Nunchaku verification error: {e}')
    sys.exit(1)
"
            
            if python -c "$marker_verification_script" 2>/dev/null; then
                log "✅ Cached Nunchaku installation verified successfully"
                return 0
            else
                log_error "⚠️ Cached installation marker exists but verification failed. Removing marker."
                rm -f "$nunchaku_marker"
            fi
        fi
        
        # Check for cached wheel file
        if [ -f "$cached_wheel" ]; then
            log "Found cached Nunchaku wheel: $cached_wheel"
            log "Installing from cached wheel..."
            
            if pip install --no-cache-dir --disable-pip-version-check "$cached_wheel"; then
                log "Nunchaku installation from cached wheel completed successfully"
                
                # Verify installation
                local verification_script="
import sys
try:
    import nunchaku
    print(f'✅ Nunchaku {nunchaku.__version__} installed successfully from cached wheel')
    print(f'Location: {nunchaku.__file__}')
    
    # Test basic functionality
    if hasattr(nunchaku, '__version__'):
        print(f'Version check: OK')
    else:
        print('Warning: Version attribute not found')
    
    sys.exit(0)
except ImportError as e:
    print(f'❌ Nunchaku import failed: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ Nunchaku verification error: {e}')
    sys.exit(1)
"
                
                if python -c "$verification_script" 2>/dev/null; then
                    log "✅ Nunchaku installation from cached wheel verified successfully"
                    touch "$nunchaku_marker"
                    log "Created installation marker: $nunchaku_marker"
                    return 0
                else
                    log_error "❌ Nunchaku installation from cached wheel verification failed"
                    return 1
                fi
            else
                log_error "❌ Failed to install Nunchaku from cached wheel"
            fi
        fi
        
        # Download and cache the wheel
        local nunchaku_wheel_url="https://huggingface.co/mit-han-lab/nunchaku/resolve/main/$nunchaku_wheel_name"
        
        log "Downloading Nunchaku wheel from: $nunchaku_wheel_url"
        log "Caching to: $cached_wheel"
        
        # Download the wheel file
        if wget -q --show-progress -O "$cached_wheel" "$nunchaku_wheel_url"; then
            log "✅ Nunchaku wheel downloaded and cached successfully"
            
            # Install from the downloaded wheel
            if pip install --no-cache-dir --disable-pip-version-check "$cached_wheel"; then
                log "Nunchaku wheel installation completed successfully"
                
                # Verify installation
                local verification_script="
import sys
try:
    import nunchaku
    print(f'✅ Nunchaku {nunchaku.__version__} installed successfully')
    print(f'Location: {nunchaku.__file__}')
    
    # Test basic functionality
    if hasattr(nunchaku, '__version__'):
        print(f'Version check: OK')
    else:
        print('Warning: Version attribute not found')
    
    sys.exit(0)
except ImportError as e:
    print(f'❌ Nunchaku import failed: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ Nunchaku verification error: {e}')
    sys.exit(1)
"
                
                if python -c "$verification_script" 2>/dev/null; then
                    log "✅ Nunchaku installation verified successfully"
                    touch "$nunchaku_marker"
                    log "Created installation marker: $nunchaku_marker"
                    return 0
                else
                    log_error "❌ Nunchaku installation verification failed"
                    return 1
                fi
            else
                log_error "❌ Nunchaku wheel installation failed"
                log_error "Attempting alternative installation methods..."
                
                # Try installing from PyPI as fallback
                log "Attempting to install Nunchaku from PyPI..."
                if pip install --no-cache-dir --disable-pip-version-check nunchaku; then
                    log "✅ Nunchaku installed from PyPI successfully"
                    touch "$nunchaku_marker"
                    log "Created installation marker: $nunchaku_marker"
                    return 0
                else
                    log_error "❌ All Nunchaku installation methods failed"
                    log_error "Nunchaku will not be available. Consider manual installation."
                    return 1
                fi
            fi
        else
            log_error "❌ Failed to download Nunchaku wheel from $nunchaku_wheel_url"
            log_error "Attempting PyPI installation as fallback..."
            
            # Try installing from PyPI as fallback
            if pip install --no-cache-dir --disable-pip-version-check nunchaku; then
                log "✅ Nunchaku installed from PyPI successfully"
                touch "$nunchaku_marker"
                log "Created installation marker: $nunchaku_marker"
                return 0
            else
                log_error "❌ All Nunchaku installation methods failed"
                log_error "Nunchaku will not be available. Consider manual installation."
                return 1
            fi
        fi
    }
    
    # Execute Nunchaku installation
    install_nunchaku
    nunchaku_status=$?
    if [[ $nunchaku_status -eq 0 ]]; then
        log "✅ Nunchaku installation completed successfully"
    else
        log_error "⚠️ Nunchaku installation had issues (Status: $nunchaku_status)"
        log_error "ComfyUI will continue without Nunchaku support"
    fi
    # --- END NUNCHAKU INSTALLATION ---

    # Final checks and marker file
    touch /tmp/sd_comfy.prepared
    echo "Stable Diffusion Comfy setup complete."
else
    echo "Stable Diffusion Comfy already prepared. Skipping setup."
    # Activate venv even if skipping setup
    if [ -f "$VENV_DIR/sd_comfy-env/bin/activate" ]; then
        source $VENV_DIR/sd_comfy-env/bin/activate
        echo "Virtual environment activated: $VENV_DIR/sd_comfy-env"
    else
        log_error "Virtual environment activation script not found!"
        exit 1
    fi
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
  
  # Rotate ComfyUI log file instead of deleting it
  if [[ -f "$LOG_DIR/sd_comfy.log" ]]; then
    # Create timestamp for old log
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    mv "$LOG_DIR/sd_comfy.log" "$LOG_DIR/sd_comfy_${timestamp}.log"
    echo "Previous ComfyUI log archived as: sd_comfy_${timestamp}.log"
    
    # Keep only the last 5 rotated logs to save space
    ls -t "$LOG_DIR"/sd_comfy_*.log 2>/dev/null | tail -n +6 | xargs -r rm
  fi
  
  # A4000-specific VRAM optimization settings (16GB)
  export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:4096,garbage_collection_threshold:0.8"
  
  # --- ENSURE CORRECT TORCH VERSIONS AT RUNTIME ---
  # It's still good practice to verify torch before launching,
  # even if we don't reinstall SageAttention here.
  echo "Verifying PyTorch ecosystem versions before launch..."
  fix_torch_versions # This will now just check unless versions are wrong
  fix_torch_status=$? 

  if [[ $fix_torch_status -ne 0 ]]; then
      log_error "fix_torch_versions function failed during pre-launch check with status $fix_torch_status."
      exit 1
  fi



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
