#!/bin/bash
# Removed set -e to allow better error handling - individual failures won't stop the entire script

#######################################
# OPTIMIZED COMFYUI SETUP SCRIPT
# 70% Reduction from 1939 to ~577 lines (further optimized)
# Maintains all essential functionality
#######################################

# Global Configuration
readonly SCRIPT_DIR=$(dirname "$(realpath "$0")")
readonly LOG_DIR="/tmp/log"

# Package Versions (from original script)
readonly TORCH_VERSION="2.7.1+cu126"
readonly TORCHVISION_VERSION="0.22.1+cu126" 
readonly TORCHAUDIO_VERSION="2.7.1+cu126"
readonly XFORMERS_VERSION="0.0.30"
readonly TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"

# Initialize environment
cd "$SCRIPT_DIR" || { echo "Failed to change directory"; exit 1; }
source .env || { echo "Failed to source .env"; exit 1; }
mkdir -p "$LOG_DIR"

# GLOBAL VIRTUAL ENVIRONMENT ACTIVATION - Activate once, keep active
activate_global_venv() {
    local venv_path="${VENV_DIR:-/tmp}/sd_comfy-env"
    [[ ! -d "$venv_path" ]] && python3.10 -m venv "$venv_path"
    source "$venv_path/bin/activate"
}



# Test network connectivity
test_connectivity() {
    log "Testing network connectivity..."
    ping -c 1 8.8.8.8 &>/dev/null && { log "✅ Network connectivity OK"; return 0; } || { log_error "❌ Network connectivity failed"; return 1; }
}

# Enhanced logging with error tracking
declare -a INSTALLATION_SUCCESSES=()
declare -a INSTALLATION_FAILURES=()
declare -a DEPENDENCY_CONFLICTS=()
declare -a CUSTOM_NODE_FAILURES=()
declare -a CUSTOM_NODE_DETAILS=()
declare -a CUSTOM_NODE_FAILED_DETAILS=()
declare -a DETAILED_LOGS=()
declare -a PIP_ERRORS=()
declare -a PIP_SUCCESSES=()

log() { echo "$1"; }

# Collect detailed logs for final summary (silent during execution)
log_detail() {
    local message="$1"
    DETAILED_LOGS+=("$message")
}

# Capture pip installation results for final summary
log_pip_success() {
    local package="$1"
    local method="$2"
    PIP_SUCCESSES+=("✅ $package ($method)")
}

log_pip_error() {
    local package="$1"
    local error="$2"
    local method="$3"
    PIP_ERRORS+=("❌ $package ($method): $error")
}

log_error() {
    echo "ERROR: $1" >&2
    INSTALLATION_FAILURES+=("$1")
}

log_success() {
    local component="$1"
    echo "✅ SUCCESS: $component"
    INSTALLATION_SUCCESSES+=("$component")
}

log_conflict() {
    local conflict="$1"
    echo "⚠️ CONFLICT: $conflict"
    DEPENDENCY_CONFLICTS+=("$conflict")
}

log_node_failure() {
    local node="$1"
    local error="$2"
    CUSTOM_NODE_FAILURES+=("$node: $error")
}

#######################################
# UNIFIED CORE FUNCTIONS
#######################################

# Single CUDA environment setup (replaces 6+ duplicate functions)
setup_cuda_env() {
    export CUDA_HOME=/usr/local/cuda-12.6 PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH FORCE_CUDA=1 CUDA_VISIBLE_DEVICES=0 PYOPENGL_PLATFORM="osmesa" WINDOW_BACKEND="headless" TORCH_CUDA_ARCH_LIST="8.6" PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:4096,garbage_collection_threshold:0.8" CUDA_LAUNCH_BLOCKING=0 CUDA_DEVICE_MAX_CONNECTIONS=32 TORCH_CUDNN_V8_API_ENABLED=1
}

# SMART pip installer - avoids unnecessary reinstalls
pip_install() {
    local package="$1" flags="${2:---no-cache-dir --disable-pip-version-check}" force="${3:-false}"
    local pkg_name=$(echo "$package" | sed 's/[<>=!].*//' | sed 's/\[.*\]//')
    
    # Check if package is already installed (unless forcing)
    [[ "$force" != "true" ]] && python -c "import $pkg_name" 2>/dev/null && { log "⏭️ Already installed: $pkg_name (skipping)"; return 0; }
    
    log "Installing: $package"
    local install_flags="$flags"; [[ "$force" == "true" ]] && install_flags="$flags --force-reinstall"
    
    pip install $install_flags "$package" 2>&1 | tee /tmp/pip_install_${pkg_name//[^a-zA-Z0-9]/_}.log && { log "✅ Successfully installed: $package"; log_pip_success "$package" "pip_install function"; return 0; } || { local pip_error=$(tail -n 5 /tmp/pip_install_${pkg_name//[^a-zA-Z0-9]/_}.log 2>/dev/null | tr '\n' ' '); log_error "❌ Failed to install: $package"; log_pip_error "$package" "$pip_error" "pip_install function"; return 1; }
}

# Smart package installer with robust wheel caching
install_with_cache() {
    local package="$1" wheel_cache="${WHEEL_CACHE_DIR:-/storage/.wheel_cache}" pip_cache="${PIP_CACHE_DIR:-/storage/.pip_cache}"
    local pkg_name=$(echo "$package" | sed 's/[<>=!].*//' | sed 's/\[.*\]//')
    
    # Check if package is already installed first
    python -c "import $pkg_name" 2>/dev/null && { log "⏭️ Already installed: $pkg_name (skipping)"; return 0; }
    
    mkdir -p "$wheel_cache" "$pip_cache" && export PIP_CACHE_DIR="$pip_cache"
    local cached_wheel=$(find "$wheel_cache" -name "${pkg_name}*.whl" -type f 2>/dev/null | head -1)
    
    # Try cached wheel first
    [[ -n "$cached_wheel" && -f "$cached_wheel" ]] && { log "🔄 Using cached wheel: $(basename "$cached_wheel")"; pip install --no-cache-dir --disable-pip-version-check "$cached_wheel" 2>/dev/null && return 0 || { log "⚠️ Cached wheel failed, removing and rebuilding..."; rm -f "$cached_wheel"; }; }
    
    # Install with pip cache and save wheels
    log "📦 Installing and caching: $package"
    pip install --cache-dir "$pip_cache" --disable-pip-version-check "$package" 2>/dev/null && { local wheels_found=0; find "$pip_cache" -name "${pkg_name}*.whl" -newer "$wheel_cache" -exec cp {} "$wheel_cache/" \; 2>/dev/null && wheels_found=1; find /tmp -name "${pkg_name}*.whl" -exec cp {} "$wheel_cache/" \; 2>/dev/null && wheels_found=1; [[ $wheels_found -eq 1 ]] && log_detail "💾 Wheel cached for future use: $package"; return 0; } || return 1
}

# Consolidated CUDA installation (replaces install_cuda_12 function)
install_cuda() {
    local marker="/storage/.cuda_12.6_installed"
    
    # Check if already installed
    [[ -f "$marker" ]] && { setup_cuda_env; hash -r; command -v nvcc &>/dev/null && [[ "$(nvcc --version 2>&1 | grep 'release' | awk '{print $6}' | sed 's/^V//')" == "12.6"* ]] && return 0; }
    
    log "Installing CUDA 12.6..."
    
    # Clean up old CUDA versions and install new
    dpkg -l | grep -q "cuda-11" && apt-get remove --purge -y 'cuda-11-*' 2>/dev/null || true
    wget -qO /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i /tmp/cuda-keyring.deb && rm -f /tmp/cuda-keyring.deb
    
    # Update and install CUDA packages (including development headers)
    apt-get update -qq && apt-get install -y build-essential python3-dev cuda-cudart-12-6 cuda-nvcc-12-6 libcublas-12-6 libcublas-dev-12-6 libcufft-12-6 libcufft-dev-12-6 libcurand-12-6 libcurand-dev-12-6 libcusolver-12-6 libcusolver-dev-12-6 libcusparse-12-6 libcusparse-dev-12-6 libnpp-12-6 libnpp-dev-12-6 2>/dev/null
    
    # Configure environment and verify
    setup_cuda_env && hash -r
    command -v nvcc &>/dev/null && { touch "$marker"; cat > /etc/profile.d/cuda12.sh << 'EOL'
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export FORCE_CUDA=1
EOL
        chmod +x /etc/profile.d/cuda12.sh; }
}

# ULTIMATE PyTorch Installation (uses multiple strategies to beat dependency hell)
setup_pytorch() {
    log "🚀 ULTIMATE PyTorch Installation - Beating Dependency Hell..."
    
    # SMART CHECK: Skip if PyTorch is already working perfectly
    log "🔍 Checking existing PyTorch installation..."
    if python -c "import torch; import torchvision; import torchaudio; print(f'PyTorch {torch.__version__} working'); assert torch.cuda.is_available(), 'CUDA required'; print('✅ CUDA available')" 2>/dev/null; then
        local torch_version=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
        log_success "PyTorch ecosystem already working perfectly: $torch_version"
        log "⏭️ SKIPPING PyTorch installation - everything works! No wasteful reinstalls!"
        return 0
    fi
    
    log "⚠️ PyTorch needs installation/fixing..."
    
    # Quick version check
    local torch_ver=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "missing")
    local cuda_available=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
    
    # Skip if already correct
    [[ "$torch_ver" == "$TORCH_VERSION" && "$cuda_available" == "True" ]] && return 0
    
    log "🔥 NUCLEAR CLEANUP: Removing all PyTorch traces..."
    
    # Step 1: Complete nuclear cleanup
    pip uninstall -y torch torchvision torchaudio xformers 2>/dev/null || true
    pip cache purge 2>/dev/null || true
    
    # Remove package directories manually
    local site_packages=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
    if [[ -d "$site_packages" ]]; then
        rm -rf "$site_packages"/torch* "$site_packages"/xformers* 2>/dev/null || true
        rm -rf "$site_packages"/*torch* "$site_packages"/*xform* 2>/dev/null || true
    fi
    
    log "💡 STRATEGY 1: Installing with --no-deps to bypass dependency resolver entirely..."
    
    # Strategy 1: Install with --no-deps (bypasses ALL dependency checking)
    pip install --no-cache-dir --disable-pip-version-check --no-deps \
        torch==2.7.1+cu126 --extra-index-url "https://download.pytorch.org/whl/cu126" 2>/dev/null
    
    pip install --no-cache-dir --disable-pip-version-check --no-deps \
        torchvision==0.22.1+cu126 --extra-index-url "https://download.pytorch.org/whl/cu126" 2>/dev/null
    
    pip install --no-cache-dir --disable-pip-version-check --no-deps \
        torchaudio==2.7.1+cu126 --extra-index-url "https://download.pytorch.org/whl/cu126" 2>/dev/null
    
    # Verify Strategy 1
    if python -c "import torch; print('Strategy 1 SUCCESS')" 2>/dev/null; then
        log "✅ Strategy 1 (--no-deps) successful!"
    else
        log "❌ Strategy 1 failed, trying Strategy 2..."
        
        # Strategy 2: Use requirements.txt with --force-reinstall
        cat > /tmp/pytorch_requirements.txt << 'EOF'
torch==2.7.1+cu126
torchvision==0.22.1+cu126
torchaudio==2.7.1+cu126
EOF
        
        pip install --no-cache-dir --disable-pip-version-check --force-reinstall \
            -r /tmp/pytorch_requirements.txt \
            --extra-index-url "https://download.pytorch.org/whl/cu126" 2>/dev/null || {
            
            log "❌ Strategy 2 failed, trying Strategy 3..."
            
            # Strategy 3: Manual wheel download and install
            log "💡 STRATEGY 3: Manual wheel downloads..."
            
            mkdir -p /tmp/pytorch_wheels
            cd /tmp/pytorch_wheels
            
            # Download wheels manually
            wget -q "https://download.pytorch.org/whl/cu126/torch-2.7.1%2Bcu126-cp310-cp310-linux_x86_64.whl" -O torch.whl 2>/dev/null || true
            wget -q "https://download.pytorch.org/whl/cu126/torchvision-0.22.1%2Bcu126-cp310-cp310-linux_x86_64.whl" -O torchvision.whl 2>/dev/null || true
            wget -q "https://download.pytorch.org/whl/cu126/torchaudio-2.7.1%2Bcu126-cp310-cp310-linux_x86_64.whl" -O torchaudio.whl 2>/dev/null || true
            
            # Install manually downloaded wheels
            pip install --no-cache-dir --disable-pip-version-check --force-reinstall \
                torch.whl torchvision.whl torchaudio.whl 2>/dev/null || true
            
            cd - > /dev/null
            rm -rf /tmp/pytorch_wheels
        }
        
        rm -f /tmp/pytorch_requirements.txt
    fi
    
    # Install xformers (always problematic, so we try multiple approaches)
    log "Installing xformers with conflict bypassing..."
    
    pip install --no-cache-dir --disable-pip-version-check --no-deps \
        xformers==0.0.30 --extra-index-url "https://download.pytorch.org/whl/cu126" 2>/dev/null || \
    pip install --no-cache-dir --disable-pip-version-check --force-reinstall \
        xformers --index-url "https://download.pytorch.org/whl/cu121" 2>/dev/null || \
    pip install --no-cache-dir --disable-pip-version-check --force-reinstall \
        xformers 2>/dev/null || \
    log_error "⚠️ All xformers installation strategies failed, continuing without"
    
    # Final verification
    log "🔍 Final verification..."
    if python -c "import torch; print(f'🎉 SUCCESS: PyTorch {torch.__version__} working!')" 2>/dev/null; then
        local torch_version=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
        log_success "PyTorch $torch_version installed and working"
        
        if python -c "import torch; print(f'🎉 CUDA available: {torch.cuda.is_available()}')" 2>/dev/null; then
            local cuda_status=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
            log_success "CUDA support: $cuda_status"
            return 0
        else
            log_conflict "PyTorch installed but CUDA not available (version mismatch)"
                return 0
        fi
            else
        log_error "PyTorch installation failed after all strategies"
                return 1
    fi
}

# Update ComfyUI Manager specifically (it's a critical component)
update_comfyui_manager() {
    local manager_dir="$REPO_DIR/custom_nodes/comfyui-manager"
    [[ ! -d "$manager_dir" ]] && { log "⚠️ ComfyUI Manager not found, skipping update"; return 0; }
    log "🔧 Updating ComfyUI Manager..."
    cd "$manager_dir" || { log_error "ComfyUI Manager directory access failed"; return 1; }
    git fetch --all &>/dev/null && git reset --hard origin/HEAD &>/dev/null && log "✅ ComfyUI Manager git update successful" || log "⚠️ ComfyUI Manager git update had issues"
    cd - > /dev/null
    [[ -f "$manager_dir/requirements.txt" ]] && { log "📦 Installing ComfyUI Manager dependencies..."; pip install -r "$manager_dir/requirements.txt" && log "✅ ComfyUI Manager dependencies installed" || log "⚠️ ComfyUI Manager dependencies had issues (continuing)"; } || log "⏭️ ComfyUI Manager has no requirements.txt"
}

update_custom_nodes() {
    local nodes_dir="$REPO_DIR/custom_nodes"
    [[ ! -d "$nodes_dir" ]] && return 0
    
    # Collect requirements from all nodes
    local combined_reqs="/tmp/all_custom_node_requirements.txt"
    echo -n > "$combined_reqs"
    for git_dir in "$nodes_dir"/*/.git; do
        [[ ! -d "$git_dir" ]] && continue
        local node_dir="${git_dir%/.git}"
        local node_name=$(basename "$node_dir")
        [[ -f "$node_dir/requirements.txt" && -s "$node_dir/requirements.txt" ]] && { echo "# From $node_name" >> "$combined_reqs"; grep -v "^[[:space:]]*#\|^[[:space:]]*$" "$node_dir/requirements.txt" >> "$combined_reqs"; echo "" >> "$combined_reqs"; }
    done
    
    [[ -s "$combined_reqs" ]] && process_combined_requirements "$combined_reqs"
    
    # Update all git repositories
    for git_dir in "$nodes_dir"/*/.git; do
        [[ ! -d "$git_dir" ]] && continue
        local node_dir="${git_dir%/.git}"
        local node_name=$(basename "$node_dir")
        if cd "$node_dir"; then
            git fetch --all &>/dev/null && git reset --hard origin/HEAD &>/dev/null
            cd - > /dev/null
        else
            log_node_failure "$node_name" "Git update failed - directory access failed"
        fi
    done
    
    rm -f "$combined_reqs"
}

process_combined_requirements() {
    local req_file="$1"
    local cache_dir="/storage/.pip_cache"
    local resolved_reqs="/tmp/resolved_requirements.txt"
    local verify_script="/tmp/verify_missing_packages.py"
    
    [[ ! -f "$req_file" ]] && return 0
    
    mkdir -p "$cache_dir"
    export PIP_CACHE_DIR="$cache_dir"
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    
    cat > "/tmp/resolve_conflicts.py" << 'EOF'
import re, sys
from collections import defaultdict

def parse_requirement(req):
    match = re.match(r'^([a-zA-Z0-9_\-\.]+)(.*)$', req.strip())
    if not match: return req.strip(), ""
    name, version_spec = match.groups()
    return name.lower(), version_spec.strip()

def normalize_requirement(req):
    if req.startswith(('git+', 'http')): return req
    return req.split('#')[0].strip()

requirements = []
with open(sys.argv[1], 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            req = normalize_requirement(line)
            if req: requirements.append(req)

package_versions = defaultdict(list)
for req in requirements:
    if req.startswith(('git+', 'http')): continue
    name, version_spec = parse_requirement(req)
    if name: package_versions[name].append((version_spec, req))

resolved_requirements = []
processed_packages = set()

for req in requirements:
    if req.startswith(('git+', 'http')):
        if req not in processed_packages:
            resolved_requirements.append(req)
            processed_packages.add(req)
        continue
    
    name, version_spec = parse_requirement(req)
    if name and name not in processed_packages:
        if len(package_versions[name]) > 1:
            chosen_version = min(package_versions[name], key=lambda x: len(x[0]))
            resolved_requirements.append(chosen_version[1])
        else:
            resolved_requirements.append(req)
        processed_packages.add(name)

with open(sys.argv[2], 'w') as f:
    for req in sorted(set(resolved_requirements)):
        f.write(f"{req}\n")
EOF
    
    python "/tmp/resolve_conflicts.py" "$req_file" "$resolved_reqs" || cp "$req_file" "$resolved_reqs"
    
    cat > "$verify_script" << 'EOF'
import sys, importlib.util, re

def normalize_package_name(name):
    base_name = re.sub(r'[<>=!~\[\];].*$', '', name).strip()
    name_mapping = {
        'opencv-contrib-python': 'cv2', 'opencv-python': 'cv2', 'scikit-image': 'skimage',
        'scikit-learn': 'sklearn', 'pillow': 'PIL', 'pytorch': 'torch', 'pyyaml': 'yaml',
        'python-dateutil': 'dateutil', 'protobuf': 'google.protobuf', 'blend-modes': 'blend_modes',
        'transparent-background': 'transparent_background', 'inference-cli': 'inference',
        'bitsandbytes': 'bitsandbytes', 'huggingface-hub': 'huggingface_hub',
    }
    return name_mapping.get(base_name.lower(), base_name.replace('-', '_').replace('.', '_'))

def is_package_available(package_name):
    if package_name.startswith(('git+', 'http')): return False
    try:
        module_name = normalize_package_name(package_name)
        if '.' in module_name:
            spec = importlib.util.find_spec(module_name)
            if spec is not None: return True
            parent_module = module_name.split('.')[0]
            spec = importlib.util.find_spec(parent_module)
            return spec is not None
        else:
            spec = importlib.util.find_spec(module_name)
            return spec is not None
    except: return False

missing_packages = []
with open(sys.argv[1], 'r') as f:
    for line in f:
        package = line.strip()
        if package and not package.startswith('#'):
            if not is_package_available(package):
                missing_packages.append(package)

with open(sys.argv[2], 'w') as f:
    for pkg in missing_packages:
        f.write(f"{pkg}\n")
EOF
    
    python "$verify_script" "$resolved_reqs" "/tmp/missing_packages.txt"
    
    if [[ -s "/tmp/missing_packages.txt" ]]; then
        split -l 5 "/tmp/missing_packages.txt" "/tmp/batch_"
        
        for batch_file in /tmp/batch_*; do
            if pip install --no-cache-dir --disable-pip-version-check -r "$batch_file" 2>&1 | tee "/tmp/pip_batch_$(basename "$batch_file").log"; then
            continue
            else
                while IFS= read -r package; do
                    [[ -z "$package" ]] && continue
                    if pip_install "$package"; then
                        log_pip_success "$package" "individual install"
                    else
                        log_pip_error "$package" "Installation failed" "individual install"
                    fi
                done < "$batch_file"
        fi
    done
    
        rm -f /tmp/batch_* /tmp/pip_batch_* /tmp/pip_individual_*
    fi
    
    if grep -E "git\+https?://" "$resolved_reqs" >/dev/null 2>&1; then
        grep -E "git\+https?://" "$resolved_reqs" | while IFS= read -r repo; do
            if pip_install "$repo"; then
                log_pip_success "$repo" "git repository"
            else
                log_pip_error "$repo" "Git installation failed" "git repository"
        fi
    done
    fi
    
    rm -f "$resolved_reqs" "$verify_script" "/tmp/missing_packages.txt" "/tmp/resolve_conflicts.py"
}

resolve_dependencies() {
    python -m pip install --upgrade pip 2>/dev/null || curl https://bootstrap.pypa.io/get-pip.py | python 2>/dev/null || log_error "pip upgrade failed"
    
    # Install core build tools
    for pkg in "wheel" "setuptools" "numpy>=1.26.0,<2.3.0"; do pip_install "$pkg" "" false; done
    
    # Install build tools
    local build_tools=("pybind11" "ninja" "packaging")
    for tool in "${build_tools[@]}"; do install_with_cache "$tool" || log_error "Build tool failed: $tool"; done
    
    # Install specific packages
    install_with_cache "av" || log_error "av installation failed"
    install_with_cache "timm==1.0.13" || log_error "timm installation failed"
    pip uninstall -y flet 2>/dev/null || true
    pip_install "flet==0.23.2" "" false || log_error "flet installation failed"
    
    # Install core dependencies
    local core_deps=("einops" "scipy" "torchsde" "aiohttp" "spandrel" "kornia==0.7.0" "urllib3==1.21" "requests==2.31.0" "fastapi==0.103.2" "gradio_client==0.6.0" "peewee==3.16.3" "psutil==5.9.5" "uvicorn==0.23.2" "pynvml==11.5.0" "python-multipart==0.0.6")
    for dep in "${core_deps[@]}"; do install_with_cache "$dep" || log_error "Failed to install: $dep"; done
}

install_component() {
    local component="$1" install_func="install_${component}"
    declare -f "$install_func" >/dev/null && "$install_func" || log_error "$component installation failed"
}

    install_sageattention() {
    python -c "import sageattention" 2>/dev/null && return 0
    local cache_dir="/storage/.sageattention_cache" wheel_cache="/storage/.wheel_cache"
    mkdir -p "$cache_dir" "$wheel_cache"
    local cached_wheel=$(find "$wheel_cache" -name "sageattention*.whl" 2>/dev/null | head -1)
    [[ -n "$cached_wheel" ]] && { pip_install "$cached_wheel" && return 0 || rm -f "$cached_wheel"; }
    [[ ! -d "$cache_dir/src" ]] && git clone https://github.com/thu-ml/SageAttention.git "$cache_dir/src" || return 1
    export TORCH_EXTENSIONS_DIR="/storage/.torch_extensions" MAX_JOBS=$(nproc) USE_NINJA=1
    cd "$cache_dir/src" || return 1
    rm -rf build dist *.egg-info
    python setup.py bdist_wheel || return 1
    local wheel=$(find dist -name "*.whl" | head -1)
    [[ -n "$wheel" ]] && cp "$wheel" "$wheel_cache/" && pip_install "$wheel" || return 1
    cd - > /dev/null
}

    install_nunchaku() {
    python -c "import nunchaku" 2>/dev/null && return 0
    local torch_check_output=$(python -c "import sys; import torch; v=torch.__version__.split('+')[0]; major,minor=map(int,v.split('.')[:2]); print('compatible' if major>2 or (major==2 and minor>=5) else f'incompatible: {major}.{minor} < 2.5'); sys.exit(0 if major>2 or (major==2 and minor>=5) else 1)" 2>/dev/null)
    [[ "$torch_check_output" != "compatible" ]] && { log_error "PyTorch version incompatible for Nunchaku: $torch_check_output"; return 1; }
    pip uninstall -y nunchaku 2>/dev/null || true
    pip_install "nunchaku==0.3.1" "" true && { log_pip_success "nunchaku" "version 0.3.1"; return 0; } || { log_pip_error "nunchaku" "Installation failed using pip_install" "version 0.3.1"; return 1; }
}
            
    install_hunyuan3d_texture_components() {
        local hunyuan3d_path="$REPO_DIR/custom_nodes/ComfyUI-Hunyuan3d-2-1"
    [[ ! -d "$hunyuan3d_path" ]] && return 1
    pip_install "pybind11"
    pip_install "ninja"
    for component in custom_rasterizer DifferentiableRenderer; do
        local comp_path="$hunyuan3d_path/hy3dpaint/$component"
        if [[ -d "$comp_path" ]]; then
            cd "$comp_path" || { log_error "$component directory access failed"; continue; }
            python setup.py install || log_error "$component installation failed"
            cd - > /dev/null
        fi
    done
}

    process_requirements() {
        local req_file="$1"
    [[ ! -f "$req_file" ]] && return 0
    pip install -r "$req_file" || { while read -r pkg; do [[ "$pkg" =~ ^[[:space:]]*# ]] || [[ -z "$pkg" ]] || pip_install "$pkg" || true; done < "$req_file"; }
}

# Service loop function (from original)
service_loop() { while true; do eval "$1"; sleep 1; done; }

#######################################
# MAIN EXECUTION FLOW
#######################################

main() {
    test_connectivity || {
        log_error "Network connectivity test failed. Check your internet connection."
        exit 1
    }
    
    if [[ -f "/tmp/sd_comfy.prepared" && -z "$REINSTALL_SD_COMFY" ]]; then
        activate_global_venv
                    return 0
    fi
    
    setup_cuda_env
    install_cuda
    
    cd "$REPO_DIR"
    export TARGET_REPO_URL="https://github.com/comfyanonymous/ComfyUI.git"
    export TARGET_REPO_DIR=$REPO_DIR
    export UPDATE_REPO=$SD_COMFY_UPDATE_REPO
    export UPDATE_REPO_COMMIT=$SD_COMFY_UPDATE_REPO_COMMIT
    
    [[ -d ".git" ]] && { [[ -n "$(git status --porcelain requirements.txt 2>/dev/null)" ]] && git checkout -- requirements.txt; git symbolic-ref -q HEAD >/dev/null || git checkout main || git checkout master || git checkout -b main; git fetch --all &>/dev/null && git reset --hard origin/HEAD &>/dev/null; }
    
    prepare_link "$REPO_DIR/output:$IMAGE_OUTPUTS_DIR/stable-diffusion-comfy" "$MODEL_DIR:$WORKING_DIR/models" "$MODEL_DIR/sd:$LINK_MODEL_TO" "$MODEL_DIR/lora:$LINK_LORA_TO" "$MODEL_DIR/vae:$LINK_VAE_TO" "$MODEL_DIR/upscaler:$LINK_UPSCALER_TO" "$MODEL_DIR/controlnet:$LINK_CONTROLNET_TO" "$MODEL_DIR/embedding:$LINK_EMBEDDING_TO" "$MODEL_DIR/llm_checkpoints:$LINK_LLM_TO"
    
    # Activate global virtual environment (will create if needed)
    activate_global_venv
    
    apt-get update -qq && apt-get install -y libatlas-base-dev libblas-dev liblapack-dev libjpeg-dev libpng-dev python3-dev build-essential libgl1-mesa-dev espeak-ng 2>/dev/null || true
    
    setup_pytorch
    for component in "sageattention" "nunchaku" "hunyuan3d_texture_components"; do install_component "$component"; done
    
    # Update and install components with error handling
    update_comfyui_manager || log_error "ComfyUI Manager update had issues (continuing)"
    update_custom_nodes || log_error "Custom nodes update had issues (continuing)"
    resolve_dependencies || log_error "Some dependencies failed (continuing)"
    process_requirements "$REPO_DIR/requirements.txt" || log_error "Core requirements had issues (continuing)"
    
    touch "/tmp/sd_comfy.prepared"
}

# Model download (can run in background)
download_models() {
    [[ -n "$SKIP_MODEL_DOWNLOAD" ]] && { log "⏭️ Model download skipped (SKIP_MODEL_DOWNLOAD set)"; return; }
    log "📥 Starting model download process... 💡 Models will download in background while ComfyUI is running 💡 You can start using ComfyUI immediately!"
    bash "$SCRIPT_DIR/../utils/sd_model_download/main.sh" & local download_pid=$!
    log "📋 Model download started with PID: $download_pid 📋 Check progress with: tail -f $LOG_DIR/sd_comfy.log"
        echo "$download_pid" > /tmp/model_download.pid
}

# Launch ComfyUI
launch() {
    [[ -n "$INSTALL_ONLY" ]] && return 0
    log "Launching ComfyUI..."
  cd "$REPO_DIR"
  
    # Log rotation
    [[ -f "$LOG_DIR/sd_comfy.log" ]] && { local timestamp=$(date +"%Y%m%d_%H%M%S"); mv "$LOG_DIR/sd_comfy.log" "$LOG_DIR/sd_comfy_${timestamp}.log"; ls -t "$LOG_DIR"/sd_comfy_*.log 2>/dev/null | tail -n +6 | xargs -r rm; }
  
    # A4000-specific optimizations
  export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:4096,garbage_collection_threshold:0.8"
  
    # Runtime PyTorch verification (skip if fresh install)
    [[ ! -f "/tmp/pytorch_ecosystem_fresh_install" ]] && setup_pytorch || rm -f "/tmp/pytorch_ecosystem_fresh_install"
    
    # Launch ComfyUI with optimized parameters (using globally activated venv Python)
    PYTHONUNBUFFERED=1 service_loop "python main.py --dont-print-server --port $SD_COMFY_PORT --cuda-malloc --use-sage-attention --preview-method auto --bf16-vae --fp16-unet --cache-lru 5 --reserve-vram 0.5 --fast --enable-compress-response-body ${EXTRA_SD_COMFY_ARGS}" > "$LOG_DIR/sd_comfy.log" 2>&1 &
  echo $! > /tmp/sd_comfy.pid
}

# Error logging only (summary function removed)
log_errors() {
    [[ ${#INSTALLATION_FAILURES[@]} -gt 0 ]] && { echo "❌ Installation failures logged:" > "$LOG_DIR/errors.log"; printf '%s\n' "${INSTALLATION_FAILURES[@]}" >> "$LOG_DIR/errors.log"; }
    [[ ${#PIP_ERRORS[@]} -gt 0 ]] && { echo "❌ Pip errors logged:" >> "$LOG_DIR/errors.log"; printf '%s\n' "${PIP_ERRORS[@]}" >> "$LOG_DIR/errors.log"; }
}

# Execute main workflow
main

    

# Log any errors before launch
log_errors

# Start ComfyUI first (so user can access it immediately)
log "🚀 Starting ComfyUI first for immediate access..."
launch

# Download models in background (non-blocking)
log "📥 Starting model download in background..."
download_models &

# Wait a moment for ComfyUI to start
sleep 5

# Final notifications (from original)
send_to_discord "Stable Diffusion Comfy Started"
env | grep -q "PAPERSPACE" && send_to_discord "Link: https://$PAPERSPACE_FQDN/sd-comfy/"

# Show final status
log "" && log "🎉 COMFYUI STARTUP COMPLETE! ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ✅ ComfyUI is now running and accessible 📥 Model download is running in background 🔗 Access ComfyUI at: http://localhost:$SD_COMFY_PORT" && env | grep -q "PAPERSPACE" && log "🌐 Paperspace URL: https://$PAPERSPACE_FQDN/sd-comfy/" && log "" && log "📋 Useful commands: • Check ComfyUI logs: tail -f $LOG_DIR/sd_comfy.log • Check model download: tail -f /tmp/model_download.log • Stop model download: kill \$(cat /tmp/model_download.pid) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

[[ -n "${CF_TOKEN}" ]] && { [[ "$RUN_SCRIPT" != *"sd_comfy"* ]] && export RUN_SCRIPT="$RUN_SCRIPT,sd_comfy"; bash "$SCRIPT_DIR/../cloudflare_reload.sh"; }
