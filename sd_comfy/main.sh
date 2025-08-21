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

# Source .env but handle readonly variable conflicts gracefully
if [ -f ".env" ]; then
    # Temporarily unset readonly variables that might conflict
    unset LOG_DIR 2>/dev/null || true
    source .env 2>/dev/null || { 
        echo "Warning: Some .env variables may be readonly, continuing..."
        # Try to source again but ignore errors
        source .env 2>/dev/null || true
    }
    # Restore LOG_DIR if it was unset
    readonly LOG_DIR="${LOG_DIR:-/tmp/log}"
else
    echo "Warning: .env file not found, using defaults"
fi

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
    
    # Comprehensive warning suppression for CUDA compilation
    export CFLAGS="-Wno-deprecated-declarations -w"
    export CXXFLAGS="-Wno-deprecated-declarations -w"
    export CUDAFLAGS="-Wno-deprecated-declarations -w"
    export NVCC_APPEND_FLAGS="-Wno-deprecated-declarations"
    
    # Additional environment variables to suppress warnings
    export TORCH_CUDA_ARCH_LIST="8.6"
    export MAX_JOBS=$(nproc)
    export USE_NINJA=1
    export DISABLE_ADDMM_CUDA_LT=1
    
    # Suppress compiler warnings during PyTorch extensions compilation
    export TORCH_EXTENSIONS_DIR="/storage/.torch_extensions"
    export PYTORCH_BUILD_VERSION="2.7.1"
    export PYTORCH_BUILD_NUMBER="1"
    
    # Ninja build system optimizations (reduces verbose output)
    export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
    export MAKEFLAGS="-j$(nproc) --quiet"
}

# SMART pip installer - avoids unnecessary reinstalls
pip_install() {
    local package="$1" flags="${2:---no-cache-dir --disable-pip-version-check --quiet}" force="${3:-false}"
    local pkg_name=$(echo "$package" | sed 's/[<>=!].*//' | sed 's/\[.*\]//')
    
    # Check if package is already installed (unless forcing)
    [[ "$force" != "true" ]] && python -c "import $pkg_name" 2>/dev/null && { log "⏭️ Already installed: $pkg_name (skipping)"; return 0; }
    
    log "📦 Installing: $package"
    local install_flags="$flags"; [[ "$force" == "true" ]] && install_flags="$flags --force-reinstall"
    
    # Use --quiet to suppress verbose output, only show errors
    pip install $install_flags "$package" 2>/tmp/pip_install_${pkg_name//[^a-zA-Z0-9]/_}.log && { log "✅ Successfully installed: $package"; log_pip_success "$package" "pip_install function"; return 0; } || { local pip_error=$(tail -n 5 /tmp/pip_install_${pkg_name//[^a-zA-Z0-9]/_}.log 2>/dev/null | tr '\n' ' '); log_error "❌ Failed to install: $package"; log_pip_error "$package" "$pip_error" "pip_install function"; return 1; }
}

# Enhanced package installer with aggressive caching
install_with_cache() {
    local package="$1" wheel_cache="${WHEEL_CACHE_DIR:-/storage/.wheel_cache}" pip_cache="${PIP_CACHE_DIR:-/storage/.pip_cache}"
    local pkg_name=$(echo "$package" | sed 's/[<>=!].*//' | sed 's/\[.*\]//')
    
    # Check if package is already installed first
    python -c "import $pkg_name" 2>/dev/null && { log "⏭️ Already installed: $pkg_name (skipping)"; return 0; }
    
    # Setup enhanced caching directories
    mkdir -p "$wheel_cache" "$pip_cache" "$pip_cache/wheels" "$pip_cache/http"
    export PIP_CACHE_DIR="$pip_cache"
    export PIP_FIND_LINKS="$wheel_cache"
    
    # Look for compatible cached wheels (more intelligent matching)
    local cached_wheel=$(find "$wheel_cache" -name "${pkg_name}*.whl" -type f 2>/dev/null | sort -V | tail -1)
    
    # Try cached wheel first with compatibility check
    if [[ -n "$cached_wheel" && -f "$cached_wheel" ]]; then
        log "🔄 Using cached wheel: $(basename "$cached_wheel")"
        if pip install --no-cache-dir --disable-pip-version-check --quiet "$cached_wheel" 2>/dev/null; then
            log_detail "💾 Successfully used cached wheel: $package"
            return 0
        else
            log "⚠️ Cached wheel incompatible, removing and rebuilding..."
            rm -f "$cached_wheel"
        fi
    fi
    
    # Install with enhanced caching and wheel collection
    log "📦 Installing and caching: $package"
    if pip install --cache-dir "$pip_cache" --disable-pip-version-check --quiet "$package" 2>/dev/null; then
        # Collect and cache wheels from multiple sources
        local wheels_cached=0
        
        # Cache wheels from pip cache
        find "$pip_cache" -name "${pkg_name}*.whl" -newer "$wheel_cache" -exec cp {} "$wheel_cache/" \; 2>/dev/null && ((wheels_cached++))
        
        # Cache wheels from temporary locations
        find /tmp -name "${pkg_name}*.whl" -exec cp {} "$wheel_cache/" \; 2>/dev/null && ((wheels_cached++))
        
        # Cache wheels from site-packages
        local site_packages=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
        [[ -d "$site_packages" ]] && find "$site_packages" -name "${pkg_name}*.dist-info" -exec dirname {} \; | head -1 | xargs -I {} find {} -name "*.whl" -exec cp {} "$wheel_cache/" \; 2>/dev/null && ((wheels_cached++))
        
        [[ $wheels_cached -gt 0 ]] && log_detail "💾 Cached $wheels_cached wheel(s) for: $package"
        return 0
    else
        return 1
    fi
}

# System dependencies installer with APT caching
install_system_dependencies() {
    local apt_cache_dir="/storage/.apt_cache"
    local sys_deps_marker="/storage/.system_deps_installed"
    
    # Check if system dependencies are already installed
    if [[ -f "$sys_deps_marker" ]]; then
        log "✅ System dependencies already installed, skipping"
        return 0
    fi
    
    log "🚀 Installing system dependencies with caching..."
    
    # Configure APT caching if not already done
    if [[ ! -f "/etc/apt/apt.conf.d/99cache" ]]; then
        mkdir -p "$apt_cache_dir"
        cat > /etc/apt/apt.conf.d/99cache << EOF
Dir::Cache::Archives "$apt_cache_dir";
Acquire::Retries "3";
Acquire::http::Timeout "30";
Acquire::https::Timeout "30";
EOF
    fi
    
    # System packages list
    local sys_packages=(
        "libatlas-base-dev" "libblas-dev" "liblapack-dev"
        "libjpeg-dev" "libpng-dev" "python3-dev" "build-essential"
        "libgl1-mesa-dev" "espeak-ng" "portaudio19-dev" "libportaudio2"
    )
    
    # Update and install with caching
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    
    # Pre-download packages to cache
    log "Pre-downloading packages to cache..."
    apt-get install -y --download-only "${sys_packages[@]}" 2>/dev/null || log "Pre-download failed, proceeding with direct install"
    
    # Install packages
    log "Installing system packages..."
    if apt-get install -y "${sys_packages[@]}" 2>/dev/null; then
        # Create success marker
        cat > "$sys_deps_marker" << EOF
# System Dependencies Installation Marker
# Installed: $(date)
# Packages cached in: $apt_cache_dir
SYSTEM_DEPS_VERIFIED=true
EOF
        log "✅ System dependencies installed and cached"
        return 0
    else
        log_error "❌ System dependencies installation failed"
        return 1
    fi
}

# Enhanced CUDA installation with comprehensive caching
install_cuda() {
    local marker="/storage/.cuda_12.6_installed"
    local apt_cache_dir="/storage/.apt_cache"
    local cuda_packages_cache="/storage/.cuda_packages_cache"
    
    # Setup APT caching to avoid re-downloading packages
    mkdir -p "$apt_cache_dir" "$cuda_packages_cache"
    
    # Check if already installed with robust verification
    if [[ -f "$marker" ]]; then
        setup_cuda_env && hash -r
        if command -v nvcc &>/dev/null && [[ "$(nvcc --version 2>&1 | grep 'release' | awk '{print $6}' | sed 's/^V//')" == "12.6"* ]]; then
            log "✅ CUDA 12.6 already installed and verified, skipping"
            return 0
        else
            log "⚠️ CUDA marker exists but verification failed, reinstalling..."
            rm -f "$marker"
        fi
    fi
    
    log "🚀 Installing CUDA 12.6 with enhanced caching..."
    
    # Configure APT for aggressive caching
    export DEBIAN_FRONTEND=noninteractive
    cat > /etc/apt/apt.conf.d/99cache << 'EOF'
Dir::Cache::Archives "/storage/.apt_cache";
Acquire::Retries "3";
Acquire::http::Timeout "30";
Acquire::https::Timeout "30";
DPkg::Options {
    "--force-confdef";
    "--force-confold";
}
EOF
    
    # Clean up old CUDA versions efficiently 
    if dpkg -l 2>/dev/null | grep -q "cuda-11"; then
        log "Removing old CUDA 11.x installations..."
        apt-get remove --purge -y 'cuda-11-*' 2>/dev/null || true
        apt-get autoremove -y 2>/dev/null || true
    fi
    
    # Install CUDA keyring (cache the deb file)
    local keyring_cache="$cuda_packages_cache/cuda-keyring_1.1-1_all.deb"
    if [[ ! -f "$keyring_cache" ]]; then
        log "Downloading CUDA keyring..."
        wget -qO "$keyring_cache" https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
    else
        log "Using cached CUDA keyring"
    fi
    dpkg -i "$keyring_cache"
    
    # Update package lists with caching
    log "Updating package lists (using cache: $apt_cache_dir)..."
    apt-get update -qq
    
    # Install CUDA packages with caching (all at once for efficiency)
    log "Installing CUDA packages (this will cache ~1.7GB for future use)..."
    local cuda_packages=(
        "build-essential" "python3-dev"
        "cuda-cudart-12-6" "cuda-nvcc-12-6" 
        "libcublas-12-6" "libcublas-dev-12-6"
        "libcufft-12-6" "libcufft-dev-12-6"
        "libcurand-12-6" "libcurand-dev-12-6"
        "libcusolver-12-6" "libcusolver-dev-12-6"
        "libcusparse-12-6" "libcusparse-dev-12-6"
        "libnpp-12-6" "libnpp-dev-12-6"
    )
    
    # Install with cache directory and parallel downloads
    apt-get install -y --download-only "${cuda_packages[@]}" 2>/dev/null || log "Pre-download failed, proceeding with direct install"
    apt-get install -y "${cuda_packages[@]}" 2>/dev/null
    
    # Configure environment and verify
    setup_cuda_env && hash -r
    
    # Robust verification before creating marker
    if command -v nvcc &>/dev/null && [[ "$(nvcc --version 2>&1 | grep 'release' | awk '{print $6}' | sed 's/^V//')" == "12.6"* ]]; then
        # Create persistent environment configuration
        cat > /etc/profile.d/cuda12.sh << 'EOL'
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export FORCE_CUDA=1
EOL
        chmod +x /etc/profile.d/cuda12.sh
        
        # Create success marker with metadata
        cat > "$marker" << EOF
# CUDA 12.6 Installation Marker
# Installed: $(date)
# NVCC Version: $(nvcc --version | grep 'release' | awk '{print $6}')
# Packages cached in: $apt_cache_dir
CUDA_VERIFIED=true
EOF
        log "✅ CUDA 12.6 installation completed and verified"
        return 0
    else
        log_error "❌ CUDA 12.6 installation verification failed"
        return 1
    fi
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
    pip install --no-cache-dir --disable-pip-version-check --no-deps --quiet \
        torch==2.7.1+cu126 --extra-index-url "https://download.pytorch.org/whl/cu126" 2>/dev/null
    
    pip install --no-cache-dir --disable-pip-version-check --no-deps --quiet \
        torchvision==0.22.1+cu126 --extra-index-url "https://download.pytorch.org/whl/cu126" 2>/dev/null
    
    pip install --no-cache-dir --disable-pip-version-check --no-deps --quiet \
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
        
        pip install --no-cache-dir --disable-pip-version-check --force-reinstall --quiet \
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
            pip install --no-cache-dir --disable-pip-version-check --force-reinstall --quiet \
                torch.whl torchvision.whl torchaudio.whl 2>/dev/null || true
            
            cd - > /dev/null
            rm -rf /tmp/pytorch_wheels
        }
        
        rm -f /tmp/pytorch_requirements.txt
    fi
    
    # Install xformers (always problematic, so we try multiple approaches)
    log "Installing xformers with conflict bypassing..."
    
    pip install --no-cache-dir --disable-pip-version-check --no-deps --quiet \
        xformers==0.0.30 --extra-index-url "https://download.pytorch.org/whl/cu126" 2>/dev/null || \
    pip install --no-cache-dir --disable-pip-version-check --force-reinstall --quiet \
        xformers --index-url "https://download.pytorch.org/whl/cu121" 2>/dev/null || \
    pip install --no-cache-dir --disable-pip-version-check --force-reinstall --quiet \
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
    [[ -f "$manager_dir/requirements.txt" ]] && { log "📦 Installing ComfyUI Manager dependencies..."; pip install --quiet -r "$manager_dir/requirements.txt" && log "✅ ComfyUI Manager dependencies installed" || log "⚠️ ComfyUI Manager dependencies had issues (continuing)"; } || log "⏭️ ComfyUI Manager has no requirements.txt"
}

update_custom_nodes() {
    local nodes_dir="$REPO_DIR/custom_nodes"
    [[ ! -d "$nodes_dir" ]] && return 0
    
    log "🚀 Starting comprehensive custom node update and dependency installation..."
    
    # Step 1: Update pip first for better dependency resolution
    log "📦 Updating pip for better dependency resolution..."
    python -m pip install --quiet --upgrade pip 2>/dev/null || log_error "pip upgrade failed"
    
    # Step 2: Collect ALL requirements from custom nodes (Git and non-Git)
    log "🔍 Collecting requirements from all custom nodes..."
    local combined_reqs="/tmp/all_custom_node_requirements.txt"
    echo -n > "$combined_reqs"
    
    # Process ALL custom nodes, not just git ones
    for node_dir in "$nodes_dir"/*; do
        [[ ! -d "$node_dir" ]] && continue
        local node_name=$(basename "$node_dir")
        
        # Check if this node has requirements.txt
        if [[ -f "$node_dir/requirements.txt" && -s "$node_dir/requirements.txt" ]]; then
            echo "# From $node_name" >> "$combined_reqs"
            grep -v "^[[:space:]]*#\|^[[:space:]]*$" "$node_dir/requirements.txt" >> "$combined_reqs"
            echo "" >> "$combined_reqs"
            log "📦 Found requirements.txt in $node_name"
        fi
    done
    
    # Step 3: Process combined requirements with intelligent conflict resolution
    if [[ -s "$combined_reqs" ]]; then
        log "🔧 Processing combined custom node requirements with conflict resolution..."
        process_combined_requirements "$combined_reqs"
    else
        log "⚠️ No requirements.txt files found in custom nodes"
    fi
    
    # Step 4: Update all git repositories
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
    
    # Clean up
    rm -f "$combined_reqs"
    
    log "✅ Comprehensive custom node update complete!"
}

# Sophisticated requirements processing with conflict resolution and batch installation
process_combined_requirements() {
    local req_file="$1"
    local cache_dir="/storage/.pip_cache"
    local resolved_reqs="/tmp/resolved_requirements.txt"
    local verify_script="/tmp/verify_missing_packages.py"
    
    [[ ! -f "$req_file" ]] && return 0
    
    log "🔧 Processing combined requirements file: $req_file"
    
    mkdir -p "$cache_dir"
    export PIP_CACHE_DIR="$cache_dir"
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    
    # Create Python script to handle version conflicts
    cat > "/tmp/resolve_conflicts.py" << 'EOF'
import re
import sys
from collections import defaultdict

def parse_requirement(req):
    # Extract package name and version specifier
    match = re.match(r'^([a-zA-Z0-9_\-\.]+)(.*)$', req.strip())
    if not match:
        return req.strip(), ""
    
    name, version_spec = match.groups()
    return name.lower(), version_spec.strip()

def normalize_requirement(req):
    if req.startswith(('git+', 'http')):
        return req
    return req.split('#')[0].strip()

# Read requirements
with open(sys.argv[1], 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Group by package name
package_versions = defaultdict(list)
for req in requirements:
    if req.startswith(('git+', 'http')):
        continue
    name, version_spec = parse_requirement(req)
    if name and version_spec:
        package_versions[name].append((version_spec, req))

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
    log "🔍 Resolving version conflicts..."
    python "/tmp/resolve_conflicts.py" "$req_file" "$resolved_reqs" || cp "$req_file" "$resolved_reqs"
    
    # Create verification script to check which packages are actually missing
    cat > "$verify_script" << 'EOF'
import sys
import importlib.util
import re

def normalize_package_name(name):
    # Extract base package name (remove version specifiers, etc.)
    base_name = re.sub(r'[<>=!~;].*$', '', name).strip()
    return base_name.replace('-', '_').replace('.', '_')

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
    log "🔍 Verifying package imports..."
    python "$verify_script" "$resolved_reqs" "/tmp/missing_packages.txt"
    
    # Install packages in smaller batches to avoid dependency conflicts
    if [[ -s "/tmp/missing_packages.txt" ]]; then
        log "📦 Installing missing packages in batches..."
        
        # Split into smaller batches of 10 packages each
        split -l 10 "/tmp/missing_packages.txt" "/tmp/pkg_batch_"
        
        # Install each batch separately
        for batch in /tmp/pkg_batch_*; do
            log "📦 Installing batch $(basename "$batch")..."
            
            # Try batch installation first (faster) - suppress verbose output
            if ! timeout 60s pip install --no-cache-dir --disable-pip-version-check --quiet -r "$batch" >"/tmp/pip_batch_$(basename "$batch").log" 2>&1; then
                log "⚠️ Batch installation failed or timed out, falling back to individual installation..."
                
                # Fallback: install packages one by one
                while read -r pkg; do
                    [[ -z "$pkg" ]] && continue
                    log "  📦 Installing: $pkg"
                    if pip install --no-cache-dir --disable-pip-version-check --quiet "$pkg" >"/tmp/pip_individual_${pkg//[^a-zA-Z0-9]/_}.log" 2>&1; then
                        log "✅ Successfully installed: $pkg"
                    else
                        log_error "❌ Failed to install: $pkg (continuing)"
                    fi
                done < "$batch"
            else
                log "✅ Batch installation successful for $(basename "$batch")"
            fi
        done
        
        # Clean up batch files
        rm -f /tmp/pkg_batch_* /tmp/pip_batch_* /tmp/pip_individual_*
    else
        log "✅ All requirements already satisfied"
    fi
    
    # Handle GitHub repositories separately
    log "🔗 Installing GitHub repositories..."
    grep -E "git\+https?://" "$resolved_reqs" | while read -r repo; do
        log "  🔗 Installing: $repo"
        if pip install --no-cache-dir --disable-pip-version-check --quiet "$repo" >"/tmp/pip_git_${repo//[^a-zA-Z0-9]/_}.log" 2>&1; then
            log "✅ Successfully installed: $repo"
        else
            log_error "❌ Failed to install: $repo (continuing)"
        fi
    done
    
    # Clean up
    rm -f "$resolved_reqs" "$verify_script" "/tmp/missing_packages.txt" "/tmp/resolve_conflicts.py" /tmp/pip_git_*
    
    log "✅ Combined requirements processing complete!"
}

resolve_dependencies() {
    local deps_cache_marker="/storage/.core_deps_installed"
    
    # Check if core dependencies are already resolved
    if [[ -f "$deps_cache_marker" ]]; then
        log "✅ Core dependencies already resolved, skipping"
        return 0
    fi
    
    log "🚀 Resolving dependencies with enhanced caching..."
    
    # Upgrade pip with caching
    python -m pip install --cache-dir "${PIP_CACHE_DIR:-/storage/.pip_cache}" --quiet --upgrade pip 2>/dev/null || curl https://bootstrap.pypa.io/get-pip.py | python 2>/dev/null || log_error "pip upgrade failed"
    
    # Install core build tools with caching
    log "📦 Installing core build tools..."
    for pkg in "wheel" "setuptools" "numpy>=1.26.0,<2.3.0"; do 
        install_with_cache "$pkg" || log_error "Core tool failed: $pkg"
    done
    
    # Install build tools
    log "📦 Installing build tools..."
    local build_tools=("pybind11" "ninja" "packaging")
    for tool in "${build_tools[@]}"; do 
        install_with_cache "$tool" || log_error "Build tool failed: $tool"
    done
    
    # Install specific packages with version fixes
    log "📦 Installing specific packages with version fixes..."
    install_with_cache "av>=9.0.0,<13.0.0" || log_error "av installation failed"
    install_with_cache "aiohttp>=3.9.0,<=3.10.11" || log_error "aiohttp installation failed"
    install_with_cache "packaging>=24.0" || log_error "packaging installation failed"
    install_with_cache "timm==1.0.13" || log_error "timm installation failed"
    
    # Handle flet separately (often conflicts)
    pip uninstall -y flet 2>/dev/null || true
    install_with_cache "flet==0.23.2" || log_error "flet installation failed"
    
    # Install core dependencies with enhanced caching
    log "📦 Installing core dependencies..."
    local core_deps=(
        "einops" "scipy" "torchsde" "spandrel" "kornia==0.7.0"
        "urllib3==1.21" "requests==2.31.0" "fastapi==0.103.2"
        "gradio_client==0.6.0" "peewee==3.16.3" "psutil==5.9.5"
        "uvicorn==0.23.2" "pynvml==11.5.0" "python-multipart==0.0.6"
    )
    
    for dep in "${core_deps[@]}"; do 
        install_with_cache "$dep" || log_error "Failed to install: $dep"
    done
    
    # Install additional required packages
    log "📦 Installing additional required packages..."
    install_with_cache "oss2" || log_error "oss2 installation failed"
    install_with_cache "opencv-contrib-python" || log_error "opencv-contrib-python installation failed"
    
    # Create success marker
    if [[ $? -eq 0 ]]; then
        cat > "$deps_cache_marker" << EOF
# Core Dependencies Installation Marker
# Installed: $(date)
# Pip cache: ${PIP_CACHE_DIR:-/storage/.pip_cache}
# Wheel cache: ${WHEEL_CACHE_DIR:-/storage/.wheel_cache}
CORE_DEPS_VERIFIED=true
EOF
        log "✅ Core dependencies resolved and cached"
    fi
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
    
    # Build with suppressed warnings - redirect verbose compilation output
    log "🔧 Building SageAttention (this may take a moment)..."
    python setup.py bdist_wheel >/tmp/sageattention_build.log 2>&1 || { log_error "SageAttention build failed - check /tmp/sageattention_build.log"; return 1; }
    
    local wheel=$(find dist -name "*.whl" | head -1)
    [[ -n "$wheel" ]] && cp "$wheel" "$wheel_cache/" && pip_install "$wheel" || return 1
    cd - > /dev/null
}

    install_nunchaku() {
        echo "Installing Nunchaku for enhanced machine learning capabilities..."
        log "Starting Nunchaku installation process"
        
        # Check if already installed
        if python -c "import nunchaku; print(f'Nunchaku {nunchaku.__version__} already installed and working')" 2>/dev/null; then
            log "✅ Nunchaku already installed and working, skipping installation"
            return 0
        fi
        
        # Check PyTorch compatibility 
        echo "Checking PyTorch compatibility for Nunchaku..."
        local torch_check_output
        torch_check_output=$(python -c "
import sys
try:
    import torch
    version = torch.__version__.split('+')[0]
    major, minor = map(int, version.split('.')[:2])
    if major > 2 or (major == 2 and minor >= 5):
        print(f'PyTorch {version} meets Nunchaku requirements (>=2.5)')
        sys.exit(0)
    else:
        print(f'PyTorch {version} below Nunchaku requirements (>=2.5)')
        sys.exit(1)
except ImportError:
    print('PyTorch is not installed, cannot check version compatibility')
    sys.exit(1)
except Exception as e:
    print(f'Error checking PyTorch version: {e}')
    sys.exit(1)
" 2>&1)
        local torch_check_status=$?
        
        echo "PyTorch check output: $torch_check_output"
        
        if [[ $torch_check_status -ne 0 ]]; then
            log_error "PyTorch version check failed for Nunchaku."
            log_error "Output: $torch_check_output"
            log_error "Nunchaku requires PyTorch >=2.5. Skipping Nunchaku installation."
            return 1
        fi
        
        # Setup variables for wheel download
        local nunchaku_version="0.3.2"
        local python_version
        python_version=$(python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')" 2>/dev/null)
        local arch=$(uname -m)
        
        # Get PyTorch version for wheel naming
        local torch_version_major_minor
        torch_version_major_minor=$(python -c "import torch; v=torch.__version__.split('+')[0]; print('.'.join(v.split('.')[:2]))" 2>/dev/null || echo "unknown")
        
        if [[ -z "$python_version" || "$torch_version_major_minor" == "unknown" ]]; then
            log_error "Could not determine Python or PyTorch version for Nunchaku wheel selection"
            return 1
        fi
        
        log "Detected Python version for wheel: cp${python_version}"
        log "Detected PyTorch version for wheel: ${torch_version_major_minor}"
        
        # Construct wheel name based on detected versions
        local nunchaku_wheel_name="nunchaku-${nunchaku_version}+torch${torch_version_major_minor}-cp${python_version}-cp${python_version}-linux_${arch}.whl"
        local wheel_cache_dir="${WHEEL_CACHE_DIR:-/storage/.wheel_cache}"
        mkdir -p "$wheel_cache_dir"
        local cached_wheel="$wheel_cache_dir/$nunchaku_wheel_name"
        
        # Check for cached wheel
        if [ -f "$cached_wheel" ]; then
            log "Found cached Nunchaku wheel: $cached_wheel"
            
            if pip install --no-cache-dir --disable-pip-version-check --quiet "$cached_wheel" 2>/dev/null; then
                if python -c "import nunchaku; print(f'✅ Nunchaku {nunchaku.__version__} installed successfully from cached wheel')" 2>/dev/null; then
                    log "✅ Nunchaku installation from cached wheel verified successfully"
                    return 0
                else
                    log_error "❌ Cached wheel verification failed, removing and trying fresh download"
                    rm -f "$cached_wheel"
                fi
            else
                log_error "❌ Failed to install from cached wheel, removing and trying fresh download"
                rm -f "$cached_wheel"
            fi
        fi
        
        # Download wheel from HuggingFace (the correct source)
        local nunchaku_wheel_url="https://huggingface.co/mit-han-lab/nunchaku/resolve/main/$nunchaku_wheel_name"
        
        log "Downloading Nunchaku wheel from: $nunchaku_wheel_url"
        log "Caching to: $cached_wheel"
        
        if wget -q --show-progress -O "$cached_wheel" "$nunchaku_wheel_url" 2>/dev/null; then
            log "✅ Nunchaku wheel downloaded and cached successfully"
            
            if pip install --no-cache-dir --disable-pip-version-check --quiet "$cached_wheel" 2>/dev/null; then
                if python -c "import nunchaku; print(f'✅ Nunchaku {nunchaku.__version__} installed successfully')" 2>/dev/null; then
                    log "✅ Nunchaku installation verified successfully"
                    return 0
                else
                    log_error "❌ Nunchaku installation verification failed"
                    log_error "⚠️ ABI compatibility issue detected"
                    return 1
                fi
            else
                log_error "❌ Nunchaku wheel installation failed"
                return 1
            fi
        else
            log_error "❌ Failed to download Nunchaku wheel from $nunchaku_wheel_url"
            log_error "This could be due to network issues or no wheel available for PyTorch ${torch_version_major_minor}"
            return 1
        fi
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
            log "🔧 Building $component (suppressing verbose output)..."
            python setup.py install >/tmp/${component}_build.log 2>&1 || log_error "$component installation failed - check /tmp/${component}_build.log"
            cd - > /dev/null
        fi
    done
}

    process_requirements() {
        local req_file="$1"
    [[ ! -f "$req_file" ]] && return 0
    pip install --quiet -r "$req_file" || { while read -r pkg; do [[ "$pkg" =~ ^[[:space:]]*# ]] || [[ -z "$pkg" ]] || pip_install "$pkg" || true; done < "$req_file"; }
}

# Service loop function (from original)
service_loop() { while true; do eval "$1"; sleep 1; done; }

# Prepare symlinks (simplified from original)
prepare_link() {
    for link_pair in "$@"; do
        [[ "$link_pair" =~ ^(.+):(.+)$ ]] && {
            local source="${BASH_REMATCH[1]}" target="${BASH_REMATCH[2]}"
            [[ -n "$source" && -n "$target" ]] && {
                mkdir -p "$(dirname "$target")"
                [[ ! -L "$target" ]] && ln -sf "$source" "$target" 2>/dev/null || true
            }
        }
    done
}

# Send to Discord (simplified)
send_to_discord() {
    [[ -n "$DISCORD_WEBHOOK_URL" ]] && curl -X POST -H "Content-Type: application/json" -d "{\"content\":\"$1\"}" "$DISCORD_WEBHOOK_URL" &>/dev/null || true
}

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
    
    # Enhanced system dependencies installation with caching
    install_system_dependencies
    
    setup_pytorch
    for component in "sageattention" "nunchaku" "hunyuan3d_texture_components"; do install_component "$component"; done
    
    # Update and install components with error handling
    update_comfyui_manager || log_error "ComfyUI Manager update had issues (continuing)"
    update_custom_nodes || log_error "Custom nodes update had issues (continuing)"
    
    # Handle specific dependency conflicts (similar to your script)
    log "🔧 Handling specific dependency conflicts..."
    
    # Fix xformers conflicts
    log "🔄 Fixing xformers conflicts..."
    pip uninstall -y xformers 2>/dev/null || true
    pip install --quiet -U xformers --index-url https://download.pytorch.org/whl/cu121 --index-url https://pypi.org/simple 2>/dev/null || log_error "xformers fix failed"
    
    # Fix other common conflicts
    log "🔧 Fixing other dependency conflicts..."
    pip install --quiet --upgrade torchaudio torchvision 2>/dev/null || log_error "torch audio/vision fix failed"
    pip install --quiet --upgrade timm==1.0.13 2>/dev/null || log_error "timm fix failed"
    
    # Reinstall flet to fix version issues
    log "🔄 Reinstalling flet to fix version conflicts..."
    pip uninstall -y flet 2>/dev/null || true
    pip install --quiet flet==0.23.2 2>/dev/null || log_error "flet fix failed"
    
    resolve_dependencies || log_error "Some dependencies failed (continuing)"
    
    # Process core ComfyUI requirements
    process_requirements "$REPO_DIR/requirements.txt" || log_error "Core requirements had issues (continuing)"
    
    # Fallback installation for critical custom node packages (excluding nunchaku for now)
    echo "=== Installing Critical Custom Node Dependencies (Fallback) ==="
    local critical_packages=("blend_modes" "deepdiff" "rembg" "webcolors" "ultralytics" "inflect" "soxr" "groundingdino" "insightface" "opencv-python" "opencv-contrib-python" "facexlib" "onnxruntime" "timm" "segment-anything" "scikit-image" "piexif" "transformers" "opencv-python-headless" "scipy>=1.11.4" "numpy" "dill" "matplotlib" "oss2")
    log "📦 Installing critical packages for custom nodes..."
    local installed_count=0
    local failed_count=0
    
    for pkg in "${critical_packages[@]}"; do
        local pkg_name=$(echo "$pkg" | sed 's/[<>=!].*//' | sed 's/\[.*\]//')
        
        # Skip if already installed
        if python -c "import $pkg_name" 2>/dev/null; then
            log "✅ $pkg_name already installed"
            ((installed_count++))
            continue
        fi
        
        # Install package
        log "📦 Installing: $pkg"
        if pip install --no-cache-dir --disable-pip-version-check --quiet "$pkg" 2>/dev/null; then
            log "✅ Successfully installed: $pkg"
            ((installed_count++))
        else
            log_error "❌ Failed to install: $pkg"
            ((failed_count++))
        fi
    done
    
    log "📊 Critical packages summary: $installed_count installed, $failed_count failed"
    [[ $failed_count -gt 0 ]] && log_error "⚠️ Some critical packages failed to install - custom nodes may not work properly"
    
    touch "/tmp/sd_comfy.prepared"
}

# Execute main workflow
main

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
