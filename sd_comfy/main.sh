#!/bin/bash
set -e

current_dir=$(dirname "$(realpath "$0")")
cd $current_dir
source .env

# Set up a trap to call the error_exit function on ERR signal
trap 'error_exit "### ERROR ###"' ERR


echo "### Setting up Stable Diffusion Comfy ###"
log "Setting up Stable Diffusion Comfy"
if [[ "$REINSTALL_SD_COMFY" || ! -f "/tmp/sd_comfy.prepared" ]]; then

    
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

    # Install PyTorch with CUDA first
    echo "Installing PyTorch with CUDA support..."
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

    cd $REPO_DIR
    try_install "xformers"
    try_install "shaderflow"

    # Add before DepthFlow installation
    export PYOPENGL_PLATFORM=egl  # Force OpenGL to use GPU
    export FORCE_CUDA=1           # Force CUDA usage
    export CUDA_VISIBLE_DEVICES=0 # Use first GPU

    # Install and setup DepthFlow with proper symlink
    echo "Installing and configuring DepthFlow..."
    # Suppress root warning during installation
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

echo "### Done ###"
