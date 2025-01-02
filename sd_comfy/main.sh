#!/bin/bash
# Remove or comment out set -e to prevent script from stopping on errors
# set -e

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
        if ! pip install $package 2>&1 | tee /tmp/pip_install.log || true; then
            failed_installations+=("$package")
            echo "Failed to install: $package"
            echo "See /tmp/pip_install.log for details"
        fi
        # Always return true to continue script
        return 0
    }

    # Function to safely process requirements file
    process_requirements() {
        local req_file="$1"
        echo "Processing requirements from: $req_file"
        
        # Initialize local pip args
        local pip_args=""
        
        while IFS= read -r requirement || [ -n "$requirement" ]; do
            # Skip empty lines and comments
            if [[ -z "$requirement" || "$requirement" =~ ^# ]]; then
                continue
            fi
            
            # Handle pip configuration options
            if [[ "$requirement" =~ ^--.*$ ]]; then
                pip_args="$pip_args $requirement"
                continue
            fi
            
            # Handle nested requirements files
            if [[ "$requirement" =~ ^-r ]]; then
                local nested_req_file=$(echo "$requirement" | cut -d' ' -f2)
                echo "Processing nested requirements file: $nested_req_file"
                process_requirements "$nested_req_file" || true
                continue
            fi
            
            # Skip local directory references
            if [[ "$requirement" =~ ^file:///storage/stable-diffusion-comfy ]]; then
                echo "Skipping local directory reference: $requirement"
                continue
            fi
            
            # Install package with accumulated pip args
            if [ ! -z "$pip_args" ]; then
                try_install "$requirement $pip_args" || true
            else
                try_install "$requirement" || true
            fi
        done < "$req_file"
        return 0
    }

    # Install base requirements first
    try_install "pip==24.0"
    try_install "--upgrade wheel setuptools"
    try_install "numpy"  # Install numpy first as it's a common dependency
    
    # Install PyTorch with CUDA first
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    cd $REPO_DIR
    try_install "xformers"
    
    # Create answer file for DepthFlow's CUDA prompt
    echo "cuda" > /tmp/depthflow_answer
    
    # Install DepthFlow with automated CUDA selection
    echo "Installing DepthFlow..."
    cat /tmp/depthflow_answer | try_install "depthflow[shaderflow]"
    rm /tmp/depthflow_answer
    
    # Handle tensorflow version compatibility
    if ! pip install "tensorflow==2.6.2" 2>/dev/null; then
        echo "Attempting to install compatible tensorflow version..."
        try_install "tensorflow>=2.8.0,<2.19.0"
    fi
    
    # Handle imgui-bundle with specific build requirements
    export CMAKE_ARGS="-DUSE_X11=ON"
    try_install "imgui-bundle --no-cache-dir"
    
    # Process requirements files
    process_requirements "requirements.txt"
    
    # Install custom nodes requirements
    if [ -f "/storage/stable-diffusion-comfy/custom_nodes/ComfyUI-Depthflow-Nodes/requirements.txt" ]; then
        process_requirements "/storage/stable-diffusion-comfy/custom_nodes/ComfyUI-Depthflow-Nodes/requirements.txt"
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
