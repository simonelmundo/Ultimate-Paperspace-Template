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
        libxi-dev || true

    # Initialize array for failed installations
    failed_installations=()

    # Function to attempt installation and track failures
    try_install() {
        local package="$1"
        echo "Installing: $package"
        if ! pip install $package 2>&1 | tee /tmp/pip_install.log; then
            failed_installations+=("$package")
            echo "Failed to install: $package"
            return 0  # Return 0 to continue script execution
        fi
        return 0
    }

    # Install pip packages individually, track failures
    try_install "pip==24.0"
    try_install "--upgrade wheel setuptools"
    
    cd $REPO_DIR
    try_install "xformers"
    try_install "torchvision torchaudio --no-deps"
    
    # Read and install requirements line by line
    while IFS= read -r requirement || [ -n "$requirement" ]; do
        if [[ ! -z "$requirement" && ! "$requirement" =~ ^# ]]; then
            try_install "$requirement"
        fi
    done < requirements.txt
    
    while IFS= read -r requirement || [ -n "$requirement" ]; do
        if [[ ! -z "$requirement" && ! "$requirement" =~ ^# ]]; then
            try_install "$requirement"
        fi
    done < "/notebooks/sd_comfy/additional_requirements.txt"

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
