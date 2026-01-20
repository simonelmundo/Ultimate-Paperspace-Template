#!/bin/bash
set -e

current_dir=$(dirname "$(realpath "$0")")
cd $current_dir
source .env

# Set up a trap to call the error_exit function on ERR signal
trap 'error_exit "### ERROR ###"' ERR


echo "### Setting up Image Browser ###"
log "Setting up Image Browser"
if [[ "$REINSTALL_IMAGE_BROWSER" || ! -f "/tmp/image_browser.prepared" ]]; then

    TARGET_REPO_URL="https://github.com/zanllp/sd-webui-infinite-image-browsing.git" \
    TARGET_REPO_DIR=$REPO_DIR \
    UPDATE_REPO="auto" \
    TARGET_REPO_BRANCH="main" \
    prepare_repo
    rm -rf $VENV_DIR/image_browser-env
    
    
    python3 -m venv /tmp/image_browser-env
    
    # Install FFmpeg 7 development libraries required for av package (v14+)
    log "📦 Installing FFmpeg 7 development libraries for av package..."
    apt-get update -qq && apt-get install -y \
        ffmpeg \
        libavformat-dev \
        libavcodec-dev \
        libavdevice-dev \
        libavutil-dev \
        libavfilter-dev \
        libswscale-dev \
        libswresample-dev \
        pkg-config \
        build-essential \
        > /dev/null 2>&1 || {
        log_error "Warning: Some FFmpeg packages failed to install"
    }
    
    source $VENV_DIR/image_browser-env/bin/activate

    pip install pip==24.0
    pip install --upgrade wheel setuptools
    
    cd $REPO_DIR
    # Try to prefer binary wheels first (faster, no compilation needed)
    # If wheels aren't available, fall back to building from source (requires FFmpeg)
    pip install --prefer-binary -r requirements.txt || {
        log "⚠️ Binary wheels not available, building from source (requires FFmpeg)..."
        pip install -r requirements.txt
    }
    
    touch /tmp/image_browser.prepared
else
    
    source $VENV_DIR/image_browser-env/bin/activate
    
fi
log "Finished Preparing Environment for Image Browser"

# Explicitly disable authentication by unsetting IIB_SECRET_KEY
# This ensures the image browser runs without password protection
unset IIB_SECRET_KEY
export IIB_SECRET_KEY=""
log "🔓 Authentication disabled (IIB_SECRET_KEY unset)"

if [[ -z "$INSTALL_ONLY" ]]; then
  echo "### Starting Image Browser ###"
  log "Starting Image Browser"
  if [ -n IMAGE_OUTPUTS_DIR ]; then
      cd $IMAGE_OUTPUTS_DIR
  else
      cd $REPO_DIR
  fi
  # Start with IIB_SECRET_KEY explicitly unset to disable authentication
  IIB_SECRET_KEY="" PYTHONUNBUFFERED=1 service_loop "python $REPO_DIR/app.py --port 7002" > $LOG_DIR/image_browser.log 2>&1 &
  echo $! > /tmp/image_browser.pid
fi


send_to_discord "Image Browser Started"

if env | grep -q "PAPERSPACE"; then
  send_to_discord "Link: https://$PAPERSPACE_FQDN/image-browser/"
fi


if [[ -n "${CF_TOKEN}" ]]; then
  if [[ "$RUN_SCRIPT" != *"image_browser"* ]]; then
    export RUN_SCRIPT="$RUN_SCRIPT,image_browser"
  fi
  bash $current_dir/../cloudflare_reload.sh
fi

echo "### Done ###"
