#!/bin/bash
# Don't exit on error

function source_env_file() {
  if [[ -e ".env" ]]; then
    source ".env"
  fi
}

function check_required_env_vars() {
  local required_vars=($(echo "$REQUIRED_ENV" | tr ',' '\n'))
  local missing_vars=()
  for var in "${required_vars[@]}"; do
    if [[ -z "${!var}" ]]; then
      missing_vars+=("$var")
    fi
  done
  if [[ ${#missing_vars[@]} -gt 0 ]]; then
    echo "The following required environment variables are missing: ${missing_vars[*]}"
    return 1
  fi
  return 0
}

export SCRIPT_ROOT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
cd $SCRIPT_ROOT_DIR
source_env_file

# Prepare Path (for local install)
mkdir -p $DATA_DIR
mkdir -p $WORKING_DIR
mkdir -p $ROOT_REPO_DIR
mkdir -p $VENV_DIR
mkdir -p $LOG_DIR

echo "Installing common dependencies"
apt-get update -qq
apt-get install -qq -y curl jq git-lfs ninja-build \
    aria2 zip python3-venv python3-dev python3.10 \
    python3.10-venv python3.10-dev python3.10-tk libgl1 > /dev/null

# Add alias to check the status of the web app
chmod +x $WORKING_DIR/status_check.py
echo "alias status='watch -n 1 /$WORKING_DIR/status_check.py'" >> ~/.bashrc

# Use Nginx to expose web app in Paperspace
apt-get install -qq -y nginx > /dev/null
cp /$WORKING_DIR/nginx/default /etc/nginx/sites-available/default
cp /$WORKING_DIR/nginx/nginx.conf /etc/nginx/nginx.conf
/usr/sbin/nginx

# Always set RUN_SCRIPT to ensure sd_comfy runs before textgen (textgen uses ComfyUI's environment)
# Order: command,image_browser,rclone,sd_comfy,textgen
export RUN_SCRIPT="command,image_browser,rclone,sd_comfy"
echo "RUN_SCRIPT set to: $RUN_SCRIPT"

run_script="$RUN_SCRIPT"

# Separate the variable by commas
IFS=',' read -ra scripts <<< "$run_script"

# Prepare required path
mkdir -p $IMAGE_OUTPUTS_DIR
if [[ ! -d $WORKING_DIR/image_outputs ]]; then
  ln -s $IMAGE_OUTPUTS_DIR $WORKING_DIR/image_outputs
fi

# Scripts deferred until ComfyUI launch (handled in sd_comfy/main.sh), comma-separated.
# image_browser runs after Comfy setup finishes, alongside ComfyUI start — not during early entry.
export DEFER_SCRIPTS="${DEFER_SCRIPTS:-image_browser}"
# Optional: other scripts that may still run non-blocking during entry (empty by default).
export BACKGROUND_SCRIPTS="${BACKGROUND_SCRIPTS:-}"

is_listed_script() {
  local name="$1"
  local list="$2"
  local item
  IFS=',' read -ra _list <<< "$list"
  for item in "${_list[@]}"; do
    [[ -n "$item" && "$item" == "$name" ]] && return 0
  done
  return 1
}

# Loop through each script and execute the corresponding case
echo "Starting script(s)"
echo "RUN_SCRIPT contains: $RUN_SCRIPT"
if [[ -n "$DEFER_SCRIPTS" ]]; then
  echo "DEFER_SCRIPTS (start with ComfyUI): $DEFER_SCRIPTS"
fi
if [[ -n "$BACKGROUND_SCRIPTS" ]]; then
  echo "BACKGROUND_SCRIPTS (non-blocking): $BACKGROUND_SCRIPTS"
fi
for script in "${scripts[@]}"
do
  echo "Processing script: $script"
  cd $SCRIPT_ROOT_DIR
  if [[ ! -d $script ]]; then
    echo "⚠️ Script folder $script not found, skipping..."
    continue
  fi
  cd $script
  source_env_file
  if ! check_required_env_vars; then
    echo "⚠️ One or more required environment variables are missing for $script, skipping..."
    continue
  fi
  if is_listed_script "$script" "$DEFER_SCRIPTS"; then
    echo "⏭️ Deferring $script until ComfyUI launch (sd_comfy will start it)"
    continue
  fi
  if is_listed_script "$script" "$BACKGROUND_SCRIPTS"; then
    echo "✅ Starting $script in background (continuing to next script)..."
    mkdir -p "$LOG_DIR"
    local_log="$LOG_DIR/${script}_entry.log"
    nohup bash control.sh reload >> "$local_log" 2>&1 &
    echo $! > "/tmp/${script}_entry.pid"
    echo "📋 $script setup log: tail -f $local_log"
    echo "✅ Queued $script (background)"
    continue
  fi
  echo "✅ Starting $script..."
  bash control.sh reload 2>&1
  echo "✅ Finished $script"
done
