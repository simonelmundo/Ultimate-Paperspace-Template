#!/bin/bash
set -e

current_dir=$(dirname "$(realpath "$0")")
source $current_dir/../helper.sh

# Set up a trap to call the error_exit function on ERR signal
trap 'error_exit "### ERROR ###"' ERR

echo "### Setting up Model Download ###"
# Dependencies (aria2 + Python modules) should be installed by caller script
# This keeps the background process focused on downloading models with aria2
# If running standalone, install dependencies here
if ! dpkg -s aria2 >/dev/null 2>&1; then
    echo "⚠️  aria2 not found - installing (this should be done by caller script)"
    apt-get install -qq aria2 -y > /dev/null 2>&1 || echo "Failed to install aria2"
fi

MODULES=("requests" "gdown" "bs4" "python-dotenv")
for module in "${MODULES[@]}"; do
    if ! pip show $module >/dev/null 2>&1; then
        echo "⚠️  $module not found - installing (this should be done by caller script)"
        pip install --quiet --no-cache-dir $module 2>/dev/null || echo "Failed to install $module"
    fi
done

if ! [ -v "MODEL_DIR" ]; then
    source $current_dir/../../.env
    export MODEL_DIR="$DATA_DIR/stable-diffusion-models"
fi
# This only happen when directly using this script
if ! [ -v "MODEL_LIST" ]; then
    env | grep -v '^_' | sed 's/\([^=]*\)=\(.*\)/\1='\''\2'\''/' > $current_dir/.env
fi

echo "### Downloading Models ###"

python $current_dir/download_model.py
echo "### Finished Model Download ###"
