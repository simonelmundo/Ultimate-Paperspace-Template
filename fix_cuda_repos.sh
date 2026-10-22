#!/bin/bash

echo "Fixing CUDA repository configuration issues..."

# Remove duplicate entries from sources.list
if grep -q "developer.download.nvidia.com" /etc/apt/sources.list; then
    echo "Removing CUDA repository entries from /etc/apt/sources.list..."
    sed -i '/developer\.download\.nvidia\.com/d' /etc/apt/sources.list
fi

# Remove any existing CUDA repository files that might conflict
if [ -f "/etc/apt/sources.list.d/cuda-ubuntu2004-x86_64.list" ]; then
    echo "Removing existing CUDA repository file..."
    rm -f /etc/apt/sources.list.d/cuda-ubuntu2004-x86_64.list
fi

# Clean up any other potential CUDA repository files
rm -f /etc/apt/sources.list.d/cuda*.list

# Clean apt cache to ensure fresh repository data
apt-get clean
rm -rf /var/lib/apt/lists/*

echo "CUDA repository cleanup completed."

# Now try to add the CUDA repository properly
echo "Adding CUDA repository key..."
wget -qO /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/cuda-keyring.deb
rm -f /tmp/cuda-keyring.deb

echo "Running apt-get update..."
apt-get update

echo "CUDA repository configuration fixed successfully!" 