#!/bin/bash

echo "Continuing Stable Diffusion Comfy installation with CUDA 11.8..."

# Remove the old marker file to force reinstallation
rm -f /storage/.cuda_12.6_installed

# Clean up any partial CUDA installations
apt-get remove --purge -y 'cuda-*' 'cuda-repo-*' 'nvidia-cuda-toolkit' || true
apt-get autoremove -y

# Clean up repository issues
sed -i '/developer\.download\.nvidia\.com/d' /etc/apt/sources.list
rm -f /etc/apt/sources.list.d/cuda*.list
apt-get clean
rm -rf /var/lib/apt/lists/*

# Add CUDA 11.8 repository
wget -qO /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/cuda-keyring.deb
rm -f /tmp/cuda-keyring.deb

# Update package lists
apt-get update

# Install CUDA 11.8 packages
echo "Installing CUDA 11.8 packages..."
apt-get install -y \
    build-essential \
    python3-dev \
    libatlas-base-dev \
    libblas-dev \
    liblapack-dev \
    libjpeg-dev \
    libpng-dev \
    libgl1- \
    cuda-cudart-11-8 \
    cuda-cudart-dev-11-8 \
    cuda-nvcc-11-8 \
    cuda-cupti-11-8 \
    cuda-cupti-dev-11-8 \
    libcublas-11-8 \
    libcublas-dev-11-8 \
    libcufft-11-8 \
    libcufft-dev-11-8 \
    libcurand-11-8 \
    libcurand-dev-11-8 \
    libcusolver-11-8 \
    libcusolver-dev-11-8 \
    libcusparse-11-8 \
    libcusparse-dev-11-8 \
    libnpp-11-8 \
    libnpp-dev-11-8

# Set up environment
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export FORCE_CUDA=1

# Verify installation
echo "Verifying CUDA 11.8 installation..."
nvcc --version

# Create marker file
touch /storage/.cuda_11.8_installed

echo "CUDA 11.8 installation completed successfully!"
echo "You can now continue with the main installation script." 