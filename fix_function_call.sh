#!/bin/bash

echo "Fixing function call syntax in main.sh..."

# Fix the function call by adding parentheses
sed -i 's/cleanup_cuda_repos$/cleanup_cuda_repos/g' sd_comfy/main.sh

echo "Function call syntax fixed!" 