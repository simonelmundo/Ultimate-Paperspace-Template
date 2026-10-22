#!/bin/bash
# List all dependencies in both virtual environments

echo "========================================"
echo "   VIRTUAL ENVIRONMENT DEPENDENCIES"
echo "========================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to list packages in a venv
list_venv_deps() {
    local venv_path="$1"
    local venv_name="$2"
    
    echo -e "${BLUE}========================================"
    echo "   $venv_name"
    echo "========================================${NC}"
    echo ""
    
    if [[ ! -d "$venv_path" ]]; then
        echo -e "${YELLOW}⚠️  Virtual environment not found: $venv_path${NC}"
        return 1
    fi
    
    local python_path="$venv_path/bin/python"
    local pip_path="$venv_path/bin/pip"
    
    echo -e "${GREEN}Python Version:${NC}"
    $python_path --version
    echo ""
    
    echo -e "${GREEN}Total packages installed:${NC}"
    $pip_path list 2>/dev/null | wc -l
    echo ""
    
    echo -e "${GREEN}All installed packages:${NC}"
    echo "----------------------------------------"
    $pip_path list 2>/dev/null
    echo ""
    
    echo -e "${GREEN}Key packages (torch, xformers, etc.):${NC}"
    echo "----------------------------------------"
    for pkg in torch torchvision torchaudio xformers transformers diffusers accelerate huggingface_hub pillow numpy scipy; do
        if $pip_path show "$pkg" &>/dev/null; then
            local version
            version=$($pip_path show "$pkg" 2>/dev/null | grep "Version:" | cut -d' ' -f2)
            echo "  $pkg: $version"
        else
            echo "  $pkg: Not installed"
        fi
    done
    echo ""
}

# ComfyUI Virtual Environment
list_venv_deps "/storage/.venvs/sd_comfy-env" "COMFYUI ENVIRONMENT (sd_comfy-env)"

# AI Toolkit Virtual Environment
list_venv_deps "/storage/.venvs/ai-toolkit-env" "AI TOOLKIT ENVIRONMENT (ai-toolkit-env)"

echo "========================================"
echo "   SUMMARY"
echo "========================================"
echo ""
echo "ComfyUI venv: /storage/.venvs/sd_comfy-env"
echo "AI Toolkit venv: /storage/.venvs/ai-toolkit-env"
echo ""
echo "To export packages to a file:"
echo "  /storage/.venvs/sd_comfy-env/bin/pip freeze > comfyui_requirements.txt"
echo "  /storage/.venvs/ai-toolkit-env/bin/pip freeze > ai_toolkit_requirements.txt"