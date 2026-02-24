#!/bin/bash
echo "============================================"
echo " Semantic Energy - Environment Setup"
echo "============================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed."
    echo "Install it via: sudo apt install python3 python3-venv python3-pip"
    exit 1
fi

# Create virtual environment
echo "[1/4] Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "[2/4] Activating virtual environment..."
source .venv/bin/activate

# Install PyTorch with CUDA
echo "[3/4] Installing PyTorch with CUDA 12.4 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if [ $? -ne 0 ]; then
    echo "[WARNING] CUDA install failed. Falling back to CPU-only PyTorch..."
    pip install torch torchvision torchaudio
fi

# Install remaining dependencies
echo "[4/4] Installing project dependencies..."
pip install -r requirements.txt

echo ""
echo "============================================"
echo " Setup complete!"
echo " Activate env:  source .venv/bin/activate"
echo " Run backend:   cd backend && python app.py"
echo " Run frontend:  cd frontend && python -m http.server 3000"
echo "============================================"
