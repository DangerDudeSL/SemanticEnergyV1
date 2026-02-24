@echo off
echo ============================================
echo  Semantic Energy - Environment Setup
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Create virtual environment
echo [1/4] Creating virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

:: Activate virtual environment
echo [2/4] Activating virtual environment...
call .venv\Scripts\activate.bat

:: Install PyTorch with CUDA
echo [3/4] Installing PyTorch with CUDA 12.4 support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo [WARNING] CUDA install failed. Falling back to CPU-only PyTorch...
    pip install torch torchvision torchaudio
)

:: Install remaining dependencies
echo [4/4] Installing project dependencies...
pip install -r requirements.txt

echo.
echo ============================================
echo  Setup complete!
echo  Run the app with: .\start.ps1
echo ============================================
pause
