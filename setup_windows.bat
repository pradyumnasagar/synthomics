@echo off
REM Setup script for Genome Sonics on Windows
REM Run in PowerShell or Command Prompt as Administrator

echo 🧬 Setting up Genome → Music ^& Art...
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.9+ from python.org
    pause
    exit /b 1
)

echo ✓ Python found

REM Create virtual environment
echo 🐍 Creating Python virtual environment...
python -m venv venv

REM Activate
call venv\Scripts\activate.bat

REM Install packages
echo 📥 Installing Python packages...
pip install --upgrade pip
pip install -r requirements.txt

REM Create output directory
if not exist output mkdir output

echo.
echo ✅ Setup complete!
echo.
echo To run the app:
echo   1. Activate: venv\Scripts\activate.bat
echo   2. Run: streamlit run app.py
echo.
echo Note: For FluidSynth audio synthesis, install with:
echo   choco install fluidsynth
echo   (or download from https://github.com/FluidSynth/fluidsynth/releases)
echo.
pause
