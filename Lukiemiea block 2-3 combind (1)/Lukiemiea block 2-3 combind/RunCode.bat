@echo off
REM -----------------------------------------------------------
REM HDS Edge Detection & Denoising Project Launcher
REM This script checks for Python, installs required packages,
REM and runs the main_combined_app.py application.
REM -----------------------------------------------------------

REM 1. Check if Python is installed
where python >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not added to PATH.
    pause
    exit /b
)

REM 2. Install required Python packages (for GUI, image processing, etc.)
echo Installing required Python packages...
pip install --upgrade pip
pip install numpy opencv-python scipy scikit-image pillow

REM 3. Run the main application
echo Launching the Combined Edge Detection & HDS Denoising Tool...
python main_combined_app.py

REM 4. Pause to keep the window open after execution
pause