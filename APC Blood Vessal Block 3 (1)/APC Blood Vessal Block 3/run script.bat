@echo off
REM Check for Python
where python >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH.
    pause
    exit /b
)

REM Install required Python packages
echo Installing required Python packages...
pip install numpy opencv-python matplotlib Pillow

REM Run the Python script
echo Running app.py...
python app.py

pause
