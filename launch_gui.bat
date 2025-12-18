@echo off
echo ========================================
echo HY-WorldPlay 1.5 GUI Launcher
echo ========================================
echo.

REM Check if conda environment exists
conda info --envs | findstr worldplay >nul 2>&1
if errorlevel 1 (
    echo Creating conda environment 'worldplay'...
    conda create --name worldplay python=3.10 -y
    if errorlevel 1 (
        echo Failed to create conda environment
        pause
        exit /b 1
    )
)

echo Activating conda environment...
call conda activate worldplay

echo.
echo Checking dependencies...
python -c "import gradio" 2>nul
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install requirements
        pause
        exit /b 1
    )
)

echo.
echo ========================================
echo Starting HY-WorldPlay GUI...
echo ========================================
echo.
echo The interface will open in your browser at:
echo http://localhost:7860
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

python app_gradio.py

pause