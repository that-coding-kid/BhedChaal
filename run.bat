@echo off
echo BhedChaal - CCTV Analysis Tool
echo =============================
echo.
echo 1. Basic Mode
echo 2. Advanced Mode
echo 3. Interactive Mode (mouse point selection)
echo 4. Simple Interactive Mode (more compatible)
echo.

choice /C 1234 /N /M "Choose a mode (1, 2, 3, or 4): "

if errorlevel 4 (
    echo Starting Simple Interactive Mode...
    python run_app.py --simple
) else if errorlevel 3 (
    echo Starting Interactive Mode...
    python run_app.py --interactive
) else if errorlevel 2 (
    echo Starting Advanced Mode...
    python run_app.py --advanced
) else (
    echo Starting Basic Mode...
    python run_app.py
)

pause 