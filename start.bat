@echo off
echo ====================================================
echo      Starting AI CCTV Detection System
echo ====================================================

REM Start the Python backend and wait a bit
start "CCTV AI Backend" cmd /k "python src/main.py"

echo Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

REM Open the dashboard in the default browser
echo Opening Dashboard at http://localhost:8000
start http://localhost:8000

echo Done! The CCTV AI is running in the other window.
echo Press any key to exit this launcher...
pause >nul
