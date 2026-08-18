@echo off
echo ====================================================
echo      Starting AI CCTV Detection System
echo ====================================================

REM Start the Python backend and wait a bit
start "CCTV AI Backend" cmd /k "python src/main.py"

echo Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

REM Open the dashboard in the default browser
echo Opening Dashboard locally at http://localhost:8000
start http://localhost:8000

REM Start Cloudflare Tunnel for public access
echo Starting Cloudflare Tunnel for Public Access...
start "Cloudflare Public Link" cmd /k "cloudflared.exe tunnel --url http://127.0.0.1:8000"

echo Done! The CCTV AI and Public Tunnel are running in the other windows.
echo Press any key to exit this launcher...
pause >nul
