@echo off
echo ====================================================
echo      Starting AI CCTV Detection System
echo ====================================================

REM Start the Python backend and wait a bit
start "CCTV AI Backend" cmd /k "python src/main.py"

echo Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

REM Open the dashboard in the default browser
REM Start Cloudflared Tunnel for seamless mobile public access
echo Starting Cloudflared Tunnel for Public Access...
start "Cloudflared Public Link" cmd /k "cloudflared.exe tunnel --url http://localhost:8000"

echo ====================================================
echo  CCTV AI Backend and Public Tunnel are running.
echo.
echo  [1] LOCAL PC ACCESS:
echo      http://localhost:8000
echo.
echo  [2] MOBILE ACCESS (LAN / Wi-Fi yang sama):
for /f "delims=" %%I in ('python -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); print(s.getsockname()[0]); s.close()"') do (
    echo      http://%%I:8000
)
echo.
echo  [3] MOBILE ACCESS (Internet / Telegram):
echo      - Cek jendela "Cloudflared Public Link" untuk mendapatkan URL (https://xxxx.trycloudflare.com)
echo      - Masukkan URL tersebut ke Dashboard - Telegram Config.
echo ====================================================
echo Press any key to exit this launcher...
pause >nul

