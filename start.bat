@echo off
title Fake News Detector - Starting...
color 0A

:: Always switch to script's own folder
cd /d "%~dp0"

echo.
echo  =============================================
echo    Fake News Detector - Starting API Server
echo  =============================================
echo.

:: Check if venv exists
if not exist "member2_ml_api\venv\Scripts\uvicorn.exe" (
    echo  [ERROR] venv not found! Run this first:
    echo          cd member2_ml_api
    echo          python -m venv venv
    echo          venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

:: Check if model files exist
if not exist "member2_ml_api\models\trained_model.pkl" (
    echo  [WARNING] trained_model.pkl not found in models/
    echo  The /predict endpoint will return 503.
    echo.
)

echo  [+] Starting FastAPI server on http://localhost:8000
echo  [+] Press Ctrl+C to stop the server
echo.
echo  Opening frontend in browser...
echo.

:: Open the frontend HTML file in default browser
start "" "%~dp0index.html"

:: Small delay so browser opens first
timeout /t 2 /nobreak >nul

:: Start uvicorn server (this blocks until Ctrl+C)
cd /d "%~dp0member2_ml_api"
"venv\Scripts\uvicorn.exe" app.main:app --host 0.0.0.0 --port 8000 --reload

echo.
echo  [!] Server stopped.
pause
