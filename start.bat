@echo off
echo ============================================================
echo AMD OCT ANALYSIS PLATFORM - LAUNCHER
echo ============================================================
echo.

REM Vérifier si Python est installé
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org
    pause
    exit /b 1
)

echo [1/3] Checking dependencies...
pip show flask >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install flask flask-cors
)

echo [2/3] Starting API Server...
start "AMD API Server" cmd /k "python api.py"
timeout /t 3 >nul

echo [3/3] Starting Web Interface...
start "AMD Web Interface" cmd /k "cd IHM && python -m http.server 8000"
timeout /t 2 >nul

echo.
echo ============================================================
echo SERVERS STARTED SUCCESSFULLY
echo ============================================================
echo.
echo API Server:   http://localhost:5000
echo Web Interface: http://localhost:8000
echo.
echo Opening web browser...
timeout /t 2 >nul
start http://localhost:8000

echo.
echo Press Ctrl+C in each window to stop the servers
echo.
pause