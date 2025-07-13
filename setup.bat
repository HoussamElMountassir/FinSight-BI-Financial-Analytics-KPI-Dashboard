@echo off
echo ===============================================
echo    FinSight BI: Banking KPI Dashboard
echo    Advanced Business Intelligence Solution
echo ===============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9 or higher from https://python.org
    pause
    exit /b 1
)

echo ✓ Python detected
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✓ Virtual environment created
) else (
    echo ✓ Virtual environment exists
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing/updating dependencies...
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo ✓ Dependencies installed successfully
echo.

REM Create necessary directories
if not exist "data" mkdir data
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "data\exports" mkdir data\exports
if not exist "logs" mkdir logs

echo ✓ Directory structure created
echo.

REM Copy environment file if it doesn't exist
if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo ✓ Environment file created from template
    )
)

echo.
echo ===============================================
echo    Setup Complete! 
echo ===============================================
echo.
echo To start the dashboard, run:
echo    startup.bat
echo.
echo Or manually run:
echo    python run_app.py
echo.
echo Dashboard will be available at:
echo    http://localhost:8050
echo.
echo ===============================================
pause
