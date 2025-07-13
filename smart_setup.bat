@echo off
setlocal enabledelayedexpansion

echo ===============================================
echo    FinSight BI: Banking KPI Dashboard
echo    Advanced Setup and Launch Script
echo ===============================================
echo.

REM Check for Python in various ways
set "PYTHON_CMD="

echo 🔍 Searching for Python installation...

REM Try standard python command
python --version >nul 2>&1
if %errorlevel% == 0 (
    set "PYTHON_CMD=python"
    echo ✓ Found Python: python
    goto :python_found
)

REM Try py launcher
py --version >nul 2>&1
if %errorlevel% == 0 (
    set "PYTHON_CMD=py"
    echo ✓ Found Python: py
    goto :python_found
)

REM Try common installation paths
set "PYTHON_PATHS=C:\Python39\python.exe C:\Python310\python.exe C:\Python311\python.exe C:\Python312\python.exe"
set "PYTHON_PATHS=%PYTHON_PATHS% C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python39\python.exe"
set "PYTHON_PATHS=%PYTHON_PATHS% C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310\python.exe"
set "PYTHON_PATHS=%PYTHON_PATHS% C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\python.exe"
set "PYTHON_PATHS=%PYTHON_PATHS% C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe"
set "PYTHON_PATHS=%PYTHON_PATHS% C:\Users\%USERNAME%\AppData\Local\Microsoft\WindowsApps\python.exe"

for %%P in (%PYTHON_PATHS%) do (
    if exist "%%P" (
        "%%P" --version >nul 2>&1
        if !errorlevel! == 0 (
            set "PYTHON_CMD=%%P"
            echo ✓ Found Python: %%P
            goto :python_found
        )
    )
)

REM Python not found
echo ❌ Python not found on your system!
echo.
echo 📥 Please install Python first:
echo    1. Go to https://python.org/downloads/
echo    2. Download Python 3.9 or higher
echo    3. During installation, check "Add Python to PATH"
echo    4. Restart this script after installation
echo.
echo 🔄 Alternative: Install from Microsoft Store
echo    1. Open Microsoft Store
echo    2. Search for "Python 3.11"
echo    3. Install the official Python package
echo    4. Restart PowerShell and try again
echo.
pause
exit /b 1

:python_found
echo.
echo ✅ Python found: %PYTHON_CMD%

REM Get Python version
for /f "tokens=2" %%i in ('"%PYTHON_CMD%" --version 2^>^&1') do set "PYTHON_VERSION=%%i"
echo    Version: %PYTHON_VERSION%

REM Check if version is adequate (basic check)
echo %PYTHON_VERSION% | findstr /r "^3\.[9-9]\|^3\.1[0-9]\|^[4-9]\." >nul
if %errorlevel% neq 0 (
    echo ⚠️  Warning: Python 3.9+ recommended. Current: %PYTHON_VERSION%
    echo    The application may still work, but consider upgrading.
    echo.
)

echo.
echo 🔧 Setting up virtual environment...

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    "%PYTHON_CMD%" -m venv venv
    if %errorlevel% neq 0 (
        echo ❌ Failed to create virtual environment
        echo    Make sure you have the venv module installed
        pause
        exit /b 1
    )
    echo ✓ Virtual environment created
) else (
    echo ✓ Virtual environment already exists
)

REM Activate virtual environment
echo.
echo 🚀 Activating virtual environment...
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo ✓ Virtual environment activated
) else (
    echo ❌ Virtual environment activation failed
    echo    venv\Scripts\activate.bat not found
    pause
    exit /b 1
)

REM Upgrade pip
echo.
echo 📦 Upgrading pip...
python -m pip install --upgrade pip --quiet
if %errorlevel% neq 0 (
    echo ⚠️  Warning: Failed to upgrade pip
) else (
    echo ✓ Pip upgraded successfully
)

REM Install requirements
echo.
echo 📦 Installing dependencies...
echo    This may take a few minutes...

REM Install core requirements first for faster feedback
echo Installing core packages...
pip install pandas numpy plotly dash dash-bootstrap-components --quiet
if %errorlevel% neq 0 (
    echo ❌ Failed to install core packages
    echo    Check your internet connection and try again
    pause
    exit /b 1
)

echo Installing remaining packages...
pip install scikit-learn scipy matplotlib seaborn --quiet
pip install sqlalchemy openpyxl python-dateutil --quiet

echo ✓ Core dependencies installed

REM Create necessary directories
echo.
echo 📁 Creating directory structure...
if not exist "data" mkdir data
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "data\exports" mkdir data\exports
if not exist "logs" mkdir logs
echo ✓ Directory structure created

REM Copy environment file
if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env" >nul 2>&1
        echo ✓ Environment file created
    )
)

REM Check if bank.csv exists
echo.
echo 📊 Checking data files...
if exist "bank.csv" (
    for %%A in (bank.csv) do (
        set "size=%%~zA"
        set /a "size_mb=!size! / 1048576"
        echo ✓ bank.csv found (!size_mb! MB)
    )
) else (
    echo ⚠️  bank.csv not found
    echo    The dashboard will run with sample data
)

echo.
echo ===============================================
echo    🎉 Setup Complete!
echo ===============================================
echo.
echo 🚀 Starting FinSight BI Dashboard...
echo.
echo    Dashboard will be available at:
echo    👉 http://localhost:8050
echo.
echo    Press Ctrl+C to stop the server
echo.
echo ===============================================

REM Start the application
python run_app.py

REM If we get here, the application has stopped
echo.
echo 📊 Dashboard has stopped.
echo.
echo 🔄 To restart the dashboard, run:
echo    startup.bat
echo.
pause
