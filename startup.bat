@echo off
echo ===============================================
echo    FinSight BI: Banking KPI Dashboard
echo    Starting Application...
echo ===============================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first to install dependencies.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if bank.csv exists
if not exist "bank.csv" (
    echo WARNING: bank.csv not found in project directory
    echo The dashboard will run with sample data
    echo.
)

echo Starting FinSight BI Dashboard...
echo.
echo Dashboard will be available at: http://localhost:8050
echo Press Ctrl+C to stop the server
echo.
echo ===============================================

REM Start the application
python run_app.py

REM If the application exits, show this message
echo.
echo Dashboard has stopped.
pause
