# FinSight BI - Smart Setup and Launch Script (PowerShell)
# Handles Python detection and installation verification

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "   FinSight BI: Banking KPI Dashboard" -ForegroundColor Yellow
Write-Host "   Smart Setup and Launch Script" -ForegroundColor Yellow  
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# Function to test Python command
function Test-PythonCommand {
    param($Command)
    try {
        $version = & $Command --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            return $version
        }
    }
    catch {
        return $null
    }
    return $null
}

# Search for Python
Write-Host "🔍 Searching for Python installation..." -ForegroundColor Blue

$pythonCmd = $null
$pythonVersion = $null

# Try different Python commands
$pythonCommands = @("python", "py", "python3")

foreach ($cmd in $pythonCommands) {
    $version = Test-PythonCommand $cmd
    if ($version) {
        $pythonCmd = $cmd
        $pythonVersion = $version
        Write-Host "✓ Found Python: $cmd" -ForegroundColor Green
        Write-Host "  Version: $version" -ForegroundColor Gray
        break
    }
}

# Try common installation paths
if (-not $pythonCmd) {
    $pythonPaths = @(
        "$env:LOCALAPPDATA\Programs\Python\Python39\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe", 
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:LOCALAPPDATA\Microsoft\WindowsApps\python.exe",
        "C:\Python39\python.exe",
        "C:\Python310\python.exe",
        "C:\Python311\python.exe",
        "C:\Python312\python.exe"
    )
    
    foreach ($path in $pythonPaths) {
        if (Test-Path $path) {
            $version = Test-PythonCommand $path
            if ($version) {
                $pythonCmd = $path
                $pythonVersion = $version
                Write-Host "✓ Found Python: $path" -ForegroundColor Green
                Write-Host "  Version: $version" -ForegroundColor Gray
                break
            }
        }
    }
}

# Python not found
if (-not $pythonCmd) {
    Write-Host "❌ Python not found on your system!" -ForegroundColor Red
    Write-Host ""
    Write-Host "📥 Please install Python first:" -ForegroundColor Yellow
    Write-Host "   1. Go to https://python.org/downloads/" -ForegroundColor White
    Write-Host "   2. Download Python 3.9 or higher" -ForegroundColor White
    Write-Host "   3. During installation, check 'Add Python to PATH'" -ForegroundColor White
    Write-Host "   4. Restart PowerShell and run this script again" -ForegroundColor White
    Write-Host ""
    Write-Host "🔄 Alternative: Install from Microsoft Store" -ForegroundColor Yellow
    Write-Host "   1. Open Microsoft Store" -ForegroundColor White
    Write-Host "   2. Search for 'Python 3.11'" -ForegroundColor White
    Write-Host "   3. Install the official Python package" -ForegroundColor White
    Write-Host "   4. Restart PowerShell and try again" -ForegroundColor White
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "✅ Python detected successfully!" -ForegroundColor Green

# Check Python version
if ($pythonVersion -match "Python (\d+)\.(\d+)") {
    $major = [int]$matches[1]
    $minor = [int]$matches[2]
    
    if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 9)) {
        Write-Host "⚠️  Warning: Python 3.9+ recommended. Current: $pythonVersion" -ForegroundColor Yellow
        Write-Host "   The application may still work, but consider upgrading." -ForegroundColor Gray
    }
}

# Setup virtual environment
Write-Host ""
Write-Host "🔧 Setting up virtual environment..." -ForegroundColor Blue

if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Gray
    & $pythonCmd -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to create virtual environment" -ForegroundColor Red
        Write-Host "   Make sure you have the venv module installed" -ForegroundColor Gray
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "✓ Virtual environment already exists" -ForegroundColor Green
}

# Activate virtual environment
Write-Host ""
Write-Host "🚀 Activating virtual environment..." -ForegroundColor Blue

if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
    Write-Host "✓ Virtual environment activated" -ForegroundColor Green
} elseif (Test-Path "venv\Scripts\activate.bat") {
    cmd /c "venv\Scripts\activate.bat"
    Write-Host "✓ Virtual environment activated (via batch)" -ForegroundColor Green
} else {
    Write-Host "❌ Virtual environment activation failed" -ForegroundColor Red
    Write-Host "   Activation script not found" -ForegroundColor Gray
    Read-Host "Press Enter to exit"
    exit 1
}

# Install dependencies
Write-Host ""
Write-Host "📦 Installing dependencies..." -ForegroundColor Blue
Write-Host "   This may take a few minutes..." -ForegroundColor Gray

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Gray
python -m pip install --upgrade pip --quiet

# Install core packages
Write-Host "Installing core packages..." -ForegroundColor Gray
python -m pip install pandas numpy plotly dash dash-bootstrap-components scikit-learn --quiet

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install core packages" -ForegroundColor Red
    Write-Host "   Check your internet connection and try again" -ForegroundColor Gray
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "✓ Dependencies installed successfully" -ForegroundColor Green

# Create directory structure
Write-Host ""
Write-Host "📁 Creating directory structure..." -ForegroundColor Blue

$directories = @("data", "data\raw", "data\processed", "data\exports", "logs")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Write-Host "✓ Directory structure created" -ForegroundColor Green

# Copy environment file
if (-not (Test-Path ".env") -and (Test-Path ".env.example")) {
    Copy-Item ".env.example" ".env"
    Write-Host "✓ Environment file created" -ForegroundColor Green
}

# Check data files
Write-Host ""
Write-Host "📊 Checking data files..." -ForegroundColor Blue

if (Test-Path "bank.csv") {
    $size = (Get-Item "bank.csv").Length
    $sizeMB = [math]::Round($size / 1MB, 1)
    Write-Host "✓ bank.csv found ($sizeMB MB)" -ForegroundColor Green
} else {
    Write-Host "⚠️  bank.csv not found" -ForegroundColor Yellow
    Write-Host "   The dashboard will run with sample data" -ForegroundColor Gray
}

Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "   🎉 Setup Complete!" -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "🚀 Starting FinSight BI Dashboard..." -ForegroundColor Yellow
Write-Host ""
Write-Host "   Dashboard will be available at:" -ForegroundColor White
Write-Host "   👉 http://localhost:8050" -ForegroundColor Cyan
Write-Host ""
Write-Host "   Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan

# Start the application
try {
    python run_app.py
}
catch {
    Write-Host ""
    Write-Host "❌ Failed to start the application" -ForegroundColor Red
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "📊 Dashboard has stopped." -ForegroundColor Yellow
Write-Host ""
Write-Host "🔄 To restart the dashboard, run:" -ForegroundColor White
Write-Host "   .\smart_setup.ps1" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to exit"
