# FinSight BI - Manual Setup Instructions

## 🚀 Manual Setup (If Scripts Don't Work)

Since you're having issues with the batch files, here's a step-by-step manual setup:

### Step 1: Install Python (if not already installed)
1. Go to https://python.org/downloads/
2. Download Python 3.9+ for Windows
3. **IMPORTANT:** Check "Add Python to PATH" during installation
4. Restart PowerShell after installation

### Step 2: Verify Python Installation
```powershell
python --version
# Should show: Python 3.x.x

pip --version
# Should show pip version
```

### Step 3: Create Virtual Environment
```powershell
# Navigate to project directory
cd D:\Projects\FinSight_BI

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies
```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install core packages
pip install pandas numpy plotly dash dash-bootstrap-components

# Install additional packages
pip install scikit-learn matplotlib seaborn sqlalchemy

# Install remaining packages
pip install python-dateutil openpyxl
```

### Step 5: Create Directories
```powershell
# Create necessary directories
New-Item -ItemType Directory -Path "data" -Force
New-Item -ItemType Directory -Path "data\raw" -Force
New-Item -ItemType Directory -Path "data\processed" -Force
New-Item -ItemType Directory -Path "data\exports" -Force
New-Item -ItemType Directory -Path "logs" -Force
```

### Step 6: Launch Dashboard
```powershell
# Run the application
python run_app.py
```

## 🎯 Expected Output

You should see something like:
```
===============================================
🏦 FinSight BI: Banking KPI Dashboard
Starting Application...
===============================================

✓ Loading data...
✓ Processing analytics...
✓ Starting dashboard server...

🚀 Dashboard will be available at: http://127.0.0.1:8050/
Press Ctrl+C to stop the server
===============================================

 * Running on http://127.0.0.1:8050
 * Debug mode: on
```

## 🌐 Access the Dashboard

Open your web browser and go to:
**http://localhost:8050**

You should see the FinSight BI Banking KPI Dashboard with:
- Executive summary KPI cards
- Interactive charts and graphs
- Customer analytics
- Risk assessment tools
- Branch performance metrics

## 🔧 Troubleshooting

### If you get "python not recognized":
1. Install Python from python.org
2. Make sure to check "Add Python to PATH"
3. Restart PowerShell
4. Try again

### If you get permission errors:
```powershell
# Run this first
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### If packages fail to install:
```powershell
# Try with user flag
pip install --user pandas numpy plotly dash dash-bootstrap-components
```

### If virtual environment fails:
```powershell
# Skip virtual environment and install globally
pip install pandas numpy plotly dash dash-bootstrap-components scikit-learn
python run_app.py
```

## 📞 Alternative Launch Methods

### Method A: Direct Python execution
```powershell
# If everything else fails, try:
python -c "import sys; sys.path.append('.'); exec(open('run_app.py').read())"
```

### Method B: Use Jupyter Notebook
```powershell
pip install jupyter
jupyter notebook
# Open notebooks/data_exploration.py in Jupyter
```

### Method C: Use VS Code
1. Open VS Code
2. Open folder: D:\Projects\FinSight_BI
3. Select Python interpreter
4. Press F5 to run run_app.py

## ✅ Success Indicators

When everything is working correctly:
1. No Python errors in PowerShell
2. Dashboard loads at http://localhost:8050
3. You see banking KPI metrics and charts
4. Interactive filters work properly
5. Data tables display customer information

## 🎉 What You'll See

The FinSight BI dashboard includes:
- **KPI Cards**: Total customers, revenue, conversion rates, risk metrics
- **Revenue Analysis**: Charts showing profitability by segment
- **Risk Distribution**: Visual risk assessment tools
- **Branch Performance**: Multi-branch comparison
- **Customer Demographics**: Age, job, education analysis
- **Interactive Filters**: Segment, branch, and risk level filtering
- **Data Export**: Download capabilities for reports

Once running, you'll have a fully functional banking analytics dashboard with machine learning insights and real-time KPI monitoring!
