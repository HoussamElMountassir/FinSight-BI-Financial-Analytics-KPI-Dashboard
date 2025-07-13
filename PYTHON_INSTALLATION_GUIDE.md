# FinSight BI - Python Installation Guide

## 🐍 Python Not Found - Installation Required

The error you encountered indicates Python is not installed or not accessible from the command line.

## 📥 **Quick Installation Solutions**

### **Option 1: Official Python (Recommended)**
1. **Download:** Go to [python.org/downloads](https://www.python.org/downloads/)
2. **Install:** Download Python 3.9+ for Windows
3. **⚠️ CRITICAL:** Check "**Add Python to PATH**" during installation
4. **Verify:** Open new PowerShell and run `python --version`

### **Option 2: Microsoft Store (Easiest)**
1. Open **Microsoft Store**
2. Search for "**Python 3.11**" or "**Python 3.12**"
3. Install the official Python package
4. Restart PowerShell and try again

### **Option 3: Chocolatey Package Manager**
```powershell
# Install Chocolatey first (if not installed)
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install Python
choco install python
```

### **Option 4: Anaconda/Miniconda**
1. Download [Anaconda](https://www.anaconda.com/products/distribution) (full) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (minimal)
2. Install with default settings
3. Use **Anaconda Prompt** instead of PowerShell

## 🚀 **Quick Launch Methods (After Python Installation)**

### **Method 1: Use Smart Setup Script**
```powershell
# Navigate to project directory
cd D:\Projects\FinSight_BI

# Run smart setup (detects Python automatically)
.\smart_setup.ps1
```

### **Method 2: Use Batch Script**
```batch
# Double-click or run in Command Prompt
smart_setup.bat
```

### **Method 3: Manual Commands**
```powershell
# After Python is installed
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install pandas numpy plotly dash dash-bootstrap-components scikit-learn
python run_app.py
```

## 🔧 **Troubleshooting Common Issues**

### **Issue 1: "Python not recognized" (after installation)**
**Solution:** Restart PowerShell/Command Prompt after Python installation

### **Issue 2: PATH not updated**
**Solution:** Add Python to PATH manually:
1. Search "Environment Variables" in Windows
2. Edit System Environment Variables
3. Add Python installation path (e.g., `C:\Python39\` and `C:\Python39\Scripts\`)

### **Issue 3: Permission errors**
**Solution:** Run PowerShell as Administrator:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### **Issue 4: Corporate firewall/proxy**
**Solution:** Configure pip for proxy:
```powershell
pip install --proxy http://your-proxy:port package-name
```

## 🎯 **Alternative: Use Pre-configured Environment**

If you have issues with Python installation, you can use these alternatives:

### **Option A: Anaconda Navigator**
1. Install Anaconda
2. Open Anaconda Navigator
3. Launch Jupyter Notebook
4. Run the analysis notebook: `notebooks/data_exploration.py`

### **Option B: Google Colab (Cloud)**
1. Upload the project files to Google Drive
2. Open Google Colab
3. Mount Drive and run the analysis

### **Option C: Docker (Advanced)**
```dockerfile
# Use official Python image
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "run_app.py"]
```

## ✅ **Verification Steps**

After installing Python, verify your setup:

```powershell
# Check Python installation
python --version
pip --version

# Navigate to project
cd D:\Projects\FinSight_BI

# Test import capabilities
python -c "import pandas, numpy, plotly; print('✅ Core libraries available')"

# Launch dashboard
python run_app.py
```

## 🌐 **Expected Result**

Once Python is properly installed and the dashboard is running, you should see:

```
===============================================
🏦 FinSight BI: Banking KPI Dashboard
Starting Application...
===============================================

✓ Python detected
✓ Virtual environment activated  
✓ Dependencies installed
✓ Data loaded successfully

🚀 Dashboard starting at: http://localhost:8050
Press Ctrl+C to stop the server
===============================================

Running on http://0.0.0.0:8050/
Debug mode: on
```

Then open your browser to: **http://localhost:8050**

## 📞 **Need Help?**

If you continue to have issues:

1. **Check Python Installation:**
   ```powershell
   where python
   python --version
   ```

2. **Use Smart Setup Script:**
   ```powershell
   .\smart_setup.ps1
   ```

3. **Try Alternative Launcher:**
   ```powershell
   py run_app.py
   ```

4. **Manual Installation:**
   - Download Python from python.org
   - ✅ **CHECK "Add Python to PATH"**
   - Restart computer
   - Try again

---

**Once Python is installed, the FinSight BI dashboard will launch successfully and provide advanced banking analytics with interactive visualizations!** 🎉📊
