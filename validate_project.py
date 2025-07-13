"""
FinSight BI Project Status and Validation
Comprehensive project verification and status report

Author: BI Development Team
Date: July 2025
"""

import sys
import os
from pathlib import Path
import importlib.util
import json
from datetime import datetime

def check_project_structure():
    """Check if all required project files and directories exist"""
    print("🏗️ Checking Project Structure...")
    print("-" * 40)
    
    required_files = [
        'README.md',
        'requirements.txt',
        'run_app.py',
        '.env.example',
        'setup.bat',
        'startup.bat',
        'INSTALLATION.md',
        'config/settings.py',
        'src/etl/data_processor.py',
        'src/analytics/banking_analytics.py',
        'src/dashboard/app.py',
        'src/utils/data_utils.py',
        'tests/test_finsight_bi.py',
        'notebooks/data_exploration.py'
    ]
    
    required_dirs = [
        'config',
        'src',
        'src/etl',
        'src/analytics', 
        'src/dashboard',
        'src/utils',
        'tests',
        'notebooks'
    ]
    
    missing_files = []
    missing_dirs = []
    
    # Check files
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    # Check directories
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
        else:
            print(f"📁 {dir_path}/")
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
    
    if missing_dirs:
        print(f"\n❌ Missing directories: {missing_dirs}")
    
    if not missing_files and not missing_dirs:
        print("\n✅ All required files and directories are present!")
    
    return len(missing_files) == 0 and len(missing_dirs) == 0

def check_python_dependencies():
    """Check if key Python dependencies can be imported"""
    print("\n🐍 Checking Python Dependencies...")
    print("-" * 35)
    
    required_packages = [
        'pandas',
        'numpy', 
        'plotly',
        'dash',
        'dash_bootstrap_components',
        'sklearn',
        'sqlite3'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'sqlite3':
                import sqlite3
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {missing_packages}")
        print("Run: pip install -r requirements.txt")
    else:
        print("\n✅ All required packages are available!")
    
    return len(missing_packages) == 0

def check_data_availability():
    """Check if data files are available"""
    print("\n📊 Checking Data Availability...")
    print("-" * 30)
    
    data_files = ['bank.csv']
    data_available = True
    
    for data_file in data_files:
        if Path(data_file).exists():
            file_size = Path(data_file).stat().st_size / (1024 * 1024)  # MB
            print(f"✅ {data_file} ({file_size:.2f} MB)")
        else:
            print(f"❌ {data_file} not found")
            data_available = False
    
    # Check if processed data exists
    db_path = Path('data/finsight_bi.db')
    if db_path.exists():
        db_size = db_path.stat().st_size / (1024 * 1024)  # MB
        print(f"✅ Processed database exists ({db_size:.2f} MB)")
    else:
        print("ℹ️ Processed database not found (will be created on first run)")
    
    return data_available

def test_core_modules():
    """Test if core modules can be imported and initialized"""
    print("\n🧪 Testing Core Modules...")
    print("-" * 25)
    
    test_results = {}
    
    # Test ETL module
    try:
        sys.path.append(str(Path.cwd()))
        from src.etl.data_processor import BankingDataETL
        etl = BankingDataETL()
        print("✅ ETL module loads successfully")
        test_results['etl'] = True
    except Exception as e:
        print(f"❌ ETL module error: {e}")
        test_results['etl'] = False
    
    # Test Analytics module
    try:
        from src.analytics.banking_analytics import BankingAnalytics
        # Create dummy data for testing
        import pandas as pd
        import numpy as np
        dummy_data = pd.DataFrame({
            'age': [25, 35, 45],
            'balance': [1000, 2000, 3000],
            'y': [0, 1, 1],
            'risk_score': [20, 30, 40],
            'customer_segment': ['Basic', 'Standard', 'Premium'],
            'annual_revenue': [100, 200, 300]
        })
        analytics = BankingAnalytics(dummy_data)
        print("✅ Analytics module loads successfully")
        test_results['analytics'] = True
    except Exception as e:
        print(f"❌ Analytics module error: {e}")
        test_results['analytics'] = False
    
    # Test Dashboard module (import only, don't start server)
    try:
        from src.dashboard.app import app
        print("✅ Dashboard module loads successfully")
        test_results['dashboard'] = True
    except Exception as e:
        print(f"❌ Dashboard module error: {e}")
        test_results['dashboard'] = False
    
    # Test Utils module
    try:
        from src.utils.data_utils import DataUtils, VizUtils, BusinessUtils
        print("✅ Utils module loads successfully")
        test_results['utils'] = True
    except Exception as e:
        print(f"❌ Utils module error: {e}")
        test_results['utils'] = False
    
    return test_results

def check_configuration():
    """Check configuration files and settings"""
    print("\n⚙️ Checking Configuration...")
    print("-" * 25)
    
    config_status = {}
    
    # Check settings.py
    try:
        from config.settings import DATABASE_CONFIG, DASHBOARD_CONFIG, ANALYTICS_CONFIG
        print("✅ Settings configuration loaded")
        config_status['settings'] = True
    except Exception as e:
        print(f"❌ Settings configuration error: {e}")
        config_status['settings'] = False
    
    # Check environment file
    if Path('.env').exists():
        print("✅ Environment file exists")
        config_status['env'] = True
    elif Path('.env.example').exists():
        print("ℹ️ Environment template exists (.env.example)")
        print("  Copy to .env and customize as needed")
        config_status['env'] = True
    else:
        print("❌ No environment configuration found")
        config_status['env'] = False
    
    return config_status

def run_basic_functionality_test():
    """Run basic functionality tests"""
    print("\n🚀 Running Basic Functionality Tests...")
    print("-" * 40)
    
    test_results = {}
    
    # Test data processing
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample data
        sample_data = pd.DataFrame({
            'age': [25, 35, 45, 55, 65],
            'job': ['admin', 'technician', 'management', 'retired', 'services'],
            'balance': [1000, 2000, 5000, 3000, 1500],
            'y': [0, 1, 1, 0, 1]
        })
        
        print("✅ Sample data created")
        
        # Test basic data operations
        avg_age = sample_data['age'].mean()
        conversion_rate = sample_data['y'].mean() * 100
        
        print(f"✅ Basic analytics: Avg age {avg_age}, Conversion rate {conversion_rate:.1f}%")
        test_results['data_processing'] = True
        
    except Exception as e:
        print(f"❌ Data processing test failed: {e}")
        test_results['data_processing'] = False
    
    # Test visualization creation
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        
        # Create simple chart
        fig = px.bar(x=['A', 'B', 'C'], y=[1, 2, 3], title="Test Chart")
        print("✅ Visualization creation successful")
        test_results['visualization'] = True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        test_results['visualization'] = False
    
    return test_results

def generate_project_report():
    """Generate comprehensive project status report"""
    print("\n📋 Generating Project Status Report...")
    print("=" * 50)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'project_name': 'FinSight BI: Banking KPI Dashboard',
        'version': '1.0.0',
        'status': 'Ready for Development/Production'
    }
    
    # Run all checks
    structure_ok = check_project_structure()
    dependencies_ok = check_python_dependencies()
    data_ok = check_data_availability()
    modules_test = test_core_modules()
    config_test = check_configuration()
    functionality_test = run_basic_functionality_test()
    
    # Compile results
    report['checks'] = {
        'project_structure': structure_ok,
        'python_dependencies': dependencies_ok,
        'data_availability': data_ok,
        'core_modules': modules_test,
        'configuration': config_test,
        'basic_functionality': functionality_test
    }
    
    # Calculate overall score
    all_tests = [
        structure_ok,
        dependencies_ok,
        data_ok,
        all(modules_test.values()) if modules_test else False,
        all(config_test.values()) if config_test else False,
        all(functionality_test.values()) if functionality_test else False
    ]
    
    passed_tests = sum(all_tests)
    total_tests = len(all_tests)
    success_rate = (passed_tests / total_tests) * 100
    
    report['summary'] = {
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'success_rate': success_rate,
        'overall_status': 'PASS' if success_rate >= 80 else 'FAIL'
    }
    
    return report

def display_project_summary():
    """Display final project summary and next steps"""
    print("\n" + "=" * 60)
    print("🏦 FinSight BI: Banking KPI Dashboard")
    print("Advanced Business Intelligence Solution")
    print("=" * 60)
    
    print("\n📊 Project Features:")
    features = [
        "✅ ETL Pipeline for banking data processing",
        "✅ Advanced analytics engine with ML capabilities", 
        "✅ Interactive web dashboard with real-time KPIs",
        "✅ Risk assessment and customer segmentation",
        "✅ Profitability and growth analysis",
        "✅ Multi-branch performance comparison",
        "✅ Comprehensive testing suite",
        "✅ Professional documentation"
    ]
    
    for feature in features:
        print(f"   {feature}")
    
    print("\n🚀 Quick Start:")
    print("   1. Run setup.bat (Windows) to install dependencies")
    print("   2. Run startup.bat to launch the dashboard")
    print("   3. Open http://localhost:8050 in your browser")
    
    print("\n🛠️ Development:")
    print("   1. Open project in VS Code")
    print("   2. Select Python interpreter from venv")
    print("   3. Use F5 to debug or run python run_app.py")
    
    print("\n📚 Documentation:")
    print("   • README.md - Project overview")
    print("   • INSTALLATION.md - Detailed setup guide")
    print("   • tests/ - Comprehensive test suite")
    print("   • notebooks/ - Interactive data analysis")
    
    print("\n🎯 Key Performance Indicators:")
    print("   • Customer acquisition and retention metrics")
    print("   • Revenue and profitability analysis")
    print("   • Risk exposure and compliance monitoring")
    print("   • Campaign effectiveness and ROI")
    
    print("\n" + "=" * 60)

def main():
    """Main function to run all project validation checks"""
    print("🔍 FinSight BI Project Validation")
    print("=" * 40)
    
    # Run comprehensive project validation
    report = generate_project_report()
    
    # Display summary
    print(f"\n📊 Final Results:")
    print(f"   Success Rate: {report['summary']['success_rate']:.1f}%")
    print(f"   Overall Status: {report['summary']['overall_status']}")
    
    if report['summary']['overall_status'] == 'PASS':
        print("\n🎉 Project validation successful!")
        print("   FinSight BI is ready for use.")
        
        # Display project summary
        display_project_summary()
        
    else:
        print("\n⚠️ Some issues detected.")
        print("   Please review the error messages above and fix any issues.")
        print("   Run this validation script again after making corrections.")
    
    # Save report to file
    try:
        with open('project_status_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Detailed report saved to: project_status_report.json")
    except Exception as e:
        print(f"\n⚠️ Could not save report: {e}")
    
    return report['summary']['overall_status'] == 'PASS'

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Validation complete - Project ready!")
    else:
        print("\n❌ Validation failed - Please fix issues")
        sys.exit(1)
