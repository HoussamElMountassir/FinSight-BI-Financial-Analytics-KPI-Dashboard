#!/usr/bin/env python3
"""
FinSight BI - Main Application Runner
Entry point for the banking analytics dashboard

Usage:
    python run_app.py [options]
    
Options:
    --mode development|production
    --port PORT_NUMBER
    --host HOST_ADDRESS
    --debug true|false
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.etl.data_processor import BankingDataETL
from src.dashboard.app import app
from src.utils.data_utils import setup_logging, ensure_directory_structure
from config.settings import DASHBOARD_CONFIG, LOGGING_CONFIG

def setup_environment():
    """Setup the application environment"""
    # Ensure directory structure exists
    ensure_directory_structure(project_root)
    
    # Setup logging
    log_file = project_root / "logs" / "finsight_bi.log"
    setup_logging(
        log_level=LOGGING_CONFIG.get('level', 'INFO'),
        log_file=str(log_file)
    )
    
    logger = logging.getLogger(__name__)
    logger.info("FinSight BI application starting...")
    
    return logger

def run_etl_pipeline():
    """Run the ETL pipeline to process data"""
    logger = logging.getLogger(__name__)
    
    try:
        # Check if bank.csv exists
        data_file = project_root / "bank.csv"
        if not data_file.exists():
            logger.warning(f"Data file not found: {data_file}")
            logger.info("Dashboard will run with sample data")
            return True
        
        # Initialize and run ETL
        etl = BankingDataETL({
            'input_file': str(data_file),
            'output_db': str(project_root / 'data' / 'finsight_bi.db')
        })
        
        logger.info("Running ETL pipeline...")
        processed_data = etl.run_full_pipeline()
        
        # Generate data quality report
        quality_report = etl.generate_data_quality_report()
        logger.info(f"Data quality score: {quality_report.get('overall_score', 'N/A'):.1f}%")
        
        logger.info(f"ETL pipeline completed successfully. Processed {len(processed_data)} records.")
        return True
        
    except Exception as e:
        logger.error(f"ETL pipeline failed: {e}")
        logger.info("Dashboard will attempt to run with existing data or sample data")
        return False

def run_dashboard(host='0.0.0.0', port=8050, debug=True):
    """Run the dashboard application"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Starting dashboard server on {host}:{port}")
        logger.info(f"Debug mode: {debug}")
        
        # Run the Dash application
        app.run(
            host=host,
            port=port,
            debug=debug
        )
        
    except Exception as e:
        logger.error(f"Failed to start dashboard: {e}")
        sys.exit(1)

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description='FinSight BI Banking Analytics Dashboard')
    parser.add_argument('--mode', choices=['development', 'production'], 
                       default='development', help='Application mode')
    parser.add_argument('--port', type=int, default=DASHBOARD_CONFIG.get('port', 8050),
                       help='Port number for the dashboard')
    parser.add_argument('--host', default=DASHBOARD_CONFIG.get('host', '0.0.0.0'),
                       help='Host address for the dashboard')
    parser.add_argument('--debug', type=bool, default=True,
                       help='Enable debug mode')
    parser.add_argument('--skip-etl', action='store_true',
                       help='Skip ETL pipeline and use existing data')
    
    args = parser.parse_args()
    
    # Setup environment
    logger = setup_environment()
    
    # Print application banner
    print("\n" + "="*60)
    print("🏦 FinSight BI: Banking KPI Dashboard")
    print("Advanced Business Intelligence for Banking Operations")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Debug: {args.debug}")
    print("="*60 + "\n")
    
    # Run ETL pipeline unless skipped
    if not args.skip_etl:
        logger.info("Starting ETL pipeline...")
        etl_success = run_etl_pipeline()
        if etl_success:
            logger.info("ETL pipeline completed successfully")
        else:
            logger.warning("ETL pipeline encountered issues")
    else:
        logger.info("ETL pipeline skipped as requested")
    
    # Start dashboard
    logger.info("Starting dashboard application...")
    
    # Adjust settings based on mode
    if args.mode == 'production':
        debug_mode = False
        logger.info("Production mode: Debug disabled")
    else:
        debug_mode = args.debug
        logger.info("Development mode: Debug enabled")
    
    print(f"\n🚀 Dashboard starting at: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server\n")
    
    # Run the dashboard
    run_dashboard(
        host=args.host,
        port=args.port,
        debug=debug_mode
    )

if __name__ == "__main__":
    main()
