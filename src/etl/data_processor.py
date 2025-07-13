"""
FinSight BI ETL Pipeline
Advanced Extract, Transform, Load pipeline for banking data

Author: BI Engineering Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BankingDataETL:
    """
    Comprehensive ETL pipeline for banking data processing
    Handles extraction, transformation, and loading of customer data
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.raw_data = None
        self.processed_data = None
        self.data_quality_report = {}
        self.encoders = {}
        self.scalers = {}
        
    def _default_config(self) -> Dict:
        """Default configuration for ETL pipeline"""
        return {
            'input_file': 'bank.csv',
            'output_db': 'data/finsight_bi.db',
            'delimiter': ';',
            'encoding': 'utf-8',
            'risk_weights': {
                'default_yes': 50,
                'loan_yes': 20,
                'housing_no': 10,
                'age_extreme': 15,
                'balance_negative': 25
            },
            'clv_params': {
                'annual_rate': 0.02,
                'retention_rate': 0.85,
                'years': 5
            }
        }
    
    # ==================== EXTRACT ====================
    
    def extract_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Extract data from CSV file with robust error handling
        """
        try:
            file_path = file_path or self.config['input_file']
            logger.info(f"Extracting data from: {file_path}")
            
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    self.raw_data = pd.read_csv(
                        file_path,
                        sep=self.config['delimiter'],
                        encoding=encoding,
                        low_memory=False
                    )
                    logger.info(f"Successfully loaded with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if self.raw_data is None:
                raise ValueError("Failed to load data with any encoding")
            
            # Clean column names
            self.raw_data.columns = self.raw_data.columns.str.strip().str.replace('"', '')
            
            # Basic data info
            logger.info(f"Extracted {len(self.raw_data)} records with {len(self.raw_data.columns)} columns")
            logger.info(f"Columns: {list(self.raw_data.columns)}")
            
            return self.raw_data
            
        except Exception as e:
            logger.error(f"Error in data extraction: {e}")
            raise
    
    # ==================== TRANSFORM ====================
    
    def transform_data(self) -> pd.DataFrame:
        """
        Comprehensive data transformation pipeline
        """
        if self.raw_data is None:
            raise ValueError("No raw data available. Run extract_data() first.")
        
        logger.info("Starting data transformation pipeline")
        df = self.raw_data.copy()
        
        # 1. Data cleaning
        df = self._clean_data(df)
        
        # 2. Data type conversion
        df = self._convert_data_types(df)
        
        # 3. Feature engineering
        df = self._create_features(df)
        
        # 4. Business logic
        df = self._apply_business_rules(df)
        
        # 5. Data validation
        self._validate_data(df)
        
        self.processed_data = df
        logger.info(f"Transformation complete. Final dataset: {len(df)} records")
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize raw data"""
        logger.info("Cleaning data...")
        
        # Remove quotes from string columns
        string_columns = df.select_dtypes(include=['object']).columns
        for col in string_columns:
            df[col] = df[col].astype(str).str.strip().str.replace('"', '')
        
        # Handle missing and unknown values
        df = df.replace(['unknown', 'Unknown', 'UNKNOWN', ''], np.nan)
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_count - len(df)
        
        if duplicates_removed > 0:
            logger.warning(f"Removed {duplicates_removed} duplicate records")
        
        return df
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert data types for proper analysis"""
        logger.info("Converting data types...")
        
        # Numeric columns
        numeric_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Boolean mappings
        boolean_mappings = {
            'default': {'yes': 1, 'no': 0},
            'housing': {'yes': 1, 'no': 0},
            'loan': {'yes': 1, 'no': 0},
            'y': {'yes': 1, 'no': 0}
        }
        
        for col, mapping in boolean_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0).astype(int)
        
        return df
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features for analysis"""
        logger.info("Creating derived features...")
        
        # Age groups
        df['age_group'] = pd.cut(
            df['age'], 
            bins=[0, 25, 35, 45, 55, 65, 100], 
            labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'],
            include_lowest=True
        )
        
        # Balance categories
        df['balance_category'] = pd.cut(
            df['balance'],
            bins=[-np.inf, 0, 500, 2000, 10000, np.inf],
            labels=['Negative', 'Very_Low', 'Low', 'Medium', 'High']
        )
        
        # Income estimation based on job
        job_income_mapping = {
            'management': 75000, 'admin.': 45000, 'technician': 55000,
            'services': 35000, 'retired': 25000, 'blue-collar': 40000,
            'unemployed': 15000, 'entrepreneur': 65000, 'housemaid': 25000,
            'self-employed': 50000, 'student': 10000
        }
        df['estimated_income'] = df['job'].map(job_income_mapping).fillna(40000)
        
        # Customer lifetime value estimation
        annual_rate = self.config['clv_params']['annual_rate']
        retention_rate = self.config['clv_params']['retention_rate']
        years = self.config['clv_params']['years']
        
        df['estimated_clv'] = np.where(
            df['balance'] > 0,
            (df['balance'] * annual_rate + df['estimated_income'] * 0.001) * 
            ((1 - retention_rate**years) / (1 - retention_rate)),
            df['estimated_income'] * 0.0005 * years
        )
        
        # Campaign efficiency metrics
        df['campaign_efficiency'] = np.where(
            df['campaign'] > 0,
            df['duration'] / df['campaign'],
            df['duration']
        )
        
        # Customer engagement score
        df['engagement_score'] = (
            (df['previous'].fillna(0) * 10) +
            (df['duration'] / 100) +
            np.where(df['poutcome'] == 'success', 20, 0)
        )
        
        # Time-based features
        month_mapping = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        df['month_num'] = df['month'].map(month_mapping)
        df['quarter'] = ((df['month_num'] - 1) // 3 + 1).astype('Int64')
        
        # Season mapping
        season_mapping = {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 
                         5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer',
                         9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'}
        df['season'] = df['month_num'].map(season_mapping)
        
        # Simulate realistic contact dates and branches
        np.random.seed(42)
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 12, 31)
        df['contact_date'] = pd.to_datetime(
            np.random.choice(pd.date_range(start_date, end_date), size=len(df))
        )
        
        # Branch assignment for multi-branch analysis
        branches = ['Downtown_Branch', 'Suburban_Branch', 'Business_District', 'Mall_Branch', 'Online_Branch']
        branch_weights = [0.25, 0.30, 0.20, 0.15, 0.10]
        df['branch'] = np.random.choice(branches, size=len(df), p=branch_weights)
        
        return df
    
    def _apply_business_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply business rules and create business metrics"""
        logger.info("Applying business rules...")
        
        # Risk scoring model
        weights = self.config['risk_weights']
        df['risk_score'] = 0
        
        # Risk factors
        df['risk_score'] += df['default'] * weights['default_yes']
        df['risk_score'] += df['loan'] * weights['loan_yes']
        df['risk_score'] += (1 - df['housing']) * weights['housing_no']
        df['risk_score'] += ((df['age'] < 25) | (df['age'] > 70)).astype(int) * weights['age_extreme']
        df['risk_score'] += (df['balance'] < 0).astype(int) * weights['balance_negative']
        
        # Risk categories
        df['risk_category'] = pd.cut(
            df['risk_score'],
            bins=[-1, 20, 40, 60, 100],
            labels=['Low', 'Medium', 'High', 'Very_High']
        )
        
        # Customer segmentation
        def customer_segment(row):
            if row['estimated_clv'] > 2000 and row['risk_score'] < 30:
                return 'Premium'
            elif row['estimated_clv'] > 1000 and row['risk_score'] < 50:
                return 'Standard'
            elif row['risk_score'] > 60:
                return 'High_Risk'
            else:
                return 'Basic'
        
        df['customer_segment'] = df.apply(customer_segment, axis=1)
        
        # Product recommendation flags
        df['recommend_savings'] = (
            (df['balance'] > 5000) & 
            (df['risk_score'] < 40) & 
            (df['age'] > 30)
        ).astype(int)
        
        df['recommend_loan'] = (
            (df['loan'] == 0) & 
            (df['balance'] > 1000) & 
            (df['risk_score'] < 30) &
            (df['estimated_income'] > 40000)
        ).astype(int)
        
        df['recommend_investment'] = (
            (df['balance'] > 10000) & 
            (df['risk_score'] < 20) & 
            (df['age'].between(30, 60))
        ).astype(int)
        
        # Campaign success prediction features
        df['likely_to_convert'] = (
            (df['engagement_score'] > 20) &
            (df['risk_score'] < 40) &
            (df['previous'] > 0)
        ).astype(int)
        
        # Profitability metrics
        df['monthly_revenue'] = df['balance'] * 0.002  # 0.2% monthly fee
        df['annual_revenue'] = df['monthly_revenue'] * 12
        
        # Churn risk (simplified)
        df['churn_risk'] = np.where(
            (df['balance'] < 100) & (df['campaign'] > 5) & (df['y'] == 0),
            1, 0
        )
        
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validate processed data quality"""
        logger.info("Validating data quality...")
        
        validation_results = {}
        
        # Required columns check
        required_columns = ['age', 'job', 'balance', 'y', 'risk_score', 'customer_segment']
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_columns'] = missing_columns
        
        # Data quality metrics
        total_records = len(df)
        validation_results['total_records'] = total_records
        validation_results['duplicate_records'] = df.duplicated().sum()
        validation_results['missing_values'] = df.isnull().sum().to_dict()
        validation_results['negative_balances'] = (df['balance'] < 0).sum()
        validation_results['invalid_ages'] = ((df['age'] < 18) | (df['age'] > 100)).sum()
        
        # Business rule validation
        validation_results['risk_score_stats'] = {
            'min': float(df['risk_score'].min()),
            'max': float(df['risk_score'].max()),
            'mean': float(df['risk_score'].mean()),
            'std': float(df['risk_score'].std())
        }
        
        validation_results['customer_segments'] = df['customer_segment'].value_counts().to_dict()
        validation_results['data_completeness'] = (1 - df.isnull().sum().sum() / df.size) * 100
        
        # Store validation results
        self.data_quality_report = validation_results
        
        # Log critical issues
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
        
        if validation_results['invalid_ages'] > 0:
            logger.warning(f"Found {validation_results['invalid_ages']} records with invalid ages")
            
        if validation_results['data_completeness'] < 95:
            logger.warning(f"Data completeness: {validation_results['data_completeness']:.2f}%")
        
        logger.info("Data validation completed")
    
    # ==================== LOAD ====================
    
    def load_to_database(self, df: Optional[pd.DataFrame] = None, db_path: Optional[str] = None) -> None:
        """Load processed data to SQLite database"""
        if df is None:
            df = self.processed_data
        
        if df is None:
            raise ValueError("No data to load. Run transform_data() first.")
        
        db_path = db_path or self.config['output_db']
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Loading data to database: {db_path}")
        
        try:
            conn = sqlite3.connect(db_path)
            
            # Load main customer data
            df.to_sql('customers', conn, if_exists='replace', index=False)
            
            # Create summary tables
            self._create_summary_tables(conn, df)
            
            # Create indexes
            self._create_indexes(conn)
            
            # Create views
            self._create_views(conn)
            
            conn.close()
            logger.info(f"Successfully loaded {len(df)} records to database")
            
        except Exception as e:
            logger.error(f"Error loading data to database: {e}")
            raise
    
    def _create_summary_tables(self, conn: sqlite3.Connection, df: pd.DataFrame) -> None:
        """Create summary tables for dashboard performance"""
        
        # Branch summary
        branch_summary = df.groupby('branch').agg({
            'age': 'count',
            'balance': ['mean', 'sum'],
            'risk_score': 'mean',
            'estimated_clv': 'sum',
            'y': 'mean',
            'annual_revenue': 'sum'
        }).round(2)
        
        branch_summary.columns = ['customer_count', 'avg_balance', 'total_balance', 
                                'avg_risk_score', 'total_clv', 'conversion_rate', 'total_revenue']
        branch_summary.reset_index().to_sql('branch_summary', conn, if_exists='replace', index=False)
        
        # Customer segment summary
        segment_summary = df.groupby('customer_segment').agg({
            'age': 'count',
            'balance': 'mean',
            'risk_score': 'mean',
            'estimated_clv': ['mean', 'sum'],
            'y': 'mean',
            'annual_revenue': 'sum'
        }).round(2)
        
        segment_summary.columns = ['customer_count', 'avg_balance', 'avg_risk_score', 
                                 'avg_clv', 'total_clv', 'conversion_rate', 'total_revenue']
        segment_summary.reset_index().to_sql('segment_summary', conn, if_exists='replace', index=False)
        
        # Monthly trends
        df['year_month'] = df['contact_date'].dt.to_period('M').astype(str)
        monthly_summary = df.groupby('year_month').agg({
            'age': 'count',
            'balance': 'mean',
            'y': 'mean',
            'campaign': 'sum',
            'annual_revenue': 'sum'
        }).round(2)
        
        monthly_summary.columns = ['customer_count', 'avg_balance', 'conversion_rate', 
                                 'total_campaigns', 'total_revenue']
        monthly_summary.reset_index().to_sql('monthly_summary', conn, if_exists='replace', index=False)
        
        # KPI summary
        kpi_summary = {
            'total_customers': len(df),
            'avg_balance': df['balance'].mean(),
            'total_balance': df['balance'].sum(),
            'conversion_rate': df['y'].mean(),
            'avg_risk_score': df['risk_score'].mean(),
            'total_clv': df['estimated_clv'].sum(),
            'total_revenue': df['annual_revenue'].sum(),
            'high_risk_customers': (df['risk_category'] == 'Very_High').sum(),
            'premium_customers': (df['customer_segment'] == 'Premium').sum()
        }
        
        kpi_df = pd.DataFrame([kpi_summary])
        kpi_df.to_sql('kpi_summary', conn, if_exists='replace', index=False)
    
    def _create_indexes(self, conn: sqlite3.Connection) -> None:
        """Create database indexes for performance"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_customers_branch ON customers(branch)",
            "CREATE INDEX IF NOT EXISTS idx_customers_segment ON customers(customer_segment)",
            "CREATE INDEX IF NOT EXISTS idx_customers_age_group ON customers(age_group)",
            "CREATE INDEX IF NOT EXISTS idx_customers_risk_category ON customers(risk_category)",
            "CREATE INDEX IF NOT EXISTS idx_customers_contact_date ON customers(contact_date)",
            "CREATE INDEX IF NOT EXISTS idx_customers_job ON customers(job)",
            "CREATE INDEX IF NOT EXISTS idx_customers_education ON customers(education)"
        ]
        
        for index_sql in indexes:
            conn.execute(index_sql)
        
        conn.commit()
    
    def _create_views(self, conn: sqlite3.Connection) -> None:
        """Create database views for common queries"""
        views = [
            """
            CREATE VIEW IF NOT EXISTS high_value_customers AS
            SELECT * FROM customers 
            WHERE customer_segment IN ('Premium', 'Standard') 
            AND risk_category IN ('Low', 'Medium')
            """,
            """
            CREATE VIEW IF NOT EXISTS campaign_performance AS
            SELECT 
                branch,
                job,
                education,
                COUNT(*) as total_contacts,
                SUM(y) as conversions,
                ROUND(AVG(y) * 100, 2) as conversion_rate,
                ROUND(AVG(balance), 2) as avg_balance,
                ROUND(AVG(risk_score), 2) as avg_risk_score
            FROM customers
            GROUP BY branch, job, education
            """,
            """
            CREATE VIEW IF NOT EXISTS risk_analysis AS
            SELECT 
                risk_category,
                COUNT(*) as customer_count,
                ROUND(AVG(balance), 2) as avg_balance,
                ROUND(SUM(annual_revenue), 2) as total_revenue,
                ROUND(AVG(y) * 100, 2) as conversion_rate
            FROM customers
            GROUP BY risk_category
            """
        ]
        
        for view_sql in views:
            conn.execute(view_sql)
        
        conn.commit()
    
    # ==================== REPORTING ====================
    
    def generate_data_quality_report(self) -> Dict:
        """Generate comprehensive data quality report"""
        if not self.data_quality_report:
            logger.warning("No data quality report available. Run transform_data() first.")
            return {}
        
        return self.data_quality_report
    
    def export_to_csv(self, df: Optional[pd.DataFrame] = None, 
                     output_path: str = "data/exports/processed_banking_data.csv") -> None:
        """Export processed data to CSV"""
        if df is None:
            df = self.processed_data
        
        if df is None:
            raise ValueError("No data to export")
        
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Data exported to: {output_path}")
    
    def run_full_pipeline(self, input_file: Optional[str] = None) -> pd.DataFrame:
        """Run complete ETL pipeline"""
        logger.info("Starting full ETL pipeline...")
        
        try:
            # Extract
            self.extract_data(input_file)
            
            # Transform
            self.transform_data()
            
            # Load
            self.load_to_database()
            
            # Export processed data
            self.export_to_csv()
            
            logger.info("ETL pipeline completed successfully")
            return self.processed_data
            
        except Exception as e:
            logger.error(f"ETL pipeline failed: {e}")
            raise

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    # Example usage
    etl = BankingDataETL()
    
    try:
        # Run full pipeline
        processed_data = etl.run_full_pipeline('../bank.csv')
        
        # Generate and display data quality report
        quality_report = etl.generate_data_quality_report()
        print("\n=== DATA QUALITY REPORT ===")
        for key, value in quality_report.items():
            print(f"{key}: {value}")
        
        print(f"\nProcessed data shape: {processed_data.shape}")
        print(f"Columns: {list(processed_data.columns)}")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
