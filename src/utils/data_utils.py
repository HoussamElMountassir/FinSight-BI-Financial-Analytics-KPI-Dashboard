"""
FinSight BI Utilities
Common utility functions for banking analytics

Author: BI Development Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# ==================== DATA UTILITIES ====================

class DataUtils:
    """Utility functions for data processing and manipulation"""
    
    @staticmethod
    def load_data_from_db(db_path: str, table_name: str = 'customers') -> pd.DataFrame:
        """Load data from SQLite database"""
        try:
            conn = sqlite3.connect(db_path)
            query = f"SELECT * FROM {table_name}"
            data = pd.read_sql_query(query, conn)
            conn.close()
            logger.info(f"Loaded {len(data)} records from {table_name}")
            return data
        except Exception as e:
            logger.error(f"Error loading data from database: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def save_data_to_csv(data: pd.DataFrame, file_path: str) -> bool:
        """Save DataFrame to CSV file"""
        try:
            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(file_path, index=False)
            logger.info(f"Data saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving data to CSV: {e}")
            return False
    
    @staticmethod
    def filter_data(data: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply multiple filters to DataFrame"""
        filtered_data = data.copy()
        
        for column, value in filters.items():
            if column in filtered_data.columns and value != "all":
                if isinstance(value, list):
                    filtered_data = filtered_data[filtered_data[column].isin(value)]
                else:
                    filtered_data = filtered_data[filtered_data[column] == value]
        
        logger.info(f"Filtered data from {len(data)} to {len(filtered_data)} records")
        return filtered_data
    
    @staticmethod
    def calculate_data_quality_score(data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive data quality score"""
        metrics = {}
        
        # Completeness
        total_cells = data.size
        missing_cells = data.isnull().sum().sum()
        metrics['completeness'] = ((total_cells - missing_cells) / total_cells) * 100
        
        # Uniqueness (for key columns if they exist)
        if 'customer_id' in data.columns:
            unique_customers = data['customer_id'].nunique()
            total_customers = len(data)
            metrics['uniqueness'] = (unique_customers / total_customers) * 100
        
        # Validity (basic checks)
        validity_checks = []
        
        if 'age' in data.columns:
            valid_ages = data['age'].between(18, 100, na=False).mean() * 100
            validity_checks.append(valid_ages)
        
        if 'balance' in data.columns:
            # Check for extreme values
            q1 = data['balance'].quantile(0.25)
            q3 = data['balance'].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            valid_balances = data['balance'].between(lower_bound, upper_bound, na=False).mean() * 100
            validity_checks.append(valid_balances)
        
        metrics['validity'] = np.mean(validity_checks) if validity_checks else 100
        
        # Overall score
        metrics['overall_score'] = np.mean([
            metrics['completeness'],
            metrics.get('uniqueness', 100),
            metrics['validity']
        ])
        
        return metrics

# ==================== VISUALIZATION UTILITIES ====================

class VizUtils:
    """Utility functions for creating visualizations"""
    
    @staticmethod
    def create_kpi_gauge(value: float, title: str, max_value: float = 100, 
                        color_threshold: Dict[str, float] = None) -> go.Figure:
        """Create KPI gauge chart"""
        if color_threshold is None:
            color_threshold = {'red': 0.3, 'yellow': 0.7, 'green': 1.0}
        
        # Determine color based on thresholds
        normalized_value = value / max_value
        if normalized_value <= color_threshold['red']:
            color = 'red'
        elif normalized_value <= color_threshold['yellow']:
            color = 'yellow'
        else:
            color = 'green'
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title},
            delta = {'reference': max_value * 0.8},
            gauge = {
                'axis': {'range': [None, max_value]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, max_value * 0.3], 'color': "lightgray"},
                    {'range': [max_value * 0.3, max_value * 0.7], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': max_value * 0.9
                }
            }
        ))
        
        fig.update_layout(height=300)
        return fig
    
    @staticmethod
    def create_comparison_chart(data: pd.DataFrame, x_col: str, y_col: str, 
                              color_col: str = None, chart_type: str = 'bar') -> go.Figure:
        """Create comparison chart with multiple types"""
        
        if chart_type == 'bar':
            fig = px.bar(data, x=x_col, y=y_col, color=color_col,
                        title=f"{y_col} by {x_col}")
        elif chart_type == 'line':
            fig = px.line(data, x=x_col, y=y_col, color=color_col,
                         title=f"{y_col} trend by {x_col}")
        elif chart_type == 'scatter':
            fig = px.scatter(data, x=x_col, y=y_col, color=color_col,
                           title=f"{y_col} vs {x_col}")
        else:
            fig = px.bar(data, x=x_col, y=y_col, color=color_col)
        
        fig.update_layout(height=400)
        return fig

# ==================== BUSINESS UTILITIES ====================

class BusinessUtils:
    """Business-specific utility functions"""
    
    @staticmethod
    def calculate_customer_segments(data: pd.DataFrame) -> pd.DataFrame:
        """Enhanced customer segmentation"""
        segmented_data = data.copy()
        
        # RFM-like segmentation using available data
        if all(col in data.columns for col in ['balance', 'y', 'campaign']):
            # Recency (inverse of campaign count - more campaigns = less recent success)
            segmented_data['recency_score'] = pd.qcut(
                segmented_data['campaign'], q=4, labels=[4, 3, 2, 1]
            ).astype(int)
            
            # Frequency (previous successful contacts)
            if 'previous' in data.columns:
                segmented_data['frequency_score'] = pd.qcut(
                    segmented_data['previous'].fillna(0), q=4, labels=[1, 2, 3, 4]
                ).astype(int)
            else:
                segmented_data['frequency_score'] = 1
            
            # Monetary (balance)
            segmented_data['monetary_score'] = pd.qcut(
                segmented_data['balance'], q=4, labels=[1, 2, 3, 4]
            ).astype(int)
            
            # Combined RFM score
            segmented_data['rfm_score'] = (
                segmented_data['recency_score'] * 100 +
                segmented_data['frequency_score'] * 10 +
                segmented_data['monetary_score']
            )
            
            # Segment labels based on RFM score
            def assign_segment(score):
                if score >= 444:
                    return 'Champions'
                elif score >= 343:
                    return 'Loyal_Customers'
                elif score >= 242:
                    return 'Potential_Loyalists'
                elif score >= 141:
                    return 'At_Risk'
                else:
                    return 'Lost_Customers'
            
            segmented_data['advanced_segment'] = segmented_data['rfm_score'].apply(assign_segment)
        
        return segmented_data
    
    @staticmethod
    def calculate_churn_probability(data: pd.DataFrame) -> pd.DataFrame:
        """Calculate churn probability based on customer behavior"""
        churn_data = data.copy()
        
        # Simplified churn scoring
        churn_score = 0
        
        # Age factor
        if 'age' in data.columns:
            churn_score += np.where(data['age'] > 65, 10, 0)
            churn_score += np.where(data['age'] < 25, 15, 0)
        
        # Balance factor
        if 'balance' in data.columns:
            churn_score += np.where(data['balance'] < 0, 25, 0)
            churn_score += np.where(data['balance'] < 100, 15, 0)
        
        # Campaign factor
        if 'campaign' in data.columns:
            churn_score += np.where(data['campaign'] > 5, 20, 0)
        
        # Conversion factor
        if 'y' in data.columns:
            churn_score += np.where(data['y'] == 0, 15, -10)
        
        churn_data['churn_score'] = churn_score
        churn_data['churn_probability'] = np.clip(churn_score / 100, 0, 1)
        
        return churn_data
    
    @staticmethod
    def calculate_roi_metrics(data: pd.DataFrame) -> Dict[str, float]:
        """Calculate various ROI metrics"""
        metrics = {}
        
        if all(col in data.columns for col in ['annual_revenue', 'campaign', 'y']):
            # Campaign ROI
            total_revenue = data['annual_revenue'].sum()
            total_campaigns = data['campaign'].sum()
            total_conversions = data['y'].sum()
            
            # Assume cost per campaign contact
            cost_per_contact = 10  # $10 per contact
            total_cost = total_campaigns * cost_per_contact
            
            metrics['campaign_roi'] = ((total_revenue - total_cost) / total_cost * 100) if total_cost > 0 else 0
            metrics['cost_per_acquisition'] = (total_cost / total_conversions) if total_conversions > 0 else 0
            metrics['revenue_per_conversion'] = (total_revenue / total_conversions) if total_conversions > 0 else 0
            
        return metrics

# ==================== REPORTING UTILITIES ====================

class ReportUtils:
    """Utility functions for generating reports"""
    
    @staticmethod
    def create_executive_summary(data: pd.DataFrame) -> Dict[str, Any]:
        """Create executive summary of key metrics"""
        summary = {}
        
        # Customer metrics
        summary['total_customers'] = len(data)
        
        if 'y' in data.columns:
            summary['conversion_rate'] = f"{data['y'].mean() * 100:.1f}%"
            summary['total_conversions'] = data['y'].sum()
        
        if 'annual_revenue' in data.columns:
            summary['total_revenue'] = f"${data['annual_revenue'].sum():,.0f}"
            summary['avg_revenue_per_customer'] = f"${data['annual_revenue'].mean():.0f}"
        
        if 'risk_score' in data.columns:
            summary['avg_risk_score'] = f"{data['risk_score'].mean():.1f}"
            summary['high_risk_customers'] = len(data[data['risk_score'] > 70])
        
        if 'balance' in data.columns:
            summary['total_deposits'] = f"${data['balance'].sum():,.0f}"
            summary['avg_balance'] = f"${data['balance'].mean():.0f}"
        
        return summary
    
    @staticmethod
    def generate_trend_analysis(data: pd.DataFrame, date_col: str = 'contact_date') -> Dict[str, Any]:
        """Generate trend analysis if date column exists"""
        if date_col not in data.columns:
            return {}
        
        trends = {}
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
            data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
        
        # Monthly trends
        data['month_year'] = data[date_col].dt.to_period('M')
        monthly_trends = data.groupby('month_year').agg({
            'y': ['sum', 'count', 'mean'] if 'y' in data.columns else ['count'],
            'annual_revenue': 'sum' if 'annual_revenue' in data.columns else 'count'
        })
        
        trends['monthly_data'] = monthly_trends
        
        # Growth rates
        if 'y' in data.columns:
            conversion_by_month = data.groupby('month_year')['y'].mean()
            if len(conversion_by_month) > 1:
                growth_rate = ((conversion_by_month.iloc[-1] - conversion_by_month.iloc[0]) / 
                             conversion_by_month.iloc[0] * 100)
                trends['conversion_growth_rate'] = f"{growth_rate:.1f}%"
        
        return trends
    
    @staticmethod
    def export_report_to_excel(data_dict: Dict[str, pd.DataFrame], file_path: str) -> bool:
        """Export multiple DataFrames to Excel with different sheets"""
        try:
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                for sheet_name, data in data_dict.items():
                    # Truncate sheet name to Excel's 31 character limit
                    sheet_name = sheet_name[:31]
                    data.to_excel(writer, sheet_name=sheet_name, index=False)
            
            logger.info(f"Report exported to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")
            return False

# ==================== PERFORMANCE UTILITIES ====================

class PerformanceUtils:
    """Utility functions for performance optimization"""
    
    @staticmethod
    def optimize_dataframe(data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        optimized_data = data.copy()
        
        # Optimize numeric columns
        for col in optimized_data.select_dtypes(include=['int64']).columns:
            col_min = optimized_data[col].min()
            col_max = optimized_data[col].max()
            
            if col_min >= 0:
                if col_max <= 255:
                    optimized_data[col] = optimized_data[col].astype('uint8')
                elif col_max <= 65535:
                    optimized_data[col] = optimized_data[col].astype('uint16')
                elif col_max <= 4294967295:
                    optimized_data[col] = optimized_data[col].astype('uint32')
            else:
                if col_min >= -128 and col_max <= 127:
                    optimized_data[col] = optimized_data[col].astype('int8')
                elif col_min >= -32768 and col_max <= 32767:
                    optimized_data[col] = optimized_data[col].astype('int16')
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    optimized_data[col] = optimized_data[col].astype('int32')
        
        # Convert object columns to category if they have low cardinality
        for col in optimized_data.select_dtypes(include=['object']).columns:
            if optimized_data[col].nunique() / len(optimized_data) < 0.5:
                optimized_data[col] = optimized_data[col].astype('category')
        
        return optimized_data
    
    @staticmethod
    def profile_performance(func, *args, **kwargs):
        """Profile function performance"""
        import time
        import tracemalloc
        
        tracemalloc.start()
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        performance_stats = {
            'execution_time': end_time - start_time,
            'memory_current': current / 1024 / 1024,  # MB
            'memory_peak': peak / 1024 / 1024  # MB
        }
        
        return result, performance_stats

# ==================== VALIDATION UTILITIES ====================

class ValidationUtils:
    """Utility functions for data validation"""
    
    @staticmethod
    def validate_banking_data(data: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate banking data for common issues"""
        issues = {
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        # Required columns check
        required_columns = ['age', 'balance', 'y']
        missing_required = [col for col in required_columns if col not in data.columns]
        if missing_required:
            issues['errors'].append(f"Missing required columns: {missing_required}")
        
        # Data type validation
        if 'age' in data.columns:
            invalid_ages = data[(data['age'] < 18) | (data['age'] > 100)].shape[0]
            if invalid_ages > 0:
                issues['warnings'].append(f"Found {invalid_ages} records with invalid ages")
        
        if 'balance' in data.columns:
            negative_balances = data[data['balance'] < -10000].shape[0]
            if negative_balances > 0:
                issues['warnings'].append(f"Found {negative_balances} records with very negative balances")
        
        # Missing data check
        missing_percentage = (data.isnull().sum() / len(data) * 100)
        high_missing = missing_percentage[missing_percentage > 20]
        if len(high_missing) > 0:
            issues['warnings'].append(f"Columns with >20% missing data: {list(high_missing.index)}")
        
        # Data quality info
        issues['info'].append(f"Total records: {len(data):,}")
        issues['info'].append(f"Total columns: {len(data.columns)}")
        issues['info'].append(f"Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return issues

# ==================== CONFIGURATION UTILITIES ====================

class ConfigUtils:
    """Utility functions for configuration management"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str) -> bool:
        """Save configuration to JSON file"""
        try:
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'database': {
                'type': 'sqlite',
                'path': 'data/finsight_bi.db'
            },
            'dashboard': {
                'host': '0.0.0.0',
                'port': 8050,
                'debug': True
            },
            'analytics': {
                'risk_threshold': 70,
                'clv_discount_rate': 0.1,
                'campaign_cost_per_contact': 10
            },
            'visualization': {
                'default_height': 400,
                'color_scheme': 'plotly',
                'show_annotations': True
            }
        }

# ==================== MAIN UTILITY FUNCTIONS ====================

def setup_logging(log_level: str = 'INFO', log_file: str = None) -> None:
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=getattr(logging, log_level.upper()), format=log_format)

def ensure_directory_structure(base_path: str) -> None:
    """Ensure all required directories exist"""
    directories = [
        'data/raw', 'data/processed', 'data/exports',
        'logs', 'models', 'reports', 'config'
    ]
    
    for directory in directories:
        (Path(base_path) / directory).mkdir(parents=True, exist_ok=True)

def get_system_info() -> Dict[str, Any]:
    """Get system information for debugging"""
    import platform
    import psutil
    
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / (1024**3),
        'disk_free_gb': psutil.disk_usage('.').free / (1024**3)
    }

if __name__ == "__main__":
    print("FinSight BI Utilities Module")
    print("This module provides utility functions for the banking analytics dashboard")
    
    # Example usage
    print("\nSystem Information:")
    sys_info = get_system_info()
    for key, value in sys_info.items():
        print(f"  {key}: {value}")
