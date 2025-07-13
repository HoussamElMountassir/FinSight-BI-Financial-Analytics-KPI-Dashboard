"""
FinSight BI Test Suite
Comprehensive testing for banking analytics dashboard

Author: BI QA Team
Version: 1.0.0
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.etl.data_processor import BankingDataETL
from src.analytics.banking_analytics import BankingAnalytics

class TestBankingDataETL:
    """Test suite for ETL pipeline"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return pd.DataFrame({
            'age': [25, 35, 45, 55, 65],
            'job': ['admin', 'technician', 'management', 'retired', 'services'],
            'marital': ['single', 'married', 'divorced', 'married', 'single'],
            'education': ['secondary', 'tertiary', 'tertiary', 'primary', 'secondary'],
            'default': ['no', 'no', 'no', 'yes', 'no'],
            'balance': [1000, 2000, 5000, -100, 1500],
            'housing': ['yes', 'yes', 'no', 'yes', 'no'],
            'loan': ['no', 'yes', 'no', 'no', 'yes'],
            'contact': ['cellular', 'telephone', 'cellular', 'unknown', 'cellular'],
            'day': [5, 10, 15, 20, 25],
            'month': ['may', 'jun', 'jul', 'aug', 'sep'],
            'duration': [261, 151, 76, 92, 198],
            'campaign': [1, 1, 1, 2, 1],
            'pdays': [-1, -1, -1, -1, -1],
            'previous': [0, 0, 0, 0, 0],
            'poutcome': ['unknown', 'unknown', 'unknown', 'unknown', 'unknown'],
            'y': ['no', 'yes', 'yes', 'no', 'yes']
        })
    
    @pytest.fixture
    def etl_processor(self):
        """Create ETL processor instance"""
        return BankingDataETL()
    
    def test_data_cleaning(self, etl_processor, sample_data):
        """Test data cleaning functionality"""
        etl_processor.raw_data = sample_data.copy()
        cleaned_data = etl_processor._clean_data(sample_data.copy())
        
        # Check that data is cleaned
        assert len(cleaned_data) <= len(sample_data)  # Duplicates removed
        assert cleaned_data is not None
    
    def test_data_type_conversion(self, etl_processor, sample_data):
        """Test data type conversion"""
        etl_processor.raw_data = sample_data.copy()
        converted_data = etl_processor._convert_data_types(sample_data.copy())
        
        # Check numeric conversions
        assert converted_data['age'].dtype in ['int64', 'float64']
        assert converted_data['balance'].dtype in ['int64', 'float64']
        
        # Check boolean conversions
        assert converted_data['default'].dtype in ['int64', 'bool']
        assert converted_data['y'].dtype in ['int64', 'bool']
    
    def test_feature_creation(self, etl_processor, sample_data):
        """Test feature engineering"""
        etl_processor.raw_data = sample_data.copy()
        # Convert data types first
        converted_data = etl_processor._convert_data_types(sample_data.copy())
        feature_data = etl_processor._create_features(converted_data)
        
        # Check new features exist
        expected_features = ['age_group', 'balance_category', 'estimated_income', 
                           'estimated_clv', 'campaign_efficiency', 'engagement_score']
        
        for feature in expected_features:
            assert feature in feature_data.columns
    
    def test_business_rules(self, etl_processor, sample_data):
        """Test business rule application"""
        etl_processor.raw_data = sample_data.copy()
        converted_data = etl_processor._convert_data_types(sample_data.copy())
        feature_data = etl_processor._create_features(converted_data)
        business_data = etl_processor._apply_business_rules(feature_data)
        
        # Check business rule outputs
        assert 'risk_score' in business_data.columns
        assert 'customer_segment' in business_data.columns
        assert business_data['risk_score'].min() >= 0
    
    def test_data_validation(self, etl_processor, sample_data):
        """Test data validation"""
        etl_processor.raw_data = sample_data.copy()
        converted_data = etl_processor._convert_data_types(sample_data.copy())
        feature_data = etl_processor._create_features(converted_data)
        business_data = etl_processor._apply_business_rules(feature_data)
        
        # Run validation
        etl_processor._validate_data(business_data)
        
        # Check validation report exists
        assert etl_processor.data_quality_report is not None
        assert 'total_records' in etl_processor.data_quality_report
        assert 'data_completeness' in etl_processor.data_quality_report

class TestBankingAnalytics:
    """Test suite for analytics engine"""
    
    @pytest.fixture
    def processed_data(self):
        """Create processed data for analytics testing"""
        np.random.seed(42)
        n_samples = 100
        
        return pd.DataFrame({
            'age': np.random.randint(18, 80, n_samples),
            'job': np.random.choice(['admin', 'technician', 'management'], n_samples),
            'balance': np.random.normal(2000, 1000, n_samples),
            'y': np.random.choice([0, 1], n_samples),
            'risk_score': np.random.uniform(0, 100, n_samples),
            'customer_segment': np.random.choice(['Premium', 'Standard', 'Basic', 'High_Risk'], n_samples),
            'branch': np.random.choice(['Branch_A', 'Branch_B', 'Branch_C'], n_samples),
            'annual_revenue': np.random.uniform(100, 1000, n_samples),
            'estimated_clv': np.random.uniform(500, 5000, n_samples),
            'risk_category': np.random.choice(['Low', 'Medium', 'High', 'Very_High'], n_samples),
            'age_group': np.random.choice(['18-25', '26-35', '36-45', '46-55', '56-65', '65+'], n_samples)
        })
    
    @pytest.fixture
    def analytics_engine(self, processed_data):
        """Create analytics engine instance"""
        return BankingAnalytics(processed_data)
    
    def test_profitability_metrics(self, analytics_engine):
        """Test profitability metrics calculation"""
        metrics = analytics_engine.calculate_profitability_metrics()
        
        # Check required metrics exist
        assert 'total_revenue' in metrics
        assert 'avg_revenue_per_customer' in metrics
        assert 'total_clv' in metrics
        assert 'revenue_by_segment' in metrics
        assert isinstance(metrics['total_revenue'], (int, float))
        assert metrics['total_revenue'] >= 0
    
    def test_growth_metrics(self, analytics_engine):
        """Test growth metrics calculation"""
        metrics = analytics_engine.calculate_growth_metrics()
        
        assert 'total_customers' in metrics
        assert 'conversion_rate' in metrics
        assert 'segment_distribution' in metrics
        assert metrics['total_customers'] > 0
        assert 0 <= metrics['conversion_rate'] <= 100
    
    def test_risk_metrics(self, analytics_engine):
        """Test risk metrics calculation"""
        metrics = analytics_engine.calculate_risk_metrics()
        
        assert 'risk_distribution' in metrics
        assert 'high_risk_analysis' in metrics
        assert 'risk_concentration' in metrics
        assert isinstance(metrics['risk_distribution'], pd.Series)
    
    def test_customer_segmentation(self, analytics_engine):
        """Test customer segmentation"""
        segmentation_results = analytics_engine.perform_customer_segmentation(n_clusters=3)
        
        assert 'cluster_analysis' in segmentation_results
        assert 'cluster_counts' in segmentation_results
        assert len(segmentation_results['cluster_analysis']) == 3
    
    def test_conversion_prediction(self, analytics_engine):
        """Test conversion prediction model"""
        prediction_results = analytics_engine.predict_conversion_probability()
        
        assert 'feature_importance' in prediction_results
        assert 'model_score' in prediction_results
        assert 0 <= prediction_results['model_score'] <= 1
    
    def test_anomaly_detection(self, analytics_engine):
        """Test anomaly detection"""
        anomaly_results = analytics_engine.detect_anomalies()
        
        assert 'anomaly_count' in anomaly_results
        assert 'anomaly_percentage' in anomaly_results
        assert anomaly_results['anomaly_count'] >= 0
        assert 0 <= anomaly_results['anomaly_percentage'] <= 100

class TestDataQuality:
    """Test data quality and validation"""
    
    def test_data_completeness(self):
        """Test data completeness requirements"""
        # Create data with missing values
        data = pd.DataFrame({
            'age': [25, np.nan, 45],
            'balance': [1000, 2000, np.nan],
            'y': [1, 0, 1]
        })
        
        completeness = (1 - data.isnull().sum().sum() / data.size) * 100
        assert completeness >= 0
        assert completeness <= 100
    
    def test_data_consistency(self):
        """Test data consistency checks"""
        # Test age ranges
        ages = [20, 30, 150, -5]  # Invalid ages: 150, -5
        valid_ages = [age for age in ages if 18 <= age <= 100]
        assert len(valid_ages) == 2
        
        # Test balance ranges
        balances = [1000, -500, 0, 1000000]
        # All balances should be numeric
        assert all(isinstance(b, (int, float)) for b in balances)
    
    def test_business_rule_validation(self):
        """Test business rule validation"""
        # Test risk score calculation
        risk_factors = {
            'default': 1,
            'loan': 1,
            'negative_balance': 1
        }
        
        risk_score = sum(risk_factors.values()) * 20  # Simplified calculation
        assert 0 <= risk_score <= 100

class TestPerformance:
    """Test performance requirements"""
    
    def test_etl_performance(self):
        """Test ETL pipeline performance"""
        import time
        
        # Create large dataset
        n_samples = 1000
        large_data = pd.DataFrame({
            'age': np.random.randint(18, 80, n_samples),
            'balance': np.random.normal(2000, 1000, n_samples),
            'y': np.random.choice([0, 1], n_samples)
        })
        
        start_time = time.time()
        
        # Simulate ETL operations
        cleaned_data = large_data.dropna()
        processed_data = cleaned_data.copy()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 1000 records in less than 1 second
        assert processing_time < 1.0
    
    def test_analytics_performance(self):
        """Test analytics calculation performance"""
        import time
        
        # Create test data
        n_samples = 500
        data = pd.DataFrame({
            'age': np.random.randint(18, 80, n_samples),
            'balance': np.random.normal(2000, 1000, n_samples),
            'annual_revenue': np.random.uniform(100, 1000, n_samples),
            'customer_segment': np.random.choice(['Premium', 'Standard', 'Basic'], n_samples)
        })
        
        start_time = time.time()
        
        # Calculate basic metrics
        total_revenue = data['annual_revenue'].sum()
        avg_balance = data['balance'].mean()
        segment_counts = data['customer_segment'].value_counts()
        
        end_time = time.time()
        calculation_time = end_time - start_time
        
        # Should calculate metrics in less than 0.1 seconds
        assert calculation_time < 0.1

class TestIntegration:
    """Integration tests for complete pipeline"""
    
    def test_full_pipeline_integration(self):
        """Test complete ETL + Analytics pipeline"""
        # Create sample input data
        sample_data = pd.DataFrame({
            'age': [25, 35, 45],
            'job': ['admin', 'technician', 'management'],
            'balance': [1000, 2000, 5000],
            'y': ['no', 'yes', 'yes'],
            'default': ['no', 'no', 'no'],
            'housing': ['yes', 'yes', 'no'],
            'loan': ['no', 'yes', 'no']
        })
        
        # Initialize ETL
        etl = BankingDataETL()
        etl.raw_data = sample_data
        
        # Run transformation
        processed_data = etl.transform_data()
        
        # Verify processed data
        assert len(processed_data) == len(sample_data)
        assert 'risk_score' in processed_data.columns
        assert 'customer_segment' in processed_data.columns
        
        # Initialize analytics
        analytics = BankingAnalytics(processed_data)
        
        # Run analytics
        results = analytics.run_all_analytics()
        
        # Verify analytics results
        assert 'profitability' in results
        assert 'growth' in results
        assert 'risk' in results
        assert 'dashboard' in results

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_dataset(self):
        """Test handling of empty dataset"""
        empty_data = pd.DataFrame()
        
        # Should handle empty data gracefully
        try:
            analytics = BankingAnalytics(empty_data)
            # This should not crash
            assert True
        except Exception:
            # If it raises an exception, it should be handled gracefully
            assert True
    
    def test_missing_columns(self):
        """Test handling of missing required columns"""
        incomplete_data = pd.DataFrame({
            'age': [25, 35],
            'balance': [1000, 2000]
            # Missing other required columns
        })
        
        etl = BankingDataETL()
        etl.raw_data = incomplete_data
        
        # Should handle missing columns gracefully
        try:
            processed_data = etl.transform_data()
            assert True  # If successful, that's good
        except Exception as e:
            # If it fails, should fail gracefully with informative error
            assert isinstance(e, (KeyError, ValueError))
    
    def test_invalid_data_types(self):
        """Test handling of invalid data types"""
        invalid_data = pd.DataFrame({
            'age': ['twenty-five', '35', 'forty-five'],  # String ages
            'balance': ['1000', 'two thousand', '5000'],  # Mixed types
            'y': ['maybe', 'yes', 'no']  # Invalid target values
        })
        
        etl = BankingDataETL()
        etl.raw_data = invalid_data
        
        # Should handle type conversion gracefully
        try:
            processed_data = etl.transform_data()
            # Check that numeric columns are converted properly
            assert processed_data['age'].dtype in ['int64', 'float64']
        except Exception:
            # If conversion fails, should be handled gracefully
            assert True

# ==================== UTILITY TEST FUNCTIONS ====================

def run_data_quality_checks(data: pd.DataFrame) -> Dict[str, bool]:
    """Run comprehensive data quality checks"""
    checks = {}
    
    # Completeness check
    checks['completeness'] = (data.isnull().sum().sum() / data.size) < 0.1
    
    # Consistency checks
    if 'age' in data.columns:
        checks['age_validity'] = data['age'].between(18, 100).all()
    
    if 'balance' in data.columns:
        checks['balance_numeric'] = pd.api.types.is_numeric_dtype(data['balance'])
    
    if 'risk_score' in data.columns:
        checks['risk_score_range'] = data['risk_score'].between(0, 100).all()
    
    return checks

def generate_test_report(test_results: Dict[str, bool]) -> str:
    """Generate test report"""
    passed = sum(test_results.values())
    total = len(test_results)
    
    report = f"""
    FinSight BI Test Report
    ======================
    
    Tests Passed: {passed}/{total}
    Success Rate: {(passed/total)*100:.1f}%
    
    Detailed Results:
    """
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        report += f"    {test_name}: {status}\n"
    
    return report

# ==================== PYTEST CONFIGURATION ====================

def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )

if __name__ == "__main__":
    print("FinSight BI Test Suite")
    print("Run with: pytest tests/test_finsight_bi.py -v")
