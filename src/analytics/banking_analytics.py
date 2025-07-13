"""
FinSight BI Analytics Engine
Advanced analytics and KPI calculations for banking dashboard

Author: BI Analytics Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class BankingAnalytics:
    """
    Comprehensive analytics engine for banking KPIs and insights
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.kpis = {}
        self.models = {}
        self.insights = {}
    
    # ==================== PROFITABILITY ANALYTICS ====================
    
    def calculate_profitability_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive profitability metrics"""
        logger.info("Calculating profitability metrics...")
        
        metrics = {}
        
        # Revenue metrics
        metrics['total_revenue'] = self.data['annual_revenue'].sum()
        metrics['avg_revenue_per_customer'] = self.data['annual_revenue'].mean()
        metrics['revenue_by_segment'] = self.data.groupby('customer_segment')['annual_revenue'].agg([
            'sum', 'mean', 'count'
        ]).round(2)
        
        # Customer Lifetime Value
        metrics['total_clv'] = self.data['estimated_clv'].sum()
        metrics['avg_clv'] = self.data['estimated_clv'].mean()
        metrics['clv_by_segment'] = self.data.groupby('customer_segment')['estimated_clv'].agg([
            'sum', 'mean', 'median', 'std'
        ]).round(2)
        
        # Campaign ROI Analysis
        campaign_metrics = self.data.groupby('campaign').agg({
            'y': ['sum', 'count', 'mean'],
            'duration': 'mean',
            'annual_revenue': 'sum'
        }).round(2)
        campaign_metrics.columns = ['conversions', 'total_contacts', 'conversion_rate', 
                                  'avg_duration', 'total_revenue']
        
        # Estimate campaign costs (simplified)
        campaign_metrics['estimated_cost'] = campaign_metrics['total_contacts'] * 10  # $10 per contact
        campaign_metrics['roi'] = ((campaign_metrics['total_revenue'] - campaign_metrics['estimated_cost']) / 
                                 campaign_metrics['estimated_cost'] * 100).round(2)
        
        metrics['campaign_performance'] = campaign_metrics
        
        # Branch profitability
        metrics['branch_profitability'] = self.data.groupby('branch').agg({
            'annual_revenue': ['sum', 'mean'],
            'estimated_clv': 'sum',
            'balance': 'sum',
            'y': 'mean'
        }).round(2)
        
        # Product profitability
        product_analysis = {}
        
        # Housing loan profitability
        housing_profit = self.data.groupby('housing').agg({
            'annual_revenue': 'mean',
            'risk_score': 'mean',
            'balance': 'mean'
        })
        product_analysis['housing_loans'] = housing_profit
        
        # Personal loan profitability
        loan_profit = self.data.groupby('loan').agg({
            'annual_revenue': 'mean',
            'risk_score': 'mean',
            'balance': 'mean'
        })
        product_analysis['personal_loans'] = loan_profit
        
        metrics['product_analysis'] = product_analysis
        
        # Profitability trends by demographics
        metrics['age_profitability'] = self.data.groupby('age_group')['annual_revenue'].agg([
            'sum', 'mean', 'count'
        ]).round(2)
        
        metrics['job_profitability'] = self.data.groupby('job')['annual_revenue'].agg([
            'sum', 'mean', 'count'
        ]).sort_values('sum', ascending=False).round(2)
        
        self.kpis['profitability'] = metrics
        return metrics
    
    # ==================== CLIENT GROWTH ANALYTICS ====================
    
    def calculate_growth_metrics(self) -> Dict[str, Any]:
        """Calculate client growth and acquisition metrics"""
        logger.info("Calculating growth metrics...")
        
        metrics = {}
        
        # Customer acquisition metrics
        metrics['total_customers'] = len(self.data)
        metrics['conversion_rate'] = (self.data['y'].mean() * 100).round(2)
        
        # Segment distribution
        metrics['segment_distribution'] = self.data['customer_segment'].value_counts()
        metrics['segment_percentages'] = (self.data['customer_segment'].value_counts(normalize=True) * 100).round(2)
        
        # Growth by demographics
        metrics['growth_by_age'] = self.data.groupby('age_group').agg({
            'y': ['sum', 'count', 'mean'],
            'balance': 'mean'
        }).round(2)
        
        metrics['growth_by_education'] = self.data.groupby('education').agg({
            'y': ['sum', 'count', 'mean'],
            'estimated_income': 'mean'
        }).round(2)
        
        metrics['growth_by_job'] = self.data.groupby('job').agg({
            'y': ['sum', 'count', 'mean'],
            'estimated_income': 'mean'
        }).sort_values(('y', 'mean'), ascending=False).round(2)
        
        # Branch performance
        metrics['branch_growth'] = self.data.groupby('branch').agg({
            'y': ['sum', 'count', 'mean'],
            'balance': 'mean',
            'estimated_clv': 'mean'
        }).round(2)
        
        # Campaign effectiveness
        metrics['campaign_effectiveness'] = self.data.groupby('campaign').agg({
            'y': 'mean',
            'duration': 'mean',
            'balance': 'mean'
        }).round(2)
        
        # Market penetration analysis
        total_contacts = len(self.data)
        successful_conversions = self.data['y'].sum()
        
        metrics['market_penetration'] = {
            'total_market_contacts': total_contacts,
            'successful_acquisitions': successful_conversions,
            'overall_penetration_rate': (successful_conversions / total_contacts * 100).round(2)
        }
        
        # Seasonal trends
        if 'contact_date' in self.data.columns:
            self.data['month'] = self.data['contact_date'].dt.month
            metrics['seasonal_trends'] = self.data.groupby('month').agg({
                'y': ['sum', 'count', 'mean']
            }).round(2)
        
        # Customer value growth
        metrics['customer_value_growth'] = self.data.groupby('customer_segment').agg({
            'balance': ['mean', 'median'],
            'estimated_clv': ['mean', 'median'],
            'annual_revenue': ['mean', 'median']
        }).round(2)
        
        self.kpis['growth'] = metrics
        return metrics
    
    # ==================== RISK ANALYTICS ====================
    
    def calculate_risk_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive risk exposure metrics"""
        logger.info("Calculating risk metrics...")
        
        metrics = {}
        
        # Overall risk distribution
        metrics['risk_distribution'] = self.data['risk_category'].value_counts()
        metrics['risk_percentages'] = (self.data['risk_category'].value_counts(normalize=True) * 100).round(2)
        
        # Risk by demographics
        metrics['risk_by_age'] = self.data.groupby('age_group')['risk_score'].agg([
            'mean', 'std', 'count'
        ]).round(2)
        
        metrics['risk_by_job'] = self.data.groupby('job')['risk_score'].agg([
            'mean', 'std', 'count'
        ]).sort_values('mean', ascending=False).round(2)
        
        metrics['risk_by_education'] = self.data.groupby('education')['risk_score'].agg([
            'mean', 'std', 'count'
        ]).round(2)
        
        # Default risk analysis
        metrics['default_analysis'] = self.data.groupby('default').agg({
            'risk_score': ['mean', 'count'],
            'balance': 'mean',
            'annual_revenue': 'mean'
        }).round(2)
        
        # Portfolio risk by branch
        metrics['branch_risk'] = self.data.groupby('branch').agg({
            'risk_score': ['mean', 'std'],
            'default': 'sum',
            'balance': 'mean'
        }).round(2)
        
        # Loan risk analysis
        metrics['loan_risk'] = self.data.groupby(['housing', 'loan']).agg({
            'risk_score': 'mean',
            'default': 'mean',
            'balance': 'mean'
        }).round(2)
        
        # High-risk customer identification
        high_risk_customers = self.data[self.data['risk_category'].isin(['High', 'Very_High'])]
        metrics['high_risk_analysis'] = {
            'count': len(high_risk_customers),
            'percentage': (len(high_risk_customers) / len(self.data) * 100).round(2),
            'avg_balance': high_risk_customers['balance'].mean().round(2),
            'total_exposure': high_risk_customers['balance'].sum().round(2)
        }
        
        # Risk concentration by segment
        metrics['risk_concentration'] = self.data.groupby('customer_segment').agg({
            'risk_score': ['mean', 'count'],
            'balance': 'sum'
        }).round(2)
        
        # Risk-adjusted returns
        self.data['risk_adjusted_revenue'] = self.data['annual_revenue'] / (1 + self.data['risk_score'] / 100)
        metrics['risk_adjusted_performance'] = self.data.groupby('customer_segment')['risk_adjusted_revenue'].agg([
            'mean', 'sum'
        ]).round(2)
        
        self.kpis['risk'] = metrics
        return metrics
    
    # ==================== ADVANCED ANALYTICS ====================
    
    def perform_customer_segmentation(self, n_clusters: int = 5) -> Dict[str, Any]:
        """Perform advanced customer segmentation using machine learning"""
        logger.info("Performing customer segmentation...")
        
        # Select features for clustering
        features = ['age', 'balance', 'duration', 'campaign', 'previous', 'risk_score', 'estimated_clv']
        
        # Prepare data
        X = self.data[features].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to data
        self.data['ml_segment'] = clusters
        
        # Analyze clusters
        cluster_analysis = self.data.groupby('ml_segment').agg({
            'age': 'mean',
            'balance': 'mean',
            'risk_score': 'mean',
            'estimated_clv': 'mean',
            'annual_revenue': 'mean',
            'y': 'mean'
        }).round(2)
        
        # Store model
        self.models['segmentation'] = {
            'model': kmeans,
            'scaler': scaler,
            'features': features
        }
        
        return {
            'cluster_analysis': cluster_analysis,
            'cluster_counts': self.data['ml_segment'].value_counts().sort_index()
        }
    
    def predict_conversion_probability(self) -> Dict[str, Any]:
        """Build model to predict conversion probability"""
        logger.info("Building conversion prediction model...")
        
        # Prepare features
        categorical_features = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
        numerical_features = ['age', 'balance', 'duration', 'campaign', 'previous', 'risk_score']
        
        # Encode categorical variables
        data_encoded = self.data.copy()
        for feature in categorical_features:
            if feature in data_encoded.columns:
                data_encoded[feature] = pd.Categorical(data_encoded[feature]).codes
        
        # Prepare features and target
        X = data_encoded[categorical_features + numerical_features].fillna(0)
        y = data_encoded['y']
        
        # Train model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Predict probabilities
        probabilities = rf_model.predict_proba(X)[:, 1]
        self.data['conversion_probability'] = probabilities
        
        # Store model
        self.models['conversion_prediction'] = {
            'model': rf_model,
            'features': X.columns.tolist(),
            'feature_importance': feature_importance
        }
        
        return {
            'feature_importance': feature_importance,
            'model_score': rf_model.score(X, y),
            'avg_probability': probabilities.mean()
        }
    
    def detect_anomalies(self) -> Dict[str, Any]:
        """Detect anomalous customer behavior"""
        logger.info("Detecting anomalies...")
        
        # Features for anomaly detection
        features = ['balance', 'duration', 'campaign', 'risk_score', 'estimated_clv']
        X = self.data[features].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Detect anomalies
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(X_scaled)
        
        # Add anomaly labels
        self.data['is_anomaly'] = (anomalies == -1).astype(int)
        
        # Analyze anomalies
        anomaly_analysis = self.data[self.data['is_anomaly'] == 1].describe()
        normal_analysis = self.data[self.data['is_anomaly'] == 0].describe()
        
        return {
            'anomaly_count': (anomalies == -1).sum(),
            'anomaly_percentage': ((anomalies == -1).sum() / len(anomalies) * 100).round(2),
            'anomaly_analysis': anomaly_analysis,
            'normal_analysis': normal_analysis
        }
    
    # ==================== TREND ANALYSIS ====================
    
    def analyze_trends(self) -> Dict[str, Any]:
        """Analyze trends and patterns in the data"""
        logger.info("Analyzing trends...")
        
        trends = {}
        
        # Campaign trends
        campaign_trends = self.data.groupby('campaign').agg({
            'y': 'mean',
            'duration': 'mean',
            'balance': 'mean'
        }).reset_index()
        
        trends['campaign_trends'] = campaign_trends
        
        # Age vs. Performance trends
        age_performance = self.data.groupby('age').agg({
            'y': 'mean',
            'balance': 'mean',
            'risk_score': 'mean'
        }).reset_index()
        
        trends['age_performance'] = age_performance
        
        # Balance vs. Conversion trends
        balance_bins = pd.qcut(self.data['balance'], q=10, duplicates='drop')
        balance_trends = self.data.groupby(balance_bins).agg({
            'y': 'mean',
            'risk_score': 'mean'
        }).reset_index()
        
        trends['balance_trends'] = balance_trends
        
        # Duration effectiveness
        duration_bins = pd.qcut(self.data['duration'], q=10, duplicates='drop')
        duration_trends = self.data.groupby(duration_bins).agg({
            'y': 'mean',
            'balance': 'mean'
        }).reset_index()
        
        trends['duration_trends'] = duration_trends
        
        return trends
    
    # ==================== COMPREHENSIVE KPI DASHBOARD ====================
    
    def generate_kpi_dashboard(self) -> Dict[str, Any]:
        """Generate comprehensive KPI dashboard data"""
        logger.info("Generating KPI dashboard...")
        
        dashboard = {}
        
        # Executive summary KPIs
        dashboard['executive_summary'] = {
            'total_customers': len(self.data),
            'total_revenue': self.data['annual_revenue'].sum().round(2),
            'avg_customer_value': self.data['estimated_clv'].mean().round(2),
            'conversion_rate': (self.data['y'].mean() * 100).round(2),
            'avg_risk_score': self.data['risk_score'].mean().round(2),
            'high_risk_percentage': ((self.data['risk_category'].isin(['High', 'Very_High'])).mean() * 100).round(2)
        }
        
        # Performance by branch
        dashboard['branch_performance'] = self.data.groupby('branch').agg({
            'y': lambda x: (x.sum(), len(x), x.mean() * 100),
            'annual_revenue': 'sum',
            'risk_score': 'mean',
            'balance': 'mean'
        }).round(2)
        
        # Customer segment analysis
        dashboard['segment_analysis'] = self.data.groupby('customer_segment').agg({
            'age': 'count',
            'annual_revenue': ['sum', 'mean'],
            'risk_score': 'mean',
            'y': lambda x: x.mean() * 100
        }).round(2)
        
        # Product performance
        dashboard['product_performance'] = {
            'housing_loans': self.data['housing'].value_counts(),
            'personal_loans': self.data['loan'].value_counts(),
            'savings_recommendations': self.data['recommend_savings'].sum(),
            'investment_recommendations': self.data['recommend_investment'].sum()
        }
        
        # Risk metrics
        dashboard['risk_metrics'] = {
            'risk_distribution': self.data['risk_category'].value_counts(),
            'default_rate': (self.data['default'].mean() * 100).round(2),
            'avg_portfolio_risk': self.data['risk_score'].mean().round(2)
        }
        
        # Growth metrics
        dashboard['growth_metrics'] = {
            'total_conversions': self.data['y'].sum(),
            'conversion_by_education': self.data.groupby('education')['y'].mean() * 100,
            'conversion_by_job': self.data.groupby('job')['y'].mean() * 100
        }
        
        return dashboard
    
    def run_all_analytics(self) -> Dict[str, Any]:
        """Run all analytics and return comprehensive results"""
        logger.info("Running all analytics...")
        
        results = {}
        
        try:
            # Core analytics
            results['profitability'] = self.calculate_profitability_metrics()
            results['growth'] = self.calculate_growth_metrics()
            results['risk'] = self.calculate_risk_metrics()
            
            # Advanced analytics
            results['segmentation'] = self.perform_customer_segmentation()
            results['conversion_prediction'] = self.predict_conversion_probability()
            results['anomaly_detection'] = self.detect_anomalies()
            results['trends'] = self.analyze_trends()
            
            # Dashboard KPIs
            results['dashboard'] = self.generate_kpi_dashboard()
            
            logger.info("All analytics completed successfully")
            
        except Exception as e:
            logger.error(f"Error in analytics: {e}")
            raise
        
        return results

# ==================== UTILITY FUNCTIONS ====================

def calculate_business_value(data: pd.DataFrame) -> Dict[str, float]:
    """Calculate overall business value metrics"""
    
    return {
        'total_portfolio_value': data['balance'].sum(),
        'total_revenue_potential': data['annual_revenue'].sum(),
        'total_clv': data['estimated_clv'].sum(),
        'risk_adjusted_value': (data['annual_revenue'] / (1 + data['risk_score'] / 100)).sum(),
        'conversion_value': data[data['y'] == 1]['estimated_clv'].sum()
    }

def generate_recommendations(analytics_results: Dict[str, Any]) -> List[str]:
    """Generate business recommendations based on analytics"""
    
    recommendations = []
    
    # Get key metrics
    exec_summary = analytics_results.get('dashboard', {}).get('executive_summary', {})
    conversion_rate = exec_summary.get('conversion_rate', 0)
    risk_percentage = exec_summary.get('high_risk_percentage', 0)
    
    # Conversion rate recommendations
    if conversion_rate < 10:
        recommendations.append("🎯 Low conversion rate detected. Consider improving campaign targeting and messaging.")
    elif conversion_rate > 20:
        recommendations.append("✅ Excellent conversion rate. Scale successful campaigns to similar customer segments.")
    
    # Risk recommendations
    if risk_percentage > 25:
        recommendations.append("⚠️ High risk exposure detected. Implement stricter risk assessment criteria.")
    
    # Segment-specific recommendations
    profitability = analytics_results.get('profitability', {})
    if 'revenue_by_segment' in profitability:
        top_segment = profitability['revenue_by_segment']['sum'].idxmax()
        recommendations.append(f"💰 Focus marketing efforts on '{top_segment}' segment - highest revenue generator.")
    
    # Branch recommendations
    dashboard = analytics_results.get('dashboard', {})
    if 'branch_performance' in dashboard:
        # Add branch-specific recommendations based on performance
        recommendations.append("🏢 Analyze top-performing branches to replicate success factors across network.")
    
    return recommendations

if __name__ == "__main__":
    # Example usage
    print("FinSight BI Analytics Engine")
    print("This module provides comprehensive analytics for banking data")
