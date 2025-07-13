"""
FinSight BI Jupyter Notebook for Data Analysis
Interactive data exploration and analytics

Run this notebook to explore the banking data interactively
"""

# Cell 1: Setup and Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("📊 FinSight BI: Interactive Data Analysis")
print("=" * 50)

# Cell 2: Load Data
# Load the banking dataset
try:
    # Try to load processed data first
    import sqlite3
    conn = sqlite3.connect('data/finsight_bi.db')
    data = pd.read_sql_query("SELECT * FROM customers", conn)
    conn.close()
    print(f"✓ Loaded processed data: {len(data)} records")
except:
    # Fall back to raw data
    try:
        data = pd.read_csv('bank.csv', delimiter=';')
        # Clean column names
        data.columns = data.columns.str.strip().str.replace('"', '')
        print(f"✓ Loaded raw data: {len(data)} records")
    except:
        print("❌ No data found. Please ensure bank.csv exists or run ETL pipeline.")
        data = pd.DataFrame()

# Display basic info
if not data.empty:
    print(f"\nDataset Shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")

# Cell 3: Data Overview
if not data.empty:
    print("📋 Data Overview")
    print("-" * 30)
    
    # Basic statistics
    print(f"Total Records: {len(data):,}")
    print(f"Total Columns: {len(data.columns)}")
    print(f"Memory Usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Missing values
    missing_data = data.isnull().sum()
    if missing_data.sum() > 0:
        print(f"\nMissing Values:")
        for col, missing in missing_data[missing_data > 0].items():
            print(f"  {col}: {missing} ({missing/len(data)*100:.1f}%)")
    else:
        print("\n✓ No missing values detected")
    
    # Data types
    print(f"\nData Types:")
    for dtype in data.dtypes.value_counts().items():
        print(f"  {dtype[0]}: {dtype[1]} columns")

# Cell 4: Customer Demographics Analysis
if not data.empty and 'age' in data.columns:
    print("\n👥 Customer Demographics Analysis")
    print("-" * 40)
    
    # Age distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Age histogram
    data['age'].hist(bins=30, ax=axes[0,0], color='skyblue', alpha=0.7)
    axes[0,0].set_title('Age Distribution')
    axes[0,0].set_xlabel('Age')
    axes[0,0].set_ylabel('Frequency')
    
    # Job distribution
    if 'job' in data.columns:
        job_counts = data['job'].value_counts().head(10)
        job_counts.plot(kind='bar', ax=axes[0,1], color='lightcoral')
        axes[0,1].set_title('Top 10 Job Categories')
        axes[0,1].tick_params(axis='x', rotation=45)
    
    # Education distribution
    if 'education' in data.columns:
        education_counts = data['education'].value_counts()
        axes[1,0].pie(education_counts.values, labels=education_counts.index, autopct='%1.1f%%')
        axes[1,0].set_title('Education Distribution')
    
    # Marital status
    if 'marital' in data.columns:
        marital_counts = data['marital'].value_counts()
        axes[1,1].pie(marital_counts.values, labels=marital_counts.index, autopct='%1.1f%%')
        axes[1,1].set_title('Marital Status Distribution')
    
    plt.tight_layout()
    plt.show()

# Cell 5: Financial Profile Analysis
if not data.empty and 'balance' in data.columns:
    print("\n💰 Financial Profile Analysis")
    print("-" * 35)
    
    # Balance statistics
    print(f"Balance Statistics:")
    print(f"  Mean: ${data['balance'].mean():,.2f}")
    print(f"  Median: ${data['balance'].median():,.2f}")
    print(f"  Std Dev: ${data['balance'].std():,.2f}")
    print(f"  Min: ${data['balance'].min():,.2f}")
    print(f"  Max: ${data['balance'].max():,.2f}")
    
    # Balance distribution
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Histogram
    data['balance'].hist(bins=50, ax=axes[0], color='gold', alpha=0.7)
    axes[0].set_title('Balance Distribution')
    axes[0].set_xlabel('Account Balance')
    axes[0].set_ylabel('Frequency')
    
    # Box plot
    data.boxplot(column='balance', ax=axes[1])
    axes[1].set_title('Balance Box Plot')
    axes[1].set_ylabel('Account Balance')
    
    # Log scale (if positive balances)
    positive_balances = data[data['balance'] > 0]['balance']
    if len(positive_balances) > 0:
        positive_balances.hist(bins=50, ax=axes[2], color='lightgreen', alpha=0.7)
        axes[2].set_yscale('log')
        axes[2].set_title('Positive Balance Distribution (Log Scale)')
        axes[2].set_xlabel('Account Balance')
    
    plt.tight_layout()
    plt.show()

# Cell 6: Campaign Analysis
if not data.empty and 'y' in data.columns:
    print("\n📞 Campaign Analysis")
    print("-" * 25)
    
    # Convert target variable if needed
    if data['y'].dtype == 'object':
        data['y_numeric'] = (data['y'] == 'yes').astype(int)
        target_col = 'y_numeric'
    else:
        target_col = 'y'
    
    # Overall conversion rate
    conversion_rate = data[target_col].mean() * 100
    print(f"Overall Conversion Rate: {conversion_rate:.2f}%")
    print(f"Total Conversions: {data[target_col].sum():,}")
    print(f"Total Contacts: {len(data):,}")
    
    # Conversion by demographics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Conversion by age group
    if 'age' in data.columns:
        age_bins = pd.cut(data['age'], bins=[0, 30, 40, 50, 60, 100], 
                         labels=['<30', '30-40', '40-50', '50-60', '60+'])
        age_conversion = data.groupby(age_bins)[target_col].mean() * 100
        age_conversion.plot(kind='bar', ax=axes[0,0], color='steelblue')
        axes[0,0].set_title('Conversion Rate by Age Group')
        axes[0,0].set_ylabel('Conversion Rate (%)')
        axes[0,0].tick_params(axis='x', rotation=45)
    
    # Conversion by job
    if 'job' in data.columns:
        job_conversion = data.groupby('job')[target_col].mean() * 100
        job_conversion.sort_values(ascending=True).plot(kind='barh', ax=axes[0,1], color='coral')
        axes[0,1].set_title('Conversion Rate by Job')
        axes[0,1].set_xlabel('Conversion Rate (%)')
    
    # Conversion by education
    if 'education' in data.columns:
        edu_conversion = data.groupby('education')[target_col].mean() * 100
        edu_conversion.plot(kind='bar', ax=axes[1,0], color='lightgreen')
        axes[1,0].set_title('Conversion Rate by Education')
        axes[1,0].set_ylabel('Conversion Rate (%)')
        axes[1,0].tick_params(axis='x', rotation=45)
    
    # Conversion by marital status
    if 'marital' in data.columns:
        marital_conversion = data.groupby('marital')[target_col].mean() * 100
        marital_conversion.plot(kind='bar', ax=axes[1,1], color='plum')
        axes[1,1].set_title('Conversion Rate by Marital Status')
        axes[1,1].set_ylabel('Conversion Rate (%)')
        axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# Cell 7: Interactive Plotly Visualizations
if not data.empty:
    print("\n📊 Interactive Visualizations")
    print("-" * 35)
    
    # Balance vs Age scatter plot
    if all(col in data.columns for col in ['age', 'balance']):
        fig = px.scatter(
            data.sample(1000) if len(data) > 1000 else data,  # Sample for performance
            x='age', 
            y='balance',
            color='job' if 'job' in data.columns else None,
            title='Customer Age vs Account Balance',
            labels={'age': 'Age', 'balance': 'Account Balance ($)'},
            hover_data=['education'] if 'education' in data.columns else None
        )
        fig.show()
    
    # Campaign duration analysis
    if 'duration' in data.columns and target_col in data.columns:
        # Bin duration for better visualization
        duration_bins = pd.cut(data['duration'], bins=10)
        duration_analysis = data.groupby(duration_bins).agg({
            target_col: ['count', 'sum', 'mean']
        }).round(3)
        duration_analysis.columns = ['Total_Contacts', 'Conversions', 'Conversion_Rate']
        duration_analysis = duration_analysis.reset_index()
        
        fig = px.bar(
            duration_analysis,
            x='duration',
            y='Conversion_Rate',
            title='Conversion Rate by Call Duration',
            labels={'duration': 'Call Duration (seconds)', 'Conversion_Rate': 'Conversion Rate'}
        )
        fig.show()

# Cell 8: Correlation Analysis
if not data.empty:
    print("\n🔗 Correlation Analysis")
    print("-" * 25)
    
    # Select numeric columns for correlation
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 1:
        # Calculate correlation matrix
        corr_matrix = data[numeric_cols].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title('Correlation Matrix of Numeric Variables')
        plt.tight_layout()
        plt.show()
        
        # Show strongest correlations
        print("\nStrongest Positive Correlations:")
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append((
                    corr_matrix.columns[i], 
                    corr_matrix.columns[j], 
                    corr_matrix.iloc[i, j]
                ))
        
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        for pair in corr_pairs[:5]:
            print(f"  {pair[0]} ↔ {pair[1]}: {pair[2]:.3f}")

# Cell 9: Risk Analysis (if risk_score exists)
if not data.empty and 'risk_score' in data.columns:
    print("\n⚠️ Risk Analysis")
    print("-" * 20)
    
    # Risk score distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Risk score histogram
    data['risk_score'].hist(bins=30, ax=axes[0], color='red', alpha=0.7)
    axes[0].set_title('Risk Score Distribution')
    axes[0].set_xlabel('Risk Score')
    axes[0].set_ylabel('Frequency')
    
    # Risk vs Balance
    if 'balance' in data.columns:
        axes[1].scatter(data['risk_score'], data['balance'], alpha=0.5, color='orange')
        axes[1].set_title('Risk Score vs Account Balance')
        axes[1].set_xlabel('Risk Score')
        axes[1].set_ylabel('Account Balance')
    
    plt.tight_layout()
    plt.show()
    
    # Risk statistics
    print(f"Risk Score Statistics:")
    print(f"  Mean: {data['risk_score'].mean():.2f}")
    print(f"  Median: {data['risk_score'].median():.2f}")
    print(f"  High Risk (>70): {(data['risk_score'] > 70).sum()} customers")

# Cell 10: Summary and Insights
if not data.empty:
    print("\n📈 Key Insights Summary")
    print("=" * 30)
    
    insights = []
    
    # Customer base insights
    insights.append(f"📊 Total customer base: {len(data):,} customers")
    
    # Age insights
    if 'age' in data.columns:
        avg_age = data['age'].mean()
        insights.append(f"👥 Average customer age: {avg_age:.1f} years")
    
    # Financial insights
    if 'balance' in data.columns:
        avg_balance = data['balance'].mean()
        insights.append(f"💰 Average account balance: ${avg_balance:,.2f}")
        
        negative_balance_pct = (data['balance'] < 0).mean() * 100
        if negative_balance_pct > 0:
            insights.append(f"⚠️ Customers with negative balance: {negative_balance_pct:.1f}%")
    
    # Campaign insights
    if target_col in data.columns:
        conversion_rate = data[target_col].mean() * 100
        insights.append(f"📞 Campaign conversion rate: {conversion_rate:.2f}%")
    
    # Job insights
    if 'job' in data.columns:
        top_job = data['job'].value_counts().index[0]
        top_job_pct = data['job'].value_counts().iloc[0] / len(data) * 100
        insights.append(f"👔 Most common job: {top_job} ({top_job_pct:.1f}%)")
    
    # Risk insights
    if 'risk_score' in data.columns:
        high_risk_pct = (data['risk_score'] > 70).mean() * 100
        insights.append(f"⚠️ High-risk customers: {high_risk_pct:.1f}%")
    
    # Display insights
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    print(f"\n✅ Analysis complete! Explore the FinSight BI dashboard for more insights.")

print("\n" + "=" * 60)
print("📝 Note: This notebook provides exploratory data analysis.")
print("   For real-time insights, use the FinSight BI dashboard:")
print("   python run_app.py")
print("=" * 60)
