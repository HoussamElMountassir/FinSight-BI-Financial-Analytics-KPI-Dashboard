"""
FinSight BI Dashboard Application
Advanced Banking KPI Dashboard with Interactive Visualizations

Author: BI Development Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback, dash_table, State
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import sqlite3
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.etl.data_processor import BankingDataETL
from src.analytics.banking_analytics import BankingAnalytics, generate_recommendations

# Initialize the Dash app
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)

app.title = "FinSight BI: Banking KPI Dashboard"

# ==================== DATA LOADING ====================

def load_sample_data():
    """Load sample data for dashboard"""
    try:
        # Check if processed data exists
        db_path = project_root / 'data' / 'finsight_bi.db'
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            data = pd.read_sql_query("SELECT * FROM customers", conn)
            conn.close()
            return data
        else:
            # Initialize ETL pipeline with sample data
            etl = BankingDataETL({
                'input_file': str(project_root / 'bank.csv'),
                'output_db': str(project_root / 'data' / 'finsight_bi.db')
            })
            return etl.run_full_pipeline()
    except Exception as e:
        print(f"Error loading data: {e}")
        # Create minimal sample data
        return pd.DataFrame({
            'age': [25, 35, 45, 55, 65],
            'job': ['admin', 'technician', 'management', 'retired', 'services'],
            'balance': [1000, 2000, 5000, 3000, 1500],
            'y': [0, 1, 1, 0, 1],
            'risk_score': [20, 30, 15, 40, 25],
            'customer_segment': ['Basic', 'Standard', 'Premium', 'High_Risk', 'Standard'],
            'branch': ['Downtown_Branch', 'Suburban_Branch', 'Business_District', 'Mall_Branch', 'Online_Branch'],
            'annual_revenue': [20, 40, 100, 60, 30],
            'estimated_clv': [500, 1000, 2500, 800, 600]
        })

# Load initial data
initial_data = load_sample_data()

# ==================== LAYOUT COMPONENTS ====================

def create_navbar():
    """Create navigation bar"""
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Dashboard", href="#dashboard")),
            dbc.NavItem(dbc.NavLink("Analytics", href="#analytics")),
            dbc.NavItem(dbc.NavLink("Reports", href="#reports")),
            dbc.DropdownMenu(
                children=[
                    dbc.DropdownMenuItem("Export Data", href="#"),
                    dbc.DropdownMenuItem("Settings", href="#"),
                    dbc.DropdownMenuItem(divider=True),
                    dbc.DropdownMenuItem("Help", href="#"),
                ],
                nav=True,
                in_navbar=True,
                label="More",
            ),
        ],
        brand="FinSight BI",
        brand_href="#",
        color="primary",
        dark=True,
        fluid=True,
    )

def create_kpi_cards():
    """Create KPI summary cards"""
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("0", id="total-customers", className="card-title text-primary"),
                    html.P("Total Customers", className="card-text"),
                    html.Small("Active customer base", className="text-muted")
                ])
            ], className="h-100")
        ], width=12, lg=2),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("$0", id="total-revenue", className="card-title text-success"),
                    html.P("Total Revenue", className="card-text"),
                    html.Small("Annual revenue", className="text-muted")
                ])
            ], className="h-100")
        ], width=12, lg=2),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("0%", id="conversion-rate", className="card-title text-info"),
                    html.P("Conversion Rate", className="card-text"),
                    html.Small("Campaign success", className="text-muted")
                ])
            ], className="h-100")
        ], width=12, lg=2),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("$0", id="avg-clv", className="card-title text-warning"),
                    html.P("Avg Customer LTV", className="card-text"),
                    html.Small("Lifetime value", className="text-muted")
                ])
            ], className="h-100")
        ], width=12, lg=2),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("0", id="avg-risk", className="card-title text-danger"),
                    html.P("Avg Risk Score", className="card-text"),
                    html.Small("Portfolio risk", className="text-muted")
                ])
            ], className="h-100")
        ], width=12, lg=2),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("0%", id="high-risk-pct", className="card-title text-dark"),
                    html.P("High Risk %", className="card-text"),
                    html.Small("Risk exposure", className="text-muted")
                ])
            ], className="h-100")
        ], width=12, lg=2)
    ], className="mb-4")

def create_control_panel():
    """Create dashboard control panel"""
    return dbc.Card([
        dbc.CardHeader(html.H5("Dashboard Controls", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Customer Segment"),
                    dcc.Dropdown(
                        id="segment-filter",
                        options=[
                            {"label": "All Segments", "value": "all"},
                            {"label": "Premium", "value": "Premium"},
                            {"label": "Standard", "value": "Standard"},
                            {"label": "Basic", "value": "Basic"},
                            {"label": "High Risk", "value": "High_Risk"}
                        ],
                        value="all"
                    )
                ], width=6),
                
                dbc.Col([
                    html.Label("Branch"),
                    dcc.Dropdown(
                        id="branch-filter",
                        options=[{"label": "All Branches", "value": "all"}],
                        value="all"
                    )
                ], width=6)
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Risk Category"),
                    dcc.Dropdown(
                        id="risk-filter",
                        options=[
                            {"label": "All Risk Levels", "value": "all"},
                            {"label": "Low Risk", "value": "Low"},
                            {"label": "Medium Risk", "value": "Medium"},
                            {"label": "High Risk", "value": "High"},
                            {"label": "Very High Risk", "value": "Very_High"}
                        ],
                        value="all"
                    )
                ], width=6),
                
                dbc.Col([
                    dbc.Button(
                        "Refresh Data", 
                        id="refresh-button", 
                        color="primary", 
                        size="sm",
                        className="mt-4"
                    )
                ], width=6)
            ])
        ])
    ], className="mb-4")

# ==================== CHART FUNCTIONS ====================

def create_revenue_chart(data):
    """Create revenue by segment chart"""
    if 'customer_segment' in data.columns and 'annual_revenue' in data.columns:
        revenue_data = data.groupby('customer_segment')['annual_revenue'].sum().reset_index()
        
        fig = px.bar(
            revenue_data,
            x='customer_segment',
            y='annual_revenue',
            title="Revenue by Customer Segment",
            labels={'annual_revenue': 'Revenue ($)', 'customer_segment': 'Customer Segment'},
            color='annual_revenue',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400, showlegend=False)
        return fig
    return go.Figure()

def create_risk_distribution_chart(data):
    """Create risk distribution chart"""
    if 'risk_category' in data.columns:
        risk_counts = data['risk_category'].value_counts()
        
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Risk Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(height=400)
        return fig
    return go.Figure()

def create_branch_performance_chart(data):
    """Create branch performance chart"""
    if 'branch' in data.columns and 'y' in data.columns:
        branch_perf = data.groupby('branch').agg({
            'y': ['sum', 'count', 'mean']
        }).round(2)
        branch_perf.columns = ['conversions', 'total_contacts', 'conversion_rate']
        branch_perf = branch_perf.reset_index()
        
        fig = px.bar(
            branch_perf,
            x='branch',
            y='conversion_rate',
            title="Conversion Rate by Branch",
            labels={'conversion_rate': 'Conversion Rate', 'branch': 'Branch'},
            text='conversion_rate'
        )
        fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig.update_layout(height=400)
        return fig
    return go.Figure()

def create_age_analysis_chart(data):
    """Create age analysis chart"""
    if 'age' in data.columns and 'balance' in data.columns:
        fig = px.scatter(
            data,
            x='age',
            y='balance',
            color='customer_segment' if 'customer_segment' in data.columns else None,
            size='annual_revenue' if 'annual_revenue' in data.columns else None,
            title="Customer Age vs Balance Analysis",
            labels={'age': 'Age', 'balance': 'Account Balance ($)'},
            hover_data=['job'] if 'job' in data.columns else None
        )
        fig.update_layout(height=400)
        return fig
    return go.Figure()

# ==================== MAIN LAYOUT ====================

app.layout = dbc.Container([
    # Data storage
    dcc.Store(id='data-store', data=initial_data.to_dict('records')),
    
    # Navigation
    create_navbar(),
    
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("🏦 FinSight BI: Banking KPI Dashboard", className="text-center mt-4 mb-2"),
            html.P("Advanced Business Intelligence for Banking Operations", 
                   className="text-center text-muted mb-4")
        ])
    ]),
    
    # Control Panel
    create_control_panel(),
    
    # KPI Cards
    create_kpi_cards(),
    
    # Charts Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Revenue Analysis")),
                dbc.CardBody([
                    dcc.Graph(id="revenue-chart")
                ])
            ])
        ], width=12, lg=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Risk Distribution")),
                dbc.CardBody([
                    dcc.Graph(id="risk-chart")
                ])
            ])
        ], width=12, lg=6)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Branch Performance")),
                dbc.CardBody([
                    dcc.Graph(id="branch-chart")
                ])
            ])
        ], width=12, lg=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Customer Demographics")),
                dbc.CardBody([
                    dcc.Graph(id="demographics-chart")
                ])
            ])
        ], width=12, lg=6)
    ], className="mb-4"),
    
    # Analytics Summary
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Business Insights & Recommendations")),
                dbc.CardBody([
                    html.Div(id="recommendations-content")
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Data Table
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Customer Data Overview")),
                dbc.CardBody([
                    dash_table.DataTable(
                        id='customer-table',
                        columns=[],
                        data=[],
                        page_size=10,
                        sort_action="native",
                        filter_action="native",
                        style_cell={'textAlign': 'left', 'fontSize': '12px'},
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgb(248, 248, 248)'
                            }
                        ],
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold'
                        }
                    )
                ])
            ])
        ])
    ])
], fluid=True)

# ==================== CALLBACKS ====================

@app.callback(
    [Output('data-store', 'data'),
     Output('branch-filter', 'options')],
    [Input('refresh-button', 'n_clicks')],
    prevent_initial_call=False
)
def refresh_data(n_clicks):
    """Refresh data and update branch filter options"""
    data = load_sample_data()
    
    # Update branch filter options
    if 'branch' in data.columns:
        branch_options = [{"label": "All Branches", "value": "all"}]
        branch_options.extend([
            {"label": branch, "value": branch} 
            for branch in sorted(data['branch'].unique())
        ])
    else:
        branch_options = [{"label": "All Branches", "value": "all"}]
    
    return data.to_dict('records'), branch_options

@app.callback(
    [Output('total-customers', 'children'),
     Output('total-revenue', 'children'),
     Output('conversion-rate', 'children'),
     Output('avg-clv', 'children'),
     Output('avg-risk', 'children'),
     Output('high-risk-pct', 'children')],
    [Input('data-store', 'data'),
     Input('segment-filter', 'value'),
     Input('branch-filter', 'value'),
     Input('risk-filter', 'value')]
)
def update_kpis(data, segment_filter, branch_filter, risk_filter):
    """Update KPI cards based on filters"""
    df = pd.DataFrame(data)
    
    # Apply filters
    if segment_filter != "all" and 'customer_segment' in df.columns:
        df = df[df['customer_segment'] == segment_filter]
    
    if branch_filter != "all" and 'branch' in df.columns:
        df = df[df['branch'] == branch_filter]
    
    if risk_filter != "all" and 'risk_category' in df.columns:
        df = df[df['risk_category'] == risk_filter]
    
    # Calculate KPIs
    total_customers = f"{len(df):,}"
    
    total_revenue = f"${df['annual_revenue'].sum():,.0f}" if 'annual_revenue' in df.columns else "$0"
    
    conversion_rate = f"{df['y'].mean() * 100:.1f}%" if 'y' in df.columns else "0%"
    
    avg_clv = f"${df['estimated_clv'].mean():,.0f}" if 'estimated_clv' in df.columns else "$0"
    
    avg_risk = f"{df['risk_score'].mean():.1f}" if 'risk_score' in df.columns else "0"
    
    if 'risk_category' in df.columns:
        high_risk_pct = f"{(df['risk_category'].isin(['High', 'Very_High']).mean() * 100):.1f}%"
    else:
        high_risk_pct = "0%"
    
    return total_customers, total_revenue, conversion_rate, avg_clv, avg_risk, high_risk_pct

@app.callback(
    [Output('revenue-chart', 'figure'),
     Output('risk-chart', 'figure'),
     Output('branch-chart', 'figure'),
     Output('demographics-chart', 'figure')],
    [Input('data-store', 'data'),
     Input('segment-filter', 'value'),
     Input('branch-filter', 'value'),
     Input('risk-filter', 'value')]
)
def update_charts(data, segment_filter, branch_filter, risk_filter):
    """Update all charts based on filters"""
    df = pd.DataFrame(data)
    
    # Apply filters
    if segment_filter != "all" and 'customer_segment' in df.columns:
        df = df[df['customer_segment'] == segment_filter]
    
    if branch_filter != "all" and 'branch' in df.columns:
        df = df[df['branch'] == branch_filter]
    
    if risk_filter != "all" and 'risk_category' in df.columns:
        df = df[df['risk_category'] == risk_filter]
    
    # Create charts
    revenue_fig = create_revenue_chart(df)
    risk_fig = create_risk_distribution_chart(df)
    branch_fig = create_branch_performance_chart(df)
    demographics_fig = create_age_analysis_chart(df)
    
    return revenue_fig, risk_fig, branch_fig, demographics_fig

@app.callback(
    Output('recommendations-content', 'children'),
    [Input('data-store', 'data')]
)
def update_recommendations(data):
    """Generate and display business recommendations"""
    df = pd.DataFrame(data)
    
    if len(df) == 0:
        return html.P("No data available for recommendations.")
    
    # Generate basic recommendations
    recommendations = []
    
    # Conversion rate analysis
    if 'y' in df.columns:
        conversion_rate = df['y'].mean() * 100
        if conversion_rate < 10:
            recommendations.append("🎯 Low conversion rate detected. Consider improving campaign targeting and messaging.")
        elif conversion_rate > 20:
            recommendations.append("✅ Excellent conversion rate. Scale successful campaigns to similar customer segments.")
    
    # Risk analysis
    if 'risk_category' in df.columns:
        high_risk_pct = df['risk_category'].isin(['High', 'Very_High']).mean() * 100
        if high_risk_pct > 25:
            recommendations.append("⚠️ High risk exposure detected. Implement stricter risk assessment criteria.")
    
    # Revenue analysis
    if 'customer_segment' in df.columns and 'annual_revenue' in df.columns:
        segment_revenue = df.groupby('customer_segment')['annual_revenue'].sum()
        if len(segment_revenue) > 0:
            top_segment = segment_revenue.idxmax()
            recommendations.append(f"💰 Focus marketing efforts on '{top_segment}' segment - highest revenue generator.")
    
    # Branch analysis
    if 'branch' in df.columns:
        recommendations.append("🏢 Analyze top-performing branches to replicate success factors across network.")
    
    # Default recommendations if none generated
    if not recommendations:
        recommendations = [
            "📊 Dashboard loaded successfully. Analyze the charts above for insights.",
            "🔍 Use the filters to drill down into specific customer segments.",
            "📈 Monitor KPIs regularly to track business performance."
        ]
    
    # Create recommendation cards
    cards = []
    for i, rec in enumerate(recommendations):
        cards.append(
            dbc.Alert(
                rec,
                color="info" if i % 2 == 0 else "success",
                className="mb-2"
            )
        )
    
    return cards

@app.callback(
    [Output('customer-table', 'columns'),
     Output('customer-table', 'data')],
    [Input('data-store', 'data'),
     Input('segment-filter', 'value'),
     Input('branch-filter', 'value')]
)
def update_data_table(data, segment_filter, branch_filter):
    """Update customer data table"""
    df = pd.DataFrame(data)
    
    # Apply filters
    if segment_filter != "all" and 'customer_segment' in df.columns:
        df = df[df['customer_segment'] == segment_filter]
    
    if branch_filter != "all" and 'branch' in df.columns:
        df = df[df['branch'] == branch_filter]
    
    # Select relevant columns for display
    display_columns = []
    for col in ['age', 'job', 'balance', 'customer_segment', 'risk_score', 'branch', 'y']:
        if col in df.columns:
            display_columns.append(col)
    
    if not display_columns:
        return [], []
    
    # Create column definitions
    columns = []
    for col in display_columns:
        col_def = {"name": col.replace('_', ' ').title(), "id": col}
        
        # Format numeric columns
        if col in ['balance', 'risk_score']:
            col_def["type"] = "numeric"
            col_def["format"] = {"specifier": ",.1f"}
        
        columns.append(col_def)
    
    # Limit data for performance
    table_data = df[display_columns].head(100).to_dict('records')
    
    return columns, table_data

# ==================== RUN APPLICATION ====================

if __name__ == '__main__':
    print("Starting FinSight BI Dashboard...")
    print("Dashboard will be available at: http://127.0.0.1:8050/")
    print("Press Ctrl+C to stop the server")
    
    app.run_server(debug=True, host='0.0.0.0', port=8050)
