# Configuration file for FinSight BI Dashboard
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXPORTS_DIR = DATA_DIR / "exports"

# Database configuration
DATABASE_CONFIG = {
    "sqlite": {
        "url": f"sqlite:///{PROJECT_ROOT}/data/finsight_bi.db",
        "echo": False
    },
    "postgresql": {
        "url": os.getenv("POSTGRES_URL", "postgresql://user:password@localhost:5432/finsight_bi"),
        "echo": False
    }
}

# Dashboard configuration
DASHBOARD_CONFIG = {
    "host": "0.0.0.0",
    "port": 8050,
    "debug": True,
    "title": "FinSight BI: Banking KPI Dashboard",
    "refresh_interval": 30  # seconds
}

# ETL Configuration
ETL_CONFIG = {
    "batch_size": 10000,
    "max_retries": 3,
    "retry_delay": 5,  # seconds
    "data_quality_threshold": 0.95
}

# Analytics Configuration
ANALYTICS_CONFIG = {
    "risk_model": {
        "default_weight": 50,
        "loan_weight": 20,
        "housing_weight": 10,
        "age_weight": 15,
        "balance_weight": 25
    },
    "clv_model": {
        "annual_interest_rate": 0.02,
        "retention_rate": 0.85,
        "discount_rate": 0.10
    },
    "segmentation": {
        "age_groups": [18, 30, 40, 50, 60, 100],
        "balance_groups": [-float('inf'), 0, 1000, 5000, 20000, float('inf')],
        "risk_groups": [0, 20, 40, 60, 100]
    }
}

# Visualization Configuration
VIZ_CONFIG = {
    "color_palette": {
        "primary": "#1f77b4",
        "secondary": "#ff7f0e",
        "success": "#2ca02c",
        "warning": "#d62728",
        "info": "#9467bd",
        "light": "#17becf",
        "dark": "#8c564b"
    },
    "chart_height": 400,
    "chart_width": 600,
    "font_family": "Arial, sans-serif"
}

# Security Configuration
SECURITY_CONFIG = {
    "secret_key": os.getenv("SECRET_KEY", "your-secret-key-here"),
    "session_timeout": 3600,  # seconds
    "max_login_attempts": 5,
    "password_min_length": 8
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": f"{PROJECT_ROOT}/logs/finsight_bi.log",
    "max_bytes": 10485760,  # 10MB
    "backup_count": 5
}

# Cache Configuration
CACHE_CONFIG = {
    "type": "simple",  # simple, redis, memcached
    "timeout": 300,  # seconds
    "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379/0")
}

# API Configuration
API_CONFIG = {
    "rate_limit": "100/hour",
    "pagination_size": 50,
    "api_version": "v1",
    "documentation_url": "/docs"
}

# Email Configuration
EMAIL_CONFIG = {
    "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
    "smtp_port": int(os.getenv("SMTP_PORT", "587")),
    "email_user": os.getenv("EMAIL_USER", ""),
    "email_password": os.getenv("EMAIL_PASSWORD", ""),
    "use_tls": True
}

# Monitoring Configuration
MONITORING_CONFIG = {
    "enable_metrics": True,
    "metrics_port": 9090,
    "health_check_interval": 60,  # seconds
    "alert_email": os.getenv("ALERT_EMAIL", "admin@company.com")
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    "customer_segments": {
        "premium": {"min_balance": 10000, "max_risk": 30},
        "standard": {"min_balance": 1000, "max_risk": 50},
        "basic": {"min_balance": 0, "max_risk": 70},
        "high_risk": {"min_balance": 0, "max_risk": 100}
    },
    "scoring_weights": {
        "balance": 0.3,
        "age": 0.2,
        "education": 0.2,
        "job": 0.2,
        "campaign_history": 0.1
    }
}

# Model Configuration
MODEL_CONFIG = {
    "random_state": 42,
    "test_size": 0.2,
    "validation_size": 0.2,
    "cv_folds": 5,
    "model_registry_path": f"{PROJECT_ROOT}/models/registry",
    "model_artifacts_path": f"{PROJECT_ROOT}/models/artifacts"
}

# Data Quality Configuration
DATA_QUALITY_CONFIG = {
    "missing_threshold": 0.1,  # 10% missing values threshold
    "duplicate_threshold": 0.05,  # 5% duplicate threshold
    "outlier_method": "iqr",  # iqr, zscore, isolation_forest
    "outlier_threshold": 3.0
}

# Deployment Configuration
DEPLOYMENT_CONFIG = {
    "environment": os.getenv("ENVIRONMENT", "development"),
    "docker_image": "finsight-bi:latest",
    "kubernetes_namespace": "finsight-bi",
    "health_check_path": "/health",
    "readiness_probe_path": "/ready"
}
