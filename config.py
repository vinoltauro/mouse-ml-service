"""
Configuration for Mouse ML Service
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Model configuration
MODEL_PATH = BASE_DIR / 'models' / 'mouse_bot_xgboost.pkl'
MODEL_VERSION = '1.0.0'

# API configuration
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', 5002))
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

# Performance configuration
ENABLE_SHAP = os.getenv('ENABLE_SHAP', 'True').lower() == 'true'
MAX_EVENTS = int(os.getenv('MAX_EVENTS', 10000))  # Max events per trajectory
REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', 30))  # seconds

# Thresholds
BOT_THRESHOLD = float(os.getenv('BOT_THRESHOLD', 0.3))  # < 30% = bot
HUMAN_THRESHOLD = float(os.getenv('HUMAN_THRESHOLD', 0.7))  # > 70% = human

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = BASE_DIR / 'logs' / 'ml_service.log'

# CORS
# ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', 'http://localhost:5001,http://localhost:3000').split(',')
# In config.py
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', 'http://localhost:5001,https://group4.kokax.com').split(',')

# Health check
SERVICE_NAME = 'Mouse ML Service'
