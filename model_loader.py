"""
Model Loader - Loads and caches XGBoost model
"""

import pickle
import shap
from pathlib import Path
from typing import Optional
import logging

from config import MODEL_PATH, ENABLE_SHAP

logger = logging.getLogger(__name__)


class ModelLoader:
    """Singleton model loader with caching"""
    
    _instance = None
    _model = None
    _scaler = None
    _explainer = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def load_model(self):
        """Load XGBoost model and scaler from pickle file"""
        if self._model is not None:
            logger.info("Using cached model")
            return self._model, self._scaler
        
        logger.info(f"Loading model from {MODEL_PATH}")
        
        try:
            with open(MODEL_PATH, 'rb') as f:
                data = pickle.load(f)
            
            self._model = data['model']
            self._scaler = data['scaler']
            
            logger.info("✅ Model loaded successfully")
            logger.info(f"   Model type: {data.get('model_type', 'unknown')}")
            
            # Load SHAP explainer if enabled
            if ENABLE_SHAP and self._explainer is None:
                self._load_shap_explainer()
            
            return self._model, self._scaler
        
        except FileNotFoundError:
            logger.error(f"❌ Model file not found: {MODEL_PATH}")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def _load_shap_explainer(self):
        """Load SHAP explainer for model interpretability"""
        try:
            logger.info("Loading SHAP explainer...")
            self._explainer = shap.TreeExplainer(self._model)
            logger.info("✅ SHAP explainer loaded")
        except Exception as e:
            logger.warning(f"⚠️  Failed to load SHAP explainer: {e}")
            self._explainer = None
    
    def get_explainer(self) -> Optional[shap.TreeExplainer]:
        """Get SHAP explainer"""
        if self._explainer is None and ENABLE_SHAP:
            self._load_shap_explainer()
        return self._explainer
    
    def get_model_info(self) -> dict:
        """Get model metadata"""
        return {
            'loaded': self._model is not None,
            'shap_enabled': self._explainer is not None,
            'model_path': str(MODEL_PATH)
        }


# Global instance
model_loader = ModelLoader()
