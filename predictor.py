"""
Predictor - Makes predictions and generates explanations
"""

import numpy as np
import time
import logging
from typing import Dict, List, Any, Optional

from feature_extractor import MouseFeatureExtractor
from model_loader import model_loader
# We import these, but we will also print them to verify values
from config import BOT_THRESHOLD, HUMAN_THRESHOLD

# Configure logging to ensure it shows up in Docker/Console logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MouseBotPredictor:
    """Predicts if mouse trajectory is human or bot"""
    
    def __init__(self):
        self.feature_extractor = MouseFeatureExtractor()
        self.model, self.scaler = model_loader.load_model()
        self.explainer = model_loader.get_explainer()
        
        logger.info("🎯 Predictor initialized")
        logger.info(f"⚙️  Active Configuration:")
        logger.info(f"   - HUMAN_THRESHOLD: {HUMAN_THRESHOLD}")
        logger.info(f"   - BOT_THRESHOLD:   {BOT_THRESHOLD}")
    
    def predict(self, events: List[Dict], explain: bool = False) -> Dict[str, Any]:
        """
        Predict if trajectory is human or bot
        """
        start_time = time.time()
        
        try:
            # Log incoming request size
            print(f"\n🔍 [PREDICT] Received {len(events)} events")

            # Extract features
            features = self.feature_extractor.extract_features(events)
            
            if not np.any(features):
                logger.warning("⚠️ Empty features extracted")
                return {
                    'error': 'Invalid trajectory - no valid features extracted',
                    'prediction': 'unknown',
                    'confidence': 0.0
                }
            
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Make prediction
            prediction_proba = self.model.predict_proba(features_scaled)[0]
            # prediction_class = self.model.predict(features_scaled)[0] # Unused directly
            
            # Get confidence and humanity score
            bot_probability = float(prediction_proba[0])
            human_probability = float(prediction_proba[1])
            humanity_score = int(human_probability * 100)
            
            # --- DEBUG LOGS FOR LOGIC ---
            print(f"📊 [MATH] Human Prob: {human_probability:.4f} | Bot Prob: {bot_probability:.4f}")
            print(f"⚖️  [THRESHOLDS] Human > {HUMAN_THRESHOLD} | Bot < {BOT_THRESHOLD}")
            
            # Determine prediction label
            if human_probability > HUMAN_THRESHOLD:
                prediction = 'human'
                confidence = human_probability
                print(f"✅ [DECISION] Classified as HUMAN ({human_probability:.4f} > {HUMAN_THRESHOLD})")
            elif human_probability < BOT_THRESHOLD:
                prediction = 'bot'
                confidence = bot_probability
                print(f"🤖 [DECISION] Classified as BOT ({human_probability:.4f} < {BOT_THRESHOLD})")
            else:
                prediction = 'uncertain'
                confidence = max(human_probability, bot_probability)
                print(f"❓ [DECISION] Classified as UNCERTAIN (Score {human_probability:.4f} is between thresholds)")
            
            result = {
                'prediction': prediction,
                'confidence': float(confidence),
                'humanityScore': humanity_score,
                'probabilities': {
                    'bot': float(bot_probability),
                    'human': float(human_probability)
                },
                'processingTimeMs': int((time.time() - start_time) * 1000)
            }
            
            # Add explanation if requested
            if explain and self.explainer is not None:
                print("🧠 [SHAP] Generating explanation...")
                explanation = self._generate_explanation(
                    features_scaled, 
                    features,
                    prediction,
                    human_probability
                )
                result['explanation'] = explanation
                print("🧠 [SHAP] Explanation generated")
            
            logger.info(f"🏁 Final Result: {prediction} ({humanity_score}/100) in {result['processingTimeMs']}ms")
            
            return result
        
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}", exc_info=True)
            return {
                'error': str(e),
                'prediction': 'error',
                'confidence': 0.0
            }
    
    def _generate_explanation(
        self, 
        features_scaled: np.ndarray,
        features_raw: np.ndarray,
        prediction: str,
        human_probability: float
    ) -> Dict[str, Any]:
        """Generate SHAP-based explanation"""
        try:
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(features_scaled)
            
            # For binary classification, take the human class SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Human class
            elif hasattr(shap_values, 'values'): 
                # Newer SHAP versions might return an Explanation object
                if len(shap_values.values.shape) == 3:
                     shap_values = shap_values.values[:,:,1]
                else:
                     shap_values = shap_values.values

            # Handle different SHAP return shapes (sometimes it's (1, features), sometimes just (features,))
            if len(shap_values.shape) > 1:
                sv = shap_values[0]
            else:
                sv = shap_values

            # Get feature names
            feature_names = self.feature_extractor.feature_names
            
            # Get top contributing features
            shap_abs = np.abs(sv)
            top_indices = np.argsort(shap_abs)[::-1][:5]  # Top 5
            
            reasons = []
            for idx in top_indices:
                feature_name = feature_names[idx]
                feature_value = features_raw[idx]
                shap_value = sv[idx]
                impact_pct = int(abs(shap_value) * 100)
                
                # Determine if this pushes towards bot or human
                direction = 'human' if shap_value > 0 else 'bot'
                severity = self._get_severity(abs(shap_value))
                
                # Get expected range for feature
                expected_range = self._get_feature_range(feature_name)
                
                reason = {
                    'feature': feature_name,
                    'value': round(float(feature_value), 2),
                    'expectedRange': expected_range,
                    'impact': impact_pct,
                    'direction': direction,
                    'severity': severity
                }
                reasons.append(reason)
            
            # Generate human-readable flags
            flags = self._generate_flags(reasons, prediction)
            
            return {
                'topReasons': reasons,
                'flags': flags,
                'shapEnabled': True
            }
        
        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            return {
                'topReasons': [],
                'flags': ['Explanation unavailable'],
                'shapEnabled': False
            }
    
    def _get_severity(self, impact: float) -> str:
        """Determine severity based on SHAP impact"""
        if impact > 0.15:
            return 'critical'
        elif impact > 0.10:
            return 'high'
        elif impact > 0.05:
            return 'medium'
        else:
            return 'low'
    
    def _get_feature_range(self, feature_name: str) -> str:
        """Get expected range for a feature (human baseline)"""
        ranges = {
            'mean_velocity': '200-400 px/s',
            'std_velocity': '50-150 px/s',
            'path_efficiency': '0.80-0.95',
            'avg_click_dwell': '60-200 ms',
            'avg_click_drift': '1-5 px',
            'avg_throughput': '3-8 bits/s',
            'max_throughput': '8-12 bits/s',
        }
        return ranges.get(feature_name, 'variable')
    
    def _generate_flags(self, reasons: List[Dict], prediction: str) -> List[str]:
        """Generate human-readable warning flags"""
        flags = []
        
        for reason in reasons:
            feature = reason['feature']
            value = reason['value']
            direction = reason['direction']
            severity = reason['severity']
            
            if severity in ['critical', 'high'] and direction == 'bot':
                if feature == 'path_efficiency' and value > 0.97:
                    flags.append(f"⚠️ Perfect straight line movement (η={value:.2f})")
                elif feature == 'avg_click_dwell' and value < 20:
                    flags.append(f"⚠️ Instant click detection ({value:.0f}ms dwell time)")
                elif feature == 'std_velocity' and value < 10:
                    flags.append(f"⚠️ Constant speed - no natural variation")
                elif feature == 'avg_throughput' and value > 15:
                    flags.append(f"⚠️ Superhuman speed ({value:.1f} bits/s)")
                elif feature == 'avg_click_drift' and value < 0.5:
                    flags.append(f"⚠️ Zero click drift - robotic precision")
        
        if len(flags) == 0:
            if prediction == 'bot':
                flags.append("Multiple subtle bot indicators detected")
            elif prediction == 'human':
                flags.append("✓ Natural human behavior pattern")
            else:
                flags.append("Borderline case - requires additional signals")
        
        return flags[:5]  # Max 5 flags


# Global instance
predictor = None

def get_predictor() -> MouseBotPredictor:
    """Get or create predictor instance"""
    global predictor
    if predictor is None:
        predictor = MouseBotPredictor()
    return predictor