"""
Mouse ML Service - Flask API
Provides bot detection predictions via REST API
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
from datetime import datetime
from config import ALLOWED_ORIGINS

# from predictor import predictor
from predictor import get_predictor

# Add this line after the imports (around line 15):
predictor = get_predictor()



# Initialize Flask app
app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes
CORS(app, origins=ALLOWED_ORIGINS)


# Global metrics
metrics = {
    'requests_total': 0,
    'predictions_human': 0,
    'predictions_bot': 0,
    'errors_total': 0,
    'avg_latency_ms': 0,
    'start_time': datetime.utcnow().isoformat()
}


@app.route('/', methods=['GET'])
def index():
    """Root endpoint - API information"""
    return jsonify({
        'service': 'Mouse Bot Detection ML Service',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'GET /': 'API information',
            'GET /health': 'Health check',
            'GET /metrics': 'Service metrics',
            'POST /predict': 'Single prediction',
            'POST /predict-explain': 'Prediction with explanations',
            'POST /predict-batch': 'Batch predictions'
        },
        'documentation': 'https://github.com/yourusername/mouse-ml-service'
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        # Check if model is loaded
        model, _, _ = predictor.model, predictor.scaler, predictor.feature_extractor
        
        if model is None:
            return jsonify({
                'status': 'unhealthy',
                'message': 'Model not loaded'
            }), 503
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'model_loaded': True
        })
    
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get service metrics"""
    return jsonify(metrics)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict if mouse movements are human or bot
    
    Request body:
    {
        "events": [
            {"x": 100, "y": 200, "timestamp": 0, "type": "move"},
            ...
        ]
    }
    
    Response:
    {
        "prediction": "human" | "bot",
        "confidence": 0.95,
        "humanityScore": 95,
        "processingTime": 42
    }
    """
    start_time = time.time()
    
    try:
        # Parse request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Request body must be JSON'
            }), 400
        
        events = data.get('events', [])
        
        if not events:
            return jsonify({
                'error': 'Missing events',
                'message': 'Request must include "events" array'
            }), 400
        
        # Make prediction
        result = predictor.predict(events, explain=False)
        
        # Update metrics
        metrics['requests_total'] += 1
        
        if result.get('error'):
            metrics['errors_total'] += 1
            return jsonify(result), 400
        
        if result['prediction'] == 'human':
            metrics['predictions_human'] += 1
        else:
            metrics['predictions_bot'] += 1
        
        # Update latency
        latency = (time.time() - start_time) * 1000
        metrics['avg_latency_ms'] = (
            (metrics['avg_latency_ms'] * (metrics['requests_total'] - 1) + latency)
            / metrics['requests_total']
        )
        
        return jsonify(result)
    
    except Exception as e:
        metrics['errors_total'] += 1
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/predict-explain', methods=['POST'])
def predict_explain():
    """
    Predict with detailed explanations
    
    Same as /predict but includes:
    - Top contributing features
    - Warning flags
    - Risk level assessment
    """
    start_time = time.time()
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Request body must be JSON'
            }), 400
        
        events = data.get('events', [])
        
        if not events:
            return jsonify({
                'error': 'Missing events',
                'message': 'Request must include "events" array'
            }), 400
        
        # Make prediction with explanations
        result = predictor.predict(events, explain=True)
        
        # Update metrics
        metrics['requests_total'] += 1
        
        if result.get('error'):
            metrics['errors_total'] += 1
            return jsonify(result), 400
        
        if result['prediction'] == 'human':
            metrics['predictions_human'] += 1
        else:
            metrics['predictions_bot'] += 1
        
        # Update latency
        latency = (time.time() - start_time) * 1000
        metrics['avg_latency_ms'] = (
            (metrics['avg_latency_ms'] * (metrics['requests_total'] - 1) + latency)
            / metrics['requests_total']
        )
        
        return jsonify(result)
    
    except Exception as e:
        metrics['errors_total'] += 1
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction for multiple trajectories
    
    Request body:
    {
        "batch": [
            {"events": [...]},
            {"events": [...]},
            ...
        ],
        "explain": false
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'batch' not in data:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Request must include "batch" array'
            }), 400
        
        batch = data['batch']
        explain = data.get('explain', False)
        
        # Extract events from each item
        batch_events = [item.get('events', []) for item in batch]
        
        # Make batch prediction
        results = predictor.batch_predict(batch_events, explain=explain)
        
        # Update metrics
        metrics['requests_total'] += len(results)
        for result in results:
            if not result.get('error'):
                if result['prediction'] == 'human':
                    metrics['predictions_human'] += 1
                else:
                    metrics['predictions_bot'] += 1
        
        return jsonify({
            'results': results,
            'count': len(results)
        })
    
    except Exception as e:
        metrics['errors_total'] += 1
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Not found',
        'message': 'The requested endpoint does not exist'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500


if __name__ == '__main__':
    # Development server
    port = int(os.environ.get('PORT', 5002))
    print(f"\n{'='*60}")
    print(f"🚀 Mouse ML Service Starting")
    print(f"{'='*60}")
    print(f"📡 Server: http://localhost:{port}")
    print(f"📚 Docs: http://localhost:{port}/")
    print(f"💚 Health: http://localhost:{port}/health")
    print(f"{'='*60}\n")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False  # Set to True for development
    )
