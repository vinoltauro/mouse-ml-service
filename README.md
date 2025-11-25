# 🖱️ Mouse Bot Detection ML Service

REST API microservice for detecting bots based on mouse movement patterns using machine learning.

## 🎯 Features

- **XGBoost Model** (98.68% F1 Score)
- **Feature Extraction** (24 behavioral features)
- **Explainable AI** (detailed reasoning for predictions)
- **RESTful API** (easy integration)
- **Docker Ready** (production deployment)
- **Azure Compatible** (deploy to cloud)

---

## 🚀 Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python app.py

# Server starts at http://localhost:5002
```

### Docker

```bash
# Build image
docker build -t mouse-ml-service .

# Run container
docker run -p 5002:5002 mouse-ml-service

# Access at http://localhost:5002
```

---

## 📡 API Endpoints

### `GET /`
API information and available endpoints

### `GET /health`
Health check

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### `GET /metrics`
Service metrics

**Response:**
```json
{
  "requests_total": 1523,
  "predictions_human": 845,
  "predictions_bot": 678,
  "avg_latency_ms": 42
}
```

### `POST /predict`
Single prediction (fast)

**Request:**
```json
{
  "events": [
    {"x": 100, "y": 200, "timestamp": 0, "type": "move"},
    {"x": 105, "y": 205, "timestamp": 16, "type": "move"}
  ]
}
```

**Response:**
```json
{
  "prediction": "human",
  "confidence": 0.85,
  "humanityScore": 85,
  "probabilities": {
    "bot": 0.15,
    "human": 0.85
  },
  "processingTime": 42,
  "eventCount": 150
}
```

### `POST /predict-explain`
Prediction with explanations

**Response includes:**
- Top contributing features
- Warning flags
- Risk assessment

```json
{
  "prediction": "bot",
  "confidence": 0.98,
  "humanityScore": 2,
  "explanations": {
    "topFeatures": [
      {"name": "path_efficiency", "value": 0.99, "importance": 0.25},
      {"name": "click_dwell", "value": 5, "importance": 0.18}
    ],
    "flags": [
      {
        "severity": "critical",
        "feature": "path_efficiency",
        "message": "Nearly perfect straight line (η=0.990)",
        "impact": "high"
      }
    ],
    "riskLevel": "critical"
  }
}
```

### `POST /predict-batch`
Batch predictions

**Request:**
```json
{
  "batch": [
    {"events": [...]},
    {"events": [...]}
  ],
  "explain": false
}
```

---

## 🧪 Testing

### cURL Examples

```bash
# Health check
curl http://localhost:5002/health

# Simple prediction
curl -X POST http://localhost:5002/predict \
  -H "Content-Type: application/json" \
  -d '{
    "events": [
      {"x":100,"y":200,"timestamp":0,"type":"move"},
      {"x":105,"y":205,"timestamp":16,"type":"move"}
    ]
  }'

# Prediction with explanation
curl -X POST http://localhost:5002/predict-explain \
  -H "Content-Type: application/json" \
  -d @sample_data.json
```

---

## 📦 Deployment

### Azure Container Instances

```bash
# Login to Azure
az login

# Create resource group
az group create --name mouse-ml-rg --location eastus

# Create container registry
az acr create --resource-group mouse-ml-rg \
  --name mousemlacrvinol --sku Basic

# Build and push image
az acr build --registry mousemlacrvinol \
  --image mouse-ml-service:latest .

# Deploy container
az container create \
  --resource-group mouse-ml-rg \
  --name mouse-ml-service \
  --image mousemlacrvinol.azurecr.io/mouse-ml-service:latest \
  --dns-name-label mouse-ml-vinol \
  --ports 5002

# Get URL
az container show --resource-group mouse-ml-rg \
  --name mouse-ml-service \
  --query ipAddress.fqdn
```

Your service will be available at:
```
http://mouse-ml-vinol.eastus.azurecontainer.io:5002
```

---

## 🔧 Configuration

Environment variables:

- `PORT` - Server port (default: 5002)
- `MODEL_PATH` - Path to model file
- `WORKERS` - Gunicorn workers (default: 2)
- `TIMEOUT` - Request timeout (default: 60s)

---

## 📊 Performance

- **Latency:** ~40-50ms per prediction
- **Throughput:** ~100 requests/second (single worker)
- **Model Size:** ~2-5 MB
- **Memory:** ~150 MB runtime

---

## 🔒 Security

- CORS enabled for all origins (configure for production)
- No authentication (add in production)
- Rate limiting recommended
- Input validation included

---

## 📈 Model Performance

- **Accuracy:** 99.58%
- **Precision:** 99.47%
- **Recall:** 97.91%
- **F1 Score:** 98.68%
- **AUC-ROC:** 99.88%

Trained on 3,551 trajectories from DELBOT dataset.

---

## 🛠️ Development

### Project Structure

```
mouse-ml-service/
├── app.py                  # Flask server
├── model_loader.py         # Model loading
├── predictor.py            # Prediction logic
├── feature_extractor.py    # Feature extraction
├── requirements.txt        # Dependencies
├── Dockerfile              # Container config
├── models/
│   └── mouse_bot_xgboost.pkl
└── README.md
```

### Adding New Features

1. Update `feature_extractor.py`
2. Retrain model
3. Update model file
4. Restart service

---

## 📝 License

MIT License - See LICENSE file

---

## 🤝 Contributing

Contributions welcome! Please open issues or pull requests.

---

## 📧 Contact

For questions or support, contact: your.email@example.com
