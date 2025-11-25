# 🚀 QUICK START GUIDE

## Files You Need to Copy

Download these files from Claude and place them in `mouse-ml-service/`:

1. ✅ `app.py` - Main Flask server
2. ✅ `model_loader.py` - Model loading
3. ✅ `predictor.py` - Prediction logic
4. ✅ `requirements.txt` - Dependencies
5. ✅ `Dockerfile` - Container config
6. ✅ `.dockerignore` - Docker ignore rules
7. ✅ `.env.example` - Environment template
8. ✅ `README.md` - Documentation
9. ✅ `test_api.py` - Test script
10. ✅ `deploy-azure.sh` - Azure deployment

Plus copy from mouse-ml-training:
11. ✅ `feature_extractor.py` - From training
12. ✅ `mouse_bot_xgboost.pkl` - From training (put in models/)

---

## 📋 Step-by-Step Commands

### STEP 1: Setup Project

```bash
cd /d/Trinity/SC
mkdir mouse-ml-service
cd mouse-ml-service

# Create subdirectories
mkdir models tests logs

# Copy files (listed above)
# ...

# Copy trained model
cp ../mouse-ml-training/mouse_bot_xgboost.pkl models/
cp ../mouse-ml-training/feature_extractor.py .
```

### STEP 2: Test Locally (Optional)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run server
python app.py

# Test in another terminal
python test_api.py
```

### STEP 3: Build Docker Image

```bash
# Build image
docker build -t mouse-ml-service .

# Run locally
docker run -p 5002:5002 mouse-ml-service

# Test
curl http://localhost:5002/health
```

### STEP 4: Deploy to Azure

```bash
# Make script executable (Linux/Mac)
chmod +x deploy-azure.sh

# Run deployment
./deploy-azure.sh

# OR manually (Windows):
# Follow commands in deploy-azure.sh one by one
```

### STEP 5: Get Your URL

After deployment completes, you'll see:

```
================================================
✅ DEPLOYMENT SUCCESSFUL!
================================================

📡 Service URL (HTTP):
   http://mouse-ml-vinol.eastus.azurecontainer.io:5002

🧪 Test endpoints:
   Health: http://mouse-ml-vinol.eastus.azurecontainer.io:5002/health
================================================
```

**Save this URL!** You'll use it in your Node.js backend.

---

## 🧪 Testing Your Deployment

```bash
# Test health
curl http://YOUR-URL:5002/health

# Test prediction
curl -X POST http://YOUR-URL:5002/predict \
  -H "Content-Type: application/json" \
  -d '{
    "events": [
      {"x":100,"y":200,"timestamp":0,"type":"move"},
      {"x":105,"y":205,"timestamp":16,"type":"move"}
    ]
  }'
```

---

## 📝 Checklist

Before deployment:
- [ ] All files copied to mouse-ml-service/
- [ ] Trained model in models/ directory
- [ ] feature_extractor.py present
- [ ] Docker installed
- [ ] Azure CLI installed (`az --version`)
- [ ] Logged into Azure (`az login`)

After deployment:
- [ ] Health check returns 200 OK
- [ ] Can make prediction requests
- [ ] URL saved for integration

---

## ⚠️ Troubleshooting

**Error: Model file not found**
```bash
# Verify model exists
ls -lh models/mouse_bot_xgboost.pkl
# Should show file size ~1-5 MB
```

**Error: Docker build fails**
```bash
# Check Docker is running
docker ps

# Rebuild without cache
docker build --no-cache -t mouse-ml-service .
```

**Error: Azure deployment fails**
```bash
# Check Azure CLI
az --version

# Re-login
az login

# Check resource group
az group list
```

**Service is slow to respond**
```bash
# Check container logs
az container logs \
  --resource-group mouse-ml-rg \
  --name mouse-ml-service

# Restart container
az container restart \
  --resource-group mouse-ml-rg \
  --name mouse-ml-service
```

---

## 🎯 Next Steps

After deployment:
1. Test all endpoints
2. Save your Azure URL
3. Integrate with Node.js backend (Phase 2)
4. Monitor performance in Azure Portal

---

## 💰 Cost Estimate

Azure Container Instance:
- CPU: 1 core
- Memory: 1.5 GB
- Cost: ~$30-40/month (always-on)
- Or: ~$0.05/hour (stop when not needed)

To stop container:
```bash
az container stop \
  --resource-group mouse-ml-rg \
  --name mouse-ml-service
```

To start container:
```bash
az container start \
  --resource-group mouse-ml-rg \
  --name mouse-ml-service
```

---

## 📧 Support

If you get stuck, check:
1. Container logs in Azure Portal
2. README.md for API documentation
3. test_api.py for usage examples
