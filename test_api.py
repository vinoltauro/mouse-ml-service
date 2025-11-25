"""
Test script for Mouse ML Service API
"""

import requests
import json
import time


# Configuration
BASE_URL = "http://localhost:5002"

# Sample mouse data (human-like)
sample_human_data = {
    "events": [
        {"x": 100, "y": 100, "timestamp": 0, "type": "move"},
        {"x": 105, "y": 103, "timestamp": 16, "type": "move"},
        {"x": 112, "y": 108, "timestamp": 32, "type": "move"},
        {"x": 120, "y": 115, "timestamp": 48, "type": "move"},
        {"x": 130, "y": 125, "timestamp": 64, "type": "move"},
        {"x": 142, "y": 137, "timestamp": 80, "type": "move"},
        {"x": 156, "y": 151, "timestamp": 96, "type": "move"},
        {"x": 172, "y": 167, "timestamp": 112, "type": "move"},
        {"x": 190, "y": 185, "timestamp": 128, "type": "move"},
        {"x": 210, "y": 205, "timestamp": 144, "type": "move"},
        {"x": 232, "y": 227, "timestamp": 160, "type": "move"},
        {"x": 256, "y": 251, "timestamp": 176, "type": "move"},
        {"x": 280, "y": 275, "timestamp": 192, "type": "down"},
        {"x": 281, "y": 276, "timestamp": 292, "type": "up"},
        {"x": 285, "y": 280, "timestamp": 308, "type": "move"},
    ]
}

# Sample bot data (straight line, instant click)
sample_bot_data = {
    "events": [
        {"x": 100, "y": 100, "timestamp": 0, "type": "move"},
        {"x": 110, "y": 110, "timestamp": 10, "type": "move"},
        {"x": 120, "y": 120, "timestamp": 20, "type": "move"},
        {"x": 130, "y": 130, "timestamp": 30, "type": "move"},
        {"x": 140, "y": 140, "timestamp": 40, "type": "move"},
        {"x": 150, "y": 150, "timestamp": 50, "type": "move"},
        {"x": 160, "y": 160, "timestamp": 60, "type": "move"},
        {"x": 170, "y": 170, "timestamp": 70, "type": "move"},
        {"x": 180, "y": 180, "timestamp": 80, "type": "move"},
        {"x": 190, "y": 190, "timestamp": 90, "type": "move"},
        {"x": 200, "y": 200, "timestamp": 100, "type": "move"},
        {"x": 210, "y": 210, "timestamp": 110, "type": "move"},
        {"x": 220, "y": 220, "timestamp": 120, "type": "down"},
        {"x": 220, "y": 220, "timestamp": 125, "type": "up"},  # 5ms click!
        {"x": 230, "y": 230, "timestamp": 135, "type": "move"},
    ]
}


def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("TEST 1: Health Check")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("✅ PASSED")
        else:
            print("❌ FAILED")
    except Exception as e:
        print(f"❌ ERROR: {e}")


def test_root():
    """Test root endpoint"""
    print("\n" + "="*60)
    print("TEST 2: Root Endpoint")
    print("="*60)
    
    try:
        response = requests.get(BASE_URL)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("✅ PASSED")
        else:
            print("❌ FAILED")
    except Exception as e:
        print(f"❌ ERROR: {e}")


def test_predict_human():
    """Test prediction with human data"""
    print("\n" + "="*60)
    print("TEST 3: Predict Human Movement")
    print("="*60)
    
    try:
        start = time.time()
        response = requests.post(
            f"{BASE_URL}/predict",
            json=sample_human_data
        )
        latency = (time.time() - start) * 1000
        
        print(f"Status Code: {response.status_code}")
        print(f"Latency: {latency:.2f}ms")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            result = response.json()
            if result['prediction'] == 'human' and result['humanityScore'] > 50:
                print("✅ PASSED - Correctly identified as human")
            else:
                print("⚠️  WARNING - Identified as bot (may be false positive)")
        else:
            print("❌ FAILED")
    except Exception as e:
        print(f"❌ ERROR: {e}")


def test_predict_bot():
    """Test prediction with bot data"""
    print("\n" + "="*60)
    print("TEST 4: Predict Bot Movement")
    print("="*60)
    
    try:
        start = time.time()
        response = requests.post(
            f"{BASE_URL}/predict",
            json=sample_bot_data
        )
        latency = (time.time() - start) * 1000
        
        print(f"Status Code: {response.status_code}")
        print(f"Latency: {latency:.2f}ms")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            result = response.json()
            if result['prediction'] == 'bot' and result['humanityScore'] < 50:
                print("✅ PASSED - Correctly identified as bot")
            else:
                print("⚠️  WARNING - Identified as human (may be false negative)")
        else:
            print("❌ FAILED")
    except Exception as e:
        print(f"❌ ERROR: {e}")


def test_predict_explain():
    """Test prediction with explanations"""
    print("\n" + "="*60)
    print("TEST 5: Predict with Explanations")
    print("="*60)
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict-explain",
            json=sample_bot_data
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            result = response.json()
            if 'explanations' in result:
                print("✅ PASSED - Explanations included")
            else:
                print("❌ FAILED - No explanations")
        else:
            print("❌ FAILED")
    except Exception as e:
        print(f"❌ ERROR: {e}")


def test_batch_predict():
    """Test batch prediction"""
    print("\n" + "="*60)
    print("TEST 6: Batch Prediction")
    print("="*60)
    
    try:
        batch_data = {
            "batch": [
                sample_human_data,
                sample_bot_data
            ],
            "explain": False
        }
        
        response = requests.post(
            f"{BASE_URL}/predict-batch",
            json=batch_data
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            result = response.json()
            if result['count'] == 2:
                print("✅ PASSED - Batch processing successful")
            else:
                print("❌ FAILED - Incorrect count")
        else:
            print("❌ FAILED")
    except Exception as e:
        print(f"❌ ERROR: {e}")


def test_metrics():
    """Test metrics endpoint"""
    print("\n" + "="*60)
    print("TEST 7: Metrics Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/metrics")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("✅ PASSED")
        else:
            print("❌ FAILED")
    except Exception as e:
        print(f"❌ ERROR: {e}")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 MOUSE ML SERVICE - API TESTS")
    print("="*60)
    print(f"Target: {BASE_URL}")
    
    test_health()
    test_root()
    test_predict_human()
    test_predict_bot()
    test_predict_explain()
    test_batch_predict()
    test_metrics()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
