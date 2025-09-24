"""
Basic test script for T-Mobile Installation Cost Prediction API
"""

import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"

# Test payload (matching ML_Payload.json format)
test_payload = {
    "name": "Campus University of Virginia VAYH050A Library",
    "origin": "Campus_University of Virginia_VAYH050A_Library.xlsx",
    "equipment": [
        {
            "type": "Cable",
            "componentGroup": "",
            "manufacturer": "Generic",
            "model": "CAT-6",
            "description": "CAT-6 - 24 AWG min. - 100m Maximum Cable Length",
            "inventory": "N/A",
            "qty": 770.35,
            "qtyType": "feet",
            "shared": True,
            "numberOfOtherOperators": 2
        },
        {
            "type": "Connector",
            "componentGroup": "",
            "manufacturer": "Generic",
            "model": "RJ-45",
            "description": "RJ-45 connector",
            "inventory": "N/A",
            "qty": 8,
            "qtyType": None
             "shared": True,
            "numberOfOtherOperators": 2
        },
        {
            "type": "Network Equipment",
            "componentGroup": "",
            "manufacturer": "Extreme Networks",
            "model": "X440-G2-12p-10GE4",
            "description": "X440-G2 12 10/100/1000BASE-T POE+, 4 1GbE unpopulated SFP upgradable to 10GbE SFP+",
            "inventory": "N/A",
            "qty": 1,
            "qtyType": None
            "shared": True,
            "numberOfOtherOperators": 2
        },
        {
            "type": "Radio Transceiver",
            "componentGroup": "",
            "manufacturer": "Airspan",
            "model": "AV1500 (Ceiling mount)",
            "description": "Airvelocity 1500 (Ceiling mount)",
            "inventory": "N/A",
            "qty": 4,
            "qtyType": None
            "shared": True,
            "numberOfOtherOperators": 2
        }
    ],
    "installation": {
        "venueType": "industrial",
        "solutionType": "das",
        "totalCoverageArea": 10000,
        "numberOfFloors": 1,
        "numberOfBuildings": 1
    }
}

def test_health_check():
    """Test health check endpoint."""
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_authentication():
    """Test authentication endpoint."""
    print("Testing authentication...")
    
    # Test with correct credentials
    auth_data = {
        "service_account": "verb-installation-cost-ml-service",
        "password": "TM0bile2025!SecurePass"
    }
    
    response = requests.post(f"{BASE_URL}/token", data=auth_data)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        token_data = response.json()
        print(f"Token received: {token_data['access_token'][:50]}...")
        return token_data['access_token']
    else:
        print(f"Authentication failed: {response.text}")
        return None

def test_prediction(token):
    """Test prediction endpoint."""
    if not token:
        print("No token available for prediction test")
        return
    
    print("Testing prediction...")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(f"{BASE_URL}/predict", headers=headers, json=test_payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        prediction_data = response.json()
        print(f"Prediction successful!")
        print(f"Final Cost: ${prediction_data['prediction']:,.2f}")
        print(f"Confidence: {prediction_data['confidence']}")
        print(f"RF Prediction: ${prediction_data['breakdown']['randomForest']:,.2f}")
        print(f"XGBoost Prediction: ${prediction_data['breakdown']['xgboost']:,.2f}")
    else:
        print(f"Prediction failed: {response.text}")

def test_unauthorized_access():
    """Test unauthorized access to prediction endpoint."""
    print("Testing unauthorized access...")
    
    response = requests.post(f"{BASE_URL}/predict", json=test_payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("T-Mobile Installation Cost Prediction API Test Suite")
    print("=" * 60)
    print()
    
    # Test 1: Health check
    test_health_check()
    
    # Test 2: Unauthorized access
    test_unauthorized_access()
    
    # Test 3: Authentication
    token = test_authentication()
    print()
    
    # Test 4: Prediction with valid token
    if token:
        test_prediction(token)
    
    print()
    print("Test suite completed!")
