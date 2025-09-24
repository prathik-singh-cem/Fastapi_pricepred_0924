# T-Mobile VERB Installation Cost Prediction API

A production-ready enterprise-secure FastAPI microservice for ML-powered installation cost prediction.

## Features

- 🔐 **Enterprise Security**: JWT-based authentication with service account credentials
- 🤖 **ML-Powered**: Uses Random Forest and XGBoost models for cost prediction
- 🎯 **Solution-Specific**: DAS-specific models with custom weighting formulas
- 📊 **Confidence Scoring**: Automatic confidence level calculation
- 🚀 **Production-Ready**: Docker support, health checks, structured logging
- 🔧 **Vault-Ready**: Prepared for HashiCorp Vault integration

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Setup

1. **Clone or navigate to the project directory**:

   ```bash
   cd /path/to/fastapi_ml_service
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:

   **Linux/macOS:**

   ```bash
   source venv/bin/activate
   ```

   **Windows:**

   ```cmd
   venv\Scripts\activate
   ```

4. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

### Alternative: Windows Batch Script

For Windows users, you can use the provided batch script:

```cmd
install_windows.bat
```

This script automates the virtual environment creation and dependency installation.

## Configuration

1. **Copy the environment template**:

   ```bash
   cp .env.example .env
   ```

   **Windows (Command Prompt):**

   ```cmd
   copy .env.example .env
   ```

2. **Edit the `.env` file** with your configuration if needed. The default values should work for most setups.

## Running the Application

1. **Ensure your virtual environment is activated**:

   **Linux/macOS:**

   ```bash
   source venv/bin/activate
   ```

   **Windows:**

   ```cmd
   venv\Scripts\activate
   ```

2. **Start the FastAPI server**:

   ```bash
   uvicorn main:app --reload
   ```

   The API will be available at `http://localhost:8000`

3. **View API documentation**:
   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

## API Usage

### 1. Authentication

Get a JWT token (valid for 15 minutes):

```bash
curl -X POST "http://localhost:8000/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "service_account=verb-installation-cost-ml-service&password=TM0bile2025!SecurePass"
```

venv\Scripts\activate
uvicorn main:app --reload
curl.exe -X POST "http://localhost:8000/token" `
  -H "Content-Type: application/x-www-form-urlencoded" `
  -d "service_account=verb-installation-cost-ml-service&password=TM0bile2025!SecurePass"


$result = Invoke-RestMethod -Uri "http://localhost:8000/predict" `
-Method Post `
-Headers @{ Authorization = "Bearer $token" } `
-ContentType "application/json" `
-Body $json

$result

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 900
}
```

### 2. Cost Prediction

Use the token to make predictions:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -H "Content-Type: application/json" \
  -d @request_payload.json
```

## API Endpoints

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|---------------|
| `/` | GET | Health check | No |
| `/health` | GET | Detailed health status | No |
| `/token` | POST | Get JWT token | No |
| `/predict` | POST | Predict installation cost | Yes |
| `/docs` | GET | API documentation (dev only) | No |

## Request/Response Format

### Prediction Request
```json
{
  "name": "Project Name",
  "origin": "source_file.xlsx",
  "equipment": [
    {
      "type": "Cable",
      "manufacturer": "Generic",
      "model": "CAT-6",
      "description": "CAT-6 cable",
      "qty": 770.35,
      "qtyType": "feet"
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
```

### Prediction Response
```json
{
  "success": true,
  "prediction": 332280,
  "confidence": "Medium",
  "breakdown": {
    "randomForest": 315000,
    "xgboost": 358200
  },
  "timestamp": "2025-07-24T19:38:27.000Z"
}
```

## Model Logic

- **DAS Solutions**: `final_price = rf_prediction * 0.8 + xgb_prediction * 0.2`
- **Other Solutions**: `final_price = (rf_prediction + xgb_prediction) / 2`

## Security Features

- JWT token authentication (15-minute expiry)
- Service account-based access control
- CORS protection
- Input validation with Pydantic models
- Prepared for HashiCorp Vault integration

## Production Deployment

### Docker

```bash
# Build image
docker build -t tmobile-ml-api .

# Run container
docker run -p 8000:8000 --env-file .env tmobile-ml-api
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Environment (development/production) | `production` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `SECRET_KEY` | JWT signing key | Required |
| `SERVICE_ACCOUNT_USERNAME` | Service account username | Required |
| `SERVICE_ACCOUNT_PASSWORD` | Service account password | Required |

## Monitoring

- Health check endpoint: `/health`
- Structured logging with timestamps
- Request/response logging
- Error tracking and reporting

## Future Enhancements

- [ ] HashiCorp Vault integration for secrets management
- [ ] Prometheus metrics endpoint
- [ ] Rate limiting
- [ ] Request caching
- [ ] Model versioning
- [ ] A/B testing framework

## Support

For technical support or questions, contact the AMDOCS Development Team.
