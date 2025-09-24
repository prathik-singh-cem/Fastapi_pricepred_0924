"""
T-Mobile Installation Cost Prediction API
A production-ready enterprise-secure FastAPI microservice for ML-powered installation cost prediction.
"""

from fastapi import FastAPI, Depends, HTTPException, status, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import logging
from datetime import datetime, timedelta
from typing import Optional
import os

from app.auth import AuthManager
from app.models import PredictionRequest, PredictionResponse, TokenResponse
from app.ml_service import MLService
from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global ML service instance
ml_service: Optional[MLService] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    global ml_service
    
    # Startup
    logger.info("Starting T-Mobile Installation Cost Prediction API...")
    try:
        ml_service = MLService()
        await ml_service.initialize()
        logger.info("ML models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ML service: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down T-Mobile Installation Cost Prediction API...")

# Create FastAPI app
app = FastAPI(
    title="T-Mobile Installation Cost Prediction API",
    description="Enterprise-secure ML microservice for installation cost prediction",
    version="1.0.0",
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT == "development" else None,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
auth_manager = AuthManager()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate JWT token and return user info."""
    try:
        payload = auth_manager.verify_token(credentials.credentials)
        return payload
    except Exception as e:
        logger.warning(f"Token validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "T-Mobile Installation Cost Prediction API",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    global ml_service
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "api": "healthy",
            "ml_models": "healthy" if ml_service and ml_service.is_ready() else "unhealthy"
        }
    }
    
    if not ml_service or not ml_service.is_ready():
        health_status["status"] = "unhealthy"
        return health_status, 503
    
    return health_status

@app.post("/token", response_model=TokenResponse)
async def login(service_account: str = Form(...), password: str = Form(...)):
    """
    Authenticate with service account credentials and receive JWT token.
    Token is valid for 15 minutes.
    """
    try:
        if not auth_manager.authenticate(service_account, password):
            logger.warning(f"Authentication failed for service account: {service_account}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid service account credentials"
            )
        
        # Generate JWT token
        token = auth_manager.create_access_token(
            data={"sub": service_account, "type": "service_account"}
        )
        
        logger.info(f"Token generated for service account: {service_account}")
        
        return TokenResponse(
            access_token=token,
            token_type="bearer",
            expires_in=900  # 15 minutes
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during authentication"
        )

@app.post("/predict", response_model=PredictionResponse)
async def predict_installation_cost(
    request: PredictionRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Predict installation cost based on equipment and installation parameters.
    Requires valid JWT token.
    """
    global ml_service
    
    try:
        if not ml_service or not ml_service.is_ready():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ML service is not available"
            )
        
        logger.info(f"Prediction request from user: {current_user.get('sub')}")
        
        # Make prediction
        result = await ml_service.predict(request)
        
        logger.info(f"Prediction completed: {result.prediction}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during prediction"
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development",
        log_level="info"
    )
