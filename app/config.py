"""
Configuration settings for T-Mobile Installation Cost Prediction API
"""

import os
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application settings
    APP_NAME: str = "T-Mobile Installation Cost Prediction API"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="production", description="Environment: development, staging, production")
    
    # Server settings
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    
    # Security settings
    SECRET_KEY: str = Field(..., description="Secret key for JWT token signing")
    ALGORITHM: str = Field(default="HS256", description="JWT algorithm")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=15, description="Token expiration time in minutes")
    
    # Service account credentials (will be moved to vault in production)
    SERVICE_ACCOUNT_USERNAME: str = Field(..., description="Service account username")
    SERVICE_ACCOUNT_PASSWORD: str = Field(..., description="Service account password")
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins"
    )
    
    # ML Model settings
    MODEL_DIR: str = Field(default="models", description="Directory containing ML model files")
    
    # Logging settings
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    
    # Future vault integration settings (placeholder)
    VAULT_URL: str = Field(default="", description="HashiCorp Vault URL")
    VAULT_TOKEN: str = Field(default="", description="HashiCorp Vault token")
    VAULT_MOUNT_POINT: str = Field(default="secret", description="Vault mount point")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()
