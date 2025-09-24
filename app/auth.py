"""
Authentication and authorization module for T-Mobile Installation Cost Prediction API
"""

import jwt
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, status
import logging

from .config import settings

logger = logging.getLogger(__name__)


class AuthManager:
    """Handles authentication and JWT token management."""
    
    def __init__(self):
        self.secret_key = settings.SECRET_KEY
        self.algorithm = settings.ALGORITHM
        self.access_token_expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
        
        # In production, these would come from a secure vault
        self.service_accounts = {
            settings.SERVICE_ACCOUNT_USERNAME: self._hash_password(settings.SERVICE_ACCOUNT_PASSWORD)
        }
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256 with salt."""
        salt = "tmobile_ml_service_salt"  # In production, use random salt per password
        return hashlib.sha256((password + salt).encode()).hexdigest()
    
    def authenticate(self, username: str, password: str) -> bool:
        """
        Authenticate service account credentials.
        
        Args:
            username: Service account username
            password: Service account password
            
        Returns:
            bool: True if authentication successful, False otherwise
        """

        logger.info("username",username)
        logger.info("password",password)

        try:
            if username not in self.service_accounts:
                logger.warning(f"Unknown service account: {username}")
                return False
            
            hashed_password = self._hash_password(password)
            if secrets.compare_digest(self.service_accounts[username], hashed_password):
                logger.info(f"Authentication successful for: {username}")
                return True
            else:
                logger.warning(f"Invalid password for service account: {username}")
                return False
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
    
    def create_access_token(self, data: Dict[str, Any]) -> str:
        """
        Create JWT access token.
        
        Args:
            data: Payload data to include in token
            
        Returns:
            str: Encoded JWT token
        """
        try:
            to_encode = data.copy()
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
            
            to_encode.update({
                "exp": expire,
                "iat": datetime.utcnow(),
                "iss": "verb-installation-cost-ml-service",
                "aud": "verb-installation-cost-ml-api"
            })
            
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            return encoded_jwt
            
        except Exception as e:
            logger.error(f"Token creation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create access token"
            )
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Verify and decode JWT token.
        
        Args:
            token: JWT token to verify
            
        Returns:
            Dict[str, Any]: Decoded token payload
            
        Raises:
            HTTPException: If token is invalid or expired
        """
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm],
                audience="verb-installation-cost-ml-api",
                issuer="verb-installation-cost-ml-service"
            )
            
            # PyJWT automatically checks expiration during decode
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token verification failed"
            )


class VaultManager:
    """
    Placeholder for future HashiCorp Vault integration.
    This will replace the hardcoded service account credentials.
    """
    
    def __init__(self):
        self.vault_url = settings.VAULT_URL
        self.vault_token = settings.VAULT_TOKEN
        self.mount_point = settings.VAULT_MOUNT_POINT
    
    async def get_service_account_credentials(self, account_name: str) -> Optional[Dict[str, str]]:
        """
        Future implementation to retrieve service account credentials from Vault.
        
        Args:
            account_name: Name of the service account
            
        Returns:
            Dict containing username and password, or None if not found
        """
        # TODO: Implement Vault integration
        # This would use hvac library to connect to HashiCorp Vault
        # and retrieve credentials from the specified mount point
        pass
    
    async def get_secret(self, secret_path: str) -> Optional[Dict[str, Any]]:
        """
        Future implementation to retrieve secrets from Vault.
        
        Args:
            secret_path: Path to the secret in Vault
            
        Returns:
            Dict containing secret data, or None if not found
        """
        # TODO: Implement Vault secret retrieval
        pass
