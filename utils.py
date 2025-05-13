"""
Utilities for API key verification, configuration management, and logging.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Set # Added Set for type hinting

from dotenv import load_dotenv
from fastapi import Depends, Header, HTTPException, status # Added status for HTTP status codes

# Assuming settings.py defines a Settings class and a global settings instance
from settings import Settings, settings as global_settings # Renamed imported settings

# --- Logging Configuration ---
# It's good practice to configure logging once at the application's entry point (e.g., main.py)
# or through a dedicated logging configuration module.
# However, if this utils.py is the central place for it, ensure it's called early.

def configure_logging(log_level: str | int = logging.INFO,
                      log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
                      ) -> None:
    """
    Configures basic logging for the application.
    Allows customization of log level and format.

    Args:
        log_level: The logging level (e.g., logging.INFO, "DEBUG").
        log_format: The format string for log messages.
    """
    # Ensure basicConfig is called only once
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=log_level,
            format=log_format,
            datefmt="%Y-%m-%d %H:%M:%S", # Added datefmt for consistency
        )
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING) # Quieter Uvicorn access logs
        logging.getLogger("uvicorn.error").setLevel(logging.INFO)
        logging.getLogger(__name__).info("Logging configured successfully.")
    else:
        logging.getLogger(__name__).debug("Logging already configured.")

# Call logging configuration when this module is imported.
# Consider moving this to main.py or an app initialization function for more control.
# configure_logging(log_level=global_settings.LOG_LEVEL if hasattr(global_settings, 'LOG_LEVEL') else logging.INFO)

logger = logging.getLogger(__name__)

# --- API Key Management ---
# Loading API keys directly at module level can be okay for simple setups,
# but for more complex applications, consider loading them within the Settings class
# or a dedicated configuration loader.

def load_api_keys_from_env() -> Set[str]:
    """
    Loads API keys from environment variables.
    Expects API_KEY (single) or API_KEYS (comma-separated list).

    Returns:
        A set of valid API keys.

    Raises:
        RuntimeError: If no API keys are configured.
    """
    raw_api_key = os.getenv("API_KEY")
    raw_api_keys_list = os.getenv("API_KEYS") # For multiple keys

    keys: Set[str] = set()

    if raw_api_key:
        keys.add(raw_api_key.strip())

    if raw_api_keys_list:
        for key in raw_api_keys_list.split(','):
            stripped_key = key.strip()
            if stripped_key:
                keys.add(stripped_key)

    if not keys:
        logger.critical("CRITICAL: No API_KEY or API_KEYS found in environment variables. API will be inaccessible.")
        raise RuntimeError("API key(s) not configured. Service cannot start securely.")
    logger.info(f"Loaded {len(keys)} API key(s) for validation.")
    return keys

# Load.env file to make environment variables available
load_dotenv()

# Initialize valid API keys once
# Consider making this part of the Settings class or a configuration object
# to avoid global state if possible.
try:
    VALID_API_KEYS: Set[str] = load_api_keys_from_env()
    # For case-insensitive comparison, store lowercase versions if needed,
    # but it's generally better to enforce case-sensitivity for API keys.
    # If case-insensitivity is a strict requirement:
    # VALID_API_KEYS_LOWER: Set[str] = {key.lower() for key in VALID_API_KEYS}
except RuntimeError as e:
    # This will stop the application from starting if keys are not set, which is good.
    logger.critical(f"Application startup failed due to missing API key configuration: {e}")
    raise

# --- Settings Dependency ---

def get_settings() -> Settings:
    """
    Dependency function to provide the global Settings instance.
    Ensures settings are loaded and validated.
    """
    # settings.validate_settings() # Assuming validation is done on Settings instantiation or elsewhere
    return global_settings

# --- API Key Verification Dependency ---

async def verify_api_key(
    x_api_key: Optional[str] = Header(None, description="The API key for authentication."),
) -> str:
    """
    Verify the API key provided in the 'X-API-Key' header.

    This is a FastAPI dependency that can be used to protect routes.

    Args:
        x_api_key: API key from the 'X-API-Key' header.

    Returns:
        The validated API key if it's valid.

    Raises:
        HTTPException:
            - 401 (Unauthorized): If the API key is missing.
            - 403 (Forbidden): If the API key is invalid.
    """
    # TODO: [1] Implement robust rate limiting (e.g., using slowapi or a Redis-backed counter).
    # TODO: [1] Implement key rotation strategy (e.g., support multiple valid keys, plan for deprecation).

    if not x_api_key:
        logger.warning("API key missing in request header 'X-API-Key'.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, # Use status codes from fastapi.status
            detail="API key missing in X-API-Key header. Authentication required.",
            headers={"WWW-Authenticate": "Header"}, # Standard for 401
        )

    # Perform a direct, case-sensitive comparison.
    # If case-insensitivity was a requirement (generally not recommended for API keys):
    # if x_api_key.lower() not in VALID_API_KEYS_LOWER:
    if x_api_key not in VALID_API_KEYS:
        logger.warning(f"Invalid API key provided: '{x_api_key[:5]}...'") # Log only a prefix for security
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key provided. Access denied.",
        )

    logger.debug(f"API key validated successfully: '{x_api_key[:5]}...'")
    return x_api_key

# Example of how to use it in main.py:
# from.utils import verify_api_key
# @app.get("/secure-data", dependencies=)
# async def get_secure_data():
#     return {"message": "This is secure data."}
