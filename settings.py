import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

class Settings:
    RISE_DB_URL = os.getenv("RISE_DB_URL")  # RISE container
    BACKTEST_DB_URL = os.getenv("BACKTEST_DB_URL")  # FlatFire container 
    API_KEY = os.getenv("API_KEY")
    PORT = int(os.getenv("PORT", "8000"))

settings = Settings()

# Validate critical settings
if not settings.RISE_DB_URL:
    raise ValueError("RISE_DB_URL environment variable is not set")
if not settings.FLATFIRE_DB_URL:
    raise ValueError("FLATFIRE_DB_URL environment variable is not set")
if not settings.API_KEY:
    raise ValueError("API_KEY environment variable is not set")
