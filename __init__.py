from dotenv import load_dotenv
import os
from decouple import Config, RepositoryEnv
from pathlib import Path
from logging import getLogger

# Load .env file
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

logger = getLogger(__name__)

class SecureConfig:
    @staticmethod
    def get_api_key(key_name, required=True):
        value = os.getenv(key_name)
        if not value and required:
            logger.error(f"Missing API Key: {key_name}")
            raise KeyError(f"API Key '{key_name}' not set in environment variables")
        return value
