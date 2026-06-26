import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class to hold environment variables for Claude."""
    CLAUDE_AZURE_ENDPOINT = os.getenv("CLAUDE_AZURE_ENDPOINT")
    CLAUDE_AZURE_KEY = os.getenv("CLAUDE_AZURE_KEY")
    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL")

    @classmethod
    def validate(cls):
        """Validate that all required configuration variables are set."""
        if not cls.CLAUDE_AZURE_ENDPOINT:
            raise ValueError("CLAUDE_AZURE_ENDPOINT is not set in the environment.")
        if not cls.CLAUDE_AZURE_KEY:
            raise ValueError("CLAUDE_AZURE_KEY is not set in the environment.")
        if not cls.CLAUDE_MODEL:
            logger.warning("CLAUDE_MODEL is not set in the environment. API call may fail if required by the endpoint.")
