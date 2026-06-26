import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class to hold environment variables."""
    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    AZURE_DOCUMENT_INTELLIGENCE_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

    @classmethod
    def validate(cls):
        """Validate that all required configuration variables are set."""
        if not cls.AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT:
            raise ValueError("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT is not set in the environment.")
        if not cls.AZURE_DOCUMENT_INTELLIGENCE_KEY:
            raise ValueError("AZURE_DOCUMENT_INTELLIGENCE_KEY is not set in the environment.")
