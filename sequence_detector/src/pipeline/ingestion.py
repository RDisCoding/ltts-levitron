import os
from typing import Any, Dict
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from src.utils.config import Config
from src.utils.logger import logger

class DocumentIngestionClient:
    """Client to handle communication with Azure AI Document Intelligence."""
    
    def __init__(self):
        Config.validate()
        self.endpoint = Config.AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT
        self.key = Config.AZURE_DOCUMENT_INTELLIGENCE_KEY
        
        self.client = DocumentIntelligenceClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.key)
        )
        logger.info("Initialized DocumentIngestionClient successfully.")

    def analyze_document(self, file_path: str) -> Dict[str, Any]:
        """
        Sends the PDF to Azure Document Intelligence and returns the structured layout response.
        We use the prebuilt-layout model to extract tables, cells, and geometry.
        """
        logger.info(f"Starting analysis for document: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with open(file_path, "rb") as f:
                poller = self.client.begin_analyze_document(
                    model_id="prebuilt-layout",
                    analyze_request=f,
                    content_type="application/pdf"
                )
            
            result = poller.result()
            logger.info(f"Successfully analyzed document: {file_path}")
            
            # The result is returned as an AnalyzeResult object, we can convert it to a dict
            return result.as_dict()
            
        except Exception as e:
            logger.error(f"Failed to analyze document via Azure API: {str(e)}")
            raise e
