import os
import json
import requests
import fitz  # PyMuPDF
from typing import Any, Dict
from src.utils.config import Config
from src.utils.logger import logger

class DocumentIngestionClient:
    """Mock Client to handle communication with an Azure Claude endpoint."""
    
    def __init__(self):
        Config.validate()
        self.endpoint = Config.CLAUDE_AZURE_ENDPOINT
        self.key = Config.CLAUDE_AZURE_KEY
        self.model = Config.CLAUDE_MODEL
        logger.info("Initialized Mock DocumentIngestionClient (Claude) successfully.")

    def analyze_document(self, file_path: str) -> Dict[str, Any]:
        """
        Extracts text from the PDF and uses Claude to hallucinate/structure the exact 
        Azure Document Intelligence JSON format for downstream processing.
        """
        logger.info(f"Starting mock analysis (Claude) for document: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        # 1. Extract text using PyMuPDF
        extracted_text = self._extract_text_from_pdf(file_path)
        logger.debug(f"Extracted {len(extracted_text)} characters from PDF.")

        # 2. Query Claude
        try:
            return self._query_claude_for_json(extracted_text)
        except Exception as e:
            logger.error(f"Failed to analyze document via Claude API: {str(e)}")
            raise e

    def _extract_text_from_pdf(self, file_path: str) -> str:
        text = ""
        try:
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text() + "\n"
        except Exception as e:
            logger.error(f"PyMuPDF failed to read {file_path}: {e}")
            raise
        return text

    def _query_claude_for_json(self, document_text: str) -> Dict[str, Any]:
        """Sends the text to Claude with strict instructions to output the Azure format."""
        
        system_prompt = (
            "You are an OCR extraction engine designed to perfectly mimic the output of "
            "Azure AI Document Intelligence. Your ONLY job is to extract the single 'Sequence of Operations' "
            "or 'Decision Matrix' table (which contains Rooms mapped to Devices/Actions with boolean markers like X, Yes, True). "
            "IGNORE all other tables such as panel schedules, relay schedules, or equipment lists to save tokens. "
            "Output pure, valid JSON matching this exact schema, with no markdown formatting:\n"
            "{\n"
            '  "tables": [\n'
            "    {\n"
            '      "rowCount": <int>,\n'
            '      "columnCount": <int>,\n'
            '      "cells": [\n'
            "        {\n"
            '          "rowIndex": <int>,\n'
            '          "columnIndex": <int>,\n'
            '          "content": "<string>",\n'
            '          "kind": "<optional string like columnHeader>",\n'
            '          "rowSpan": <int default 1>,\n'
            '          "columnSpan": <int default 1>\n'
            "        }\n"
            "      ]\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "DO NOT include markdown code blocks like ```json. ONLY output the raw JSON."
        )

        headers = {
            "Content-Type": "application/json",
            "api-key": self.key,
            "x-api-key": self.key,
            "anthropic-version": "2023-06-01"
        }

        payload = {
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": f"Extract ONLY the first 3 rows of the sequence of operations decision matrix table from this text (to avoid hitting token limits):\n\n{document_text}"
                }
            ],
            "max_tokens": 8192,
            "temperature": 0.0
        }
        
        if self.model:
            payload["model"] = self.model

        logger.info("Sending request to Claude endpoint...")
        response = requests.post(self.endpoint, headers=headers, json=payload)
        
        if response.status_code != 200:
            logger.error(f"Claude API Error: {response.status_code} - {response.text}")
            raise Exception(f"Claude API Error: {response.status_code}")

        response_data = response.json()
        
        # Extract the text content from the response
        try:
            # Handle typical Anthropic response structure
            if "content" in response_data and isinstance(response_data["content"], list):
                raw_json_str = response_data["content"][0]["text"]
            # Handle OpenAI-like wrapper structure if Azure uses it
            elif "choices" in response_data:
                raw_json_str = response_data["choices"][0]["message"]["content"]
            else:
                logger.error(f"Unexpected response format: {response_data}")
                raise ValueError("Unexpected response format from Claude.")
                
            # Clean up potential markdown blocks if Claude disobeys
            raw_json_str = raw_json_str.strip()
            if raw_json_str.startswith("```json"):
                raw_json_str = raw_json_str[7:]
            if raw_json_str.endswith("```"):
                raw_json_str = raw_json_str[:-3]
                
            return json.loads(raw_json_str.strip())
            
        except json.JSONDecodeError as e:
            logger.error(f"Claude did not return valid JSON: {raw_json_str}")
            raise Exception("Failed to parse Claude output as JSON.")
