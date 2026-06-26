import logging
import sys

def setup_logger(name: str = "sequence_detector") -> logging.Logger:
    """Sets up and returns a customized logger."""
    logger = logging.getLogger(name)
    
    # Only configure if not already configured to avoid duplicate handlers
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)
        
        # Force utf-8 encoding for Windows consoles
        if sys.platform == 'win32':
            sys.stdout.reconfigure(encoding='utf-8')
            
        # Create console handler with a higher log level
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        
        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        
        # Add the handlers to the logger
        logger.addHandler(ch)
        
    return logger

logger = setup_logger()
