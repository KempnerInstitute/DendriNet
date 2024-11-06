"""
logging_config.py
=================
This module contains the logging configuration for the dendritic_modeling 
package.
"""

import sys
import logging

LOGGER_NAME = 'dendritic_modeling'
logger = logging.getLogger(LOGGER_NAME)

def setup_logging(file_path="dendritic_modeling.log"):
    global logger
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)
    
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(levelname)s  %(message)s')
        console_handler.setFormatter(console_format)
    
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        file_handler.setFormatter(file_format)
    
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    update_file_handler(file_path)

def update_file_handler(file_path):
    """
    Updates the file handler to the new file path. Removes any existing
    file handler first.
    """
    global logger
    # Remove existing file handlers
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)

    # Create a new file handler with the updated path
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
