import sys
import logging

LOGGER_NAME = 'dendritic_modeling'

def setup_logging():
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)
    
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(levelname)s  %(message)s')
        console_handler.setFormatter(console_format)
    
        file_handler = logging.FileHandler('dendritic_modeling.log')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        file_handler.setFormatter(file_format)
    
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger