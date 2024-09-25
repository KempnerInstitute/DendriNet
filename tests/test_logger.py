import sys
import logging
import unittest
from unittest.mock import patch, MagicMock
from dendritic_modeling.logging_config import setup_logging, LOGGER_NAME


class TestLoggingConfig(unittest.TestCase):
    
    @patch('dendritic_modeling.logging_config.logging.getLogger')  # Mock getLogger
    @patch('dendritic_modeling.logging_config.logging.FileHandler')  # Mock FileHandler
    @patch('dendritic_modeling.logging_config.logging.StreamHandler')  # Mock StreamHandler
    def test_setup_logging(self, mock_stream_handler, mock_file_handler, mock_get_logger):
        # Mocking the logger and handlers
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # Mock the stream and file handlers
        mock_stream = MagicMock()
        mock_file = MagicMock()
        mock_stream_handler.return_value = mock_stream
        mock_file_handler.return_value = mock_file
        
        mock_logger.hasHandlers.return_value = False


        # Call the function under test
        logger = setup_logging()
        
        # Assertions

        # 1. Ensure getLogger was called with LOGGER_NAME
        mock_get_logger.assert_called_with(LOGGER_NAME)

        # 2. Ensure setLevel was called with DEBUG level
        mock_logger.setLevel.assert_called_with(logging.DEBUG)

        # 3. Ensure StreamHandler and FileHandler were created
        mock_stream_handler.assert_called_once_with(sys.stdout)
        mock_file_handler.assert_called_once_with('dendritic_modeling.log')

        # 4. Ensure the handlers were added to the logger
        mock_logger.addHandler.assert_any_call(mock_stream)
        mock_logger.addHandler.assert_any_call(mock_file)

        # 5. Check formatters were set on both handlers
        mock_stream.setFormatter.assert_called_once()
        mock_file.setFormatter.assert_called_once()

        # 6. Ensure handlers are added only once
        self.assertEqual(mock_logger.addHandler.call_count, 2)


if __name__ == '__main__':
    unittest.main()
