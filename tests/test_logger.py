import unittest

from dendritic_modeling import LoggerManager


class TestLoggerManager(unittest.TestCase):
    def setUp(self):
        # Reset singleton instance before each test
        LoggerManager._instance = None

    def test_singleton_instance(self):
        """Test that only one instance of LoggerManager is created."""
        instance1 = LoggerManager().get_logger()
        instance2 = LoggerManager().get_logger()
        self.assertIs(instance1, instance2, "LoggerManager should be a singleton")

    def test_logging(self):
        """Test logging message output."""
        logger = LoggerManager().get_logger()
        # Use the logger name rather than root
        with self.assertLogs(logger.name, level="INFO") as cm:
            logger.info("Test message")
        self.assertIn("Test message", str(cm.output))


if __name__ == "__main__":
    unittest.main()
