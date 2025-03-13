# Set up the logging environment
from dendritic_modeling.logging_config import LoggerManager

logger_manager = LoggerManager()
logger = logger_manager.get_logger()

# Import utility functions
from dendritic_modeling.utils import safe_backward_hook, safe_register_hooks, register_safe_hook, SafeHookModule