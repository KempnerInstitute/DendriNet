"""
logging_config.py
=================
Logging setup for programs that use the dendritic_modeling package.

Importing the package configures nothing: its loggers have only a
``NullHandler``, as library logging should. A program (the training,
analysis and sweep command lines) calls :func:`setup_logging` to print the
package's records to stdout and, optionally, to write them to
``dendritic_modeling.log`` in a log directory. The program's own module logs
as ``__main__`` when it runs as a script or with ``python -m``, outside the
package's logger tree, so its records go to the same handlers.
"""

import logging
import os
import sys

LOGGER_NAME = "dendritic_modeling"
MAIN_LOGGER_NAME = "__main__"
LOG_FILE_NAME = LOGGER_NAME + ".log"
_CONSOLE_FORMAT = "%(levelname)s %(filename)s:%(lineno)d  %(message)s"
_FILE_FORMAT = "%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s"


def _package_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def _program_loggers() -> tuple[logging.Logger, logging.Logger]:
    """The package logger and the running program's ``__main__`` logger."""
    return _package_logger(), logging.getLogger(MAIN_LOGGER_NAME)


def _set_file_handler(file_path: str, level: int = logging.DEBUG) -> None:
    """Write the program's records to ``file_path``, closing any previous file."""
    closed = set()
    for logger in _program_loggers():
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
                if id(handler) not in closed:
                    handler.close()
                    closed.add(id(handler))
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    handler = logging.FileHandler(file_path)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FILE_FORMAT))
    for logger in _program_loggers():
        logger.addHandler(handler)


def setup_logging(
    log_dir=None,
    debug=False,
    *,
    console=True,
    file_level=logging.DEBUG,
):
    """Configure the package's logging for a program; calling it again is safe.

    Args:
        log_dir: Directory for ``dendritic_modeling.log``. Without it, the
            ``DENDRITIC_LOG_DIR`` environment variable is used (a relative
            path resolves against the current directory); without either, no
            file is written.
        debug: Print DEBUG records to the console (INFO otherwise).
        console: Print records to stdout.
        file_level: Lowest level written to the file.

    Returns:
        logging.Logger: The package logger.
    """
    loggers = _program_loggers()
    for logger in loggers:
        logger.setLevel(logging.DEBUG)
    if console:
        handler = next(
            (
                h
                for logger in loggers
                for h in logger.handlers
                if getattr(h, "_dendritic_console", False)
            ),
            None,
        )
        if handler is None:
            handler = logging.StreamHandler(sys.stdout)
            handler._dendritic_console = True
            handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT))
        handler.setLevel(logging.DEBUG if debug else logging.INFO)
        for logger in loggers:
            if handler not in logger.handlers:
                logger.addHandler(handler)
    directory = log_dir or os.environ.get("DENDRITIC_LOG_DIR")
    if directory:
        _set_file_handler(
            os.path.join(os.path.abspath(directory), LOG_FILE_NAME), file_level
        )
    return loggers[0]


class LoggerManager:
    """Run-directory file logging used by the training scripts.

    Constructing it performs no I/O; :meth:`set_log_directory` writes the
    package's and the program's records to ``dendritic_modeling.log`` in the
    run directory, closing the previous run's file.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.logger = _package_logger()
            cls._instance.file_path = None
        return cls._instance

    def set_log_file(self, file_path):
        """Write the package's and the program's records to ``file_path``."""
        self.file_path = file_path
        for logger in _program_loggers():
            logger.setLevel(logging.DEBUG)
        _set_file_handler(file_path)
        self.logger.info(f"Log file set to {file_path}.")

    def get_logger(self):
        return self.logger

    def set_log_directory(self, log_dir):
        """Write the run's records to ``dendritic_modeling.log`` in ``log_dir``."""
        self.set_log_file(os.path.join(log_dir, LOG_FILE_NAME))


def get_logger(name=None):
    """Return the logger ``name`` (the package logger by default)."""
    return logging.getLogger(name or LOGGER_NAME)
