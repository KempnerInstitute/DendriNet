"""Importing the package configures no logging; programs call setup_logging."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


def test_importing_the_package_adds_only_a_null_handler(tmp_path):
    probe = (
        "import json, logging, dendritic_modeling, dendritic_modeling.networks;"
        "logger = logging.getLogger('dendritic_modeling');"
        "print(json.dumps({'handlers': [type(h).__name__ for h in logger.handlers],"
        " 'level': logger.level}))"
    )
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    env.pop("DENDRITIC_LOG_DIR", None)
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    state = json.loads(completed.stdout.strip().splitlines()[-1])
    assert state == {"handlers": ["NullHandler"], "level": logging.NOTSET}
    assert list(tmp_path.iterdir()) == []


def test_setup_logging_is_idempotent_and_writes_the_requested_directory(
    tmp_path, monkeypatch
):
    from dendritic_modeling.utils.logging_config import (
        LOG_FILE_NAME,
        LoggerManager,
        setup_logging,
    )

    logger = logging.getLogger("dendritic_modeling")
    main_logger = logging.getLogger("__main__")
    saved = {item: (list(item.handlers), item.level) for item in (logger, main_logger)}
    try:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("DENDRITIC_LOG_DIR", "logs")
        setup_logging()
        setup_logging(debug=True)
        consoles = [h for h in logger.handlers if getattr(h, "_dendritic_console", 0)]
        assert len(consoles) == 1 and consoles[0].level == logging.DEBUG
        assert (tmp_path / "logs" / LOG_FILE_NAME).exists()

        first = next(h for h in logger.handlers if isinstance(h, logging.FileHandler))
        LoggerManager().set_log_directory(str(tmp_path / "run"))
        files = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(files) == 1 and files[0] is not first
        assert first.stream is None  # the replaced file was closed
        logging.getLogger("dendritic_modeling.child").info("recorded")
        files[0].flush()
        assert "recorded" in (tmp_path / "run" / LOG_FILE_NAME).read_text()
    finally:
        for item, (handlers, level) in saved.items():
            for handler in list(item.handlers):
                if handler not in handlers:
                    item.removeHandler(handler)
                    handler.close()
            item.setLevel(level)


def test_a_program_module_logs_through_the_configured_handlers(tmp_path):
    program = (
        "import logging\n"
        "from dendritic_modeling.utils.logging_config import LoggerManager, setup_logging\n"
        "logger = logging.getLogger(__name__)\n"
        "setup_logging()\n"
        "LoggerManager().set_log_directory('run')\n"
        "logger.info('program record')\n"
        "logging.getLogger('dendritic_modeling.child').info('package record')\n"
    )
    script = tmp_path / "program.py"
    script.write_text(program)
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    env.pop("DENDRITIC_LOG_DIR", None)
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "program record" in completed.stdout
    assert completed.stdout.count("package record") == 1
    run_log = (tmp_path / "run" / "dendritic_modeling.log").read_text()
    assert "program record" in run_log and "package record" in run_log
