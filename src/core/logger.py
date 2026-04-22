import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


_LOGGING_CONFIGURED = False


def configure_logging(base_dir: Path) -> dict[str, Path]:
    global _LOGGING_CONFIGURED

    logs_dir = base_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    app_log_path = logs_dir / "app.log"
    error_log_path = logs_dir / "error.log"

    if _LOGGING_CONFIGURED:
        return {
            "logs_dir": logs_dir,
            "app_log_path": app_log_path,
            "error_log_path": error_log_path,
        }

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    app_file_handler = RotatingFileHandler(
        app_log_path,
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    app_file_handler.setLevel(logging.INFO)
    app_file_handler.setFormatter(formatter)

    error_file_handler = RotatingFileHandler(
        error_log_path,
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(app_file_handler)
    root_logger.addHandler(error_file_handler)
    root_logger.addHandler(stream_handler)

    _LOGGING_CONFIGURED = True

    return {
        "logs_dir": logs_dir,
        "app_log_path": app_log_path,
        "error_log_path": error_log_path,
    }


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)