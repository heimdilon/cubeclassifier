"""Logging configuration for cube classifier"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def _coerce_level(level: Union[int, str]) -> int:
    if isinstance(level, str):
        return getattr(logging, level.upper(), logging.INFO)
    return int(level)


def setup_logger(
    name: str = "cube_classifier",
    log_file: Optional[str] = None,
    level: Union[int, str] = logging.INFO,
) -> logging.Logger:
    """
    Setup logger with console and optional file output

    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    resolved_level = _coerce_level(level)
    logger = logging.getLogger(name)
    logger.setLevel(resolved_level)
    logger.propagate = False

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    simple_formatter = logging.Formatter("%(levelname)s: %(message)s")

    # Console handler (simple format)
    console_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            console_handler = handler
            break

    if console_handler is None:
        console_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(console_handler)

    console_handler.setLevel(resolved_level)
    console_handler.setFormatter(simple_formatter)

    # File handler (detailed format)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_log_path = log_path.resolve()

        file_handler = None
        for handler in logger.handlers:
            if (
                isinstance(handler, logging.FileHandler)
                and Path(handler.baseFilename).resolve() == resolved_log_path
            ):
                file_handler = handler
                break

        if file_handler is None:
            file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            logger.addHandler(file_handler)

        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(detailed_formatter)

    return logger


def configure_logger(
    log_file: Optional[str] = None, level: Union[int, str] = logging.INFO
) -> logging.Logger:
    return setup_logger(name="cube_classifier", log_file=log_file, level=level)


# Default logger instance
logger = setup_logger(name="cube_classifier", log_file=None, level=logging.INFO)
