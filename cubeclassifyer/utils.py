"""Logging configuration for cube classifier"""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "cube_classifier", log_file: str = None, level: int = logging.INFO
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
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    simple_formatter = logging.Formatter("%(levelname)s: %(message)s")

    # Console handler (simple format)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    # File handler (detailed format)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

    return logger


# Default logger instance
logger = setup_logger(
    name="cube_classifier", log_file="logs/training.log", level=logging.INFO
)
