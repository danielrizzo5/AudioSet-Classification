"""Logging utilities for the application.

This module provides utilities for configuring and managing logging
throughout the application, centralizing logging configuration.
"""

import sys

from loguru import logger


def configure_logger(level: str = "INFO") -> None:
    """Configure Loguru and intercept all standard logging.

    Args:
        level: The log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logger.remove()

    def not_error(record):
        return record["level"].no < logger.level("ERROR").no

    format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level}</level> | "
        "<cyan>{name}:{function}:{line}</cyan> - "
        "{message}"
    )

    logger.add(
        sys.stdout,
        level=level.upper(),
        format=format,
        diagnose=False,
        backtrace=True,
        enqueue=True,  # Thread-safe logging
        catch=True,
        filter=not_error,
        colorize=True,
    )

    logger.add(
        sys.stderr,
        level="ERROR",
        format=format,
        diagnose=False,
        backtrace=True,
        enqueue=True,  # Thread-safe logging
        catch=True,
        colorize=True,
    )
