"""Logging utilities for the application."""

import sys

from loguru import logger


def configure_logger(level: str = "INFO") -> None:
    """Configure Loguru with stdout (non-errors) and stderr (errors) sinks."""
    logger.remove()

    def not_error(record) -> bool:  # noqa: ANN001
        return record["level"].no < logger.level("ERROR").no

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level}</level> | "
        "<cyan>{name}:{function}:{line}</cyan> - "
        "{message}"
    )

    logger.add(
        sys.stdout,
        level=level.upper(),
        format=fmt,
        diagnose=False,
        backtrace=True,
        catch=True,
        filter=not_error,
        colorize=True,
    )

    logger.add(
        sys.stderr,
        level="ERROR",
        format=fmt,
        diagnose=False,
        backtrace=True,
        catch=True,
        colorize=True,
    )
