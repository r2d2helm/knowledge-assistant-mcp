"""
Logging configuration module for Knowledge Assistant MCP Server.

Configures structlog with appropriate processors for development.
"""

import logging

import structlog


def configure_logging() -> None:
    """Configure structlog for the application.

    Uses ConsoleRenderer for readable colored output in development.
    """
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.typing.FilteringBoundLogger:
    """Get a configured structlog logger.

    Args:
        name: Optional logger name (typically __name__)

    Returns:
        A bound structlog logger
    """
    if name:
        return structlog.get_logger(name)
    return structlog.get_logger()
