"""Rotating file logs at ~/.deskbot/logs plus rich console output."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from rich.logging import RichHandler

from deskbot import paths

_configured = False


def setup_logging(level: str = "INFO") -> logging.Logger:
    global _configured
    logger = logging.getLogger("deskbot")

    if _configured:
        return logger

    paths.ensure_dirs()
    logger.setLevel(level.upper())
    logger.propagate = False

    file_handler = RotatingFileHandler(
        paths.LOGS_DIR / "deskbot.log", maxBytes=2_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(file_handler)

    console_handler = RichHandler(show_time=False, show_path=False, markup=True)
    logger.addHandler(console_handler)

    _configured = True
    return logger
