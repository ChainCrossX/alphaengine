"""Structured JSON logger. Every decision in the system writes here."""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "extra_fields"):
            payload.update(record.extra_fields)  # type: ignore
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(
    name: str = "alphaengine",
    level: str = "INFO",
    json_format: bool = True,
    file_path: str | None = None,
    rotate_bytes: int = 10_485_760,
    rotate_backups: int = 7,
) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = JsonFormatter() if json_format else logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if file_path:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            file_path, maxBytes=rotate_bytes, backupCount=rotate_backups
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


def log_event(logger: logging.Logger, level: str, msg: str, **fields) -> None:
    """Helper to attach structured fields to a log record."""
    record = logger.makeRecord(
        logger.name,
        getattr(logging, level.upper(), logging.INFO),
        fn="",
        lno=0,
        msg=msg,
        args=(),
        exc_info=None,
    )
    record.extra_fields = fields  # type: ignore
    logger.handle(record)
