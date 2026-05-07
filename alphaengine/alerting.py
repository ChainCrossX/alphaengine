"""
Alerting. Slack and Discord webhooks plus optional Twilio SMS for critical events.
All channels are opt-in via config; if a channel is disabled the dispatcher is a no-op.
"""
from __future__ import annotations

import json
import os
from typing import Any

import requests

from .logger import get_logger

log = get_logger(__name__)


class Alerter:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def emit(self, event_type: str, message: str, **fields: Any) -> None:
        if not self.cfg.get("events", {}).get(event_type, False):
            return
        payload = {"event": event_type, "message": message, **fields}
        text = self._format(payload)
        self._slack(text)
        self._discord(text)
        if event_type in ("circuit_breaker", "error"):
            self._sms(text)

    @staticmethod
    def _format(payload: dict[str, Any]) -> str:
        lines = [f"[{payload['event']}] {payload['message']}"]
        for k, v in payload.items():
            if k in ("event", "message"):
                continue
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def _slack(self, text: str) -> None:
        cfg = self.cfg.get("slack", {})
        if not cfg.get("enabled"):
            return
        url = cfg.get("webhook_url")
        if not url:
            return
        try:
            requests.post(url, json={"text": text}, timeout=5)
        except Exception as e:
            log.warning(f"slack alert failed: {e}")

    def _discord(self, text: str) -> None:
        cfg = self.cfg.get("discord", {})
        if not cfg.get("enabled"):
            return
        url = cfg.get("webhook_url")
        if not url:
            return
        try:
            requests.post(url, json={"content": text}, timeout=5)
        except Exception as e:
            log.warning(f"discord alert failed: {e}")

    def _sms(self, text: str) -> None:
        cfg = self.cfg.get("twilio", {})
        if not cfg.get("enabled"):
            return
        try:
            from twilio.rest import Client
            client = Client(cfg["account_sid"], cfg["auth_token"])
            client.messages.create(
                from_=cfg["from_number"],
                to=cfg["to_number"],
                body=text[:1500],
            )
        except Exception as e:
            log.warning(f"twilio sms failed: {e}")
