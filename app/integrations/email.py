"""Transactional email over SMTP (admin alerts, digests)."""

from __future__ import annotations

from email.message import EmailMessage

import structlog

from app.core.config import Settings, get_settings
from app.core.metrics import metrics

logger = structlog.get_logger(__name__)


class EmailClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.stats = metrics.provider("smtp")

    @property
    def configured(self) -> bool:
        return bool(self.settings.smtp_host)

    async def send(self, to: list[str], subject: str, text: str, *, html: str | None = None) -> bool:
        if not to:
            return False
        if not self.configured:
            logger.info("email_not_configured", subject=subject, recipients=to)
            return False
        import time

        import aiosmtplib

        message = EmailMessage()
        message["From"] = self.settings.smtp_from
        message["To"] = ", ".join(to)
        message["Subject"] = subject
        message.set_content(text)
        if html:
            message.add_alternative(html, subtype="html")

        started = time.perf_counter()
        try:
            await aiosmtplib.send(
                message,
                hostname=self.settings.smtp_host,
                port=self.settings.smtp_port,
                username=self.settings.smtp_user,
                password=self.settings.smtp_password.get_secret_value() if self.settings.smtp_password else None,
                start_tls=self.settings.smtp_use_tls,
                timeout=20,
            )
        except (aiosmtplib.SMTPException, OSError) as exc:
            self.stats.record((time.perf_counter() - started) * 1000, error=str(exc))
            logger.error("email_send_failed", error=str(exc), subject=subject)
            return False
        self.stats.record((time.perf_counter() - started) * 1000)
        return True


__all__ = ["EmailClient"]
