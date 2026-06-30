"""Best-effort email notifications over SMTP.

No Qt, no hardware SDK — just the stdlib. Configured entirely through ``config``
(env-overridable), so no credentials live in the repo or in session files. If
SMTP is not fully configured the functions are silent no-ops, and sending never
raises: a missing/unreachable mail server can never break an acquisition.

To enable (e.g. with a Gmail App Password — a regular password will be rejected):
    set DSI_NOTIFY_EMAIL=1
    set DSI_SMTP_USER=you@gmail.com
    set DSI_SMTP_PASSWORD=<16-char app password>
    (optional) set DSI_NOTIFY_TO=someone@example.com   # defaults to the user's address
"""

import smtplib
from email.message import EmailMessage

from config import (
    NOTIFY_EMAIL_ENABLED, NOTIFY_EMAIL_TO, NOTIFY_SMTP_HOST, NOTIFY_SMTP_PORT,
    NOTIFY_SMTP_USER, NOTIFY_SMTP_PASSWORD, NOTIFY_SMTP_TIMEOUT_S,
)


def is_configured():
    """True only if notifications are enabled and an SMTP account is set."""
    return bool(NOTIFY_EMAIL_ENABLED and NOTIFY_SMTP_HOST and NOTIFY_SMTP_USER
                and NOTIFY_SMTP_PASSWORD and NOTIFY_EMAIL_TO)


def send_email(subject, body):
    """Send a plain-text notification email. Returns True on success, False
    otherwise. Never raises — failures are swallowed so the caller (an
    acquisition worker) is never affected by a mail problem."""
    if not is_configured():
        return False
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = NOTIFY_SMTP_USER
        msg["To"] = NOTIFY_EMAIL_TO
        msg.set_content(body)
        with smtplib.SMTP(NOTIFY_SMTP_HOST, NOTIFY_SMTP_PORT, timeout=NOTIFY_SMTP_TIMEOUT_S) as server:
            server.starttls()
            server.login(NOTIFY_SMTP_USER, NOTIFY_SMTP_PASSWORD)
            server.send_message(msg)
        return True
    except Exception:
        return False
