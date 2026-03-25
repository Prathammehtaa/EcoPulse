# src/alerts.py
from __future__ import annotations

import logging
from typing import Callable, List, Optional

from airflow.hooks.base import BaseHook
from airflow.models import Variable
from airflow.utils.email import send_email

logger = logging.getLogger(__name__)

# Default IDs / keys (override by passing params if you want)
DEFAULT_SLACK_CONN_ID = "slack_webhook"
DEFAULT_EMAILS_VAR = "ECO_PULSE_NOTIFY_EMAILS"


def get_recipients(
    variable_name: str = DEFAULT_EMAILS_VAR,
    default: Optional[List[str]] = None,
) -> List[str]:
    """
    Read comma-separated emails from an Airflow Variable.
    """
    default = default or ["stephyromichan1@gmail.com"]
    raw = Variable.get(variable_name, default_var="")
    emails = [e.strip() for e in raw.split(",") if e.strip()]
    return emails or default


def _get_slack_webhook_from_connection(conn_id: str) -> str:
    """
    Slack Incoming Webhook (as in Airflow UI screenshot):
      - conn.schema = https
      - conn.host   = hooks.slack.com/services
      - conn.password = T000/B000/XXX (token only)
    Builds: https://hooks.slack.com/services/T000/B000/XXX
    """
    conn = BaseHook.get_connection(conn_id)

    schema = (conn.schema or "https").strip()
    host = (conn.host or "").strip()
    token = (conn.password or "").strip()

    if not host:
        raise ValueError(f"Slack connection '{conn_id}' is missing host (Slack Webhook Endpoint).")
    if not token:
        raise ValueError(f"Slack connection '{conn_id}' is missing token (Webhook Token).")

    # host in screenshot has no scheme, so add it
    if host.startswith("http://") or host.startswith("https://"):
        base = host.rstrip("/")
    else:
        base = f"{schema}://{host}".rstrip("/")

    return f"{base}/{token.lstrip('/')}"


def post_to_slack(
    text: str,
    conn_id: str = DEFAULT_SLACK_CONN_ID,
    fallback_variable: Optional[str] = None,
    timeout: int = 10,
) -> None:
    """
    Post a message to Slack using an Incoming Webhook stored in an Airflow Connection.
    Safe: never raises.

    Optionally provide fallback_variable to support old setups:
      fallback_variable="ECO_PULSE_SLACK_WEBHOOK"
    """
    try:
        webhook = ""

        # Preferred: Connection
        try:
            webhook = _get_slack_webhook_from_connection(conn_id)
        except Exception as e:
            if fallback_variable:
                webhook = Variable.get(fallback_variable, default_var="").strip()
                if webhook:
                    logger.info(
                        "Slack connection '%s' not usable (%s); using fallback variable '%s'.",
                        conn_id,
                        e,
                        fallback_variable,
                    )
            else:
                logger.info("Slack connection '%s' not usable; skipping Slack alert. (%s)", conn_id, e)
                return

        if not webhook:
            logger.info("Slack webhook not configured (conn_id=%s); skipping Slack alert.", conn_id)
            return

        # Lazy import so your DAG parsing doesn't fail if provider isn't installed
        from airflow.providers.slack.hooks.slack_webhook import SlackWebhookHook

        SlackWebhookHook(
            slack_webhook_conn_id="slack_webhook",   
            webhook_token=webhook,
            timeout=timeout,
        ).send(text=text)

    except Exception as e:
        # Never raise (avoid masking the original Airflow failure)
        logger.warning("Slack post failed (suppressed): %s", e)


def notify_task_failure(
    context,
    recipients: Optional[List[str]] = None,
    slack_conn_id: str = DEFAULT_SLACK_CONN_ID,
    slack_fallback_variable: Optional[str] = None,
) -> None:
    """
    Per-task failure callback:
      - Slack (concise)
      - Email (detailed)

    Safe: does not raise if Slack/email fails.
    Uses airflow.utils.email.send_email (avoids EmailOperator.execute() warnings in callbacks).
    """
    try:
        ti = context.get("task_instance")
        dag = context.get("dag")
        run_id = context.get("run_id")
        ts = context.get("ts")
        exception = context.get("exception")

        dag_id = getattr(dag, "dag_id", "unknown")
        task_id = getattr(ti, "task_id", "unknown")
        try_number = getattr(ti, "try_number", "n/a")
        log_url = getattr(ti, "log_url", "n/a")

        # Slack (concise)
        slack_msg = (
            f":x: *EcoPulse task failed*\n"
            f"*DAG:* {dag_id}\n"
            f"*Task:* {task_id}\n"
            f"*Run:* {run_id}\n"
            f"*When:* {ts}\n"
            f"*Try:* {try_number}\n"
            f"*Logs:* {log_url}\n"
            f"*Error:* `{str(exception)[:900]}`"
        )
        post_to_slack(slack_msg, conn_id=slack_conn_id, fallback_variable=slack_fallback_variable)

        # Email (detailed)
        recipients = recipients or get_recipients()
        subject = f"[FAILED] EcoPulse DAG {dag_id} | Task {task_id}"
        html_content = f"""
        <p>Hi team,</p>

        <p><b>EcoPulse pipeline task failed.</b></p>

        <ul>
            <li><b>DAG:</b> {dag_id}</li>
            <li><b>Task:</b> {task_id}</li>
            <li><b>Run ID:</b> {run_id}</li>
            <li><b>Execution Time:</b> {ts}</li>
            <li><b>Try Number:</b> {try_number}</li>
            <li><b>Log URL:</b> <a href="{log_url}">{log_url}</a></li>
        </ul>

        <p><b>Exception:</b></p>
        <pre>{exception}</pre>

        <p>Best,<br/>Stephy</p>
        """
        send_email(to=recipients, subject=subject, html_content=html_content)

    except Exception as e:
        logger.warning("notify_task_failure encountered an error (suppressed): %s", e)


def notify_dag_failure(
    context,
    slack_conn_id: str = DEFAULT_SLACK_CONN_ID,
    slack_fallback_variable: Optional[str] = None,
) -> None:
    """
    DAG-run level failure callback (one-per-run signal).
    """
    try:
        dag = context.get("dag")
        run_id = context.get("run_id")
        ts = context.get("ts")
        dag_id = getattr(dag, "dag_id", "unknown")

        msg = (
            f":rotating_light: *EcoPulse DAG RUN FAILED*\n"
            f"*DAG:* {dag_id}\n"
            f"*Run:* {run_id}\n"
            f"*When:* {ts}\n"
            f"Check Airflow for failed task(s) and logs."
        )
        post_to_slack(msg, conn_id=slack_conn_id, fallback_variable=slack_fallback_variable)
    except Exception as e:
        logger.warning("notify_dag_failure encountered an error (suppressed): %s", e)


def notify_success_slack(
    context,
    slack_conn_id: str = DEFAULT_SLACK_CONN_ID,
    slack_fallback_variable: Optional[str] = None,
) -> None:
    """
    One-time success message (place this as a final PythonOperator task).
    """
    try:
        dag = context.get("dag")
        run_id = context.get("run_id")
        ts = context.get("ts")
        dag_id = getattr(dag, "dag_id", "unknown")

        msg = (
            f":white_check_mark: *EcoPulse pipeline succeeded*\n"
            f"*DAG:* {dag_id}\n"
            f"*Run:* {run_id}\n"
            f"*When:* {ts}\n"
            f"All steps completed (grid + weather → merge/features → validation)."
        )
        post_to_slack(msg, conn_id=slack_conn_id, fallback_variable=slack_fallback_variable)
    except Exception as e:
        logger.warning("notify_success_slack encountered an error (suppressed): %s", e)


def make_success_slack_callable(
    slack_conn_id: str = DEFAULT_SLACK_CONN_ID,
    slack_fallback_variable: Optional[str] = None,
) -> Callable:
    """
    Convenience for PythonOperator(python_callable=...), because Airflow passes kwargs.
    """
    def _fn(**context):
        notify_success_slack(
            context,
            slack_conn_id=slack_conn_id,
            slack_fallback_variable=slack_fallback_variable,
        )

    return _fn