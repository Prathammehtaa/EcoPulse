# src/alerts.py
from __future__ import annotations

import logging
from typing import Callable, List, Optional

import requests
from airflow.models import Variable
from airflow.operators.email import EmailOperator

logger = logging.getLogger(__name__)


def get_recipients(
    variable_name: str = "ECO_PULSE_NOTIFY_EMAILS",
    default: Optional[List[str]] = None,
) -> List[str]:
    """
    Read comma-separated emails from an Airflow Variable.
    """
    default = default or ["stephyromichan1@gmail.com"]
    raw = Variable.get(variable_name, default_var="")
    emails = [e.strip() for e in raw.split(",") if e.strip()]
    return emails or default


def post_to_slack(text: str, webhook_var: str = "ECO_PULSE_SLACK_WEBHOOK") -> None:
    """
    Post a message to Slack using an Incoming Webhook stored in an Airflow Variable.
    If not configured, this is a no-op.
    """
    webhook = Variable.get(webhook_var, default_var="")
    if not webhook:
        logger.info("Slack webhook variable '%s' not set; skipping Slack alert.", webhook_var)
        return

    try:
        resp = requests.post(webhook, json={"text": text}, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        # Never raise (avoid masking the original Airflow failure)
        logger.warning("Slack post failed: %s", e)


def notify_task_failure(context, recipients: Optional[List[str]] = None) -> None:
    """
    Per-task failure callback:
    - Slack (concise)
    - Email (detailed)
    Safe: does not raise if Slack/email fails.
    """
    try:
        ti = context.get("task_instance")
        dag = context.get("dag")
        run_id = context.get("run_id")
        ts = context.get("ts")
        exception = context.get("exception")

        # Slack (concise)
        slack_msg = (
            f":x: *EcoPulse task failed*\n"
            f"*DAG:* {dag.dag_id if dag else 'unknown'}\n"
            f"*Task:* {ti.task_id if ti else 'unknown'}\n"
            f"*Run:* {run_id}\n"
            f"*When:* {ts}\n"
            f"*Try:* {getattr(ti, 'try_number', 'n/a')}\n"
            f"*Logs:* {getattr(ti, 'log_url', 'n/a')}\n"
            f"*Error:* `{str(exception)[:900]}`"
        )
        post_to_slack(slack_msg)

        # Email (detailed)
        recipients = recipients or get_recipients()
        subject = f"[FAILED] EcoPulse DAG {dag.dag_id if dag else 'unknown'} | Task {ti.task_id if ti else 'unknown'}"
        html_content = f"""
        <p>Hi team,</p>

        <p><b>EcoPulse pipeline task failed.</b></p>

        <ul>
            <li><b>DAG:</b> {dag.dag_id if dag else 'unknown'}</li>
            <li><b>Task:</b> {ti.task_id if ti else 'unknown'}</li>
            <li><b>Run ID:</b> {run_id}</li>
            <li><b>Execution Time:</b> {ts}</li>
            <li><b>Try Number:</b> {getattr(ti, 'try_number', 'n/a')}</li>
            <li><b>Log URL:</b> <a href="{getattr(ti, 'log_url', '#')}">{getattr(ti, 'log_url', '')}</a></li>
        </ul>

        <p><b>Exception:</b></p>
        <pre>{exception}</pre>

        <p>Best,<br/>Stephy</p>
        """

        EmailOperator(
            task_id="email_failure_notification",
            to=recipients,
            subject=subject,
            html_content=html_content,
        ).execute(context=context)

    except Exception as e:
        logger.warning("notify_task_failure encountered an error (suppressed): %s", e)


def notify_dag_failure(context) -> None:
    """
    DAG-run level failure callback (one-per-run signal).
    Note: Airflow may invoke this for DAG-level failures; it complements per-task alerts.
    """
    try:
        dag = context.get("dag")
        run_id = context.get("run_id")
        ts = context.get("ts")

        msg = (
            f":rotating_light: *EcoPulse DAG RUN FAILED*\n"
            f"*DAG:* {dag.dag_id if dag else 'unknown'}\n"
            f"*Run:* {run_id}\n"
            f"*When:* {ts}\n"
            f"Check Airflow for failed task(s) and logs."
        )
        post_to_slack(msg)
    except Exception as e:
        logger.warning("notify_dag_failure encountered an error (suppressed): %s", e)


def notify_success_slack(context) -> None:
    """
    One-time success message (place this as a final PythonOperator task).
    """
    try:
        dag = context.get("dag")
        run_id = context.get("run_id")
        ts = context.get("ts")

        msg = (
            f":white_check_mark: *EcoPulse pipeline succeeded*\n"
            f"*DAG:* {dag.dag_id if dag else 'unknown'}\n"
            f"*Run:* {run_id}\n"
            f"*When:* {ts}\n"
            f"All steps completed (grid + weather → merge/features → validation)."
        )
        post_to_slack(msg)
    except Exception as e:
        logger.warning("notify_success_slack encountered an error (suppressed): %s", e)


def make_success_slack_callable() -> Callable:
    """
    Convenience for PythonOperator(python_callable=...), because Airflow passes kwargs.
    """
    def _fn(**context):
        notify_success_slack(context)
    return _fn