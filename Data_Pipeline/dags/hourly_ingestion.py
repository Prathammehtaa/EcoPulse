# dags/hourly_ingestion_dag.py
from __future__ import annotations

import os
import sys
from datetime import timedelta, datetime

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

# ------------------------------------------------------
# Add src/ to PYTHONPATH
# ------------------------------------------------------
DAG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(DAG_DIR)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# ------------------------------------------------------
# Imports from src/
# ------------------------------------------------------
from signals_historical_ingestion import main as ingestion_main
from alerts import (
    get_recipients,
    notify_task_failure,
    notify_dag_failure,
    make_success_slack_callable,
)
from airflow.utils.email import send_email


def notify_success_email(**context):
    """
    Success email using the SAME path as your failure emails:
    airflow.utils.email.send_email + recipients from alerts.get_recipients()
    """
    dag = context.get("dag")
    dag_id = getattr(dag, "dag_id", "unknown")
    run_id = context.get("run_id")
    ts = context.get("ts")

    subject = f"[SUCCESS] EcoPulse Hourly Ingestion Complete | {dag_id}"
    html_content = f"""
    <p>Hi team,</p>
    <p><b>EcoPulse hourly grid ingestion completed successfully.</b></p>
    <ul>
        <li><b>DAG:</b> {dag_id}</li>
        <li><b>Run ID:</b> {run_id}</li>
        <li><b>Execution Time:</b> {ts}</li>
    </ul>
    <p>Best,<br/>EcoPulse Dev Team</p>
    """
    send_email(to=get_recipients(), subject=subject, html_content=html_content)


default_args = {
    "owner": "ecopulse",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=30),
    # Per-task failure notifications (Slack + Email)
    "on_failure_callback": notify_task_failure,
}

with DAG(
    dag_id="ecopulse_hourly_ingestion",
    description="Hourly idempotent grid ingestion for EcoPulse",
    default_args=default_args,
    start_date=datetime(2026, 2, 25),
    schedule="0 * * * *",  # every hour at minute 0
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(minutes=45),
    # One-per-run DAG failure notification (Slack)
    on_failure_callback=notify_dag_failure,
    tags=["ecopulse", "ingestion", "hourly"],
) as dag:

    start = EmptyOperator(task_id="start")

    run_hourly_ingestion = PythonOperator(
        task_id="run_hourly_grid_ingestion",
        python_callable=lambda: ingestion_main(mode="hourly"),
    )

    # One Slack success (final task)
    slack_success = PythonOperator(
        task_id="slack_success",
        python_callable=make_success_slack_callable(),
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # One Email success (final task) - uses alerts.get_recipients()
    email_success = PythonOperator(
        task_id="email_success",
        python_callable=notify_success_email,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    end = EmptyOperator(task_id="end")

    start >> run_hourly_ingestion >> slack_success >> email_success >> end