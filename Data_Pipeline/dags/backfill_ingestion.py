from __future__ import annotations

import os
import sys
from datetime import timedelta

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.models import Variable
from signals_historical_ingestion import main


# ------------------------------------------------------
# Add src/ to PYTHONPATH
# ------------------------------------------------------
DAG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(DAG_DIR)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)



# ------------------------------------------------------
# Email Recipients
# ------------------------------------------------------
def get_recipients():
    raw = Variable.get("ECO_PULSE_NOTIFY_EMAILS", default_var="")
    emails = [e.strip() for e in raw.split(",") if e.strip()]
    return emails or ["stephyromichan1@gmail.com"]


# ------------------------------------------------------
# DAG Definition
# ------------------------------------------------------
default_args = {
    "owner": "ecopulse",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="ecopulse_grid_backfill_historical",
    description="Historical backfill for EcoPulse GRID signals",
    default_args=default_args,
    start_date=days_ago(1),
    schedule=None,  # manual trigger
    catchup=False,
    max_active_runs=1,
    tags=["ecopulse", "grid", "backfill"],
) as dag:

    grid_ingestion = PythonOperator(
        task_id="grid_historical_ingestion",
        python_callable=main,  # directly calling your script's main()
    )

    notify_success = EmailOperator(
        task_id="email_success",
        to=get_recipients(),
        subject="EcoPulse Grid Historical Backfill Complete",
        html_content="""
        <p>Hi team,</p>

        <p>The <b>EcoPulse GRID historical ingestion</b> has completed successfully.</p>

        <ul>
            <li><b>DAG:</b> {{ dag.dag_id }}</li>
            <li><b>Run ID:</b> {{ run_id }}</li>
            <li><b>Execution Time:</b> {{ ts }}</li>
        </ul>

        <p>Raw grid data has been ingested and stored per configuration.</p>

        <p>Best,<br/>Stephy</p>
        """,
    )

    grid_ingestion >> notify_success
