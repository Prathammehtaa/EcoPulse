# dags/ecopulse_full_backfill_pipeline.py
from __future__ import annotations

import os
import sys
from datetime import timedelta

from airflow import DAG
from airflow.operators.email import EmailOperator
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.task_group import TaskGroup
from datetime import datetime

# ------------------------------------------------------
# Add src/ to PYTHONPATH
# ------------------------------------------------------
DAG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(DAG_DIR)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# ------------------------------------------------------
# Imports
# ------------------------------------------------------
from signals_historical_ingestion import main as grid_ingestion_main
from grid_preprocessing import main as grid_preprocessing_main
from weather_historical_ingestion import main as weather_ingestion_main
from weather_preprocessing import main as weather_preprocessing_main
from merge_and_features import main as merge_features_main
from schema_validation_task import run_tfdv_schema_validation
from alerts import get_recipients, notify_task_failure, notify_dag_failure, make_success_slack_callable

from airflow.utils.email import send_email

default_args = {
    "owner": "ecopulse",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "on_failure_callback": notify_task_failure,     
    "execution_timeout": timedelta(hours=2),         
}

def notify_success_email(**context):
    """
    Success email using the SAME path as your failure emails:
    airflow.utils.email.send_email + recipients from alerts.get_recipients()
    """
    dag = context.get("dag")
    dag_id = getattr(dag, "dag_id", "unknown")
    run_id = context.get("run_id")
    ts = context.get("ts")

    subject = f"[SUCCESS] EcoPulse Backfill Ingestion Complete | {dag_id}"
    html_content = f"""
    <p>Hi team,</p>
    <p><b>EcoPulse Backfill grid ingestion completed successfully.</b></p>
    <ul>
        <li><b>DAG:</b> {dag_id}</li>
        <li><b>Run ID:</b> {run_id}</li>
        <li><b>Execution Time:</b> {ts}</li>
    </ul>
    <p>Best,<br/>EcoPulse Dev Team</p>
    """
    send_email(to=get_recipients(), subject=subject, html_content=html_content)

with DAG(
    dag_id="ecopulse_full_backfill_pipeline",
    description="Full historical backfill pipeline for EcoPulse (Grid + Weather + Features)",
    default_args=default_args,
    start_date=datetime(2026, 2, 23),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=6),              
    on_failure_callback=notify_dag_failure,         
    tags=["ecopulse", "backfill", "full-pipeline"],
) as dag:

    start = EmptyOperator(task_id="start")

    # ---------------------------
    # Grid branch
    # ---------------------------
    with TaskGroup(group_id="grid_pipeline") as grid_pipeline:
        grid_ingestion = PythonOperator(
            task_id="grid_historical_ingestion",
            python_callable=grid_ingestion_main,
            sla=timedelta(hours=2),
        )

        grid_preprocessing = PythonOperator(
            task_id="grid_preprocessing",
            python_callable=grid_preprocessing_main,
            sla=timedelta(hours=2),
        )

        grid_ingestion >> grid_preprocessing

    # ---------------------------
    # Weather branch
    # ---------------------------
    with TaskGroup(group_id="weather_pipeline") as weather_pipeline:
        weather_ingestion = PythonOperator(
            task_id="weather_historical_ingestion",
            python_callable=weather_ingestion_main,
            sla=timedelta(hours=2),
        )

        weather_preprocessing = PythonOperator(
            task_id="weather_preprocessing",
            python_callable=weather_preprocessing_main,
            sla=timedelta(hours=2),
        )

        weather_ingestion >> weather_preprocessing

    join_before_merge = EmptyOperator(
        task_id="join_before_merge",
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    merge_and_features = PythonOperator(
        task_id="merge_and_feature_engineering",
        python_callable=merge_features_main,
        sla=timedelta(hours=2),
    )


    schema_validation_tfdv = PythonOperator(
    task_id="schema_validation_tfdv",
    python_callable=run_tfdv_schema_validation,
    sla=timedelta(hours=2),
)

    #One Slack success only once (final task)
    slack_success = PythonOperator(
        task_id="slack_success",
        python_callable=make_success_slack_callable(),
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    email_success = PythonOperator(
        task_id="email_success",
        python_callable=notify_success_email,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    end = EmptyOperator(task_id="end")

    start >> [grid_pipeline, weather_pipeline]
    [grid_pipeline, weather_pipeline] >> join_before_merge
    join_before_merge >> merge_and_features >> schema_validation_tfdv >> slack_success >> notify_success_email >> end