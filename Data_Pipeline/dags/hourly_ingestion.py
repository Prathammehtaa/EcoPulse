# dags/hourly_ingestion.py
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
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
from signals_historical_ingestion import main as grid_ingestion_main
from weather_historical_ingestion import main as weather_ingestion_main
from hourly_anomaly_checks import (
    run_grid_hourly_anomaly_checks,
    run_weather_hourly_anomaly_checks,
)
from alerts import (
    notify_task_failure,
    notify_dag_failure,
    make_success_slack_callable,
)

default_args = {
    "owner": "ecopulse",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=1),
    "on_failure_callback": notify_task_failure,
}

with DAG(
    dag_id="ecopulse_hourly_ingestion",
    description="Hourly grid + weather ingestion with lightweight anomaly checks for EcoPulse",
    default_args=default_args,
    start_date=datetime(2026, 2, 25),
    schedule="0 * * * *",
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=2),
    on_failure_callback=notify_dag_failure,
    tags=["ecopulse", "hourly", "ingestion", "anomaly-checks"],
) as dag:

    start = EmptyOperator(task_id="start")

    # ---------------------------
    # Grid branch
    # ---------------------------
    with TaskGroup(group_id="grid_pipeline") as grid_pipeline:
        grid_ingestion = PythonOperator(
            task_id="grid_hourly_ingestion",
            python_callable=lambda: grid_ingestion_main(mode="hourly"),
            sla=timedelta(hours=1),
        )

        grid_hourly_anomaly_checks = PythonOperator(
            task_id="grid_hourly_anomaly_checks",
            python_callable=run_grid_hourly_anomaly_checks,
            sla=timedelta(hours=1),
        )

        grid_ingestion >> grid_hourly_anomaly_checks

    # ---------------------------
    # Weather branch
    # ---------------------------
    with TaskGroup(group_id="weather_pipeline") as weather_pipeline:
        weather_ingestion = PythonOperator(
            task_id="weather_hourly_ingestion",
            python_callable=lambda: weather_ingestion_main(mode="hourly"),
            sla=timedelta(hours=1),
        )

        weather_hourly_anomaly_checks = PythonOperator(
            task_id="weather_hourly_anomaly_checks",
            python_callable=run_weather_hourly_anomaly_checks,
            sla=timedelta(hours=1),
        )

        weather_ingestion >> weather_hourly_anomaly_checks

    join_after_anomaly_checks = EmptyOperator(
        task_id="join_after_anomaly_checks",
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    slack_success = PythonOperator(
        task_id="slack_success",
        python_callable=make_success_slack_callable(),
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    end = EmptyOperator(task_id="end")

    start >> [grid_pipeline, weather_pipeline]
    [grid_pipeline, weather_pipeline] >> join_after_anomaly_checks
    join_after_anomaly_checks >> slack_success >> end