from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

# Runs inside the airflow container; our code is mounted at /opt/project/backend
PY = "python -m"
BASE = "cd /opt/project/backend &&"

with DAG(
    dag_id="ingest_docs",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["rag"],
) as dag:

    ingest = BashOperator(
        task_id="reindex_chroma",
        bash_command=f"{BASE} {PY} app.services.rag_service",
    )
