# airflow/dags/retrain_intent.py
from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

BASE = "cd /opt/project/backend &&"

with DAG(
    dag_id="retrain_intent",
    start_date=datetime(2024,1,1),
    schedule_interval=None,   
    catchup=False,
    tags=["intent"],
) as dag:

    train = BashOperator(
        task_id="train",
        bash_command=f"{BASE} python scripts/train_intent.py",
    )

    evaluate = BashOperator(
        task_id="evaluate",
        bash_command=f"{BASE} python scripts/evaluate_intent.py",
    )

    train >> evaluate
