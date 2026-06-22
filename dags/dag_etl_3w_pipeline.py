"""
DAG: dag_etl_3w_pipeline
Descricao: Orquestra o pipeline ETL completo do dataset 3W Petrobras
           nas tres camadas da arquitetura medalhao (Bronze -> Silver -> Gold).

Fluxo das tasks:
  start
    |
  bronze_ingestao          <- data/processed/ (Parquet) -> Bronze Parquet
    |
  silver_limpeza           <- Bronze -> Silver (validacao fisica + nulos)
    |
  gold_feature_engineering <- Silver -> Gold (rolling, lag, delta, Z-score)
    |
  notifica_conclusao       <- Log de finalizacao com metricas basicas

Agendamento padrao: diariamente a meia-noite (pode ser ajustado via UI do Airflow)
URL da UI: http://localhost:8080  |  admin / admin

Variaveis de ambiente esperadas (injetadas pelo docker-compose.yml):
  PROJECT_DIR  - caminho raiz do workspace dentro do container Airflow
  GITHUB_TOKEN - token opcional para download do dataset 3W
"""

from __future__ import annotations

import json
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

# ---------------------------------------------------------------------------
# Configuracoes gerais
# ---------------------------------------------------------------------------
PROJECT_DIR = os.environ.get("PROJECT_DIR", "/opt/airflow/workspace")
SCRIPTS_DIR = Path(PROJECT_DIR) / "scripts"
LOG = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "alan.pinheiro",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
    "execution_timeout": timedelta(minutes=30),   # timeout padrao por task
}

# ---------------------------------------------------------------------------
# Funcoes Python executadas pelas tasks
# ---------------------------------------------------------------------------

def _verificar_ambiente(**context):
    """
    Verifica se os diretorios e dependencias necessarias estao presentes.
    Falha explicitamente antes de iniciar o ETL se algo estiver faltando.
    """
    import sys

    LOG.info("=== Verificacao de ambiente ===")
    LOG.info("PROJECT_DIR: %s", PROJECT_DIR)

    # Verifica script ETL
    etl_script = SCRIPTS_DIR / "etl_bronze_silver_gold.py"
    if not etl_script.exists():
        raise FileNotFoundError(
            f"Script ETL nao encontrado: {etl_script}\n"
            "Verifique se o volume './scripts' esta corretamente mapeado no docker-compose.yml"
        )
    LOG.info("Script ETL encontrado: %s", etl_script)

    # Verifica dependencias Python
    try:
        import pandas
        import pyarrow
        import numpy
        LOG.info("Dependencias OK: pandas=%s, pyarrow=%s, numpy=%s",
                 pandas.__version__, pyarrow.__version__, numpy.__version__)
    except ImportError as exc:
        raise ImportError(f"Dependencia faltando: {exc}") from exc

    # Verifica diretorio de dados processados (Parquet)
    processed_dir = Path(PROJECT_DIR) / "data" / "processed"
    if not processed_dir.exists():
        raise FileNotFoundError(
            f"Diretorio data/processed/ nao encontrado: {processed_dir}\n"
            "Execute primeiro os scripts de ingestao para popular os dados.\n"
            "Exemplo: python scripts/Ingestao_well_Classes_Gith_3w/ingestao_classe_0.py"
        )

    parquet_files = list(processed_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"Nenhum arquivo .parquet encontrado em: {processed_dir}\n"
            "O pipeline Bronze le os dados de data/processed/ (Parquet).\n"
            "Execute os scripts de ingestao antes de acionar esta DAG."
        )

    LOG.info("data/processed/ OK: %d arquivos Parquet encontrados.", len(parquet_files))
    LOG.info("Verificacao de ambiente concluida com sucesso.")
    return {"status": "ok", "project_dir": PROJECT_DIR, "parquet_files": len(parquet_files)}


def _executar_bronze(**context):
    """
    Executa a camada Bronze do pipeline ETL.
    Responsabilidade: RAW CSV -> Bronze Parquet
    - Padronizacao de schema (float64 sensores, int8 class, datetime timestamp)
    - Remocao de registros com class < 0
    - Adicao de campo de auditoria bronze_at
    """
    import subprocess
    import sys

    LOG.info("=== Iniciando Camada Bronze ===")
    etl_script = str(SCRIPTS_DIR / "etl_bronze_silver_gold.py")

    result = subprocess.run(
        [sys.executable, etl_script, "--step", "bronze", "--verbose"],
        capture_output=True,
        text=True,
        cwd=PROJECT_DIR,
        env={**os.environ, "PROJECT_DIR": PROJECT_DIR},
    )

    if result.stdout:
        LOG.info("STDOUT:\n%s", result.stdout)
    if result.stderr:
        LOG.warning("STDERR:\n%s", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"Camada Bronze falhou com codigo {result.returncode}.\n"
            f"STDERR: {result.stderr[-2000:]}"
        )

    LOG.info("=== Camada Bronze concluida ===")

    # Metricas basicas para XCom
    bronze_dir = Path(PROJECT_DIR) / "data" / "bronze"
    parquet_files = list(bronze_dir.glob("*.parquet")) if bronze_dir.exists() else []
    metrics = {"parquet_files": len(parquet_files), "status": "success"}
    context["ti"].xcom_push(key="bronze_metrics", value=metrics)
    return metrics


def _executar_silver(**context):
    """
    Executa a camada Silver do pipeline ETL.
    Responsabilidade: Bronze Parquet -> Silver Parquet
    - Validacao de faixas fisicas dos 8 sensores
    - Forward fill + backward fill por (ID_Poco, class)
    - Preenchimento residual com mediana da classe
    - Remocao de timestamps duplicados
    - Adicao de campo de auditoria silver_at
    """
    import subprocess
    import sys

    LOG.info("=== Iniciando Camada Silver ===")
    etl_script = str(SCRIPTS_DIR / "etl_bronze_silver_gold.py")

    result = subprocess.run(
        [sys.executable, etl_script, "--step", "silver", "--verbose"],
        capture_output=True,
        text=True,
        cwd=PROJECT_DIR,
        env={**os.environ, "PROJECT_DIR": PROJECT_DIR},
    )

    if result.stdout:
        LOG.info("STDOUT:\n%s", result.stdout)
    if result.stderr:
        LOG.warning("STDERR:\n%s", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"Camada Silver falhou com codigo {result.returncode}.\n"
            f"STDERR: {result.stderr[-2000:]}"
        )

    LOG.info("=== Camada Silver concluida ===")

    silver_dir = Path(PROJECT_DIR) / "data" / "silver"
    parquet_files = list(silver_dir.glob("*.parquet")) if silver_dir.exists() else []
    metrics = {"parquet_files": len(parquet_files), "status": "success"}
    context["ti"].xcom_push(key="silver_metrics", value=metrics)
    return metrics


# ---------------------------------------------------------------------------
# Funcoes Gold: dynamic task mapping (uma task por poco)
# Airflow 2.9.2 - @task decorator + .expand()
#
# Fluxo:
#   silver_limpeza
#       |
#   gold_listar_pocos          <- lista IDs de pocos da Silver (XCom: list[str])
#       |
#   gold_poco (N instancias)   <- feature engineering por poco (dynamic mapping)
#       |
#   gold_finalizar             <- stats globais + normalizacao + train/val/test
# ---------------------------------------------------------------------------

def _coletar_metricas_gold() -> dict:
    """Le metricas dos parquets Gold ja escritos em disco."""
    gold_dir = Path(PROJECT_DIR) / "data" / "gold"
    metrics = {"status": "success"}
    if gold_dir.exists():
        for split in ("train", "val", "test"):
            fpath = gold_dir / f"{split}.parquet"
            if fpath.exists():
                try:
                    import pandas as pd
                    df = pd.read_parquet(fpath, columns=["class"])
                    metrics[f"{split}_rows"] = len(df)
                    metrics[f"{split}_classes"] = sorted(df["class"].unique().tolist())
                except Exception:
                    metrics[f"{split}_rows"] = "erro ao ler"
        scaler = gold_dir / "scaler_stats.json"
        metrics["scaler_stats_presente"] = scaler.exists()
    return metrics


def _notificar_conclusao(**context):
    """
    Task final: consolida metricas de todas as camadas e registra no log.
    Em um ambiente de producao, este e o ponto ideal para enviar notificacoes
    (Slack, e-mail, webhook) ou gravar um registro de execucao no PostgreSQL.
    """
    ti = context["ti"]
    bronze = ti.xcom_pull(key="bronze_metrics", task_ids="bronze_ingestao") or {}
    silver = ti.xcom_pull(key="silver_metrics", task_ids="silver_limpeza") or {}
    # gold_finalizar usa @task: o valor de retorno e gravado como XCom "return_value"
    gold   = ti.xcom_pull(task_ids="gold_finalizar") or {}

    run_id  = context["run_id"]
    dag_id  = context["dag"].dag_id
    ts      = context["logical_date"].isoformat()

    LOG.info("=" * 60)
    LOG.info("PIPELINE ETL 3W CONCLUIDO COM SUCESSO")
    LOG.info("DAG: %s | Run: %s | Horario: %s", dag_id, run_id, ts)
    LOG.info("-" * 60)
    LOG.info("Bronze  -> arquivos Parquet: %s", bronze.get("parquet_files", "N/A"))
    LOG.info("Silver  -> arquivos Parquet: %s", silver.get("parquet_files", "N/A"))
    LOG.info("Gold    -> treino: %s linhas | val: %s linhas | teste: %s linhas",
             gold.get("train_rows", "N/A"),
             gold.get("val_rows", "N/A"),
             gold.get("test_rows", "N/A"))
    LOG.info("Gold    -> scaler_stats.json: %s", gold.get("scaler_stats_presente", False))
    LOG.info("=" * 60)

    return {
        "pipeline": "etl_3w",
        "status": "success",
        "bronze": bronze,
        "silver": silver,
        "gold": gold,
    }


# ---------------------------------------------------------------------------
# Definicao da DAG
# ---------------------------------------------------------------------------
with DAG(
    dag_id="dag_etl_3w_pipeline",
    description="Pipeline ETL Bronze -> Silver -> Gold para o dataset 3W Petrobras",
    schedule_interval="0 2 * * *",   # toda madrugada as 02:00 (ajuste conforme necessario)
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,               # impede execucoes paralelas do mesmo pipeline
    default_args=DEFAULT_ARGS,
    tags=["3w", "etl", "petroleo", "offshore", "bronze", "silver", "gold"],
    doc_md="""
## Pipeline ETL 3W Petrobras

Orquestra o processamento completo do dataset 3W nas tres camadas:

| Task | Camada | Responsabilidade |
|------|--------|-----------------|
| `verificar_ambiente` | Pre-check | Valida ambiente e dependencias |
| `bronze_ingestao` | Bronze | Ingestao, schema padrao, auditoria |
| `silver_limpeza` | Silver | Validacao fisica, limpeza, nulos |
| `gold_listar_pocos` | Gold | Lista IDs de pocos da Silver |
| `gold_poco` (N inst.) | Gold | Feature eng. por poco (dynamic mapping) |
| `gold_finalizar` | Gold | Stats globais, Z-Score, train/val/test |
| `notifica_conclusao` | Pos | Metricas e notificacao |

**Agendamento:** `0 2 * * *` (todo dia as 02:00)

**Acionar manualmente:**
```bash
docker compose exec airflow-scheduler airflow dags trigger dag_etl_3w_pipeline
```
    """,
) as dag:

    # Task 0: Verificacao de ambiente
    verificar_ambiente = PythonOperator(
        task_id="verificar_ambiente",
        python_callable=_verificar_ambiente,
        doc_md="Verifica se o script ETL existe e as dependencias Python estao instaladas.",
    )

    # Task 1: Camada Bronze
    bronze_ingestao = PythonOperator(
        task_id="bronze_ingestao",
        python_callable=_executar_bronze,
        doc_md="Executa `etl_bronze_silver_gold.py --step bronze`: ingestao e padronizacao de schema.",
    )

    # Task 2: Camada Silver
    silver_limpeza = PythonOperator(
        task_id="silver_limpeza",
        python_callable=_executar_silver,
        doc_md="Executa `etl_bronze_silver_gold.py --step silver`: validacao fisica e limpeza.",
    )

    # ---------------------------------------------------------------------------
    # Tasks 3a/3b/3c: Camada Gold - dynamic task mapping por poco
    #
    # gold_listar_pocos  -> lista IDs de pocos da Silver via subprocess
    # gold_poco          -> feature eng. de UM poco (uma instancia por poco)
    # gold_finalizar     -> stats globais + normaliza + escreve train/val/test
    # ---------------------------------------------------------------------------

    @task(task_id="gold_listar_pocos")
    def _gold_listar_pocos():
        """Lista os IDs de pocos disponiveis na Silver (chama --step gold_list)."""
        import subprocess, sys

        LOG.info("=== Gold: listando pocos da Silver ===")
        etl_script = str(SCRIPTS_DIR / "etl_bronze_silver_gold.py")

        result = subprocess.run(
            [sys.executable, etl_script, "--step", "gold_list"],
            capture_output=True, text=True,
            cwd=PROJECT_DIR,
            env={**os.environ, "PROJECT_DIR": PROJECT_DIR},
        )

        if result.stderr:
            LOG.info("STDERR gold_list:\n%s", result.stderr[-2000:])
        if result.returncode != 0:
            raise RuntimeError(
                f"gold_list falhou (cod {result.returncode}):\n{result.stderr[-1000:]}"
            )

        # A ultima linha do stdout contem o JSON com a lista de pocos
        wells = []
        for line in reversed(result.stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("["):
                wells = json.loads(line)
                break

        if not wells:
            raise ValueError(
                f"gold_list nao retornou lista valida.\nSTDOUT: {result.stdout[-500:]}"
            )

        LOG.info("Gold: %d pocos encontrados: %s", len(wells), wells)
        return wells

    @task(task_id="gold_poco", execution_timeout=timedelta(hours=1), retries=0)
    def _gold_processar_poco(well_id: str):
        """Feature engineering para um unico poco (uma instancia por poco).
        Carrega so as linhas do poco via pyarrow filter (evita OOM).
        """
        import subprocess, sys

        LOG.info("=== Gold-Poco [%s]: iniciando ===", well_id)
        etl_script = str(SCRIPTS_DIR / "etl_bronze_silver_gold.py")

        result = subprocess.run(
            [sys.executable, etl_script, "--step", "gold_poco",
             "--well", well_id, "--verbose"],
            capture_output=True, text=True,
            cwd=PROJECT_DIR,
            env={**os.environ, "PROJECT_DIR": PROJECT_DIR},
        )

        if result.stderr:
            LOG.info("STDERR poco %s:\n%s", well_id, result.stderr[-3000:])
        if result.returncode != 0:
            raise RuntimeError(
                f"Poco [{well_id}] falhou (cod {result.returncode}):\n"
                f"{result.stderr[-1000:]}"
            )

        LOG.info("=== Gold-Poco [%s]: concluido ===", well_id)
        return {"well_id": well_id, "status": "ok"}

    @task(task_id="gold_finalizar", execution_timeout=timedelta(hours=1), retries=0)
    def _gold_finalizar(resultados: list):
        """Calcula stats globais, normaliza e escreve train/val/test.
        So e executada apos TODOS os gold_poco concluirem (dependencia via resultados).
        """
        import subprocess, sys

        LOG.info("=== Gold-Final: iniciando (apos %d pocos) ===", len(resultados))
        etl_script = str(SCRIPTS_DIR / "etl_bronze_silver_gold.py")

        result = subprocess.run(
            [sys.executable, etl_script, "--step", "gold_final", "--verbose"],
            capture_output=True, text=True,
            cwd=PROJECT_DIR,
            env={**os.environ, "PROJECT_DIR": PROJECT_DIR},
        )

        if result.stderr:
            LOG.info("STDERR gold_final:\n%s", result.stderr[-3000:])
        if result.returncode != 0:
            raise RuntimeError(
                f"gold_final falhou (cod {result.returncode}):\n"
                f"{result.stderr[-2000:]}"
            )

        LOG.info("=== Gold-Final: concluido ===")

        metrics = _coletar_metricas_gold()
        LOG.info("Metricas Gold: %s", metrics)
        return metrics

    # Instancia as tasks Gold com dynamic task mapping
    pocos_list = _gold_listar_pocos()
    por_poco   = _gold_processar_poco.expand(well_id=pocos_list)
    gold_final = _gold_finalizar(resultados=por_poco)

    # Task 4: Notificacao
    notifica_conclusao = PythonOperator(
        task_id="notifica_conclusao",
        python_callable=_notificar_conclusao,
        doc_md="Consolida metricas e registra conclusao do pipeline no log.",
    )

    # Dependencias
    # silver_limpeza -> pocos_list -> por_poco (auto, via expand) -> gold_final (auto, via resultados)
    verificar_ambiente >> bronze_ingestao >> silver_limpeza >> pocos_list
    gold_final >> notifica_conclusao
