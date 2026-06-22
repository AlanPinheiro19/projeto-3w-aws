"""
DAG: dag_gold_rebuild
Descricao: Reprocessa SOMENTE a camada Gold (Silver ja esta pronta).
           Util para re-rodar feature engineering sem refazer Bronze/Silver.

Fluxo das tasks:
  gold_listar_pocos
      |
  gold_poco[0..N]   <- dynamic task mapping (um por poco)
      |
  gold_finalizar    <- stats globais + Z-score + train/val/test

Acionar via Airflow UI: http://localhost:8080
  DAG ID: dag_gold_rebuild -> Trigger DAG

Ou via CLI (dentro do container):
  docker compose exec airflow-scheduler \
      airflow dags trigger dag_gold_rebuild

Variaveis de ambiente esperadas (injetadas pelo docker-compose.yml):
  PROJECT_DIR  - caminho raiz do workspace dentro do container
"""

from __future__ import annotations

import json
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.decorators import task
from airflow.utils.dates import days_ago

# ---------------------------------------------------------------------------
# Configuracoes
# ---------------------------------------------------------------------------
PROJECT_DIR = os.environ.get("PROJECT_DIR", "/opt/airflow/workspace")
SCRIPTS_DIR = Path(PROJECT_DIR) / "scripts"
LOG = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "alan.pinheiro",
    "depends_on_past": False,
    "email_on_failure": "alan.pinheiro@usp.br",
    "email_on_retry": False,
    "retries": 0,                            # sem retry automatico (OOM nao adianta retry)
    "execution_timeout": timedelta(hours=2), # Gold pode demorar com datasets grandes
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_etl(args: list[str], step_name: str) -> str:
    """Executa o script ETL com os argumentos fornecidos e retorna stdout."""
    import subprocess, sys

    etl_script = str(SCRIPTS_DIR / "etl_bronze_silver_gold.py")
    cmd = [sys.executable, etl_script] + args

    LOG.info("Executando: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=PROJECT_DIR,
        env={**os.environ, "PROJECT_DIR": PROJECT_DIR},
    )

    if result.stdout:
        # Limita log para nao lotar o banco de metadados do Airflow
        LOG.info("STDOUT [%s] (ultimas 3000 chars):\n%s", step_name, result.stdout[-3000:])
    if result.stderr:
        LOG.info("STDERR [%s] (ultimas 3000 chars):\n%s", step_name, result.stderr[-3000:])

    if result.returncode != 0:
        raise RuntimeError(
            f"[{step_name}] falhou com codigo {result.returncode}.\n"
            f"STDERR (ultimas 2000):\n{result.stderr[-2000:]}"
        )

    return result.stdout


def _coletar_metricas_gold() -> dict:
    """Le linhas e classes dos parquets Gold escritos em disco."""
    gold_dir = Path(PROJECT_DIR) / "data" / "gold"
    metrics: dict = {"status": "success"}

    if not gold_dir.exists():
        metrics["aviso"] = "gold_dir nao encontrado"
        return metrics

    for split in ("train", "val", "test"):
        fpath = gold_dir / f"{split}.parquet"
        if fpath.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(fpath, columns=["class"])
                metrics[f"{split}_rows"]    = len(df)
                metrics[f"{split}_classes"] = sorted(int(c) for c in df["class"].unique())
            except Exception as exc:
                metrics[f"{split}_rows"] = f"erro: {exc}"
        else:
            metrics[f"{split}_rows"] = 0

    scaler = gold_dir / "scaler_stats.json"
    metrics["scaler_stats_presente"] = scaler.exists()

    LOG.info("Metricas Gold: %s", metrics)
    return metrics


# ---------------------------------------------------------------------------
# Definicao da DAG
# ---------------------------------------------------------------------------
with DAG(
    dag_id="dag_gold_rebuild",
    description="Reprocessa apenas a camada Gold (Bronze/Silver ja prontos)",
    schedule_interval=None,   # so manual (Trigger DAG na UI)
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["3w", "gold", "feature-engineering", "manual"],
    doc_md="""
## DAG: dag_gold_rebuild

Reprocessa **somente** a camada Gold do dataset 3W.
Use quando Bronze e Silver ja estao prontos e so o Gold precisa ser refeito
(ex: apos correcao de bug no feature engineering, ajuste de hiperparametros de janelas, etc.).

### Fluxo

```
gold_listar_pocos
       |
gold_poco[0]  gold_poco[1]  ...  gold_poco[N]
       \\             |              /
        gold_finalizar
               |
        gold_relatorio
```

### Como acionar

**Via UI:** http://localhost:8080 → DAG `dag_gold_rebuild` → ▶ Trigger DAG

**Via CLI:**
```bash
docker compose exec airflow-scheduler \\
    airflow dags trigger dag_gold_rebuild
```

### Outputs

| Arquivo | Descricao |
|---------|-----------|
| `data/gold/train.parquet` | 70% das linhas, features normalizadas |
| `data/gold/val.parquet`   | 15% |
| `data/gold/test.parquet`  | 15% |
| `data/gold/scaler_stats.json` | media/std por feature para inferencia |
    """,
) as dag:

    # ------------------------------------------------------------------
    # Task 1: Lista pocos disponiveis na Silver
    # ------------------------------------------------------------------
    @task(task_id="gold_listar_pocos")
    def gold_listar_pocos() -> list[str]:
        """Lista IDs de pocos presentes na Silver via --step gold_list."""
        LOG.info("=== Gold-Rebuild: listando pocos ===")
        LOG.info("PROJECT_DIR: %s", PROJECT_DIR)

        stdout = _run_etl(["--step", "gold_list"], "gold_list")

        # A ultima linha com "[" e o JSON da lista
        wells: list[str] = []
        for line in reversed(stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("["):
                try:
                    wells = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue

        if not wells:
            # Fallback: tenta ler diretamente da Silver
            LOG.warning("gold_list nao retornou JSON. Tentando fallback via Silver...")
            silver_dir = Path(PROJECT_DIR) / "data" / "silver"
            if silver_dir.exists():
                import pyarrow.parquet as pq
                for f in silver_dir.glob("*.parquet"):
                    try:
                        tbl = pq.read_table(f, columns=["ID_Poco"])
                        wells = sorted(tbl.column("ID_Poco").to_pylist().__class__(
                            set(str(v) for v in tbl.column("ID_Poco").to_pylist() if v)
                        ))
                        break
                    except Exception:
                        pass

        if not wells:
            raise ValueError(
                f"Nenhum poco encontrado.\n"
                f"Verifique se data/silver/ existe e contem parquets com coluna ID_Poco.\n"
                f"STDOUT gold_list:\n{stdout[-500:]}"
            )

        LOG.info("Gold-Rebuild: %d pocos encontrados: %s", len(wells), wells)
        return wells

    # ------------------------------------------------------------------
    # Task 2: Feature engineering por poco (dynamic mapping)
    # ------------------------------------------------------------------
    @task(
        task_id="gold_poco",
        execution_timeout=timedelta(hours=1),
        retries=0,
        pool="default_pool",    # usa pool padrao (1 slot = sequencial, evita OOM)
    )
    def gold_processar_poco(well_id: str) -> dict:
        """
        Feature engineering para um unico poco.
        Carrega so as linhas do poco via pyarrow filter (evita OOM).
        Inclui fix: fillna(0) nas features, dropna so para sensores todos-NaN.
        """
        LOG.info("=== Gold-Poco [%s]: iniciando ===", well_id)

        stdout = _run_etl(
            ["--step", "gold_poco", "--well", well_id, "--verbose"],
            f"gold_poco[{well_id}]",
        )

        # Extrai numero de linhas do stdout para logging
        n_linhas = "?"
        for line in reversed(stdout.splitlines()):
            if "linhas ->" in line or "linhas escritas" in line.lower():
                n_linhas = line.strip()
                break

        LOG.info("=== Gold-Poco [%s]: concluido. %s ===", well_id, n_linhas)
        return {"well_id": well_id, "status": "ok", "info": n_linhas}

    # ------------------------------------------------------------------
    # Task 3: Finaliza Gold (stats globais + normaliza + train/val/test)
    # ------------------------------------------------------------------
    @task(
        task_id="gold_finalizar",
        execution_timeout=timedelta(hours=1),
        retries=0,
    )
    def gold_finalizar(resultados: list) -> dict:
        """
        Calcula estatisticas globais Z-score, normaliza todos os pocos
        e escreve train.parquet / val.parquet / test.parquet.
        So roda apos TODOS os gold_poco concluirem.
        """
        ok    = [r for r in resultados if isinstance(r, dict) and r.get("status") == "ok"]
        falha = [r for r in resultados if not (isinstance(r, dict) and r.get("status") == "ok")]

        LOG.info("=== Gold-Final: %d pocos OK, %d com falha ===", len(ok), len(falha))
        if falha:
            LOG.warning("Pocos com falha: %s", falha)

        if not ok:
            raise RuntimeError(
                "Nenhum poco processado com sucesso. Verifique os logs de gold_poco."
            )

        _run_etl(["--step", "gold_final", "--verbose"], "gold_final")

        LOG.info("=== Gold-Final: concluido ===")
        return _coletar_metricas_gold()

    # ------------------------------------------------------------------
    # Task 4: Relatorio final
    # ------------------------------------------------------------------
    @task(task_id="gold_relatorio")
    def gold_relatorio(metrics: dict) -> None:
        """Imprime relatorio final com contagens de linhas e classes."""
        LOG.info("=" * 60)
        LOG.info("GOLD REBUILD — CONCLUIDO")
        LOG.info("=" * 60)
        LOG.info("  train : %s linhas | classes: %s",
                 metrics.get("train_rows", "N/A"),
                 metrics.get("train_classes", "N/A"))
        LOG.info("  val   : %s linhas | classes: %s",
                 metrics.get("val_rows", "N/A"),
                 metrics.get("val_classes", "N/A"))
        LOG.info("  test  : %s linhas | classes: %s",
                 metrics.get("test_rows", "N/A"),
                 metrics.get("test_classes", "N/A"))
        LOG.info("  scaler_stats.json: %s", metrics.get("scaler_stats_presente", False))
        LOG.info("=" * 60)

        train_rows = metrics.get("train_rows", 0)
        if isinstance(train_rows, int) and train_rows == 0:
            raise RuntimeError(
                "train.parquet tem 0 linhas apos gold_finalizar. "
                "Verifique os logs de gold_poco e gold_finalizar."
            )

    # ------------------------------------------------------------------
    # Montagem do grafo
    # ------------------------------------------------------------------
    pocos       = gold_listar_pocos()
    por_poco    = gold_processar_poco.expand(well_id=pocos)
    metricas    = gold_finalizar(resultados=por_poco)
    gold_relatorio(metrics=metricas)
