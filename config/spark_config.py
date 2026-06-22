"""
Module: spark_config.py
Description: Configuracoes e constantes para o ambiente local de desenvolvimento 3W.
             O SparkSession e criado sob demanda (lazy) para evitar falhas de import
             em scripts que nao utilizam Spark (ex: scripts de ingestao via requests).
Author: Alan Pinheiro da Silva
Date: 2025-06-12
"""

import logging
import os
from pathlib import Path
from typing import Optional

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuracoes de caminhos
# Suporta sobrescrita via variavel de ambiente PROJECT_DIR para rodar em Docker
# (container usa /home/glue_user/workspace, Windows usa D:/Alan/Alan/Pos GRADUAÇÂO USP/eEDB-007 - Trabalho de Conclusão/projeto-3w-aws)
# -----------------------------------------------------------------------------

BASE_DIR = Path(os.environ.get("PROJECT_DIR", "D:/Alan/Alan/Pos GRADUAÇÂO USP/eEDB-007 - Trabalho de Conclusão/projeto-3w-aws"))
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
PROCESSED_DIR = DATA_DIR / "processed"
LOGS_DIR = BASE_DIR / "logs"


# -----------------------------------------------------------------------------
# PostgreSQL
# -----------------------------------------------------------------------------

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
POSTGRES_DATABASE = os.environ.get("POSTGRES_DB", "well_data")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "glue_user")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "glue_password")
POSTGRES_JDBC_URL = (
    f"jdbc:postgresql://{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DATABASE}"
)


# -----------------------------------------------------------------------------
# S3 / MinIO
# Quando MINIO_ENDPOINT definido, boto3 aponta para MinIO local em vez da AWS
# -----------------------------------------------------------------------------

S3_BUCKET = os.environ.get("S3_BUCKET", "3w-offshore-events")
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")


# -----------------------------------------------------------------------------
# GitHub API
# -----------------------------------------------------------------------------

GITHUB_API_BASE = "https://api.github.com/repos/petrobras/3W/contents/dataset"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/petrobras/3W/main/dataset"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")


# -----------------------------------------------------------------------------
# Configuracoes de processamento
# -----------------------------------------------------------------------------

DEFAULT_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
DEFAULT_WELLS = ['00002', '00004', '00006']
REQUEST_TIMEOUT = 60
RATE_LIMIT_DELAY = 0.3


# -----------------------------------------------------------------------------
# Estrutura de diretorios
# -----------------------------------------------------------------------------

def setup_directories() -> None:
    """Cria a estrutura de diretorios necessaria para o projeto."""
    directories = [RAW_DIR, BRONZE_DIR, SILVER_DIR, GOLD_DIR, PROCESSED_DIR, LOGS_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info("Diretorio verificado: %s", directory)


# -----------------------------------------------------------------------------
# SparkSession - criacao lazy (importa PySpark somente quando chamado)
# Scripts de ingestao nao precisam de Spark; importar no topo quebraria esses scripts
# -----------------------------------------------------------------------------

def create_spark_session(
    app_name: str = "3W-LocalDevelopment",
    local_mode: bool = True,
    driver_memory: str = "4g",
    executor_memory: str = "2g",
) -> "SparkSession":
    """
    Cria e retorna uma SparkSession configurada para desenvolvimento local.
    A importacao do PySpark acontece aqui (lazy) para nao impedir o import
    deste modulo em contextos sem Spark instalado.

    Args:
        app_name: Nome da aplicacao no Spark UI
        local_mode: Se True, usa local[*] (todos os cores disponveis)
        driver_memory: Memoria do driver Spark
        executor_memory: Memoria do executor Spark

    Returns:
        SparkSession configurada
    """
    try:
        from pyspark.sql import SparkSession
        from pyspark.conf import SparkConf
    except ImportError as exc:
        raise ImportError(
            "PySpark nao encontrado. Execute dentro do container Glue "
            "ou instale: pip install pyspark"
        ) from exc

    logger.info("Criando SparkSession: %s", app_name)

    conf = SparkConf().setAppName(app_name)

    if local_mode:
        conf.setMaster("local[*]")
        conf.set("spark.sql.shuffle.partitions", "4")
        conf.set("spark.default.parallelism", "4")

    conf.set("spark.sql.adaptive.enabled", "true")
    conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.driver.memory", driver_memory)
    conf.set("spark.executor.memory", executor_memory)
    conf.set("spark.sql.parquet.writeLegacyFormat", "true")

    # Configuracao MinIO como S3 local
    conf.set("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
    conf.set("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
    conf.set("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
    conf.set("spark.hadoop.fs.s3a.path.style.access", "true")
    conf.set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    logger.info("SparkSession criada. Versao Spark: %s", spark.version)

    return spark


# -----------------------------------------------------------------------------
# Utilitarios PostgreSQL
# -----------------------------------------------------------------------------

def get_postgres_connection_properties() -> dict:
    """Retorna propriedades JDBC para conexao com PostgreSQL."""
    return {
        "user": POSTGRES_USER,
        "password": POSTGRES_PASSWORD,
        "driver": "org.postgresql.Driver",
    }


def get_postgres_url() -> str:
    """Retorna a URL JDBC do PostgreSQL."""
    return POSTGRES_JDBC_URL