"""
Modulo: ingestao_local.py
Descricao: Ingestao local do dataset 3W Petrobras a partir do GitHub.
           Realiza o download dos arquivos Parquet e os organiza por classe e poco.
Autor: Alan Pinheiro da Silva
Data: 2025-06-12
"""

import sys
import time
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import requests

# Adiciona o diretório raiz ao path para importar o módulo config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.spark_config import (
    setup_directories, RAW_DIR, REQUEST_TIMEOUT, RATE_LIMIT_DELAY,
    DEFAULT_CLASSES, DEFAULT_WELLS, logger
)


# -----------------------------------------------------------------------------
# Funções Utilitárias
# -----------------------------------------------------------------------------

def extract_metadata_from_filename(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extrai o ID do poco e o timestamp de medicao a partir do nome do arquivo.

    Formato esperado: WELL-{id_poco}_{YYYYMMDDHHMMSS}.parquet
    Exemplo: WELL-00002_20140921183228.parquet

    Parametros:
        filename: Nome do arquivo Parquet

    Retorna:
        Tupla contendo (id_poco, string_timestamp) ou (None, None)
    """
    pattern = r'^WELL-(\d+)_(\d{14})\.parquet$'
    match = re.match(pattern, filename, re.IGNORECASE)

    if match:
        return match.group(1), match.group(2)

    logger.warning(f"Nome de arquivo fora do padrao esperado: {filename}")
    return None, None


def format_timestamp(timestamp_str: str) -> str:
    """
    Converte string de timestamp bruto para formato legivel de data/hora.

    Parametros:
        timestamp_str: String no formato YYYYMMDDHHMMSS

    Retorna:
        String formatada YYYY-MM-DD HH:MM:SS
    """
    if not timestamp_str or len(timestamp_str) != 14:
        return timestamp_str or ""

    try:
        year = timestamp_str[0:4]
        month = timestamp_str[4:6]
        day = timestamp_str[6:8]
        hour = timestamp_str[8:10]
        minute = timestamp_str[10:12]
        second = timestamp_str[12:14]
        return f"{year}-{month}-{day} {hour}:{minute}:{second}"
    except Exception as e:
        logger.error(f"Erro ao formatar timestamp {timestamp_str}: {e}")
        return timestamp_str


def fetch_class_file_list(classe: str) -> List[Dict]:
    """
    Consulta a API do GitHub e retorna a lista de arquivos de uma classe especifica.

    Parametros:
        classe: Numero da classe (0-9)

    Retorna:
        Lista de dicionarios com metadados dos arquivos
    """
    from config.spark_config import GITHUB_TOKEN
    api_url = f"https://api.github.com/repos/petrobras/3W/contents/dataset/{classe}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    try:
        logger.info("Consultando lista de arquivos da classe %s", classe)
        response = requests.get(api_url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        logger.error("Falha ao consultar classe %s: %s", classe, exc)
        return []


def filter_well_files(files: List[Dict], wells: List[str]) -> List[Dict]:
    """
    Filtra a lista de arquivos para incluir apenas os pocos especificados.

    Parametros:
        files: Lista de metadados de arquivos retornada pela API do GitHub
        wells: Lista de IDs de pocos a incluir (ex: ['00001', '00002'])

    Retorna:
        Lista filtrada de metadados de arquivos
    """
    filtered = []

    for file_info in files:
        if not file_info.get('name', '').endswith('.parquet'):
            continue

        file_name = file_info['name'].upper()
        for well in wells:
            if file_name.startswith(f"WELL-{well}"):
                filtered.append(file_info)
                break

    return filtered


def download_file(file_url: str, destination_path: Path) -> Tuple[bool, int]:
    """
    Realiza o download de um arquivo a partir de uma URL para o destino local.

    Parametros:
        file_url: URL de origem do arquivo
        destination_path: Caminho local onde o arquivo sera salvo

    Retorna:
        Tupla contendo (flag_sucesso, tamanho_em_bytes)
    """
    from config.spark_config import GITHUB_TOKEN
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    try:
        response = requests.get(file_url, headers=headers, timeout=REQUEST_TIMEOUT, stream=True)
        response.raise_for_status()

        with open(destination_path, 'wb') as fh:
            for chunk in response.iter_content(chunk_size=8192):
                fh.write(chunk)

        file_size = destination_path.stat().st_size
        return True, file_size

    except requests.exceptions.RequestException as exc:
        logger.error("Falha no download de %s: %s", destination_path.name, exc)
        return False, 0


def download_class_data(classe: str, wells: List[str], limit: Optional[int] = None) -> int:
    """
    Realiza o download de todos os arquivos Parquet de uma classe e poco(s) especificos.

    Parametros:
        classe: Numero da classe (0-9)
        wells: Lista de IDs de pocos a baixar
        limit: Numero maximo de arquivos por classe (util para testes)

    Retorna:
        Numero de arquivos baixados com sucesso
    """
    class_dir = RAW_DIR / f"classe_{classe}"
    class_dir.mkdir(parents=True, exist_ok=True)

    # Busca e filtra os arquivos da classe
    all_files = fetch_class_file_list(classe)
    target_files = filter_well_files(all_files, wells)

    if limit:
        target_files = target_files[:limit]

    logger.info("Classe %s: %d arquivos encontrados para os pocos %s", classe, len(target_files), wells)

    if not target_files:
        return 0

    success_count = 0

    for idx, file_info in enumerate(target_files, 1):
        file_name = file_info['name']
        file_url = file_info['download_url']
        local_path = class_dir / file_name

        if local_path.exists():
            logger.info("[%d/%d] Ja existe, pulando: %s", idx, len(target_files), file_name)
            success_count += 1
            continue

        logger.info("[%d/%d] Baixando: %s", idx, len(target_files), file_name)

        success, file_size = download_file(file_url, local_path)

        if success:
            success_count += 1
            logger.info("  Salvo: %.1f KB", file_size / 1024)
        else:
            logger.error("  Falha: %s", file_name)

        time.sleep(RATE_LIMIT_DELAY)

    return success_count


# -----------------------------------------------------------------------------
# Execucao Principal
# -----------------------------------------------------------------------------

def run_ingestion_pipeline(classes: List[str] = None,
                           wells: List[str] = None,
                           limit: Optional[int] = None) -> None:
    """
    Executa o pipeline completo de ingestao de dados.

    Parametros:
        classes: Lista de classes a processar (padrao: todas, 0-9)
        wells: Lista de IDs de pocos a processar (padrao: 00001, 00002, 00006)
        limit: Numero maximo de arquivos por classe (para testes)
    """
    logger.info("=" * 60)
    logger.info("INICIANDO PIPELINE DE INGESTAO DE DADOS")
    logger.info("Classes: %s", classes or DEFAULT_CLASSES)
    logger.info("Pocos: %s", wells or DEFAULT_WELLS)
    logger.info("Destino: %s", RAW_DIR)
    logger.info("=" * 60)

    # Cria estrutura de diretorios
    setup_directories()

    # Define valores padrao
    classes_to_process = classes or DEFAULT_CLASSES
    wells_to_process = wells or DEFAULT_WELLS

    total_files = 0

    for classe in classes_to_process:
        logger.info("--- Processando Classe %s ---", classe)
        downloaded = download_class_data(classe, wells_to_process, limit)
        total_files += downloaded
        logger.info("Classe %s: %d arquivos baixados", classe, downloaded)

    logger.info("=" * 60)
    logger.info("PIPELINE DE INGESTAO CONCLUIDO")
    logger.info(f"Total de arquivos baixados: {total_files}")
    logger.info(f"Localizacao dos dados: {RAW_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        # Para teste rapido, limita a 10 arquivos por classe
        run_ingestion_pipeline(limit=10)
    except KeyboardInterrupt:
        logger.warning("Pipeline interrompido pelo usuario")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Erro inesperado: {e}")
        sys.exit(1)