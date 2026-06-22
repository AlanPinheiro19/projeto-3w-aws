"""
Script: ingestao_unificada.py
Descricao: Ingestao unificada do dataset 3W para ambiente local.
           Substitui os 10 scripts individuais por classe (ingestao_classe0_local.py ... ingestao_classe9_local.py).
           Parametrizavel via linha de comando ou variaveis de ambiente.

Uso:
    # Todas as classes, pocos padrao, limite de 5 arquivos por classe (para teste):
    python ingestao_unificada.py --limit 5

    # Apenas classes 0, 1 e 3, poco 00002:
    python ingestao_unificada.py --classes 0 1 3 --wells 00002

    # Todas as classes e todos os pocos, sem limite:
    python ingestao_unificada.py --all-wells

    # Com token GitHub para evitar rate limit (60 req/h sem token):
    GITHUB_TOKEN=ghp_xxxx python ingestao_unificada.py

Dependencias:
    pip install requests pandas pyarrow
"""

import sys
import re
import time
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import requests
import pandas as pd

# Adiciona o diretorio raiz ao path para importar config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.spark_config import (
    setup_directories,
    RAW_DIR,
    PROCESSED_DIR,
    REQUEST_TIMEOUT,
    RATE_LIMIT_DELAY,
    DEFAULT_CLASSES,
    DEFAULT_WELLS,
    GITHUB_TOKEN,
    logger,
)

GITHUB_API_BASE = "https://api.github.com/repos/petrobras/3W/contents/dataset"

EVENT_LABELS = {
    0: "Normal",
    1: "Abrupt Increase of BSW",
    2: "Spurious Closure of DHSV",
    3: "Severe Slugging",
    4: "Flow Instability",
    5: "Rapid Productivity Loss",
    6: "Quick Restriction in PCK",
    7: "Scaling in PCK",
    8: "Hydrate in Production Line",
    9: "Undefined/Other",
}


# ---------------------------------------------------------------------------
# Utilitarios
# ---------------------------------------------------------------------------

def _github_headers() -> Dict:
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    return headers


def extract_metadata_from_filename(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extrai ID do poco e timestamp do nome do arquivo.
    Formato esperado: WELL-{id_poco}_{YYYYMMDDHHMMSS}.parquet
    """
    pattern = r'^WELL-(\d+)_(\d{14})\.parquet$'
    match = re.match(pattern, filename, re.IGNORECASE)
    if match:
        return match.group(1), match.group(2)
    logger.warning("Nome de arquivo fora do padrao esperado: %s", filename)
    return None, None


def format_timestamp(timestamp_str: str) -> str:
    """Converte YYYYMMDDHHMMSS para YYYY-MM-DD HH:MM:SS."""
    if not timestamp_str or len(timestamp_str) != 14:
        return timestamp_str or ""
    try:
        return (
            f"{timestamp_str[0:4]}-{timestamp_str[4:6]}-{timestamp_str[6:8]} "
            f"{timestamp_str[8:10]}:{timestamp_str[10:12]}:{timestamp_str[12:14]}"
        )
    except Exception:
        return timestamp_str


# ---------------------------------------------------------------------------
# Acesso ao GitHub
# ---------------------------------------------------------------------------

def fetch_class_file_list(classe: str) -> List[Dict]:
    """Retorna lista de arquivos da classe especificada via GitHub API."""
    url = f"{GITHUB_API_BASE}/{classe}"
    try:
        logger.info("Consultando GitHub API - classe %s", classe)
        response = requests.get(url, headers=_github_headers(), timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        logger.error("Falha ao consultar classe %s: %s", classe, exc)
        return []


def filter_well_files(files: List[Dict], wells: List[str]) -> List[Dict]:
    """Filtra apenas os arquivos .parquet dos pocos especificados."""
    result = []
    for item in files:
        name = item.get("name", "")
        if not name.endswith(".parquet"):
            continue
        for well_id in wells:
            if name.upper().startswith(f"WELL-{well_id}"):
                result.append(item)
                break
    return result


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path) -> Tuple[bool, int]:
    """Baixa um arquivo para o disco. Retorna (sucesso, tamanho_bytes)."""
    try:
        response = requests.get(url, headers=_github_headers(), timeout=REQUEST_TIMEOUT, stream=True)
        response.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            for chunk in response.iter_content(chunk_size=8192):
                fh.write(chunk)
        return True, dest.stat().st_size
    except requests.exceptions.RequestException as exc:
        logger.error("Download falhou para %s: %s", dest.name, exc)
        return False, 0


# ---------------------------------------------------------------------------
# Processamento / enriquecimento
# ---------------------------------------------------------------------------

def enrich_parquet(raw_path: Path, classe: str) -> Tuple[bool, int, int]:
    """
    Le o arquivo Parquet bruto, adiciona colunas de metadados e salva
    na pasta processed/.

    Colunas adicionadas:
        ID_Poco         : ID do poco extraido do nome do arquivo
        classe_evento   : numero da classe (int)
        nome_evento     : descricao textual do evento
        data_medicao    : timestamp inicial da serie (YYYY-MM-DD HH:MM:SS)

    Retorna (sucesso, linhas, colunas).
    """
    try:
        df = pd.read_parquet(raw_path)

        id_poco, ts_raw = extract_metadata_from_filename(raw_path.name)

        # Normaliza o index de timestamp para coluna
        if df.index.name == "timestamp" or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            if "index" in df.columns:
                df = df.rename(columns={"index": "timestamp"})

        df["ID_Poco"] = id_poco or "UNKNOWN"
        df["classe_evento"] = int(classe)
        df["nome_evento"] = EVENT_LABELS.get(int(classe), "Desconhecido")
        df["data_medicao"] = format_timestamp(ts_raw) if ts_raw else ""

        out_path = PROCESSED_DIR / raw_path.name
        df.to_parquet(out_path, index=False)

        return True, df.shape[0], df.shape[1]
    except Exception as exc:
        logger.error("Falha ao processar %s: %s", raw_path.name, exc)
        return False, 0, 0


# ---------------------------------------------------------------------------
# Pipeline por classe
# ---------------------------------------------------------------------------

def process_class(classe: str, wells: List[str], limit: Optional[int] = None) -> Dict:
    """
    Executa o pipeline completo para uma classe:
      1. Busca lista de arquivos no GitHub
      2. Filtra pelos pocos especificados
      3. Faz download na pasta raw/classe_{n}/
      4. Enriquece e salva em processed/

    Retorna dicionario com estatisticas.
    """
    class_raw_dir = RAW_DIR / f"classe_{classe}"
    class_raw_dir.mkdir(parents=True, exist_ok=True)

    all_files = fetch_class_file_list(classe)
    target_files = filter_well_files(all_files, wells)

    if limit:
        target_files = target_files[:limit]

    logger.info(
        "Classe %s | %s | %d arquivos encontrados para pocos %s",
        classe,
        EVENT_LABELS.get(int(classe), "?"),
        len(target_files),
        wells,
    )

    stats = {
        "classe": classe,
        "nome_evento": EVENT_LABELS.get(int(classe), "Desconhecido"),
        "encontrados": len(target_files),
        "baixados": 0,
        "processados": 0,
        "linhas_total": 0,
        "bytes_total": 0,
        "falhas": [],
    }

    for idx, file_info in enumerate(target_files, 1):
        name = file_info["name"]
        url = file_info["download_url"]
        local_path = class_raw_dir / name

        # Pula se ja existe
        if local_path.exists():
            logger.info("[%d/%d] Ja existe, pulando: %s", idx, len(target_files), name)
            stats["baixados"] += 1
            # Ainda processa se nao ha versao processed
            processed_path = PROCESSED_DIR / name
            if not processed_path.exists():
                ok, rows, _ = enrich_parquet(local_path, classe)
                if ok:
                    stats["processados"] += 1
                    stats["linhas_total"] += rows
            else:
                stats["processados"] += 1
            continue

        logger.info("[%d/%d] Baixando: %s", idx, len(target_files), name)
        ok, size = download_file(url, local_path)

        if ok:
            stats["baixados"] += 1
            stats["bytes_total"] += size
            logger.info("  Salvo: %.1f KB", size / 1024)

            ok_proc, rows, _ = enrich_parquet(local_path, classe)
            if ok_proc:
                stats["processados"] += 1
                stats["linhas_total"] += rows
                logger.info("  Processado: %d linhas", rows)
        else:
            stats["falhas"].append(name)

        time.sleep(RATE_LIMIT_DELAY)

    return stats


# ---------------------------------------------------------------------------
# Consolidacao
# ---------------------------------------------------------------------------

def consolidate_processed() -> pd.DataFrame:
    """
    Combina todos os arquivos em processed/ em um unico DataFrame.
    Salva: all_classes_combined.parquet e all_classes_combined.csv
    """
    logger.info("Consolidando arquivos processados...")
    parquet_files = [
        f for f in PROCESSED_DIR.glob("*.parquet")
        if f.name != "all_classes_combined.parquet"
    ]

    if not parquet_files:
        logger.warning("Nenhum arquivo processado encontrado em %s", PROCESSED_DIR)
        return pd.DataFrame()

    dfs = []
    for pf in parquet_files:
        try:
            dfs.append(pd.read_parquet(pf))
        except Exception as exc:
            logger.error("Falha ao ler %s: %s", pf.name, exc)

    df_combined = pd.concat(dfs, ignore_index=True)

    out_parquet = PROCESSED_DIR / "all_classes_combined.parquet"
    out_csv = PROCESSED_DIR / "all_classes_combined.csv"
    df_combined.to_parquet(out_parquet, index=False)
    df_combined.to_csv(out_csv, index=False)

    logger.info("Consolidado: %d linhas x %d colunas", *df_combined.shape)
    logger.info("Salvo em: %s", out_parquet)
    return df_combined


def print_summary(results: List[Dict], df: pd.DataFrame) -> None:
    """Exibe o relatorio final da ingestao."""
    logger.info("=" * 65)
    logger.info("RELATORIO FINAL DE INGESTAO")
    logger.info("=" * 65)
    logger.info("%-6s %-32s %8s %9s %12s", "Classe", "Evento", "Arqs.", "Linhas", "Falhas")
    logger.info("-" * 65)
    for r in results:
        logger.info(
            "%-6s %-32s %8d %9d %12d",
            r["classe"], r["nome_evento"][:32],
            r["baixados"], r["linhas_total"], len(r["falhas"]),
        )
    logger.info("-" * 65)
    logger.info("Total de linhas no dataset combinado: %d", len(df))

    if not df.empty and "classe_evento" in df.columns:
        logger.info("Distribuicao de classes:")
        dist = df["classe_evento"].value_counts().sort_index()
        for cls, cnt in dist.items():
            pct = cnt / len(df) * 100
            label = EVENT_LABELS.get(int(cls), "?")
            logger.info("  Classe %d (%s): %d linhas (%.1f%%)", cls, label, cnt, pct)

    logger.info("Diretorios:")
    logger.info("  Raw:       %s", RAW_DIR)
    logger.info("  Processed: %s", PROCESSED_DIR)
    logger.info("=" * 65)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    classes: List[str],
    wells: List[str],
    limit: Optional[int],
    skip_consolidate: bool,
) -> None:
    setup_directories()
    logger.info("Iniciando ingestao | classes=%s | pocos=%s | limit=%s", classes, wells, limit)

    all_results = []
    for classe in classes:
        result = process_class(classe, wells, limit)
        all_results.append(result)

    if not skip_consolidate:
        df_combined = consolidate_processed()
        print_summary(all_results, df_combined)
    else:
        logger.info("Consolidacao ignorada (--skip-consolidate)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingestao unificada do dataset 3W (Petrobras)"
    )
    parser.add_argument(
        "--classes", nargs="+", default=DEFAULT_CLASSES,
        metavar="N", help="Classes a processar (default: 0 1 2 ... 9)"
    )
    parser.add_argument(
        "--wells", nargs="+", default=DEFAULT_WELLS,
        metavar="ID", help="IDs dos pocos (default: 00002 00004 00006)"
    )
    parser.add_argument(
        "--all-wells", action="store_true",
        help="Baixa todos os pocos de cada classe (ignora --wells)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limite de arquivos por classe (util para testes rapidos)"
    )
    parser.add_argument(
        "--skip-consolidate", action="store_true",
        help="Nao gera o arquivo consolidado ao final"
    )
    args = parser.parse_args()

    wells_arg = ["*"] if args.all_wells else args.wells

    try:
        run(
            classes=args.classes,
            wells=wells_arg,
            limit=args.limit,
            skip_consolidate=args.skip_consolidate,
        )
    except KeyboardInterrupt:
        logger.warning("Ingestao interrompida pelo usuario.")
        sys.exit(1)
    except Exception as exc:
        logger.error("Erro inesperado: %s", exc)
        raise
