"""
Script: ingestao_classe8_wells_only.py
Descricao: Ingestion of Class 8 Parquet files from Petrobras 3W dataset.
            Filters only files starting with 'WELL-00002' (case-insensitive).
Author: Alan Pinheiro da Silva
Date: 2025-01-08
"""

import requests
import pandas as pd
from pathlib import Path
from io import BytesIO
import time
import logging
from typing import List, Dict, Tuple

# -----------------------------------------------------------------------------
# CONFIGURACAO
# -----------------------------------------------------------------------------

CLASSE = "8"
BASE_DIR = Path("C:/projeto-3w-aws/data")
LOCAL_DIR = BASE_DIR / f"classe{CLASSE}_wells_02_only"
API_URL = f"https://api.github.com/repos/petrobras/3W/contents/dataset/{CLASSE}"

# Configuracao de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Criar diretorio de destino
LOCAL_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# FUNCOES AUXILIARES
# -----------------------------------------------------------------------------

def fetch_file_list(api_url: str) -> List[Dict]:
    """
    Fetch file list from GitHub API.
    
    Args:
        api_url: GitHub API endpoint URL
        
    Returns:
        List of file metadata dictionaries
    """
    logger.info(f"Fetching file list from: {api_url}")
    response = requests.get(api_url)
    response.raise_for_status()
    return response.json()


def filter_parquet_files(files: List[Dict]) -> List[Dict]:
    """
    Filter only Parquet files.
    
    Args:
        files: List of file metadata from GitHub API
        
    Returns:
        Filtered list containing only Parquet files
    """
    return [f for f in files if f['name'].endswith('.parquet')]


def filter_well_files(parquet_files: List[Dict]) -> List[Dict]:
    """
    Filter files that start with 'WELL-00002' (case-insensitive).
    
    Args:
        parquet_files: List of Parquet file metadata
        
    Returns:
        Filtered list containing only files starting with WELL-00002
    """
    return [f for f in parquet_files if f['name'].upper().startswith('WELL-00002')]


def download_file(url: str, destination: Path, timeout: int = 60) -> Tuple[bool, int]:
    """
    Download a single file from URL to local destination.
    
    Args:
        url: Source URL
        destination: Local path to save file
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (success_flag, file_size_bytes)
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        
        with open(destination, 'wb') as f:
            f.write(response.content)
        
        return True, len(response.content)
    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed for {destination.name}: {e}")
        return False, 0


def validate_parquet(file_path: Path) -> Tuple[bool, int, int]:
    """
    Validate Parquet file by attempting to read it.
    
    Args:
        file_path: Path to Parquet file
        
    Returns:
        Tuple of (is_valid, rows, columns)
    """
    try:
        df = pd.read_parquet(file_path)
        return True, df.shape[0], df.shape[1]
    except Exception as e:
        logger.error(f"Validation failed for {file_path.name}: {e}")
        return False, 0, 0


def load_all_parquets(directory: Path) -> pd.DataFrame:
    """
    Load all Parquet files from directory into single DataFrame.
    
    Args:
        directory: Directory containing Parquet files
        
    Returns:
        Combined DataFrame
    """
    df_list = []
    for parquet_file in directory.glob("*.parquet"):
        df = pd.read_parquet(parquet_file)
        df_list.append(df)
        logger.info(f"Loaded: {parquet_file.name} -> {df.shape}")
    
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    return pd.DataFrame()


# -----------------------------------------------------------------------------
# PIPELINE PRINCIPAL
# -----------------------------------------------------------------------------

def run_ingestion_pipeline() -> None:
    """
    Execute the complete ingestion pipeline for Class 8 WELL files.
    """
    logger.info("Starting ingestion pipeline for Class 8")
    logger.info(f"Destination directory: {LOCAL_DIR}")
    
    # Step 1: Fetch file list
    files = fetch_file_list(API_URL)
    logger.info(f"Total items in API response: {len(files)}")
    
    # Step 2: Filter Parquet files
    parquet_files = filter_parquet_files(files)
    logger.info(f"Parquet files found: {len(parquet_files)}")
    
    # Step 3: Filter WELL files
    well_files = filter_well_files(parquet_files)
    logger.info(f"WELL files found: {len(well_files)}")
    
    if len(well_files) == 0:
        logger.warning("No WELL files found. Exiting.")
        logger.info("Sample of available Parquet files:")
        for f in parquet_files[:5]:
            logger.info(f"  - {f['name']}")
        return
    
    # Log first 10 WELL files
    logger.info("WELL files to download:")
    for f in well_files[:10]:
        logger.info(f"  - {f['name']} ({f['size']} bytes)")
    
    # Step 4: Download files
    success_count = 0
    failed_count = 0
    total_bytes = 0
    
    for idx, file_info in enumerate(well_files, 1):
        url = file_info['download_url']
        name = file_info['name']
        local_path = LOCAL_DIR / name
        
        logger.info(f"[{idx}/{len(well_files)}] Downloading: {name}")
        
        success, size_bytes = download_file(url, local_path)
        if success:
            success_count += 1
            total_bytes += size_bytes
            logger.info(f"  Saved: {local_path} ({size_bytes / 1024:.1f} KB)")
        else:
            failed_count += 1
        
        time.sleep(0.5)  # Rate limiting for GitHub API
    
    # Step 5: Validate downloaded files
    logger.info("Validating downloaded Parquet files...")
    valid_count = 0
    for parquet_file in LOCAL_DIR.glob("*.parquet"):
        is_valid, rows, cols = validate_parquet(parquet_file)
        if is_valid:
            valid_count += 1
            logger.info(f"  Valid: {parquet_file.name} ({rows} rows, {cols} cols)")
        else:
            logger.warning(f"  Invalid: {parquet_file.name}")
    
      # Step 6: Combine all data (optional)
    logger.info("Combining all Parquet files into single DataFrame...")
    df_combined = load_all_parquets(LOCAL_DIR)
    
    if not df_combined.empty:
        combined_path = LOCAL_DIR / "all_wells_combined.parquet"
        df_combined.to_parquet(combined_path)
        logger.info(f"Combined DataFrame saved: {combined_path}")
        logger.info(f"Final shape: {df_combined.shape}")
        
    # Step 7: Summary report
    logger.info("=" * 50)
    logger.info("INGESTION PIPELINE COMPLETED")
    logger.info("=" * 50)
    logger.info(f"Total WELL files found: {len(well_files)}")
    logger.info(f"Successfully downloaded: {success_count}")
    logger.info(f"Failed downloads: {failed_count}")
    logger.info(f"Valid Parquet files: {valid_count}")
    logger.info(f"Total data downloaded: {total_bytes / (1024*1024):.2f} MB")
    logger.info(f"Output directory: {LOCAL_DIR}")
    logger.info("=" * 50)


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    run_ingestion_pipeline()