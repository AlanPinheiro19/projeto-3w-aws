"""
Script: ingestao_todas_classes.py
Descricao: Ingestion of ALL Classes (0 to 9) Parquet files from Petrobras 3W dataset.
            Filters files for wells: WELL-00002, WELL-00004, WELL-00006.
            Extracts ID_Poco, data_medicao, and classe_evento from filename/path.
Author: Alan Pinheiro da Silva
Date: 2025-01-08
Version: 3.0 - All classes (0-9)
"""

import requests
import pandas as pd
import os
from pathlib import Path
import time
import logging
import re
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------------------------------------------------------------
# CONFIGURACAO
# -----------------------------------------------------------------------------

# Classes to ingest (0 = normal operation, 1-9 = undesirable events)
CLASSES = [str(i) for i in range(10)]  # ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Wells to filter
WELLS_TO_FILTER = ["WELL-00002", "WELL-00004", "WELL-00006"]

# Base directories
BASE_DIR = Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parent.parent.parent)) / "data"
RAW_DIR = BASE_DIR / "raw"  # Raw downloaded files
PROCESSED_DIR = BASE_DIR / "processed"  # Processed files with metadata

# GitHub API base URL
GITHUB_API_BASE = "https://api.github.com/repos/petrobras/3W/contents/dataset"

# Processing configuration
REQUEST_TIMEOUT = 60
RATE_LIMIT_DELAY = 0.3
MAX_WORKERS = 1  # Keep at 1 to avoid GitHub rate limits

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create directories
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# FUNCOES AUXILIARES
# -----------------------------------------------------------------------------

def extract_metadata_from_filename(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract ID_Poco and data_medicao from filename.
    
    Format: WELL-{ID_Poco}_{YYYYMMDDHHMMSS}.parquet
    Example: WELL-00002_20140921183228.parquet
    
    Returns:
        Tuple of (id_poco, data_medicao)
    """
    pattern = r'^WELL-(\d+)_(\d{14})\.parquet$'
    match = re.match(pattern, filename, re.IGNORECASE)
    
    if match:
        return match.group(1), match.group(2)
    return None, None


def format_data_medicao(timestamp_str: str) -> str:
    """
    Format data_medicao string to readable datetime format.
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
    except Exception:
        return timestamp_str


def fetch_class_files(classe: str) -> List[Dict]:
    """
    Fetch file list for a specific class from GitHub API.
    
    Args:
        classe: Class number (0-9)
    
    Returns:
        List of file metadata dictionaries
    """
    api_url = f"{GITHUB_API_BASE}/{classe}"
    
    try:
        logger.info(f"Fetching files for class {classe}...")
        response = requests.get(api_url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch class {classe}: {e}")
        return []


def filter_well_files(files: List[Dict], wells: List[str]) -> List[Dict]:
    """
    Filter files for specific wells.
    """
    filtered = []
    for f in files:
        if not f['name'].endswith('.parquet'):
            continue
        
        file_upper = f['name'].upper()
        for well in wells:
            if file_upper.startswith(well.upper()):
                filtered.append(f)
                break
    
    return filtered


def download_file(url: str, destination: Path) -> Tuple[bool, int]:
    """
    Download a single file from URL to local destination.
    """
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        with open(destination, 'wb') as f:
            f.write(response.content)
        
        return True, len(response.content)
    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed for {destination.name}: {e}")
        return False, 0


def process_parquet_file(file_path: Path, classe: str) -> Tuple[bool, int, int, Dict]:
    """
    Process Parquet file: add metadata columns (ID_Poco, data_medicao, classe_evento).
    
    Returns:
        Tuple of (success, rows, columns, metadata)
    """
    try:
        # Read Parquet
        df = pd.read_parquet(file_path)
        
        # Extract metadata from filename
        id_poco, data_medicao_raw = extract_metadata_from_filename(file_path.name)
        
        # Add new columns
        df['ID_Poco'] = id_poco if id_poco else 'UNKNOWN'
        df['classe_evento'] = int(classe)
        df['data_medicao_raw'] = data_medicao_raw if data_medicao_raw else ''
        df['data_medicao'] = format_data_medicao(data_medicao_raw) if data_medicao_raw else ''
        
        # Reset index if it's a timestamp
        if df.index.name is not None or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            if 'index' in df.columns:
                df = df.rename(columns={'index': 'timestamp'})
        
        # Save processed file
        output_path = PROCESSED_DIR / file_path.name
        df.to_parquet(output_path, index=False)
        
        metadata = {
            'filename': file_path.name,
            'classe': classe,
            'id_poco': id_poco,
            'rows': df.shape[0],
            'columns': df.shape[1]
        }
        
        return True, df.shape[0], df.shape[1], metadata
        
    except Exception as e:
        logger.error(f"Processing failed for {file_path.name}: {e}")
        return False, 0, 0, {}


def process_class(classe: str, wells: List[str]) -> Dict:
    """
    Process all files for a specific class.
    
    Returns:
        Dictionary with statistics for this class
    """
    logger.info("=" * 60)
    logger.info(f"PROCESSING CLASS {classe}")
    logger.info("=" * 60)
    
    # Create class-specific raw directory
    class_raw_dir = RAW_DIR / f"classe_{classe}"
    class_raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch files for this class
    files = fetch_class_files(classe)
    
    if not files:
        logger.warning(f"No files found for class {classe}")
        return {'classe': classe, 'found': 0, 'downloaded': 0, 'processed': 0, 'rows': 0}
    
    # Filter for target wells
    well_files = filter_well_files(files, wells)
    logger.info(f"Class {classe}: Found {len(well_files)} files for wells {wells}")
    
    if not well_files:
        return {'classe': classe, 'found': 0, 'downloaded': 0, 'processed': 0, 'rows': 0}
    
    # Download and process each file
    downloaded = 0
    processed = 0
    total_rows = 0
    total_size = 0
    
    for idx, file_info in enumerate(well_files, 1):
        file_name = file_info['name']
        file_url = file_info['download_url']
        local_path = class_raw_dir / file_name
        
        logger.info(f"[{idx}/{len(well_files)}] Class {classe}: Downloading {file_name}")
        
        # Download file
        success, size_bytes = download_file(file_url, local_path)
        
        if success:
            downloaded += 1
            total_size += size_bytes
            logger.info(f"  Downloaded: {size_bytes / 1024:.1f} KB")
            
            # Process file (add metadata)
            proc_success, rows, cols, metadata = process_parquet_file(local_path, classe)
            
            if proc_success:
                processed += 1
                total_rows += rows
                logger.info(f"  Processed: {rows} rows, {cols} cols")
            else:
                logger.warning(f"  Processing failed for {file_name}")
        else:
            logger.warning(f"  Download failed for {file_name}")
        
        time.sleep(RATE_LIMIT_DELAY)
    
    return {
        'classe': classe,
        'found': len(well_files),
        'downloaded': downloaded,
        'processed': processed,
        'rows': total_rows,
        'size_bytes': total_size
    }


def consolidate_all_data() -> pd.DataFrame:
    """
    Load all processed Parquet files into a single DataFrame.
    """
    logger.info("=" * 60)
    logger.info("CONSOLIDATING ALL CLASSES")
    logger.info("=" * 60)
    
    df_list = []
    
    for parquet_file in PROCESSED_DIR.glob("*.parquet"):
        if parquet_file.name == "all_classes_combined.parquet":
            continue
        
        try:
            df = pd.read_parquet(parquet_file)
            df_list.append(df)
            logger.info(f"Loaded: {parquet_file.name} -> {df.shape}")
        except Exception as e:
            logger.error(f"Failed to load {parquet_file.name}: {e}")
    
    if not df_list:
        logger.warning("No processed files found")
        return pd.DataFrame()
    
    # Combine all DataFrames
    df_combined = pd.concat(df_list, ignore_index=True)
    
    # Save consolidated files
    combined_parquet = PROCESSED_DIR / "all_classes_combined.parquet"
    combined_csv = PROCESSED_DIR / "all_classes_combined.csv"
    
    df_combined.to_parquet(combined_parquet, index=False)
    df_combined.to_csv(combined_csv, index=False)
    
    logger.info(f"Consolidated Parquet saved: {combined_parquet}")
    logger.info(f"Consolidated CSV saved: {combined_csv}")
    logger.info(f"Final shape: {df_combined.shape[0]:,} rows x {df_combined.shape[1]} columns")
    
    return df_combined


def print_final_summary(results: List[Dict], df_combined: pd.DataFrame) -> None:
    """
    Print detailed summary of all classes ingestion.
    """
    logger.info("=" * 70)
    logger.info("FINAL SUMMARY - ALL CLASSES INGESTION")
    logger.info("=" * 70)
    
    # Class summary
    logger.info("\nBy Class:")
    logger.info("-" * 50)
    logger.info(f"{'Class':<8} {'Found':<10} {'Downloaded':<12} {'Processed':<12} {'Rows':<15}")
    logger.info("-" * 50)
    
    total_found = 0
    total_downloaded = 0
    total_processed = 0
    total_rows = 0
    
    for r in results:
        logger.info(f"{r['classe']:<8} {r['found']:<10} {r['downloaded']:<12} {r['processed']:<12} {r['rows']:<15,}")
        total_found += r['found']
        total_downloaded += r['downloaded']
        total_processed += r['processed']
        total_rows += r['rows']
    
    logger.info("-" * 50)
    logger.info(f"{'TOTAL':<8} {total_found:<10} {total_downloaded:<12} {total_processed:<12} {total_rows:<15,}")
    
    # Wells summary
    if not df_combined.empty and 'ID_Poco' in df_combined.columns:
        logger.info("\nBy Well:")
        logger.info("-" * 40)
        well_counts = df_combined['ID_Poco'].value_counts()
        for well, count in well_counts.items():
            logger.info(f"  WELL-{well}: {count:,} rows")
    
    # Class distribution
    if not df_combined.empty and 'classe_evento' in df_combined.columns:
        logger.info("\nBy Event Class:")
        logger.info("-" * 40)
        class_counts = df_combined['classe_evento'].value_counts().sort_index()
        for cls, count in class_counts.items():
            class_name = "Normal Operation" if cls == 0 else f"Event Type {cls}"
            logger.info(f"  Class {cls} ({class_name}): {count:,} rows ({count/len(df_combined)*100:.1f}%)")
    
    # Date range
    if not df_combined.empty and 'data_medicao' in df_combined.columns:
        logger.info("\nDate Range:")
        logger.info("-" * 40)
        logger.info(f"  From: {df_combined['data_medicao'].min()}")
        logger.info(f"  To:   {df_combined['data_medicao'].max()}")
    
    logger.info("\n" + "=" * 70)
    logger.info(f"Raw files directory: {RAW_DIR}")
    logger.info(f"Processed files directory: {PROCESSED_DIR}")
    logger.info("=" * 70)


# -----------------------------------------------------------------------------
# PIPELINE PRINCIPAL
# -----------------------------------------------------------------------------

def run_ingestion_pipeline() -> None:
    """
    Execute the complete ingestion pipeline for all classes (0-9).
    """
    logger.info("=" * 70)
    logger.info("STARTING INGESTION PIPELINE - ALL CLASSES (0 TO 9)")
    logger.info(f"Target wells: {', '.join(WELLS_TO_FILTER)}")
    logger.info(f"Classes to process: {', '.join(CLASSES)}")
    logger.info(f"Raw directory: {RAW_DIR}")
    logger.info(f"Processed directory: {PROCESSED_DIR}")
    logger.info("=" * 70)
    
    start_time = time.time()
    results = []
    
    # Process each class
    for classe in CLASSES:
        result = process_class(classe, WELLS_TO_FILTER)
        results.append(result)
    
    # Consolidate all processed data
    df_combined = consolidate_all_data()
    
    # Print final summary
    print_final_summary(results, df_combined)
    
    # Execution time
    elapsed = time.time() - start_time
    logger.info(f"\nTotal execution time: {elapsed / 60:.1f} minutes")


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        run_ingestion_pipeline()
    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise