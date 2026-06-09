"""
Script: ingestao_classe1_wells_only.py
Descricao: Ingestion of Class 1 Parquet files from Petrobras 3W dataset.
            Filters only files starting with 'WELL' (case-insensitive).
            Extracts ID_Poco and data_medicao from filename and adds as columns.
Author: Alan Pinheiro da Silva
Date: 2025-01-08
Version: 2.0
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO
import time
import logging
import re
from typing import List, Dict, Tuple, Optional

# -----------------------------------------------------------------------------
# CONFIGURACAO
# -----------------------------------------------------------------------------

CLASSE = "1"
BASE_DIR = Path("C:/projeto-3w-aws/data")
LOCAL_DIR = BASE_DIR / f"classe{CLASSE}_wells_only"
PROCESSED_DIR = BASE_DIR / f"classe{CLASSE}_processed"  # Diretorio para arquivos processados
API_URL = f"https://api.github.com/repos/petrobras/3W/contents/dataset/{CLASSE}"

# Configuracao de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Criar diretorios
LOCAL_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# FUNCOES AUXILIARES
# -----------------------------------------------------------------------------

def extract_metadata_from_filename(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract ID_Poco and data_medicao from filename.
    
    Format: WELL-{ID_Poco}_{YYYYMMDDHHMMSS}.parquet
    Example: WELL-00011_20140921183228.parquet
    
    Args:
        filename: Parquet filename (e.g., WELL-00011_20140921183228.parquet)
        
    Returns:
        Tuple of (id_poco, data_medicao)
        - id_poco: string like "00011"
        - data_medicao: string like "20140921183228"
    """
    # Regex pattern to match WELL-{id}_{timestamp}
    pattern = r'^WELL-(\d+)_(\d{14})\.parquet$'
    match = re.match(pattern, filename, re.IGNORECASE)
    
    if match:
        id_poco = match.group(1)      # e.g., "00011"
        data_medicao = match.group(2)  # e.g., "20140921183228"
        return id_poco, data_medicao
    else:
        logger.warning(f"Filename does not match expected pattern: {filename}")
        return None, None


def format_data_medicao(timestamp_str: str) -> str:
    """
    Format data_medicao string to a more readable datetime format.
    
    Args:
        timestamp_str: String like "20140921183228" (YYYYMMDDHHMMSS)
        
    Returns:
        Formatted string like "2014-09-21 18:32:28"
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
    Filter files that start with 'WELL' (case-insensitive).
    
    Args:
        parquet_files: List of Parquet file metadata
        
    Returns:
        Filtered list containing only files starting with WELL
    """
    return [f for f in parquet_files if f['name'].upper().startswith('WELL')]


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


def process_parquet_file(file_path: Path, output_path: Path) -> Tuple[bool, Dict]:
    """
    Process a single Parquet file: add ID_Poco and data_medicao columns.
    
    Args:
        file_path: Source Parquet file path
        output_path: Destination path for processed file
        
    Returns:
        Tuple of (success_flag, metadata_dict)
    """
    try:
        # Extract metadata from filename
        id_poco, data_medicao_raw = extract_metadata_from_filename(file_path.name)
        
        if id_poco is None or data_medicao_raw is None:
            logger.warning(f"Skipping file with invalid name format: {file_path.name}")
            return False, {}
        
        # Read the Parquet file
        df = pd.read_parquet(file_path)
        
        # Add new columns
        df['ID_Poco'] = id_poco
        df['data_medicao_raw'] = data_medicao_raw
        df['data_medicao'] = format_data_medicao(data_medicao_raw)
        
        # Optionally, if timestamp is the index, reset it to a column
        if df.index.name is not None or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            if 'index' in df.columns:
                df = df.rename(columns={'index': 'timestamp'})
        
        # Save processed Parquet
        df.to_parquet(output_path, index=False)
        
        metadata = {
            'filename': file_path.name,
            'id_poco': id_poco,
            'data_medicao': data_medicao_raw,
            'original_shape': df.shape,
            'columns': list(df.columns)
        }
        
        logger.info(f"  Processed: {file_path.name} -> {df.shape[0]} rows, {df.shape[1]} cols")
        return True, metadata
        
    except Exception as e:
        logger.error(f"Error processing {file_path.name}: {e}")
        return False, {}


def load_all_processed_parquets(directory: Path) -> pd.DataFrame:
    """
    Load all processed Parquet files from directory into single DataFrame.
    
    Args:
        directory: Directory containing processed Parquet files
        
    Returns:
        Combined DataFrame
    """
    df_list = []
    for parquet_file in directory.glob("*.parquet"):
        if parquet_file.name == "all_wells_combined.parquet":
            continue
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
    Execute the complete ingestion pipeline for Class 1 WELL files.
    """
    logger.info("=" * 60)
    logger.info("STARTING INGESTION PIPELINE FOR CLASS 1")
    logger.info(f"Raw download directory: {LOCAL_DIR}")
    logger.info(f"Processed output directory: {PROCESSED_DIR}")
    logger.info("=" * 60)
    
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
    
    # Log first 10 WELL files with extracted metadata
    logger.info("WELL files to download (with extracted metadata):")
    for f in well_files[:10]:
        id_poco, data_med = extract_metadata_from_filename(f['name'])
        logger.info(f"  - {f['name']} (ID_Poco: {id_poco}, Data: {data_med})")
    
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
        
        time.sleep(0.3)  # Rate limiting for GitHub API
    
    # Step 5: Process each downloaded file (add ID_Poco and data_medicao)
    logger.info("=" * 60)
    logger.info("PROCESSING FILES - ADDING METADATA COLUMNS")
    logger.info("=" * 60)
    
    processed_count = 0
    processing_errors = 0
    all_metadata = []
    
    for parquet_file in LOCAL_DIR.glob("*.parquet"):
        output_file = PROCESSED_DIR / parquet_file.name
        
        logger.info(f"Processing: {parquet_file.name}")
        
        success, metadata = process_parquet_file(parquet_file, output_file)
        
        if success:
            processed_count += 1
            all_metadata.append(metadata)
            logger.info(f"  Output saved: {output_file}")
        else:
            processing_errors += 1
    
    # Step 6: Combine all processed data into single DataFrame
    logger.info("=" * 60)
    logger.info("COMBINING PROCESSED DATA")
    logger.info("=" * 60)
    
    df_combined = load_all_processed_parquets(PROCESSED_DIR)
    
    if not df_combined.empty:
        combined_path = PROCESSED_DIR / "all_wells_combined.parquet"
        df_combined.to_parquet(combined_path, index=False)
        
        # Also save as CSV for easier inspection (optional)
        csv_path = PROCESSED_DIR / "all_wells_combined.csv"
        df_combined.to_csv(csv_path, index=False)
        
        logger.info(f"Combined DataFrame saved: {combined_path}")
        logger.info(f"Combined CSV saved: {csv_path}")
        logger.info(f"Final shape: {df_combined.shape}")
        
        # Show column information
        logger.info("Columns in combined DataFrame:")
        for col in df_combined.columns:
            logger.info(f"  - {col}: {df_combined[col].dtype}")
        
        # Show unique ID_Poco values
        if 'ID_Poco' in df_combined.columns:
            unique_wells = df_combined['ID_Poco'].unique()
            logger.info(f"Unique wells in dataset: {len(unique_wells)}")
            logger.info(f"  IDs: {sorted(unique_wells)[:10]}...")
        
        # Show date range of data_medicao
        if 'data_medicao' in df_combined.columns:
            logger.info("Date range of measurements:")
            logger.info(f"  Min: {df_combined['data_medicao'].min()}")
            logger.info(f"  Max: {df_combined['data_medicao'].max()}")
    
    # Step 7: Summary report
    logger.info("=" * 60)
    logger.info("INGESTION PIPELINE COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Total WELL files found: {len(well_files)}")
    logger.info(f"Successfully downloaded: {success_count}")
    logger.info(f"Failed downloads: {failed_count}")
    logger.info(f"Successfully processed: {processed_count}")
    logger.info(f"Processing errors: {processing_errors}")
    logger.info(f"Total data downloaded: {total_bytes / (1024*1024):.2f} MB")
    logger.info(f"Raw download directory: {LOCAL_DIR}")
    logger.info(f"Processed output directory: {PROCESSED_DIR}")
    
    if not df_combined.empty:
        logger.info(f"Combined dataset shape: {df_combined.shape[0]} rows x {df_combined.shape[1]} columns")
    
    logger.info("=" * 60)


# -----------------------------------------------------------------------------
# FUNCAO PARA ANALISE RAPIDA APOS PROCESSAMENTO
# -----------------------------------------------------------------------------

def quick_analysis() -> None:
    """
    Perform quick analysis on processed data after ingestion.
    """
    combined_path = PROCESSED_DIR / "all_wells_combined.parquet"
    
    if not combined_path.exists():
        logger.warning("Combined file not found. Run ingestion pipeline first.")
        return
    
    logger.info("=" * 60)
    logger.info("QUICK ANALYSIS OF PROCESSED DATA")
    logger.info("=" * 60)
    
    df = pd.read_parquet(combined_path)
    
    logger.info(f"Total records: {len(df):,}")
    logger.info(f"Total columns: {len(df.columns)}")
    
    # Check for null values
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if not null_cols.empty:
        logger.info("Columns with null values:")
        for col, count in null_cols.items():
            logger.info(f"  - {col}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Show class distribution (if class column exists)
    if 'class' in df.columns:
        logger.info("Class distribution:")
        for cls, count in df['class'].value_counts().items():
            logger.info(f"  Class {cls}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Show state distribution (if state column exists)
    if 'state' in df.columns:
        logger.info("State distribution:")
        for state, count in df['state'].value_counts().items():
            logger.info(f"  State {state}: {count:,} ({count/len(df)*100:.1f}%)")


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        run_ingestion_pipeline()
        
        # Optional: run quick analysis after processing
        run_quick_analysis = input("\nRun quick analysis on processed data? (y/n): ").lower()
        if run_quick_analysis == 'y':
            quick_analysis()
            
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise