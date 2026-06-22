"""
Script: ingestao_classe0_wells_only_s3.py
Descricao: Ingestion of Class 0 Parquet files from Petrobras 3W dataset.
            Filters only files starting with 'WELL-00002'.
            Uploads data directly to S3 bucket.
Author: Alan Pinheiro da Silva
Date: 2025-01-08
Version: 3.2 - Simplified
"""

import requests
import pandas as pd
import boto3
from io import BytesIO
import time
import logging
import re
from typing import List, Dict, Tuple, Optional
from botocore.exceptions import ClientError

# -----------------------------------------------------------------------------
# CONFIGURACAO
# -----------------------------------------------------------------------------

S3_BUCKET = "tcc-ptbr-3w-datalake-2026"
S3_PREFIX = "raw/0/"
AWS_REGION = "us-east-1"
API_URL = "https://api.github.com/repos/petrobras/3W/contents/dataset/0"
WELL_FILTER = "WELL-00002"
REQUEST_TIMEOUT = 60
RATE_LIMIT_DELAY = 0.3

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# FUNCOES
# -----------------------------------------------------------------------------

def get_s3_client() -> boto3.client:
    """Initialize S3 client."""
    return boto3.client('s3', region_name=AWS_REGION)


def extract_metadata_from_filename(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract ID_Poco and data_medicao from filename.
    Example: WELL-00002_20140921183228.parquet
    """
    pattern = r'^WELL-(\d+)_(\d{14})\.parquet$'
    match = re.match(pattern, filename, re.IGNORECASE)
    
    if match:
        return match.group(1), match.group(2)
    return None, None


def fetch_file_list() -> List[Dict]:
    """Fetch file list from GitHub API."""
    logger.info(f"Fetching file list from GitHub...")
    response = requests.get(API_URL, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def filter_target_files(files: List[Dict]) -> List[Dict]:
    """Filter Parquet files matching WELL_FILTER."""
    return [f for f in files 
            if f['name'].endswith('.parquet') 
            and f['name'].upper().startswith(WELL_FILTER.upper())]


def process_and_upload(file_url: str, file_name: str, s3_client: boto3.client) -> Tuple[bool, int, int]:
    """
    Download, process, and upload to S3.
    Returns (success, rows, size_bytes)
    """
    try:
        # Download from GitHub
        response = requests.get(file_url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        # Read Parquet
        df = pd.read_parquet(BytesIO(response.content))
        
        # Extract metadata and add columns
        id_poco, data_medicao = extract_metadata_from_filename(file_name)
        if id_poco:
            df['ID_Poco'] = id_poco
            df['data_medicao'] = data_medicao
        
        # Reset index if needed
        if df.index.name is not None or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            if 'index' in df.columns:
                df = df.rename(columns={'index': 'timestamp'})
        
        # Upload to S3
        buffer = BytesIO()
        df.to_parquet(buffer, index=False, compression='snappy')
        buffer.seek(0)
        
        s3_key = f"{S3_PREFIX}{file_name}"
        s3_client.upload_fileobj(buffer, S3_BUCKET, s3_key)
        
        return True, df.shape[0], len(buffer.getvalue())
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return False, 0, 0


# -----------------------------------------------------------------------------
# PIPELINE PRINCIPAL
# -----------------------------------------------------------------------------

def run_ingestion() -> None:
    """Execute the ingestion pipeline."""
    
    logger.info("=" * 50)
    logger.info(f"INGESTING DATA FOR {WELL_FILTER}")
    logger.info(f"Destination: s3://{S3_BUCKET}/{S3_PREFIX}")
    logger.info("=" * 50)
    
    # Initialize
    s3_client = get_s3_client()
    
    # Get file list
    files = fetch_file_list()
    target_files = filter_target_files(files)
    
    logger.info(f"Found {len(target_files)} files matching {WELL_FILTER}")
    
    if len(target_files) == 0:
        logger.warning("No files found. Exiting.")
        return
    
    # Process each file
    success = 0
    failed = 0
    total_rows = 0
    
    for idx, file_info in enumerate(target_files, 1):
        file_name = file_info['name']
        file_url = file_info['download_url']
        
        logger.info(f"[{idx}/{len(target_files)}] Processing: {file_name}")
        
        ok, rows, size = process_and_upload(file_url, file_name, s3_client)
        
        if ok:
            success += 1
            total_rows += rows
            logger.info(f"  Uploaded: {rows} rows, {size/1024:.1f} KB")
        else:
            failed += 1
        
        time.sleep(RATE_LIMIT_DELAY)
    
    # Summary
    logger.info("=" * 50)
    logger.info("INGESTION COMPLETED")
    logger.info(f"Success: {success}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total rows: {total_rows:,}")
    logger.info(f"Location: s3://{S3_BUCKET}/{S3_PREFIX}")
    logger.info("=" * 50)


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        run_ingestion()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")