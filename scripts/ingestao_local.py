"""
Module: ingestao_local.py
Description: Local ingestion of Petrobras 3W dataset from GitHub.
             Downloads Parquet files and organizes them by class and well.
Author: Alan Pinheiro da Silva
Date: 2025-06-12
"""

import sys
import time
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import requests

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.spark_config import (
    setup_directories, RAW_DIR, REQUEST_TIMEOUT, RATE_LIMIT_DELAY,
    DEFAULT_CLASSES, DEFAULT_WELLS, logger
)


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def extract_metadata_from_filename(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract well ID and measurement timestamp from filename.
    
    Expected format: WELL-{well_id}_{YYYYMMDDHHMMSS}.parquet
    Example: WELL-00002_20140921183228.parquet
    
    Args:
        filename: Parquet file name
    
    Returns:
        Tuple containing (well_id, timestamp_string) or (None, None)
    """
    pattern = r'^WELL-(\d+)_(\d{14})\.parquet$'
    match = re.match(pattern, filename, re.IGNORECASE)
    
    if match:
        return match.group(1), match.group(2)
    
    logger.warning(f"Filename does not match expected pattern: {filename}")
    return None, None


def format_timestamp(timestamp_str: str) -> str:
    """
    Convert raw timestamp string to readable datetime format.
    
    Args:
        timestamp_str: String in format YYYYMMDDHHMMSS
    
    Returns:
        Formatted string YYYY-MM-DD HH:MM:SS
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
        logger.error(f"Error formatting timestamp {timestamp_str}: {e}")
        return timestamp_str


def fetch_class_file_list(classe: str) -> List[Dict]:
    """
    Fetch file list for a specific class from GitHub API.

    Args:
        classe: Class number (0-9)

    Returns:
        List of file metadata dictionaries
    """
    from config.spark_config import GITHUB_TOKEN
    api_url = f"https://api.github.com/repos/petrobras/3W/contents/dataset/{classe}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    try:
        logger.info("Fetching file list for class %s", classe)
        response = requests.get(api_url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        logger.error("Failed to fetch class %s: %s", classe, exc)
        return []


def filter_well_files(files: List[Dict], wells: List[str]) -> List[Dict]:
    """
    Filter files to include only specified wells.
    
    Args:
        files: List of file metadata from GitHub API
        wells: List of well IDs to include (e.g., ['00002', '00004'])
    
    Returns:
        Filtered list of file metadata
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
    Download a file from URL to local destination.

    Args:
        file_url: Source URL
        destination_path: Local path to save file

    Returns:
        Tuple of (success_flag, file_size_bytes)
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
        logger.error("Download failed for %s: %s", destination_path.name, exc)
        return False, 0


def download_class_data(classe: str, wells: List[str], limit: Optional[int] = None) -> int:
    """
    Download all Parquet files for a specific class and well(s).
    
    Args:
        classe: Class number (0-9)
        wells: List of well IDs to download
        limit: Maximum number of files to download (for testing)
    
    Returns:
        Number of successfully downloaded files
    """
    class_dir = RAW_DIR / f"classe_{classe}"
    class_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch and filter files
    all_files = fetch_class_file_list(classe)
    target_files = filter_well_files(all_files, wells)
    
    if limit:
        target_files = target_files[:limit]
    
    logger.info("Class %s: Found %d files for wells %s", classe, len(target_files), wells)

    if not target_files:
        return 0

    success_count = 0

    for idx, file_info in enumerate(target_files, 1):
        file_name = file_info['name']
        file_url = file_info['download_url']
        local_path = class_dir / file_name

        if local_path.exists():
            logger.info("[%d/%d] Skipping existing: %s", idx, len(target_files), file_name)
            success_count += 1
            continue

        logger.info("[%d/%d] Downloading: %s", idx, len(target_files), file_name)

        success, file_size = download_file(file_url, local_path)

        if success:
            success_count += 1
            logger.info("  Saved: %.1f KB", file_size / 1024)
        else:
            logger.error("  Failed: %s", file_name)

        time.sleep(RATE_LIMIT_DELAY)

    return success_count


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def run_ingestion_pipeline(classes: List[str] = None, 
                           wells: List[str] = None, 
                           limit: Optional[int] = None) -> None:
    """
    Execute the complete data ingestion pipeline.
    
    Args:
        classes: List of classes to process (default: all 0-9)
        wells: List of well IDs to process (default: 00002, 00004, 00006)
        limit: Maximum files per class (for testing)
    """
    logger.info("=" * 60)
    logger.info("STARTING DATA INGESTION PIPELINE")
    logger.info("Classes: %s", classes or DEFAULT_CLASSES)
    logger.info("Wells: %s", wells or DEFAULT_WELLS)
    logger.info("Destination: %s", RAW_DIR)
    logger.info("=" * 60)
    
    # Setup directories
    setup_directories()
    
    # Set defaults
    classes_to_process = classes or DEFAULT_CLASSES
    wells_to_process = wells or DEFAULT_WELLS
    
    total_files = 0
    
    for classe in classes_to_process:
        logger.info("--- Processing Class %s ---", classe)
        downloaded = download_class_data(classe, wells_to_process, limit)
        total_files += downloaded
        logger.info("Class %s: Downloaded %d files", classe, downloaded)
    
    logger.info("=" * 60)
    logger.info("INGESTION PIPELINE COMPLETED")
    logger.info(f"Total files downloaded: {total_files}")
    logger.info(f"Data location: {RAW_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        # For testing, use limit=10 to download only first 10 files per class
        run_ingestion_pipeline(limit=10)
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)