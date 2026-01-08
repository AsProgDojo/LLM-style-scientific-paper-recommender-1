#!/usr/bin/env python3
"""
Download random sample of scientific papers from PubMed Central Open Access Commercial XML files (oa_comm)
from the public AWS S3 bucket.

Reads oa_comm.filelist.csv (downloaded separately via aws s3 cp)
Samples N keys deterministically
Writes manifest CSV for reproducibility
Downloads files with retries and verifies the final count

Usage:



"""
import csv
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

S3_BUCKET = 's3://pmc-oa-opendata'

@dataclass(frozen=True)
class FileRow:
    key: str #download location of the paper
    accession_id : Optional[str] = None #for tracking paper throughout the program
    license : Optional[str] = None

def read_filelist_csv(path : Path) -> List[FileRow]:
    """
    Read from the oa_comm.filelist.csv file. and return the rows with AccessionId, License and Key column
    where the last one represents the AWS S3 key.
    """

    rows : List[FileRow] = []
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError('CSV has no header/fieldnames')

        fieldnames = {name.lower() : name for name in reader.fieldnames}
        key_col = fieldnames['key']
        accession_id_col = fieldnames['accessionid']
        license_col = fieldnames['license']

        if not key_col or not accession_id_col or not license_col in fieldnames:
            raise ValueError(
                f'CSV missing one or more of required fields (Key, AccessionId, License). Found: {reader.fieldnames}'
            )

        for row in reader:
            key = row.get(key_col, "").strip()
            if not key:
                continue
            accession_id = (row.get(accession_id_col, "").strip() if row.get(accession_id_col) else None)
            license_type = (row.get(license_col, "").strip() if row.get(license_col) else None)

            rows.append(FileRow(key, accession_id, license_type))

    if not rows:
        raise ValueError('No rows loaded from filelist CSV')
    return rows

def get_random_sample(rows: List[FileRow], n : int, seed : int) -> List[FileRow]:
    """Get random sample from the list of rows each representing scientific paper's metadata."""
    if n < 0:
        raise ValueError('n cannot be negative')
    elif n > len(rows):
        raise ValueError(f'Requested n = {n} but the number of rows is {len(rows)}')

    rng = random.Random(seed)
    return rng.sample(rows, n) # sample without replacement

def ensure_dir_exists(path: Path) -> None:
    """Check if a directory exists, if not, create it."""
    path.mkdir(parents=True, exist_ok=True)

def run_cmd(cmd: List[str]) -> int:
    """Run a command, return exit code"""
    proc = subprocess.run(cmd)
    return proc.returncode

def download_one(key : str, dest : Path) -> bool:
    """
    Download one S3 object using AWS CLI (public bucket, no-sign-request)
    Returns True on success, False on fail
    """
    ensure_dir_exists(dest)
    s3_uri = f's3://{S3_BUCKET}/{key}'
    cmd = ['aws', 's3', 'cp', s3_uri, str(dest), '--no-sign-request']
    return run_cmd(cmd) == 0

