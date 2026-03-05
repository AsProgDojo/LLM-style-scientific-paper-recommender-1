#!/usr/bin/env python3
"""
Download random sample of scientific papers from PubMed Central Open Access Commercial XML files (oa_comm)
from the public AWS S3 bucket.

Reads oa_comm.filelist.csv (downloaded separately via aws s3 cp)
Samples N keys deterministically
Writes manifest CSV for reproducibility
Downloads files with retries and verifies the final count

"""
import argparse
import csv
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

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

        fieldnames = {name : name for name in reader.fieldnames}
        required = ['Key', 'AccessionID', 'License']
        missing = [r for r in required if r not in fieldnames]

        if missing:
            raise ValueError(
                f'CSV missing one or more of required fields (Key, AccessionID, License). Found: {reader.fieldnames}'
            )

        key_col = fieldnames['Key']
        accession_id_col = fieldnames['AccessionID']
        license_col = fieldnames['License']

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
    s3_uri = f'{S3_BUCKET}/{key}'
    cmd = ['aws', 's3', 'cp', s3_uri, str(dest), '--no-sign-request']
    return run_cmd(cmd) == 0

def local_path_for_key(out_dir : Path, key : str) -> Path:
    """Store as  out_dir/<key> persevering directory structure e.g. out_dir/oa_comm/xml/all/PMC12345.xml"""
    return out_dir / key

def write_reproducible_output(out_path : Path, rows : List[FileRow]) -> None:
    """Make file that holds the same rows as the sample data. Used for reproducibility"""
    with out_path.open('w', encoding='utf-8', newline='') as f:
        fieldnames = ['Key', 'AccessionID', 'License']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            r = {'Key' : r.key, 'AccessionID' : r.accession_id, 'License' : r.license}
            writer.writerow(r)

def make_reproducible_output_path(out_dir : Path) -> Path:
    """Make path where the reproducible output is stored."""
    return out_dir/'reproducible_output.csv'


def count_xml_files(out_dir : Path) -> int:
    return sum(1 for _ in out_dir.rglob('*.xml'))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--filelist', required=True, type=Path, help="Path to oa_comm.filelist.csv")
    ap.add_argument('--out_dir', required=True, type=Path, help="Path to output directory")
    ap.add_argument('--n', type=int, default=10, help="Number of papers to download")
    ap.add_argument('--seed', type=int, default = 42, help="Random seed")
    ap.add_argument("--existing", type=bool, default=False, help="Choose to download from different .csv file with necessary columns")
    args = ap.parse_args()

    ensure_dir_exists(args.out_dir)
    existing = args.existing

    print(f'Reading filelist: {args.filelist}...')
    rows = read_filelist_csv(args.filelist)
    print(f'Total number of rows: {len(rows)}')

    if not existing:
        print(f'Sampling N: {args.n}, with seed={args.seed}...')
        sample = get_random_sample(rows, args.n, args.seed)

        print('Saving the sample data...')
        reproducible_output_path = make_reproducible_output_path(args.out_dir)
        write_reproducible_output(reproducible_output_path, sample)
        print(f'Sample data saved to {reproducible_output_path}')
    if existing:
        sample = rows

    print('Start downloading...')
    failures : List[Tuple[str, str]] = []
    for i, r in enumerate(sample, start=1):
        dest = local_path_for_key(args.out_dir, r.key)
        download_ok = download_one(r.key, dest)

        if not download_ok:
            failures.append((r.key, r.accession_id))

        if i % 100 == 0:
            print(f'Progress: {i}/{len(sample)} | Downloaded ok: {i - len(failures)} | Failed: {len(failures)}')

    print(f'Download finished. Failed count: {len(failures)}')

    if failures:
        print('Writing failures into a separate file...')
        failures_path = args.out_dir/'failed_downloads.csv'
        with failures_path.open('w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Key', 'Destination'])
            for k, d in failures:
                writer.writerow([k, d])
        print(f'Wrote failures file to: {failures_path}')

    xml_count = count_xml_files(args.out_dir)
    print(f'Successfully downloaded: {xml_count} XML files')
    if xml_count < args.n:
        print(f'WARNING: Downloaded less XML files than requested. Check failed_downloads.csv')

    return 0

if __name__ == '__main__':
    raise SystemExit(main())