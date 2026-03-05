from argparse import ArgumentParser
import subprocess

FILELIST_SOURCE = 's3://pmc-oa-opendata/oa_comm/xml/metadata/csv/oa_comm.filelist.csv'

def run_cmd(cmd):
    proc = subprocess.run(cmd)
    return proc.returncode

if __name__ == '__main__':

    ap = ArgumentParser()
    ap.add_argument('--out_dir', default='.', help='Output directory where the PMC paper metadata will be saved')
    args = ap.parse_args()

    filelist_destination = args.out_dir
    cmd = ['aws', 's3', 'cp', FILELIST_SOURCE, filelist_destination, '--no-sign-request']
    if run_cmd(cmd) == 0:
        print(f'Successfully downloaded PMC paper metadata')
    else:
        print('Error downloading PMC paper metadata. Try running the program again')


