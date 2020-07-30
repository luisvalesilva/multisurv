#!/usr/bin/env python

"""
    Compute variance
    ~~~~~~~~~~~~~~~~
    Compute gene value variance across all samples and save to file.
"""

import time
import os
from io import StringIO

import click
import requests
import json
import pandas as pd

@click.command()
@click.option('-i', '--input_file_dir', default=None,
              type=click.Path(exists=True),
              help='Directory containing input files.')
@click.option('-s', '--chunk_size', default=10000, type=int,
              help='Chunk size (number of lines, i.e. genes) to process at a' +
              ' time. Default: 10000')
@click.option('-o', '--output_file', default=None, type=click.Path(),
              help='Path to output file. Default: None')
@click.option('-t', '--n_samples', default=None, type=int,
              help='Number of samples to run (allows running a subset for' +
              ' testing). Default: None')
@click.version_option(version='0.0.1', prog_name='Compute gene variance')
def main(input_file_dir, chunk_size, output_file, n_samples):
    """Run variance calculation pipeline."""
    start = time.time()
    print_header()

    print('Downloading metadata from GDC database...')

    mRNA_files = request_file_info()
    mRNA_files = mRNA_files[
        mRNA_files['cases.0.project.project_id'].str.startswith('TCGA')]
    mRNA_files = mRNA_files[
        mRNA_files['file_name'].str.endswith('FPKM-UQ.txt.gz')]
    mRNA_files = mRNA_files[
        mRNA_files['cases.0.samples.0.sample_type'] == 'Primary Tumor']

    # When there is more than one file for a single patient just keep the first 
    # (this is assuming they are just replicates and all similar)
    mRNA_files = mRNA_files[~mRNA_files.duplicated(
        subset=['cases.0.submitter_id'], keep='first')]

    file_map = make_patient_file_map(
        mRNA_files, base_dir=input_file_dir)

    # Subset
    if n_samples is not None:
        print('Keeping only subset of {n_samples} samples...')
        file_map = {key: file_map[key] for i, key in enumerate(file_map)
                    if i < n_samples}

    eg_file = file_map[list(file_map.keys())[0]]
    total_n_lines = len(list(pd.read_csv(
        eg_file, sep='\t', header=None, index_col=0, names=['count']).index))

    print('Process gene chunks:')
    variance_table = pd.DataFrame()

    for bin in [range(i, i + chunk_size)
                for i in range(0, total_n_lines, chunk_size)]:
        print('>>>', bin)
        chunks = load_data_chunk(file_map, total_n_lines, list(bin))
        chunks = merge_dfs(chunks)

        if variance_table.empty:
            variance_table = get_var(chunks)
        else:
            variance_table = pd.concat([variance_table, get_var(chunks)])

    # Remove rows with ambiguous reads, not aligned, etc., included by HTSeq
    # (they start with '__')
    variance_table = variance_table[~variance_table.index.str.startswith('__')]
    variance_table.shape

    #-------------------------------------------------------------------------#
    # Save to file
    print()
    print('Saving result to file:')
    print(f'"{output_file}"')
    variance_table.to_csv(output_file, sep='\t', index=True)

    print_footer(start)


def print_header():
    print()
    print(' ' * 15, '*' * 25)
    print(' ' * 15, '  COMPUTE GENE VARIANCE')
    print(' ' * 15, '*' * 25)
    print()

def print_footer(start):
    hrs, mins, secs = elapsed_time(start)
    print()
    print(' ' * 18, f'Completed in {hrs}hrs {mins}m {secs}s')
    print(' ' * 15, '*' * 25)

def elapsed_time(start):
    """Compute time since provided start time.

    Parameters
    ----------
    start: float
        Output of time.time().
    Returns
    -------
    Elapsed hours, minutes and seconds (as tuple of int).
    """
    time_elapsed = time.time() - start
    hrs = time_elapsed // 3600
    secs = time_elapsed % 3600
    mins = secs // 60
    secs = secs % 60

    return int(hrs), int(mins), int(secs)

def request_file_info():
    fields = [
        "file_name",
        "cases.submitter_id",
        "cases.samples.sample_type",
        "cases.project.project_id",
        "cases.project.primary_site",
        ]

    fields = ",".join(fields)

    files_endpt = "https://api.gdc.cancer.gov/files"

    filters = {
        "op": "and",
        "content":[
            {
            "op": "in",
            "content":{
                "field": "files.experimental_strategy",
                "value": ['RNA-Seq']
                }
            }
        ]
    }

    params = {
        "filters": filters,
        "fields": fields,
        "format": "TSV",
        "size": "200000"
        }

    response = requests.post(
        files_endpt,
        headers={"Content-Type": "application/json"},
        json=params)

    return pd.read_csv(StringIO(response.content.decode("utf-8")), sep="\t")

def make_patient_file_map(df, base_dir):
    return {row['cases.0.submitter_id']: os.path.join(
        base_dir, row.id, row.file_name)
            for _, row in df.iterrows()}

def load_data_chunk(patient_file_map, total_lines, chunk_lines):
    n = len(patient_file_map)

    rows_to_skip = [x for x in list(range(0, total_lines))
                    if not x in chunk_lines]

    dfs = []
    for i, patient in enumerate(patient_file_map):
        print('\r' + f'   Load tables: {str(i + 1)}/{n}', end='')
        df = pd.read_csv(patient_file_map[patient], sep='\t', header=None,
                         index_col=0, names=['FPKM-UQ'], skiprows=rows_to_skip)
        df.columns = [patient]
        dfs.append(df)

    print()

    return dfs

def merge_dfs(table_list):
    n = len(table_list)

    final_table = pd.DataFrame()

    for i, table in enumerate(table_list):
        print('\r' + f'   Merge tables: {str(i + 1)}/{n}', end='')
        if final_table.empty:
            final_table = table
        else:
            final_table = final_table.join(table)

    print()

    return final_table

def get_var(table):
    print('   Compute count variance...')

    return table.var(axis=1)


if __name__ == '__main__':
    main()
