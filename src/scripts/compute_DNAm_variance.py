#!/usr/bin/env python

"""
    Compute variance
    ~~~~~~~~~~~~~~~~
    Compute DNA methylation probe value variance across all samples and save to
    file.
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
@click.option('-l', '--labels_file', default=None,
              type=click.Path(exists=True),
              help='File containing labels for included patients.')
@click.option('-p', '--probe_file', default=None,
              type=click.Path(exists=True),
              help='File containing selected probes.')
@click.option('-o', '--output_file', default=None, type=click.Path(),
              help='Path to output file. Default: None')
@click.option('-t', '--n_samples', default=None, type=int,
              help='Number of samples to run (allows running a subset for' +
              ' testing). Default: None')
@click.version_option(version='0.0.1', prog_name='Compute DNAm probe variance')

def main(input_file_dir, chunk_size, labels_file, probe_file, output_file,
         n_samples):
    """Run variance calculation pipeline."""
    start = time.time()
    print_header()

    print('Downloading metadata from GDC database...')

    DNAm_files = request_file_info()
    DNAm_files = DNAm_files[
        DNAm_files['cases.0.project.project_id'].str.startswith('TCGA')]
    DNAm_files = DNAm_files[
        DNAm_files['file_name'].str.endswith('gdc_hg38.txt')]
    DNAm_files = DNAm_files[
        DNAm_files['cases.0.samples.0.sample_type'] == 'Primary Tumor']

    # When there is more than one file for a single patient just keep the first 
    # (this is assuming they are just replicates and all similar)
    DNAm_files = DNAm_files[~DNAm_files.duplicated(
        subset=['cases.0.submitter_id'], keep='first')]

    file_map = make_patient_file_map(
        DNAm_files, base_dir=input_file_dir)

    # Drop unused patients
    labels = pd.read_csv(labels_file, sep='\t')
    file_map = {k: file_map[k] for k in file_map
                if k in list(labels['submitter_id'])}

    # Subset
    if n_samples is not None:
        print(f'Keeping only subset of {n_samples} samples...')
        file_map = {key: file_map[key] for i, key in enumerate(file_map)
                    if i < n_samples}

    # Selected probes
    probe_set = pd.read_csv(probe_file, sep='\t', header=None,
                            names=['Probes'])
    dfs = load_all_data(file_map, probe_set.Probes)

    print('Process probe chunks:')
    variance_table = pd.DataFrame()

    for bin in [range(i, i + chunk_size)
                for i in range(0, len(probe_set), chunk_size)]:
        print('   >>>', bin)
        probe_subset = probe_set.Probes[bin[0]:bin[-1]]
        subset_dfs = [x[x.index.isin(probe_subset)] for x in dfs]
        chunk = merge_dfs(subset_dfs)

        if variance_table.empty:
            variance_table = chunk.var(axis=1)
        else:
            variance_table = pd.concat([variance_table, chunk.var(axis=1)])

    #-------------------------------------------------------------------------#
    # Save to file
    print()
    print('Saving result to file:')
    print(f'"{output_file}"')
    variance_table.to_csv(output_file, sep='\t', index=True)

    print_footer(start)


def print_header():
    print()
    print(' ' * 10, '*' * 25)
    print(' ' * 12, 'COMPUTE DNAm VARIANCE')
    print(' ' * 10, '*' * 25)
    print()

def print_footer(start):
    hrs, mins, secs = elapsed_time(start)
    print()
    print(' ' * 10, f'Completed in {hrs}hrs {mins}m {secs}s')
    print('*' * 46)

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
                "value": ['Methylation Array']
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

def load_all_data(patient_file_map, probe_set):
    n = len(patient_file_map)

    dfs = []
    for i, patient in enumerate(patient_file_map):
        print('\r' + f'   Load tables: {str(i + 1)}/{n}', end='')
        df = pd.read_csv(patient_file_map[patient], sep='\t', index_col=0,
                         usecols=['Composite Element REF', 'Beta_value'])
        # Keep only selected probes
        df = df[df.index.isin(probe_set)]
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

if __name__ == '__main__':
    main()
