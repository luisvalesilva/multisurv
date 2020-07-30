#!/usr/bin/env python

"""
    WSI patcher
    ~~~~~~~~~~~
    Sample patches from WSIs.
"""

import sys
import time
import os
from collections import Counter

import click
import requests
import json
import pandas as pd

# Add "src" directory to PATH
sys.path.append(os.path.split(os.getcwd())[0])
import patcher

@click.command()
@click.option('-i', '--input_dir', default=None,
              type=click.Path(exists=True),
              help='Directory containing input files.')
@click.option('-l', '--labels_file', default=None,
              type=click.Path(exists=True),
              help='File containing labels for included patients.')
@click.option('-o', '--output_dir', default=None, type=click.Path(),
              help='Path to parent output directory (containing "train",' +
            '"val", and "test" subdirectories. Default: None')
@click.option('-n', '--n_patches', default=None, type=int,
              help='Total number of patches to sample. Default: None')
@click.version_option(version='0.0.1', prog_name='Sample patches from WSIs')

def main(input_dir, labels_file, output_dir, n_patches):
    """Run WSI patching pipeline."""
    start = time.time()
    print_header()

    print('Collect paths to all slide files...')
    slide_paths = get_slide_paths(input_dir)

    labels = pd.read_csv(labels_file, sep='\t')

    # Drop slides for unused patients
    patients = list(labels.submitter_id)
    slide_paths = [path for path in slide_paths
                   if get_patient_id(path) in patients]
    n_slides = len(slide_paths)

    # Check existing patch files and drop respective WSIs from list
    slide_paths, n_dropped = drop_completed_slides(
        output_dir, slide_paths, patients)

    if n_dropped > 0:
        print(f'Dropped {n_dropped} already completed slides' + \
              f' (from a total of {n_slides})')

    print(f'Sample patches from slides ({n_patches}/slide):')

    patcher.OfflinePatcher(
        slide_files=slide_paths,
        n_patches=n_patches,
        labels_table=labels,
        target_dir=output_dir,
        patch_size=(512, 512),
        slide_level=0,
        get_random_tissue_patch=True).run()

    print_footer(start)


def print_header():
    print()
    print('*' * 42)
    print(' ' * 12, 'WSI PATCH SAMPLER')
    print('*' * 42)
    print()

def print_footer(start):
    hrs, mins, secs = elapsed_time(start)
    print(' ' * 6, f'Completed in {hrs}hrs {mins}m {secs}s')
    print('*' * 42)

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

def get_slide_paths(input_dir):
    slide_paths = []

    for leaf_dir in os.listdir(input_dir):
        path = os.path.join(input_dir, leaf_dir)
        for file in os.listdir(path):
            if file.endswith('.svs'):
                slide_path = os.path.join(path, file)
                slide_paths.append(slide_path)

    return slide_paths

def get_patient_id(filepath):
    return '-'.join(os.path.basename(filepath).split('-')[:3])

def drop_completed_slides(directory, slide_paths, patients, minimum_n=99):
    existing_patches = []
    for x in ['train', 'val', 'test']:
        existing_patches += os.listdir(os.path.join(directory, x))

    patient_patch_count = Counter([get_patient_id(x)
                                   for x in existing_patches])
    completed_patients = set([c for c in patient_patch_count
                              if patient_patch_count[c] >= minimum_n])
    kept_patients = [patient for patient in patients
                     if patient not in completed_patients]
    n = len(slide_paths)
    slide_paths = [path for path in slide_paths
                   if get_patient_id(path) in kept_patients]

    return slide_paths, n - len(slide_paths)

if __name__ == '__main__':
    main()
