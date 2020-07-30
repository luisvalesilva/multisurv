"""Utility methods."""


import os
import time

import numpy as np
import pandas as pd
import torch
import torchvision
from lifelines import KaplanMeierFitter

import dataset
import transforms as patch_transforms
from baseline_models import Baselines 
from evaluation import Evaluation
from result_table_writer import ResultTable
import plotting as plot


# Storage location of data and trained model weights
INPUT_DATA_DIR = '/mnt/dataA/TCGA/processed/'
TRAINED_MODEL_DIR = '/mnt/dataA/multisurv_models/'


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

def get_label_map(data_file, split_group='train'):
    """Make dictionary of patient labels.

    Parameters
    ----------
    split_group: str
        Train-val-test split group name in the survival label table to subset
        data by.
    Returns
    -------
    Dict with time and event (censor variable) values for each patient id key.
    """
    df = pd.read_csv(data_file, sep='\t')

    if split_group is not None:
        groups = list(df['group'].unique())
        assert split_group in groups, f'Accepted "split_group"s are: {groups}'

        df = df.loc[df['group'] == split_group]

    keys = list(df['submitter_id'])
    values = zip(list(df['time']), list(df['event']))

    return dict(zip(keys, values))

def get_dataloaders(data_location, labels_file, modalities,
                    wsi_patch_size=None, n_wsi_patches=None, batch_size=None,
                    exclude_patients=None, return_patient_id=False):
    """Instantiate PyTorch DataLoaders.

    Parameters
    ----------
    Returns
    -------
    Dict of Pytorch Dataloaders.
    """
    data_dirs = {
        'clinical': os.path.join(data_location, 'Clinical'),
        'wsi': os.path.join(data_location, 'WSI'),
        'mRNA': os.path.join(data_location, 'RNA-seq'),
        'miRNA': os.path.join(data_location, 'miRNA-seq'),
        'DNAm': os.path.join(data_location, 'DNAm/5k'),
        'CNV': os.path.join(data_location, 'CNV'),
    }

    data_dirs = {mod: data_dirs[mod] for mod in modalities}
    if batch_size is None:
        if 'wsi' in data_dirs.keys() and n_wsi_patches > 1:
            batch_size = 2**5
        else:
            batch_size = 2**7

    patient_labels = {'train': get_label_map(labels_file, 'train'),
                      'val': get_label_map(labels_file, 'val'),
                      'test': get_label_map(labels_file, 'test')}

    if 'wsi' in data_dirs.keys():
        transforms = {
            'train': torchvision.transforms.Compose([
                patch_transforms.ToPIL(),
                torchvision.transforms.CenterCrop(wsi_patch_size),
                torchvision.transforms.ColorJitter(
                    brightness=64/255, contrast=0.5, saturation=0.25,
                    hue=0.04),
                patch_transforms.ToNumpy(),
                patch_transforms.RandomRotate(),
                patch_transforms.RandomFlipUpDown(),
                patch_transforms.ToTensor(),
            ]),
            # No data augmentation for validation
            'val': torchvision.transforms.Compose([
                patch_transforms.ToPIL(),
                torchvision.transforms.CenterCrop(wsi_patch_size),
                patch_transforms.ToNumpy(),
                patch_transforms.ToTensor(),
            ]),
            'test': torchvision.transforms.Compose([
                patch_transforms.ToPIL(),
                torchvision.transforms.CenterCrop(wsi_patch_size),
                patch_transforms.ToNumpy(),
                patch_transforms.ToTensor(),
        ])}
    else:
        transforms = {'train': None, 'val': None, 'test': None}

    datasets = {x: dataset.MultimodalDataset(
        label_map=patient_labels[x],
        data_dirs=data_dirs,
        n_patches=n_wsi_patches,
        patch_size=wsi_patch_size,
        transform=transforms[x],
        exclude_patients=exclude_patients,
        return_patient_id=return_patient_id)
                for x in ['train', 'val', 'test']}

    print('Data modalities:')
    for mod in modalities:
        print('  ', mod)
    print()
    print('Dataset sizes (# patients):')
    for x in datasets.keys():
        print(f'   {x}: {len(datasets[x])}')
    print()
    print('Batch size:', batch_size)

    # Use "drop_last=True" to drop the last incomplete batch
    # to avoid undefined loss values due to lack of sufficient
    # orderable observation pairs caused by data censorship
    # When running all data with batch = 64:
    #    8880 % 64 = 48
    # When running 20 cancer data with batch = 64:
    #    7369 % 64 = 9

    dataloaders = {'train': torch.utils.data.DataLoader(
        datasets['train'], batch_size=batch_size,
        shuffle=True, num_workers=4, drop_last=True),
                   'val': torch.utils.data.DataLoader(
        datasets['val'], batch_size=batch_size * 2,
        shuffle=False, num_workers=4, drop_last=True),
                   'test': torch.utils.data.DataLoader(
        datasets['test'], batch_size=batch_size * 2,
        shuffle=False, num_workers=4, drop_last=True)}

    return dataloaders

def compose_run_tag(model, lr, dataloaders, log_dir, suffix=''):
    """Compose run tag to use as file name prefix.

    Used for Tensorboard log file and model weights.

    Parameters
    ----------
    Returns
    -------
    Run tag string.
    """
    def add_string(string, addition, sep='_'):
        if not string:
            return addition
        else: return string + sep + addition

    data = None
    for modality in model.data_modalities:
        data = add_string(data, modality)
        if modality == 'wsi':
            n = dataloaders['train'].dataset.np
            size = dataloaders['train'].dataset.psize[0]
            string = f'1patch{size}px' if n == 1 else f'{n}patches{size}px'
            data = add_string(data, string, sep='')

    run_tag = f'{data}_lr{lr}'

    if model.fusion_method:
        if model.fusion_method != 'max' and len(model.data_modalities) == 1:
            run_tag += f'_{model.fusion_method}Aggr'

    run_tag += suffix
    print(f'Run tag: "{run_tag}"')

    # Stop if TensorBoard log directory already exists
    tb_log_dir = os.path.join(log_dir, run_tag)
    assert not os.path.isdir(tb_log_dir), ('Tensorboard log directory ' +
                                           f'already exists:\n"{tb_log_dir}"')
    return run_tag

def discretize_time_by_duration_quantiles(t, e, n):
    """Discretize time by equidistant survival probabilities.

    Based on the distribution of the event times. Estimate the survival
    function using the Kaplan-Meier estimator and make a grid of equidistant
    estimates (corresponding to quantiles of the input times). The result is a
    grid with cuts determined by event density.

    Proposed in Kvamme and Borgan, 2019 (arXiv:1910.06724).

    Parameters
    ----------
    t: list
        Patient times in the dataset.
    e: list
        Patient events in the dataset.
    n: int
        Number of intervals.
    Returns
    -------
    Time interval cuts (as numpy array).
    """
    kmf = KaplanMeierFitter()
    kmf.fit(durations=t, event_observed=e)
    km_estimates = kmf.survival_function_.values[:, 0]
    km_times = kmf.survival_function_.index.values
    s_cuts = np.linspace(km_estimates.min(), km_estimates.max(), n)
    cuts_idx = np.searchsorted(km_estimates[::-1], s_cuts)[::-1]
    cuts = km_times[::-1][cuts_idx]
    cuts = np.unique(cuts)
    cuts[0] = 0

    # Convert from years to days
    cuts *= 365

    return cuts
