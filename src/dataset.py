"""PyTorch multimodal dataset class."""


import os
import random
import csv
import warnings

import torch
from torch.utils.data import Dataset
from skimage import io


class _BaseDataset(Dataset):
    def __init__(self,
                 label_map,
                 data_dirs={
                     'clinical': None, 'wsi': None, 'mRNA': None,
                     'miRNA': None, 'DNAm': None, 'CNV': None
                     },
                 exclude_patients=None):
        self.label_map = label_map
        self.data_dirs = data_dirs
        self.patient_ids = list(set(label_map.keys()))

        valid_mods = ['clinical', 'wsi', 'mRNA', 'miRNA', 'DNAm', 'CNV']
        assert all(k in valid_mods for k in self.data_dirs.keys()), \
                f'Accepted data modalitites (in "data_dirs") are: {valid_mods}'

        assert not all(v is None for v in self.data_dirs.values()), \
                f'At least one input data modality is required: {valid_mods}'

        # Check missing data: drop patients missing all data
        patients_missing_all_data = self._patients_missing_all_data()

        if patients_missing_all_data:
            print(f'Excluding {len(patients_missing_all_data)} patient(s)' +
                  ' missing all data.')
            self.patient_ids = [pid for pid in self.patient_ids
                                if pid not in patients_missing_all_data]

        if exclude_patients is not None:
            self.patient_ids = [pid for pid in self.patient_ids
                                if pid not in exclude_patients]
            kept = len(self.patient_ids)
            print(f'Keeping {kept} patient(s) not in exclude list.')

    def __len__(self):
        return len(self.patient_ids)

    def _get_patient_ids(self, path_to_data):
        files = os.listdir(path_to_data)
        file_ext = os.path.splitext(files[0])[1]

        pids = set([os.path.splitext(file)[0] for file in files])

        return pids

    def _patients_missing_all_data(self):
        missing_all_data = []

        for data_dir in self.data_dirs.values():
            if data_dir is not None:
                pids_in_data = self._get_patient_ids(data_dir)
                missing_data = [pid for pid in self.patient_ids
                                if pid not in pids_in_data]

                # Break if a data modality has all data
                if not missing_data:
                    break
                elif not missing_all_data:
                    missing_all_data = missing_data
                else:  # Keep patients missing all checked data so far
                    missing_all_data = [pid for pid in missing_all_data
                                        if pid in missing_data]

        return missing_all_data


class MultimodalDataset(_BaseDataset):
    """TCGA dataset iterating over patient IDs.

    Note: Data is filled in with all zeros for every patient before checking
    availability, as a mechanism to allow data dropout. Because of this, any
    patient originally missing all input will be run with all-zero data. To
    exclude such examples, missing data is checked at instantiation and any
    patients missing all data are dropped.

    Parameters
    ----------
    label_map: dict
        Patient labels as dictionary (patient id: (time, is_censored)).
    data_dirs: dict
        Data directories in the format {'clinical': 'path/to/dir',
                                        'wsi': 'path/to/dir',
                                        'mRNA': 'path/to/dir',
                                        'miRNA': 'path/to/dir',
                                        'DNAm': 'path/to/dir'}
                                        'CNV': 'path/to/dir'}
    n_patches: int
        Number of WSI patches to load per label (i.e. patient). Required if
        data_dirs['wsi'] is not "None".
    patch_size: int
        Number of pixels defining the side length of the square patches.
        Required if data_dirs['wsi'] is not "None".
    transform: callable
        Optional transform to apply to WSI patches. Required if
        data_dirs['wsi'] is not "None".
    dropout: float [0, 1]
        Probability of dropping one data modality (applied if at least two are
        available).
    exclude_patients: list of str
        Optional list of patient ids to exclude.
    return_patient_id: bool
        Whether to add patient id to output.
    """
    def __init__(self, label_map,
                 data_dirs={'clinical': None, 'wsi': None,
                            'mRNA': None, 'miRNA': None,
                            },
                 n_patches=None, patch_size=None, transform=None, dropout=0,
                 exclude_patients=None, return_patient_id=False):
        super().__init__(label_map, data_dirs, exclude_patients)
        self.modality_loaders = {
            'clinical': self._get_clinical,
            'wsi': self._get_patches,
            'mRNA': self._get_data,
            'miRNA': self._get_data,
            'DNAm': self._get_data,
            'CNV': self._get_data,
            }
        assert 0 <= dropout <= 1, '"dropout" must be in [0, 1].'
        self.dropout = dropout
        if sum([v is not None for v in self.data_dirs.values()]) == 1:
            if self.dropout > 0:
                warnings.warn('Input data is unimodal: "dropout" set to 0.')
                self.dropout = 0

        self.data_dirs = {mod: (data_dirs[mod]
                                if mod in data_dirs.keys() else None)
                          for mod in self.modality_loaders}
        if self.data_dirs['wsi'] is not None:
            assert n_patches is not None and n_patches > 0, \
                    '"n_patches" must be greater than 0 when inputting WSIs .'
            self.np = n_patches
            self.psize = patch_size, patch_size
            assert transform is not None, \
                    '"transform" is required when inputting WSIs .'
            self.transform = transform
        else:
            self.np = self.psize = self.transform = None

        self.return_patient_id = return_patient_id

    def _read_patient_file(self, path):
        with open(path, 'r') as f:
            f = csv.reader(f, delimiter='\t')

            values = []
            for row in enumerate(f):
                values.append(row[1][0])

        return values

    def _get_clinical(self, data_dir, patient_id):
        patient_files = {os.path.splitext(f)[0]: f
                         for f in os.listdir(data_dir)}

        data_file = os.path.join(data_dir, patient_files[patient_id])
        data = self._read_patient_file(data_file)
        categorical = torch.tensor(
            [int(float(value)) for value in data[:9]], dtype=torch.int)
        continuous = torch.tensor(
            [float(value) for value in data[9:]], dtype=torch.float)

        return categorical, continuous

    def _get_patches(self, data_dir, patient_id):
        """Read WSI patches for selected patient.

        Patient files list absolute paths to available WSI patches for the
        repective patient.
        """
        patient_file = os.path.join(data_dir, patient_id + '.txt')

        try:
            patch_files = self._read_patient_file(patient_file)
        except:  # If data is missing create all-zero tensor
            return torch.zeros([self.np, 3, self.psize[0], self.psize[0]])

        # Select n patches at random
        patch_files = random.sample(patch_files, self.np)
        patches = [io.imread(p) for p in patch_files]

        if self.transform is not None:
            patches = torch.stack([self.transform(patch)
                                   for patch in patches])

        return patches

    def _get_data(self, data_dir, patient_id):
        patient_files = {os.path.splitext(f)[0]: f
                         for f in os.listdir(data_dir)}

        if patient_id in patient_files:
            data_file = os.path.join(data_dir, patient_files[patient_id])
            data = self._read_patient_file(data_file)
            data = torch.tensor([float(value) for value in data])
        else:  # Return all-zero tensor if data is missing
            eg_file = os.path.join(data_dir, list(patient_files.values())[0])
            nfeatures = len(self._read_patient_file(eg_file))
            data = torch.zeros(nfeatures)

        return data

    def _drop_data(self, data):
        available_modalities = []

        # Check available modalities in current mini-batch
        for modality, values in data.items():
            if isinstance(values, (list, tuple)):  # Clinical data
                values = values[1]  # Use continuous features
            if len(torch.nonzero(values)) > 0:  # Keep if data is available
                available_modalities.append(modality)

        # Drop data modality
        n_mod = len(available_modalities)

        if n_mod > 1:
            if random.random() < self.dropout:
                drop_modality = random.choice(available_modalities)
                if isinstance(data[drop_modality], (list, tuple)):
                    # Clinical data
                    data[drop_modality] = tuple(
                        torch.zeros_like(x) for x in data[drop_modality])
                else:
                    data[drop_modality] = torch.zeros_like(data[drop_modality])

        return data

    def get_patient_data(self, patient_id):
        time, event = self.label_map[patient_id]
        data = {}

        # Load selected patient's data
        for modality in self.data_dirs:
            data_source = self.data_dirs[modality]
            if data_source is not None:
                data[modality] = self.modality_loaders[modality](
                    data_source, patient_id)
                if isinstance(data[modality], (list, tuple)):  # Clinical data
                    data[modality] = tuple(x.float()
                                           for x in data[modality])
                else:
                    data[modality] = data[modality].float()

        # Data dropout
        if self.dropout > 0:
            n_modalities = len([k for k in data])
            if n_modalities > 1:
                data = self._drop_data(data)

        return data, time, event

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        data, time, event = self.get_patient_data(patient_id)

        if self.return_patient_id:
            return data, time, event, patient_id

        return data, time, event
