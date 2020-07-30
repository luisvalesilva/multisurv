"""Model predictions."""


import warnings

import torch
import numpy as np
from lifelines.utils import concordance_index


class Predictor:
    """Add prediction functionality."""
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def _check_dropout(self, dataset):
        dropout = dataset.dropout
        if dropout > 0:
            warnings.warn(f'Data dropout set to {dropout} in input dataset')

    def _data_to_device(self, data):
        for modality in data:
            if isinstance(data[modality], (list, tuple)):  # Clinical data
                data[modality] = tuple([v.to(self.device)
                                        for v in data[modality]])
            else:
                data[modality] = data[modality].to(self.device)

        return data

    def _clone(self, data):
        data_clone = {} 

        for modality in data:
            if isinstance(data[modality], (list, tuple)):  # Clinical data
                data_clone[modality] = tuple([v.clone()
                                              for v in data[modality]])
            else:
                data_clone[modality] = data[modality].clone()

        return data_clone

    def _convert_to_survival(self, conditional_probabilities):
        return np.cumprod(conditional_probabilities)

    def predict(self, patient_data, prediction_year=None, intervals=None):
        """Predict patient survival probability at provided time point."""
        if prediction_year is not None:
            assert intervals is not None, '"intervals" is required to' + \
                    ' compute prediction at a specific "prediction_year".'

        data = self._clone(patient_data)

        # Model expects batch dimension
        for modality in data:
            if isinstance(data[modality], (list, tuple)):  # Clinical data
                data[modality] = tuple(
                    [m.unsqueeze(0) for m in data[modality]])
            else:
                data[modality] = data[modality].unsqueeze(0)

        data = self._data_to_device(data)
        self.model.eval()

        with torch.set_grad_enabled(False):
            feature_representations, probabilities = self.model(data)

        survival_prob = self._convert_to_survival(probabilities.cpu())

        if prediction_year is not None:
            survival_prob = np.interp(
                prediction_year * 365, intervals,
                # Add probability 1.0 at t0 to match number of intervals
                torch.cat((torch.tensor([1]).float(), survival_prob)))
        
        return feature_representations, survival_prob

    def predict_dataset(self, dataset, verbose=True):
        """Predict survival probability for provided set of patients."""
        self._check_dropout(dataset)

        if verbose:
            print('Analyzing patients')
            n = len(dataset)

        pids = []
        result = {}
        times = []
        events = []
        probabilities = []

        # Get all patient data and predictions
        for i, patient in enumerate(dataset):
            if verbose:
               print('\r' + f'{str((i + 1))}/{n}', end='')

            assert len(patient) == 4, 'DataSet output tuple has' + \
                f' length {len(patient)}. Please instantiate DataSet' + \
                ' class with "return_patient_id=True".'
            data, time, event, patient_id = patient
            pids.append(patient_id)

            data = self._data_to_device(data)
            _, predictions = self.predict(data)

            times.append(time)
            events.append(event)
            probabilities.append(predictions)

        result['patient_data'] = {}

        for i, pid in enumerate(pids):
            result['patient_data'][pid] = (
                probabilities[i], times[i], events[i])

        return result
