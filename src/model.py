"""Abstract model class."""


import os

import numpy as np
import torch
from torch.optim import Adam

from multisurv import MultiSurv
from lr_range_test import LRRangeTest
from coach import ModelCoach
from predictor import Predictor
from sub_models import freeze_layers
from loss import Loss


class _BaseModelWithData:
    """Abstract model with input data."""
    def __init__(self, dataloaders, fusion_method=None,
                 output_intervals=None, unimodal_state_files=None,
                 freeze_up_to=None, device=None):
        self.fusion_method = fusion_method
        self.dataloaders = dataloaders
        self.unimodal_state_files = unimodal_state_files
        self.freeze_up_to = freeze_up_to
        self.device = device
        self.output_intervals = output_intervals
        eg_dataloader = list(dataloaders.values())[0]
        data_dirs = eg_dataloader.dataset.data_dirs
        self.data_modalities = [modality for modality in data_dirs
                                if data_dirs[modality] is not None]
        self._instantiate_model()
        self.model_blocks = [name for name, _ in self.model.named_children()]


    def _instantiate_model(self, move_to_device=True):
        print('Instantiating MultiSurv model...')
        self.model = MultiSurv(
            data_modalities=self.data_modalities,
            fusion_method=self.fusion_method,
            n_output_intervals=len(self.output_intervals) - 1,
            device=self.device)

        if self.unimodal_state_files is not None:
            self.pretrained_weights = self._get_pretrained_unimodal_weights()
            print('(loading pretrained unimodal model weights...)')
            self.model.load_state_dict(self.pretrained_weights)

        if move_to_device:
            self.model = self.model.to(self.device)

        if self.freeze_up_to is not None:
            freeze_layers(self.model, self.freeze_up_to)

    def _get_pretrained_unimodal_weights(self):
        for modality in self.data_modalities:
            # Load and collect saved weights
            pretrained_dict = torch.load(self.unimodal_state_files[modality])
            # Filter out unnecessary keys
            model_weight_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in self.model.state_dict()}
            # Overwrite entries in the existing state dict
            model_weight_dict.update(pretrained_dict)

        return model_weight_dict


class Model(_BaseModelWithData):
    """Top abstract model class."""
    def __init__(self, dataloaders, fusion_method='max',
                 output_intervals=torch.arange(0., 365 * 31, 365),
                 auxiliary_criterion=None, unimodal_state_files=None,
                 freeze_up_to=None, device=None):
        super().__init__(dataloaders, fusion_method, output_intervals,
                         unimodal_state_files, freeze_up_to, device)
        self.optimizer = Adam
        self.loss = Loss()
        self.aux_loss = auxiliary_criterion

    def test_lr_range(self):
        self._instantiate_model()

        self.lr_test = LRRangeTest(
            dataloader=self.dataloaders['train'],
            optimizer=self.optimizer(self.model.parameters(), lr=1e-4),
            criterion=self.loss, auxiliary_criterion=self.aux_loss,
            output_intervals=self.output_intervals, model=self.model,
            device=self.device)
        self.lr_test.run(init_value=1e-6, final_value=10., beta=0.98)

    def plot_lr_range(self, trim=4):
        try:
            self.lr_test.plot(trim)
        except AttributeError as error:
            print(f'Error: {error}.')
            print(f'       Please run {".test_lr_range"} first.')

    def fit(self, lr, num_epochs, info_freq, log_dir, lr_factor=0.1,
            scheduler_patience=5):
        self._instantiate_model()
        optimizer = self.optimizer(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode='max', factor=lr_factor,
            patience=scheduler_patience, verbose=True, threshold=1e-3,
            threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

        model_coach = ModelCoach(
            model=self.model, dataloaders=self.dataloaders,
            optimizer=optimizer, criterion=self.loss,
            auxiliary_criterion=self.aux_loss,
            output_intervals=self.output_intervals, device=self.device)

        model_coach.train(num_epochs, scheduler, info_freq, log_dir)

        self.model = model_coach.model
        self.best_model_weights = model_coach.best_wts
        self.best_concord_values = model_coach.best_perf
        self.current_concord = model_coach.current_perf

    def save_weights(self, saved_epoch, prefix, weight_dir):
        valid_keys = self.best_model_weights.keys()
        assert saved_epoch in list(valid_keys) + ['current'], \
                f'Valid "saved_epoch" options: {list(valid_keys)}' \
                f'\n(use "current" to save current state)'

        print('Saving model weights to file:')
        if saved_epoch == 'current':
            epoch = list(self.current_concord.keys())[0]
            value = self.current_concord[epoch]
            file_name = os.path.join(
                weight_dir,
                f'{prefix}_{epoch}_concord{value:.2f}.pth')
        else:
            file_name = os.path.join(
                weight_dir,
                f'{prefix}_{saved_epoch}_' + \
                f'concord{self.best_concord_values[saved_epoch]:.2f}.pth')
            self.model.load_state_dict(self.best_model_weights[saved_epoch])

        torch.save(self.model.state_dict(), file_name)
        print('   ', file_name)

    def load_weights(self, path):
        print('Load model weights:')
        print(path)
        self.model.load_state_dict(torch.load(path))
        self.model = self.model.to(self.device)

    def predict(self, input_data, prediction_year=None):
        predictor = Predictor(self.model, self.device)
        # Use midpoints of MultiSurv output intervals
        midpoints = self.output_intervals
        midpoints[1:] = midpoints[1:] - np.diff(midpoints)[0] / 2
        prediction = predictor.predict(input_data, prediction_year, midpoints)
        feature_representations, risk = prediction

        return feature_representations, risk

    def predict_dataset(self, dataset, verbose=True):
        predictor = Predictor(self.model, self.device)

        return predictor.predict_dataset(dataset, verbose)
