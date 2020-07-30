"""Abstract model class."""


import os
import warnings
import time
import copy
from itertools import combinations

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from pycox.evaluation import EvalSurv

import utils


class ModelCoach:
    """Model fitting functionality."""
    def __init__(self, model, dataloaders, optimizer, criterion,
                 auxiliary_criterion, output_intervals, device=None):
        self.model = model
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.criterion = criterion.to(device)
        self.aux_criterion = (auxiliary_criterion.to(device)
                              if auxiliary_criterion is not None
                              else auxiliary_criterion)
        # Save 3 best model weights
        self.best_perf = {'epoch a': 0.0, 'epoch c': 0.0, 'epoch b': 0.0}
        self.best_wts = {'epoch a': None, 'epoch c': None, 'epoch b': None}
        self.current_perf = {'epoch a': 0}
        self.output_intervals = output_intervals
        self.device = device

    def _data_to_device(self, data):
        for modality in data:
            if isinstance(data[modality], (list, tuple)):  # Clinical data
                data[modality] = tuple([v.to(self.device)
                                        for v in data[modality]])
            else:
                data[modality] = data[modality].to(self.device)

        return data

    def _compute_auxiliary_loss(self, features):
        "Embedding vector distance loss."
        losses = []

        y = torch.ones(1).to(self.device)
        for x1, x2 in combinations(features, 2):
            losses.append(self.aux_criterion(x1, x2, y))

        loss = torch.tensor(losses).mean()

        return loss.to(self.device)

    def _compute_loss(self, risk, time, event, modality_features):
        loss = self.criterion(risk=risk, times=time, events=event,
                breaks=self.output_intervals.double().to(self.device),
                              device=self.device)

        is_multimodal = len(self.model.data_modalities) > 1

        if (not is_multimodal and self.aux_criterion is not None):
            warnings.warn('Input data is unimodal: auxiliary' +
                          ' loss is not applicable.')

        if is_multimodal and self.aux_criterion is not None:
            # Embedding vector distance loss
            auxiliary_loss = self._compute_auxiliary_loss(modality_features)
            loss = (1.0 * auxiliary_loss) + (0.05 * loss)

        return loss

    def _log_info(self, phase, logger, epoch, epoch_loss, epoch_concord):
        info = {phase + '_loss': epoch_loss,
                phase + '_concord': epoch_concord}

        for tag, value in info.items():
            logger.add_scalar(tag, value, epoch)

    def _process_data_batch(self, data, phase, retain_graph=None):
        if len(data) == 3:
            data, time, event = data
        elif len(data) == 4:  # With patient id (not needed here)
            data, time, event, pid = data
        data = self._data_to_device(data)
        time = time.to(self.device)
        event = event.to(self.device)

        with torch.set_grad_enabled(phase == 'train'):
            feature_representations, risk = self.model(data)
            modality_features = feature_representations['modalities']
            loss = self._compute_loss(
                risk, time, event, modality_features)

            if phase == 'train':
                # Zero out parameter gradients
                self.optimizer.zero_grad()
                if retain_graph is not None:
                    loss.backward(retain_graph=retain_graph)
                else:
                    loss.backward()
                self.optimizer.step()

        return loss, risk, time, event

    def _predictions_to_pycox(self, preds, time_points=None):
        # Convert to survival probabilities
        surv_probs = torch.cumprod(preds, 1)
        df = pd.DataFrame(torch.transpose(surv_probs, 0, 1).cpu().numpy())

        if time_points is None:
            time_points = torch.arange(0.5, 30, 1)

        # Replace automatic index by time points
        df.insert(0, 'time', time_points)
        df = df.set_index('time')

        return df

    def _run_training_loop(self, num_epochs, scheduler, info_freq, log_dir):
        logger = SummaryWriter(log_dir)
        log_info = True

        if info_freq is not None:
            def print_header():
                sub_header = ' Epoch     Loss     Ctd     Loss     Ctd'
                print('-' * (len(sub_header) + 2))
                print('             Training        Validation')
                print('           ------------     ------------')
                print(sub_header)
                print('-' * (len(sub_header) + 2))

            print()

            print_header()

        for epoch in range(1, num_epochs + 1):
            if info_freq is None:
                print_info = False
            else:
                print_info = epoch == 1 or epoch % info_freq == 0

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                running_losses = []

                if print_info or log_info:
                    running_durations = torch.FloatTensor().to(self.device)
                    running_censors = torch.LongTensor().to(self.device)
                    running_risks = torch.FloatTensor().to(self.device)

                # Iterate over data
                for data in self.dataloaders[phase]:
                    batch_result = self._process_data_batch(data, phase)
                    loss, risk, time, event = batch_result

                    # Stats
                    running_losses.append(loss.item())
                    running_durations = torch.cat((running_durations,
                                                   time.data.float()))
                    running_censors = torch.cat((running_censors,
                                                 event.long().data))
                    running_risks = torch.cat((running_risks, risk.detach()))

                epoch_loss = torch.mean(torch.tensor(running_losses))

                surv_probs = self._predictions_to_pycox(
                    running_risks, time_points=None)
                running_durations = running_durations.cpu().numpy()
                running_censors = running_censors.cpu().numpy()
                epoch_concord = EvalSurv(
                    surv_probs, running_durations, running_censors,
                    censor_surv='km'
                ).concordance_td('adj_antolini')

                if print_info:
                    if phase == 'train':
                        message = f' {epoch}/{num_epochs}'
                    space = 10 if phase == 'train' else 27
                    message += ' ' * (space - len(message))
                    message += f'{epoch_loss:.4f}' 
                    space = 19 if phase == 'train' else 36
                    message += ' ' * (space - len(message))
                    message += f'{epoch_concord:.3f}' 
                    if phase == 'val':
                        print(message)

                if log_info:
                    self._log_info(
                        phase=phase, logger=logger, epoch=epoch,
                        epoch_loss=epoch_loss, epoch_concord=epoch_concord)

                if phase == 'val':
                    if scheduler:
                        scheduler.step(epoch_concord)

                    # Record current performance
                    k = list(self.current_perf.keys())[0]
                    self.current_perf[
                        'epoch' + str(epoch)] = self.current_perf.pop(k)
                    self.current_perf['epoch' + str(epoch)] = epoch_concord
                    # Deep copy the model
                    for k, v in self.best_perf.items():
                        if epoch_concord >= v:
                            self.best_perf[
                                'epoch' + str(epoch)] = self.best_perf.pop(k)
                            self.best_perf[
                                'epoch' + str(epoch)] = epoch_concord
                            self.best_wts[
                                'epoch' + str(epoch)] = self.best_wts.pop(k)
                            self.best_wts[
                                'epoch' + str(epoch)] = copy.deepcopy(
                                    self.model.state_dict())
                            break

    def train(self, num_epochs, scheduler, info_freq, log_dir):
        """Train multimodal PyTorch model."""
        start_time = time.time()

        # Handle keyboard interrupt
        try:
            self._run_training_loop(num_epochs, scheduler, info_freq, log_dir)

            hrs, mins, secs = utils.elapsed_time(start_time)
            print()
            message = '>>>>> Training completed in'
            if hrs > 0:
                message += f' {hrs}h'
            if mins > 0:
                message += f' {mins}m'
            print(message + f' {secs}s')
            print('>>>>> Best validation C-indices:')
            for k, v in self.best_perf.items():
                print(f'     {v} ({k})')
        except KeyboardInterrupt:
            hrs, mins, secs = utils.elapsed_time(start_time)
            print()
            print('>>> Keyboard interrupt! <<<')
            print(f'(trained for {hrs}h {mins}m {secs}s)')
            print()
            print('Best validation concordance values:')
            for k, v in self.best_perf.items():
                print(f'     {round(v, 4)} ({k})')
