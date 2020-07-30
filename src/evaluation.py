"""Performance evaluation."""

import warnings

import numpy as np
import pandas as pd

import torch
from sklearn.utils import resample
from lifelines.utils import concordance_index
from pycox.evaluation import EvalSurv


class _BaseEvaluation:
    """Evaluation functionality common to all model types."""
    def __init__(self, model, dataset):
        self.model = model
        self.type = self._model_type()
        self.data = dataset

        if self.type == 'MultiSurv':
            if 'dataset' not in str(type(self.data)):
                raise ValueError('MultiSurv model requires a PyTorch dataset.')
        else:
            if 'DataFrame' not in str(type(self.data)):
                raise ValueError(f'{self.type} model requires data' +
                                 ' in pandas DataFrame.')

        self.patient_predictions = None
        self.c_index = None
        self.c_index_td = None
        self.ibs = None
        self.inbll = None

    def _model_type(self):
       if 'lifelines' in str(type(self.model)):
           model_type = 'lifelines'
       elif 'pysurvival' in str(type(self.model)):
           model_type = 'pysurvival'
       elif 'pycox' in str(type(self.model)):
           model_type = 'pycox'
       elif 'Model' in str(type(self.model)):
           if 'MultiSurv' in str(type(self.model.model)):
               model_type = 'MultiSurv'
       else:
           raise ValueError('"model" is not recognized.')

       return model_type

    def _unpack_data(self, data):
        times = [data[patient]['time'] for patient in data]
        events = [data[patient]['event'] for patient in data]
        predictions = [data[patient]['probabilities'] for patient in data]

        return times, events, predictions

    def _compute_c_index(self, data):
        times, events, predictions = self._unpack_data(data)
        probs_by_interval = torch.stack(predictions).permute(1, 0)
        c_index = [concordance_index(event_times=times,
                                     predicted_scores=interval_probs,
                                     event_observed=events)
                   for interval_probs in probs_by_interval]

        return c_index

    def _predictions_to_pycox(self, data, time_points=None):
        predictions = {k: v['probabilities'] for k, v in data.items()}
        df = pd.DataFrame.from_dict(predictions)

        # Use predictions at same "time_points" for all models
        # Use MultiSurv's default output interval midpoints as default
        if time_points is None:
            time_points = torch.arange(0.5, 30, 1)

        # Replace automatic index by time points
        df.insert(0, 'time', time_points)
        df = df.set_index('time')

        return df


class _BaselineModelEvaluation(_BaseEvaluation):
    """Evaluation of baseline models."""
    def __init__(self, model, dataset):
        super().__init__(model, dataset)

    def _collect_patient_ids(self):
        return self.data.index

    def _predict_risk(self, patient):
        patient_data = self.data.loc[patient]
        time, event = patient_data['time'], patient_data['event']
        data = patient_data.iloc[2:]

        if self.type == 'lifelines':
            pred = self.model.predict_partial_hazard(data)[0]
        elif self.type == 'pysurvival':
            pred = self.model.predict_risk(data)[0]
        else:
            raise NotImplementedError

        return time, event, pred

    def _predict(self):
        data = self.data.iloc[:, 2:]

        if self.type == 'lifelines':
            pred = self.model.predict_survival_function(data)
            pred_times = pred.index
        elif self.type == 'pysurvival':
            pred = self.model.predict_survival(data)
            pred_times = self.model.times
            pred = pd.DataFrame(pred.transpose())
        elif self.type == 'pycox':
            pred = self.model.predict_surv_df(
                np.array(data.values.astype('float32')))
            pred_times = pred.index
        else:
            raise NotImplementedError

        # Interpolate survival prediction times to match MultiSurv's default
        interp_pred = np.empty((0, torch.arange(0.5, 30, 1).shape[0]))

        for patient in pred.columns:
            interp_patient = np.interp(torch.arange(0.5, 30, 1),
                                       pred_times, pred[patient])
            interp_pred = np.append(interp_pred, [interp_patient], axis=0)

        times, events = self.data['time'], self.data['event']

        return times, events, torch.from_numpy(interp_pred)


class _MultiSurvEvaluation(_BaseEvaluation):
    """Evaluation of MultiSurv models."""
    def __init__(self, model, dataset, device):
        super().__init__(model, dataset)
        self.device = device

    def _data_to_device(self, data):
        for modality in data:       
            if isinstance(data[modality], (list, tuple)):  # Clinical data 
                data[modality] = tuple([v.to(self.device)
                                        for v in data[modality]])  
            else:
                data[modality] = data[modality].to(self.device)

        return data

    def _collect_patient_ids(self):
        return self.data.patient_ids
    
    def _predict(self, patient):
        data, time, event = self.data.get_patient_data(patient)
        _, preds = self.model.predict(self._data_to_device(data),
                                      prediction_year=None)
                                                                                
        return time, event, preds


class Evaluation(_BaseEvaluation):
    """Functionality to compute model evaluation metrics.

    Parameters
    ----------
    model: PyTorch or baseline model
        Model to predict patient survival.
    dataset: dataset.MultimodalDataset or pandas.core.frame.DataFrame
        Dataset of patients.
    device: torch.device
        The device to run the model on (CPU or GPU).
    """
    def __init__(self, model, dataset, device=None):
        super().__init__(model, dataset)
        self.device = device

        if self.type == 'MultiSurv':
            self.ev = _MultiSurvEvaluation(model, dataset, self.device)
        else:
            if device is not None:
                warnings.warn('"device" is only used with MultiSurv model.' +
                              ' Ignored here.')
            self.ev = _BaselineModelEvaluation(model, dataset)

    def _collect_patient_predictions(self):
        if (self.device is not None and self.type is not 'MultiSurv'):
            warnings.warn('"device" is only used with MultiSurv model.' +
                        ' Ignored here.')

        # Get all patient labels and predictions
        patient_data = {}

        pids = self.ev._collect_patient_ids()

        if self.type == 'MultiSurv':
            for i, patient in enumerate(pids):
                print(f'\r' + f'Collect patient predictions:' +
                      f' {str((i + 1))}/{len(pids)}', end='')

                time, event, pred = self.ev._predict(patient)
                patient_data[patient] = {'time': time,
                                         'event': event,
                                         'probabilities': pred}
        else:
            print(f'Collect patient predictions...', end='')
            times, events, pred = self.ev._predict()

            for i, patient in enumerate(pids): 
                patient_data[patient] = {'time': times[i],
                                         'event': events[i],
                                         'probabilities': pred[i]}

        print()
        print()

        return patient_data

    def _compute_pycox_metrics(self,data, time_points=None,
                               drop_last_times=25):
        times, events, _ = self._unpack_data(data)
        times, events = np.array(times), np.array(events)
        predictions = self._predictions_to_pycox(data, time_points)

        ev = EvalSurv(predictions, times, events, censor_surv='km')
        # Using "antolini" method instead of "adj_antolini" resulted in Ctd
        # values different from C-index for proportional hazard methods (for
        # CNV data); this is presumably due to the tie handling, since that is
        # what the pycox authors "adjust" (see code comments at:
        # https://github.com/havakv/pycox/blob/6ed3973954789f54453055bbeb85887ded2fb81c/pycox/evaluation/eval_surv.py#L171)
        # c_index_td = ev.concordance_td('antolini')
        c_index_td = ev.concordance_td('adj_antolini')

        # time_grid = np.array(predictions.index)
        # Use 100-point time grid based on data
        time_grid = np.linspace(times.min(), times.max(), 100)
        # Since the score becomes unstable for the highest times, drop the last
        # time points?
        if drop_last_times > 0:
            time_grid = time_grid[:-drop_last_times]
        ibs = ev.integrated_brier_score(time_grid)
        inbll = ev.integrated_nbll(time_grid)

        return c_index_td, ibs, inbll

    def compute_metrics(self, time_points=None):
        """Calculate evaluation metrics."""
        if self.patient_predictions is None:
            # Get all patient labels and predictions
            self.patient_predictions = self._collect_patient_predictions()

        if self.c_index is None:
            self.c_index = self._compute_c_index(self.patient_predictions)

        if self.c_index_td is None:
            td_metrics = self._compute_pycox_metrics(self.patient_predictions,
                                                     time_points)
            self.c_index_td, self.ibs, self.inbll = td_metrics

    def run_bootstrap(self, n=1000, time_points=None):
        """Calculate bootstrapped metrics.

        Parameters
        ----------
        n: int
            Number of boostrapped samples.
        time_points: torch.Tensor
            Time points of the predictions.
        Returns
        -------
        Metrics calculated on original dataset and bootstrap samples.
        """
        n = int(n)
        if n <= 0:
            raise ValueError('"n" must be greater than 0.')

        if self.c_index is None:
            try:
                self.compute_metrics(time_points)
            except ZeroDivisionError as err:
                return err, 'C-index could not be calculated.'

        # Run bootstrap
        print('Bootstrap')
        print('-' * 9)

        self.boot_c_index = {}
        self.boot_c_index_td, self.boot_ibs, self.boot_inbll = [], [], []
        skipped = 0

        for i in range(n):
            print('\r' + f'{str((i + 1))}/{n}', end='')
            # Get bootstrap sample (same size as dataset)
            boot_ids = resample(self.ev._collect_patient_ids(), replace=True)
            sample_data = {patient: self.patient_predictions[patient]
                           for patient in boot_ids}

            # When running samples with small number of patients (e.g. some
            # individual cancer types) sometimes there are no admissible pairs
            # to compute the C-index (or other metrics).
            # In those cases continue and print a warning at the end
            try:                                                                
                current_cindex = self._compute_c_index(data=sample_data)
                td_metrics = self._compute_pycox_metrics(
                    sample_data, time_points)
                current_ctd, current_ibs, current_inbll = td_metrics
            except ZeroDivisionError as error:                                  
                err = error                                                     
                skipped += 1                                                    
                continue                                                        

            if not self.boot_c_index:
                for j in range(len(current_cindex)):
                    self.boot_c_index.update({str(j): []})

            for k, x in enumerate(current_cindex):
                self.boot_c_index[str(k)].append(x)

            self.boot_c_index_td.append(current_ctd)
            self.boot_ibs.append(current_ibs)
            self.boot_inbll.append(current_inbll)

        print()

        if skipped > 0:
            warnings.warn(
                f'Skipped {skipped} bootstraps ({err}).')

    def format_results(self, method='percentile'):
        """Calculate bootstrap confidence intervals.

        The empirical bootstrap method uses the distribution of differences
        between the metric calculated on the original dataset and on the
        bootstrap samples. The percentile method uses the distribution of the
        metrics calculated on the bootstrap samples directly.

        Parameters
        ----------
        method: str 
            Bootstrap method to calculate confidence intervals (one of
            "percentile" and "empirical").
        Returns
        -------
        Metrics with 95% bootstrap confidence intervals.
        """
        assert self.c_index is not None, 'Results not available.' + \
                ' Please call "compute_metrics" or "run_bootstrap" first.'

        bootstrap_methods = ['percentile', 'empirical']
        assert method in bootstrap_methods, + \
                '"method" must be one of {bootstrap_methods}.'

        c_index = self.c_index[0]
        c_index = round(c_index , 3)
        ctd = round(self.c_index_td, 3)
        ibs = round(self.ibs, 3)
        inbll = round(self.inbll, 3)

        output = {}

        output['C-index'] = str(c_index)
        output['Ctd'] = str(ctd)
        output['IBS'] = str(ibs)
        output['INBLL'] = str(inbll)

        def _get_differences(metric_value, bootstrap_values):
            differences = [x - metric_value for x in bootstrap_values]

            return sorted(differences) 

        def _get_empirical_percentiles(values, metric_value):
            values = _get_differences(metric_value, values)
            percent = np.percentile(values, [2.5, 97.5])
            low, high = tuple(round(metric_value + x, 3)
                              for x in [percent[0], percent[1]])

            return f'({low}-{high})'

        def _get_percentiles(values):
            percent = np.percentile(values, [2.5, 97.5])
            low, high = round(percent[0], 3), round(percent[1], 3)

            return f'({low}-{high})'

        try:
            if method == 'empirical':
                c_index_percent = _get_empirical_percentiles(
                    self.boot_c_index['0'], self.c_index[0])
                c_index_td_percent = _get_empirical_percentiles(
                    self.boot_c_index_td, self.c_index_td)
                ibs_percent = _get_empirical_percentiles(
                    self.boot_ibs, self.ibs)
                inbll_percent = _get_empirical_percentiles(
                        self.boot_inbll, self.inbll)
            else:
                c_index_percent = _get_percentiles(self.boot_c_index['0'])
                c_index_td_percent = _get_percentiles(self.boot_c_index_td)
                ibs_percent = _get_percentiles(self.boot_ibs)
                inbll_percent = _get_percentiles(self.boot_inbll)

            output['C-index'] += ' ' + str(c_index_percent)
            output['Ctd'] += ' ' + str(c_index_td_percent)
            output['IBS'] += ' ' + str(ibs_percent)
            output['INBLL'] += ' ' + str(inbll_percent)
        except:
            return output

        return output

    def show_results(self, method='percentile'):
        results = self.format_results(method)

        if '(' in results['C-index']:  # bootstrap results available?
            print('          Value (95% CI)')
            print('-' * 29)

        for algo, res in results.items():
            print(algo + ' ' * (10 - len(algo)) + res)
