"""Baseline models."""
                                                                                   
from lifelines import CoxPHFitter
from pysurvival.models.survival_forest import RandomSurvivalForestModel
import torchtuples as tt
from pycox.models import CoxPH
from pycox.models.cox_time import MLPVanillaCoxTime
from pycox.models import CoxTime
from pycox.models import LogisticHazard
from pycox.models import DeepHitSingle
from pycox.models import MTLR


class _BaseData:
    def __init__(self, algorithm, data):
        self.algorithm = algorithm

        self._lifelines_methods = ['CPH']
        self._pysurvival_methods = ['RSF']
        self._pycox_methods = ['DeepSurv', 'CoxTime', 'DeepHit',
                               'MTLR', 'Nnet-survival']
        self._discrete_time_methods = ['DeepHit', 'MTLR', 'Nnet-survival']
        methods = self._lifelines_methods + \
                  self._pysurvival_methods + \
                  self._pycox_methods

        if not algorithm in methods:
            raise ValueError(f'{algorithm} is not a recognized algorithm.')
        
        self.data = data

        if self.algorithm in self._pycox_methods:
            self.x, self.y, self.val = self._process_for_pycox()


    def _process_for_pycox(self):
        def _get_data(df):
            return df[df.columns[2:]].values.astype('float32')

        def _get_target(df):
            return (df['time'].values.astype('float32'),
                    df['event'].values.astype('float32'))

        x = {group: _get_data(self.data[group]) for group in self.data}
        y = {group: _get_target(self.data[group]) for group in self.data}
        val = tt.tuplefy(x['val'], y['val'])

        return x, y, val


class _BaseModel(_BaseData):
    def __init__(self, algorithm, data):
        super().__init__(algorithm, data)

    def _get_discrete_time_net(self, label_transf, net_args):
        self.y['train'] = label_transf.fit_transform(*self.y['train'])
        self.y['val'] = label_transf.transform(*self.y['val'])
        self.val = (self.x['val'], self.y['val'])

        net = tt.practical.MLPVanilla(
            out_features=label_transf.out_features, **net_args)

        return net

    def _model_factory(self, n_trees=None, n_input_features=None,
                       n_neurons=None):
        if self.algorithm == 'CPH':
            return CoxPHFitter()
        elif self.algorithm == 'RSF':
            return RandomSurvivalForestModel(num_trees=n_trees)
        elif self.algorithm in self._pycox_methods:
            net_args = {
                'in_features': n_input_features,
                'num_nodes': n_neurons,
                'batch_norm': True,
                'dropout': 0.1,
            }
            
            if self.algorithm == 'DeepSurv':
                net = tt.practical.MLPVanilla(
                    out_features=1, output_bias=False, **net_args)
                model = CoxPH(net, tt.optim.Adam)

                return model
            if self.algorithm == 'CoxTime':
                net = MLPVanillaCoxTime(**net_args)
                model = CoxTime(net, tt.optim.Adam)

                return model
            if self.algorithm in self._discrete_time_methods:
                num_durations = 30
                print(f'   {num_durations} equidistant intervals')
            if self.algorithm == 'DeepHit':
                labtrans = DeepHitSingle.label_transform(num_durations)
                net = self._get_discrete_time_net(labtrans, net_args)
                model = DeepHitSingle(net, tt.optim.Adam, alpha=0.2, sigma=0.1,
                                      duration_index=labtrans.cuts)

                return model
            if self.algorithm == 'MTLR':
                labtrans = MTLR.label_transform(num_durations)
                net = self._get_discrete_time_net(labtrans, net_args)
                model = MTLR(net, tt.optim.Adam, duration_index=labtrans.cuts)

                return model
            if self.algorithm == 'Nnet-survival':
                labtrans = LogisticHazard.label_transform(num_durations)
                net = self._get_discrete_time_net(labtrans, net_args)
                model = LogisticHazard(net, tt.optim.Adam(0.01),
                                       duration_index=labtrans.cuts)

                return model
        else:
            raise Exception('Unrecognized model.')


class Baselines(_BaseModel):
    """Fit baseline models.

    Parameters
    ----------
    algorithm: str
        Algorithm name.
    data: dict of pandas DataFrames
        Training, validation and test datasets (dict keys "train", "val", and
        "test").
    """
    def __init__(self, algorithm, data, n_trees=None, n_neurons=None):
        super().__init__(algorithm, data)
        model_factory_args = {}

        if self.algorithm == 'RSF':
            model_factory_args['n_trees'] = n_trees
        elif self.algorithm in self._pycox_methods:
            model_factory_args['n_input_features'] = self.x['train'].shape[1]
            model_factory_args['n_neurons'] = n_neurons
        
        self.model = self._model_factory(**model_factory_args)

    def fit(self, **kwargs):
        if self.algorithm == 'CPH':
            self.model.fit(
                self.data['train'], duration_col='time', event_col='event',
                **kwargs)
        elif self.algorithm == 'RSF':
            self.model.fit(
                self.data['train'].iloc[:, 2:].values,
                self.data['train']['time'].values,
                self.data['train']['event'].values)
        elif self.algorithm in self._pycox_methods:
            lrfinder = self.model.lr_finder(
                self.x['train'], self.y['train'], tolerance=2)

            # Set LR as ~half of highest LR before training loss explosion
            lr = lrfinder.get_best_lr() * 0.4
            if len(str(lr)) > 5:
                lr = round(lr, 4)
            print('   Learning rate', lr)
            print('   Batch size', kwargs['batch_size'])
            self.model.optimizer.set_lr(lr)
            print()

            if self.algorithm == 'CoxTime':
                val_data = self.val.repeat(10).cat()
            else:
                val_data = self.val

            self.training_log = self.model.fit(
                self.x['train'], self.y['train'], epochs=500,
                callbacks=[tt.callbacks.EarlyStopping()],
                val_data=val_data, **kwargs)
            
            if not self.algorithm in self._discrete_time_methods:
                _ = self.model.compute_baseline_hazards()
