"""Manage results tables."""


import os

import pandas as pd

class ResultTable:
    def __init__(self):
        """Manage table of evaluation results."""
        self.fn = 'results.csv'

        if not os.path.isfile(self.fn):
            self.table = self._create_table()
            self.table.to_csv(self.fn)
        else:
            self.table = pd.read_csv(
                self.fn, index_col=['Algorithm', 'Metric'])

    def _create_table(self):
        print('Result table file does not exist. Creating...')
        algorithms = ['CPH', 'RSF', 'DeepSurv', 'CoxTime', 'DeepHit',
                      'MTLR', 'Nnet-survival', 'MultiSurv']
        metrics = ['C-index', 'Ctd', 'IBS', 'INBLL']
        data_modalities = ['clinical', 'mRNA', 'DNAm', 'miRNA', 'CNV', 'wsi']

        midx = pd.MultiIndex.from_product([algorithms, metrics],
                                          names=['Algorithm', 'Metric'])
        df = pd.DataFrame(index=midx, columns=data_modalities)

        return df

    def write_value(self, value, algorithm, metric, data_modality):
        self.table.loc[(algorithm, metric), data_modality] = value
        self.table.to_csv(self.fn)

    def write_result_dict(self, result_dict, algorithm, data_modality):
        for metric, value in result_dict.items():
            self.write_value(value=value, algorithm=algorithm,
                             metric=metric, data_modality=data_modality)
