"""Deep Learning-based multimodal data model for survival prediction."""

import warnings

import torch

from sub_models import FC, ClinicalNet, CnvNet, WsiNet, Fusion


class MultiSurv(torch.nn.Module):
    """Deep Learning model for MULTImodal pan-cancer SURVival prediction."""
    def __init__(self, data_modalities, fusion_method='max',
                 n_output_intervals=None, device=None):
        super(MultiSurv, self).__init__()
        self.data_modalities = data_modalities
        self.mfs = modality_feature_size = 512
        valid_mods = ['clinical', 'wsi', 'mRNA', 'miRNA', 'DNAm', 'CNV']
        assert all(mod in valid_mods for mod in data_modalities), \
                f'Accepted input data modalitites are: {valid_mods}'

        assert len(data_modalities) > 0, 'At least one input must be provided.'

        if fusion_method == 'cat':
            self.num_features = 0
        else:
            self.num_features = self.mfs

        self.submodels = {}

        # Clinical -----------------------------------------------------------#
        if 'clinical' in self.data_modalities:
            self.clinical_submodel = ClinicalNet(
                output_vector_size=self.mfs)
            self.submodels['clinical'] = self.clinical_submodel

            if fusion_method == 'cat':
                self.num_features += self.mfs

        # WSI patches --------------------------------------------------------#
        if 'wsi' in self.data_modalities:
            self.wsi_submodel = WsiNet(output_vector_size=self.mfs)
            self.submodels['wsi'] = self.wsi_submodel

            if fusion_method == 'cat':
                self.num_features += self.mfs

        # mRNA ---------------------------------------------------------------#
        if 'mRNA' in self.data_modalities:
            self.mRNA_submodel = FC(1000, self.mfs, 3)
            self.submodels['mRNA'] = self.mRNA_submodel

            if fusion_method == 'cat':
                self.num_features += self.mfs

        # miRNA --------------------------------------------------------------#
        if 'miRNA' in self.data_modalities:
            self.miRNA_submodel = FC(1881, self.mfs, 3, scaling_factor=2)
            self.submodels['miRNA'] = self.miRNA_submodel

            if fusion_method == 'cat':
                self.num_features += self.mfs

        # DNAm ---------------------------------------------------------------#
        if 'DNAm' in self.data_modalities:
            self.DNAm_submodel = FC(5000, self.mfs, 5, scaling_factor=2)
            self.submodels['DNAm'] = self.DNAm_submodel

            if fusion_method == 'cat':
                self.num_features += self.mfs

        # DNAm ---------------------------------------------------------------#
        if 'CNV' in self.data_modalities:
            self.CNV_submodel = CnvNet(output_vector_size=self.mfs)
            self.submodels['CNV'] = self.CNV_submodel

            if fusion_method == 'cat':
                self.num_features += self.mfs

        # Instantiate multimodal aggregator ----------------------------------#
        if len(data_modalities) > 1:
            self.aggregator = Fusion(fusion_method, self.mfs, device)
        else:
            if fusion_method is not None:
                warnings.warn('Input data is unimodal: no fusion procedure.')

        # Fully-connected and risk layers ------------------------------------#
        n_fc_layers = 4
        n_neurons = 512

        self.fc_block = FC(
            in_features=self.num_features, out_features=n_neurons,
            n_layers=n_fc_layers)

        self.risk_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=n_neurons,
                            out_features=n_output_intervals),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        multimodal_features = tuple()

        # Run data through modality sub-models (generate feature vectors) ----#
        for modality in x:
            multimodal_features += (self.submodels[modality](x[modality]),)

        # Feature fusion/aggregation -----------------------------------------#
        if len(multimodal_features) > 1:
            x = self.aggregator(torch.stack(multimodal_features))
            feature_repr = {'modalities': multimodal_features, 'fused': x}
        else:  # skip if running unimodal data
            x = multimodal_features[0]
            feature_repr = {'modalities': multimodal_features[0]}

        # Outputs ------------------------------------------------------------#
        x = self.fc_block(x)
        risk = self.risk_layer(x)

        # Return non-zero features (not missing input data)
        output_features = tuple()

        for modality in multimodal_features:
            modality_features = torch.stack(
                [batch_element for batch_element in modality
                 if batch_element.sum() != 0])
            output_features += modality_features,

        feature_repr['modalities'] = output_features

        return feature_repr, risk
