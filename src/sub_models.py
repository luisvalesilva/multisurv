"""MultiSurv sub-models."""

from bisect import bisect_left

import torch
import torch.nn as nn
from torchvision import models

from embrace_net import EmbraceNet
from attention import Attention


def freeze_layers(model, up_to_layer=None):
    if up_to_layer is not None:
        # Freeze all layers
        for i, param in model.named_parameters():
            param.requires_grad = False

        # Release all layers after chosen layer
        frozen_layers = []
        for name, child in model.named_children():
            if up_to_layer in frozen_layers:
                for params in child.parameters():
                    params.requires_grad = True
            else:
                frozen_layers.append(name)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.model = models.resnext50_32x4d(pretrained=True)
        freeze_layers(self.model, up_to_layer='layer3')
        self.n_features = self.model.fc.in_features

        # Remove classifier (last layer)
        self.model = nn.Sequential(*(list(self.model.children())[:-1]))

    def forward(self, x):
        out = self.model(x)

        return out


class FC(nn.Module):
    "Fully-connected model to generate final output."
    def __init__(self, in_features, out_features, n_layers, dropout=True,
                 batchnorm=False, scaling_factor=4):
        super(FC, self).__init__()
        if n_layers == 1:
            layers = self._make_layer(in_features, out_features, dropout,
                                      batchnorm)
        elif n_layers > 1:
            n_neurons = self._pick_n_neurons(in_features)
            if n_neurons < out_features:
                n_neurons = out_features

            if n_layers == 2:
                layers = self._make_layer(
                    in_features, n_neurons, dropout, batchnorm=True)
                layers += self._make_layer(
                    n_neurons, out_features, dropout, batchnorm)
            else:
                for layer in range(n_layers):
                    last_layer_i = range(n_layers)[-1]

                    if layer == 0:
                        n_neurons *= scaling_factor
                        layers = self._make_layer(
                            in_features, n_neurons, dropout, batchnorm=True)
                    elif layer < last_layer_i:
                        n_in = n_neurons
                        n_neurons = self._pick_n_neurons(n_in)
                        if n_neurons < out_features:
                            n_neurons = out_features
                        layers += self._make_layer(
                            n_in, n_neurons, dropout, batchnorm=True)
                    else:
                        layers += self._make_layer(
                            n_neurons, out_features, dropout, batchnorm)
        else:
            raise ValueError('"n_layers" must be positive.')

        self.fc = nn.Sequential(*layers)

    def _make_layer(self, in_features, out_features, dropout, batchnorm):
        layer = nn.ModuleList()
        if dropout:
            layer.append(nn.Dropout())
        layer.append(nn.Linear(in_features, out_features))
        layer.append(nn.ReLU(inplace=True))
        if batchnorm:
            layer.append(nn.BatchNorm1d(out_features))

        return layer

    def _pick_n_neurons(self, n_features):
        # Pick number of features from list immediately below n input
        n_neurons = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        idx = bisect_left(n_neurons, n_features)

        return n_neurons[0 if idx == 0 else idx - 1]

    def forward(self, x):
        return self.fc(x)


class ClinicalNet(nn.Module):
    """Clinical data extractor.

    Handle continuous features and categorical feature embeddings.
    """
    def __init__(self, output_vector_size, embedding_dims=[
        (33, 17), (2, 1), (8, 4), (3, 2), (3, 2), (3, 2), (3, 2), (3, 2),
        (20, 10)]):
        super(ClinicalNet, self).__init__()
        # Embedding layer
        self.embedding_layers = nn.ModuleList([nn.Embedding(x, y)
                                               for x, y in embedding_dims])

        n_embeddings = sum([y for x, y in embedding_dims])
        n_continuous = 1

        # Linear Layers
        self.linear = nn.Linear(n_embeddings + n_continuous, 256)

        # Embedding dropout Layer
        self.embedding_dropout = nn.Dropout()

        # Continuous feature batch norm layer
        self.bn_layer = nn.BatchNorm1d(n_continuous)

        # Output Layer
        self.output_layer = FC(256, output_vector_size, 1)

    def forward(self, x):
        categorical_x, continuous_x = x
        categorical_x = categorical_x.to(torch.int64)

        x = [emb_layer(categorical_x[:, i])
             for i, emb_layer in enumerate(self.embedding_layers)]
        x = torch.cat(x, 1)
        x = self.embedding_dropout(x)

        continuous_x = self.bn_layer(continuous_x)

        x = torch.cat([x, continuous_x], 1)
        out = self.output_layer(self.linear(x))

        return out


class CnvNet(nn.Module):
    """Gene copy number variation data extractor."""
    def __init__(self, output_vector_size, embedding_dims=[(3, 2)] * 2000):
        super(CnvNet, self).__init__()
        self.embedding_layers = nn.ModuleList([nn.Embedding(x, y)
                                               for x, y in embedding_dims])
        n_embeddings = 2 * 2000
        self.fc = FC(in_features=n_embeddings, out_features=output_vector_size,
                     n_layers=5, scaling_factor=1)

    def forward(self, x):
        x = x.to(torch.int64)

        x = [emb_layer(x[:, i])
             for i, emb_layer in enumerate(self.embedding_layers)]
        x = torch.cat(x, 1)
        out = self.fc(x)

        return out

class WsiNet(nn.Module):
    "WSI patch feature extractor and aggregator."
    def __init__(self, output_vector_size):
        super(WsiNet, self).__init__()
        self.feature_extractor = ResNet()
        self.num_image_features = self.feature_extractor.n_features
        # Multiview WSI patch aggregation
        self.fc = FC(self.num_image_features, output_vector_size , 1)

    def forward(self, x):
        view_pool = []

        # Extract features from each patch
        for v in x:
            v = self.feature_extractor(v)
            v = v.view(v.size(0), self.num_image_features)

            view_pool.append(v)

        # Aggregate features from all patches
        patch_features = torch.stack(view_pool).max(dim=1)[0]

        out = self.fc(patch_features)

        return out


class Fusion(nn.Module):
    "Multimodal data aggregator."
    def __init__(self, method, feature_size, device):
        super(Fusion, self).__init__()
        self.method = method
        methods = ['cat', 'max', 'sum', 'prod', 'embrace', 'attention']

        if self.method not in methods:
            raise ValueError('"method" must be one of ', methods)

        if self.method == 'embrace':
            if device is None:
                raise ValueError(
                    '"device" is required if "method" is "embrace"')

            self.embrace = EmbraceNet(device=device)

        if self.method == 'attention':
            if not feature_size:
                raise ValueError(
                    '"feature_size" is required if "method" is "attention"')
            self.attention = Attention(size=feature_size)

    def forward(self, x):
        if self.method == 'attention':
            out = self.attention(x)
        if self.method == 'cat':
            out = torch.cat([m for m in x], dim=1)
        if self.method == 'max':
            out = x.max(dim=0)[0]
        if self.method == 'sum':
            out = x.sum(dim=0)
        if self.method == 'prod':
            out = x.prod(dim=0)
        if self.method == 'embrace':
            out = self.embrace(x)

        return out
