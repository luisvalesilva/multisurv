"""EmbraceNet multimodal fusion architecture.

Choi and Lee, Information Fusion 2019.
"""

import torch


class EmbraceNet(torch.nn.Module):
    """Embracement modality feature aggregation layer."""
    def __init__(self, device='cuda:0'):
        """Embracement modality feature aggregation layer.

        Note: EmbraceNet needs to deal with mini batch elements differently
        (check missing data and adjust sampling probailities accordingly). This
        way, we take the unusual measure of considering the batch dimension in
        every operation.

        Parameters
        ----------
        device: "torch.device" object
            Device to which input data is allocated (sampling index tensor is
            allocated to the same device).
        """
        super(EmbraceNet, self).__init__()
        self.device = device

    def _get_selection_probabilities(self, d, b):
        p = torch.ones(d.size(0), b)  # Size modalities x batch

        # Handle missing data
        for i, modality in enumerate(d):
            for j, batch_element in enumerate(modality):
                if len(torch.nonzero(batch_element)) < 1:
                    p[i, j] = 0

        # Equal chances to all available modalities in each mini batch element
        m_vector = torch.sum(p, dim=0)
        p /= m_vector

        return p

    def _get_sampling_indices(self, p, c, m):
        r = torch.multinomial(
            input=p.transpose(0, 1), num_samples=c, replacement=True)
        r = torch.nn.functional.one_hot(r.long(), num_classes=m)
        r = r.permute(2, 0, 1)

        return r


    def forward(self, x):
        m, b, c = x.size()

        p = self._get_selection_probabilities(x, b)
        r = self._get_sampling_indices(p, c, m).float().to(self.device)

        d_prime = r * x
        e = d_prime.sum(dim=0)

        return e
