from itertools import combinations

from matplotlib import pyplot as plt
import torch

from coach import ModelCoach


class LRRangeTest:
    """Optimal learning rate range test.

    "LR range test" strategy (Smith 2017; https://arxiv.org/pdf/1506.01186.pdf)
    popularized by the MOOC from Fast.ai (and their fastai package). The code
    here was modeled after a blog post by Sylvain Gugger
    (https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html).
    """
    def __init__(self, dataloader, optimizer, criterion, auxiliary_criterion,
                 model, output_intervals, device):
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion.to(device)
        self.aux_criterion = (auxiliary_criterion.to(device)
                              if auxiliary_criterion is not None
                              else auxiliary_criterion)
        self.model = model.to(device)
        self.output_intervals = output_intervals
        self.device = device
        self.lrs = []
        self.losses = []

    def _data_to_device(self, data):
        for modality in data:
            if isinstance(data[modality], (list, tuple)):  # Clinical data
                data[modality] = tuple([v.to(self.device)
                                        for v in data[modality]])
            else:
                data[modality] = data[modality].to(self.device)

        return data

    def run(self, init_value=1e-8, final_value=10., beta=0.98):
        "Run test."
        power = (1 / (len(self.dataloader) - 1))
        mult = (final_value / init_value) ** power
        lr = init_value
        self.optimizer.param_groups[0]['lr'] = lr
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0

        # ModelCoach expects dict of dataloaders
        model_coach = ModelCoach(
            model=self.model, dataloaders={'train': self.dataloader},
            optimizer=self.optimizer, criterion=self.criterion,
            auxiliary_criterion=self.aux_criterion,
            output_intervals=self.output_intervals, device=self.device)

        print('>>> Compute loss at increasing LR values')

        for data in self.dataloader:
            batch_num += 1
            print('\r' + f'    Iterate over mini-batches: {str(batch_num)}',
                  end='')

            # Get the loss for this mini-batch of inputs/outputs
            loss, _, _, _ = model_coach._process_data_batch(
                data, phase='train', retain_graph=True)

            # Compute the smoothed loss
            avg_loss = (beta * avg_loss + (1 - beta) *
                        loss.data.item())
            smoothed_loss = avg_loss / (1 - beta ** batch_num)

            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                print()
                print('    Exploding loss; finish test.')
                return self

            # Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            # Store the values
            self.losses.append(smoothed_loss)
            self.lrs.append(lr)

            # Do the optimizer step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update the lr for the next step
            lr *= mult
            self.optimizer.param_groups[0]['lr'] = lr

        print()
        print('    Completed test.')

        return self

    def plot(self, trim=4):
        "Plot test results."
        lrs, losses = self.lrs, self.losses
        lrs, losses = lrs[10:-trim], losses[10:-trim]

        fig = plt.figure(figsize=(5, 6))
        ax = fig.add_subplot(2, 1, 1)
        line, = ax.plot(lrs, losses, color='red')
        ax.set_xscale('log')
        plt.grid(True, which='both', axis='x', color='0.90')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')
        plt.title('Learning rate range test')
        plt.show()
