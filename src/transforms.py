"""Dataset transforms.

Image trasnformation functionality to use with PyTorch.
"""

import random

import numpy as np
import torch
from PIL import Image


class ToPIL:
    """Convert patch in sample from Numpy ndarray to PIL Image."""

    def __call__(self, image):
        return Image.fromarray(image)


class ToNumpy:
    """Convert PIL image patch in sample to Numpy ndarray."""

    def __call__(self, image):
        # Drop A of PIL RGBA, if present
        return np.array(image)[:, :, :3]


class RandomFlipUpDown:
    """Vertically flip the patch randomly with probability 0.5."""

    def __init__(self, probability=0.5):
        """
        Parameters
        ----------
        probability: float
            Probability to apply flip with.
        """
        self.probability = probability

    def __call__(self, image):
        if random.random() < self.probability:
            image = np.flipud(image)

            return image

        return image


class RandomRotate:
    """Rotate the given patch and mask by random multiple of 90 degrees."""

    def __call__(self, image):
        random_k = random.randint(0, 4)
        image = np.rot90(image, random_k)

        return image


class ToTensor:
    """Convert patch ndarray to Tensor."""

    def __init__(self, output_dict=True):
        """
        Parameters
        ----------
        output_dict:bool
            Whether to ouptut dict of patch and label batches. If false outputs
            tuple.
        """
        self.output_dict = output_dict

    def __call__(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # also, scale pixels to range [0.0, 1.0]
        image = image.transpose((2, 0, 1)) * 1./255
        image = torch.from_numpy(image).float()

        return image
