import random

import torch
import torch.nn.functional as F
from torch import nn as nn


def saturated_sigmoid(x: torch.Tensor):
    """Saturating sigmoid: 1.2 * sigmoid(x) - 0.1 cut to [0, 1]."""
    y = torch.sigmoid(x)
    # noinspection PyTypeChecker
    saturated_sigmoid_ = torch.clamp(1.2 * y - 0.1, 0.0, 1.0)
    return saturated_sigmoid_


class BinarizationLayer(nn.Module):

    def __init__(self, noise_mean=0.0, noise_stddev=1.0):
        super().__init__()
        self.noise_mean = noise_mean
        self.noise_stddev = noise_stddev

    def forward(self, sigma, binarize=None):
        noise = 0
        if self.training:
            noise = torch.normal(self.noise_mean, self.noise_stddev, sigma.size(), device=sigma.device)
        sigma_noised = sigma + noise
        binary_vals = (sigma_noised > 0.0).float()  # gb
        if self.training:  # train with real values
            sat_sigma = saturated_sigmoid(sigma_noised)  # ga
            if binarize is None:
                binarize = random.random() > 0.5
            if binarize:  # train with binary values
                x = binary_vals + sat_sigma - sat_sigma.detach()
            else:
                x = sat_sigma
        else:  # pass only binary values
            sat_sigma = saturated_sigmoid(sigma_noised)
            x = binary_vals
        return x, sat_sigma


class SelectionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, noise_mean=0.0, noise_stddev=1.0, reduction_rate=2):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        mid_out_channels = max(out_channels // reduction_rate, 1)
        self.fc1 = nn.Linear(in_channels, mid_out_channels)
        self.bn = nn.BatchNorm1d(mid_out_channels)
        self.fc2 = nn.Linear(mid_out_channels, out_channels)
        self.binarization = BinarizationLayer(noise_mean, noise_stddev)

    def forward(self, x, binarize=None):
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(self.bn(x))
        sigma = self.fc2(x)
        x = self.binarization(sigma, binarize=binarize)
        return x


class NewBinarySelectionLayer(SelectionLayer):

    def __init__(self, in_channels, noise_mean=0.0, noise_stddev=1.0, reduction_rate=2, do_batchnorm=False):
        super().__init__(in_channels, 1, noise_mean, noise_stddev, reduction_rate)
        delattr(self, 'fc1')
        delattr(self, 'fc2')
        self.do_batchnorm = do_batchnorm
        if not do_batchnorm:
            delattr(self, 'bn')
        self.fc = nn.Linear(in_channels, 1)

    def forward(self, x, binarize=None):
        x = self.gap(x)
        x = torch.flatten(x, 1)
        if self.do_batchnorm:
            x = self.bn(x)
        sigma = self.fc(x)
        if self.training:
            if binarize:
                x_b = self.binarization(sigma, binarize=binarize)
                x = self.binarization(sigma, binarize=not binarize)
                return x, x_b, True
            else:
                x = self.binarization(sigma, binarize=not binarize)
                return x, None, False
        else:
            x_b = self.binarization(sigma)
            return x_b, None, True

class NewBinarizationLayer(nn.Module):

    def __init__(self, noise_mean=0.0, noise_stddev=1.0):
        super().__init__()
        self.noise_mean = noise_mean
        self.noise_stddev = noise_stddev

    def forward(self, sigmas, binarize=None):
        noise = 0
        if self.training:
            noise = torch.normal(self.noise_mean, self.noise_stddev, sigmas.size(), device=sigmas.device)
        sigma_noised = sigmas + noise
        binary_vals = torch.zeros_like(sigmas, dtype=torch.long)
        binary_vals[:, 0] = (sigmas[:, 0] > sigmas[:, 1])
        binary_vals[:, 1] = 1 - binary_vals[:, 0]
        if self.training:  # train with real values
            sat_sigma = saturated_sigmoid(sigma_noised)  # ga
            if binarize is None:
                binarize = random.random() > 0.5
            if binarize:  # train with binary values
                x = binary_vals + sat_sigma - sat_sigma.detach()
            else:
                x = sat_sigma
        else:  # pass only binary values
            x = binary_vals
        return x

class EmbeddingBinarySelectionLayer(SelectionLayer):

    def __init__(self, in_channels, noise_mean=0.0, noise_stddev=1.0, reduction_rate=2, do_batchnorm=False):
        super().__init__(in_channels, 1, noise_mean, noise_stddev, reduction_rate)
        delattr(self, 'fc1')
        delattr(self, 'fc2')
        self.do_batchnorm = do_batchnorm
        if not do_batchnorm:
            delattr(self, 'bn')
        self.fc = nn.Linear(in_channels, 3)
        self.binarization = BinarizationLayer(noise_mean, noise_stddev)
        self.register_buffer('ONES', torch.ones(2))

    def forward(self, x, binarize=None):
        x = self.gap(x)
        x = torch.flatten(x, 1)
        if self.do_batchnorm:
            x = self.bn(x)
        sigmas = self.fc(x)
        if self.training:
            x_b, _ = self.binarization(sigmas[:, 2], binarize=True)
            x, _ = self.binarization(sigmas[:, 0:2], binarize=False)
            return x, x_b, None
        else:
            x_b, x_b_r = self.binarization(sigmas[:, 2])
            _, sat_sigma = self.binarization(sigmas[:, 0:2])
            return sat_sigma, x_b, x_b_r

class BinarySelectionLayer(SelectionLayer):

    def __init__(self, in_channels, noise_mean=0.0, noise_stddev=1.0, reduction_rate=2, do_batchnorm=False):
        super().__init__(in_channels, 1, noise_mean, noise_stddev, reduction_rate)
        delattr(self, 'fc1')
        delattr(self, 'fc2')
        self.do_batchnorm = do_batchnorm
        if not do_batchnorm:
            delattr(self, 'bn')
        self.fc = nn.Linear(in_channels, 1)

    def forward(self, x, binarize=None):
        x = self.gap(x)
        x = torch.flatten(x, 1)
        if self.do_batchnorm:
            x = self.bn(x)
        sigma = self.fc(x)
        x = self.binarization(sigma, binarize=binarize)
        return x


class ClassificationLayer(nn.Module):
    def __init__(self, num_in_channels, num_out_channels, add_noise=False, no_gap=False):
        super().__init__()
        if no_gap:
            self.gap = nn.Identity()
            self.fc = nn.LazyLinear(num_out_channels)
        else:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(num_in_channels, num_out_channels)
        self.add_noise = add_noise

    def forward(self, x):
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if self.training and self.add_noise:
            x += torch.randn(x.size(), device=x.device)
        return x


class BinaryClassificationLayer(ClassificationLayer):

    def __init__(self, num_in_channels, add_noise=False):
        super().__init__(num_in_channels, 1, add_noise)


class HardBinaryClassificationLayer(BinaryClassificationLayer):

    def __init__(self, num_in_channels, add_noise=False):
        super().__init__(num_in_channels, add_noise)

    def forward(self, x, binarize=False):
        x = super().forward(x)
        x = torch.sigmoid(x)
        if not self.training and binarize:
            # x = (x >= 0.0).float()
            x = (x >= 0.5).float()
        return x
