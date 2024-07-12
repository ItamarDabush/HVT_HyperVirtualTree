from typing import List, Union, Callable, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_layers.selection_layers import BinarySelectionLayer
from models.network_in_network import NetworkInNetwork
from models.wide_resnet import WideResNet
from utils.binary_tree import Node
from utils.constants import INPUT_SIZE

CIFAR10_INPUT_CHANNELS = INPUT_SIZE['CIFAR10'][0]
CIFAR10_INPUT_SIZE = INPUT_SIZE['CIFAR10'][1]
SIZE_AFTER_TWO_MAXPOOL = 8
INITIAL_SIGMA=0.5
SCALE_FACTOR=64

ConfigTuple = Tuple[Union[int, str, Tuple[int, int]], ...]
ConfigList = List[Any]

class SubHyperDecisioNet(nn.Module):

    def __init__(self, hyper, multi_hyper, hyper_depth=1, initial_hin=0, node_code: Tuple[int, ...] = ()):
        super().__init__()
        self.node_code = node_code
        self.hyper = hyper
        self.multi_hyper = multi_hyper
        if self.hyper:
            self.features = nn.Sequential(nn.Linear(hyper_depth, 230496), nn.ReLU(inplace=True),
                                          (nn.Conv2d(96, 96, 1, padding=0)), nn.ReLU(inplace=True),
                                          (nn.Conv2d(96, 96, 1, padding=0)), nn.ReLU(inplace=True),
                                          nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                                          nn.Dropout(0.5, inplace=True),
                                          (nn.Conv2d(96, 96, 3, padding=1)), nn.ReLU(inplace=True),
                                          (nn.Conv2d(96, 96, 1, padding=0)), nn.ReLU(inplace=True),
                                          (nn.Conv2d(96, 10, 1, padding=0)))
        else:
            self.features = nn.Sequential((nn.Conv2d(96, 192, 5, padding=2)), nn.ReLU(inplace=True),
                                          (nn.Conv2d(192, 192, 1, padding=0)), nn.ReLU(inplace=True),
                                          (nn.Conv2d(192, 192, 1, padding=0)), nn.ReLU(inplace=True),
                                          nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                                          nn.Dropout(0.5),
                                          (nn.Conv2d(192, 192, 3, padding=1)), nn.ReLU(inplace=True),
                                          (nn.Conv2d(192, 192, 1, padding=0)), nn.ReLU(inplace=True),
                                          (nn.Conv2d(192, 10, 1, padding=0)))

        if self.hyper:
            self.register_buffer('INITIAL_SIGMA', torch.tensor([INITIAL_SIGMA+initial_hin/SCALE_FACTOR]))

        if self.multi_hyper:
            self.register_buffer('INITIAL_SIGMA', torch.tensor([INITIAL_SIGMA]))
            self.register_buffer('SCALE_FACTOR', torch.tensor([SCALE_FACTOR]))

    def forward(self, x, h_in=None, **kwargs):
        if self.hyper:
            for i, feature in enumerate(self.features):
                if 'Linear' in feature.__str__() and i == 0:
                    if self.multi_hyper:
                        h_in = (self.INITIAL_SIGMA+h_in/self.SCALE_FACTOR)
                        outputs = []
                        tot_weights = feature(h_in)
                        for i, sigma in enumerate(h_in):
                            curr_tot_weights = tot_weights[i]
                            current_image = x[i:i+1]
                            current_weights = torch.reshape(
                                curr_tot_weights[:230400],
                                (96, 96, 5, 5))
                            current_bias = curr_tot_weights[230400:]
                            output = F.conv2d(current_image, current_weights, current_bias, padding=2)
                            outputs.append(output)
                        x = torch.cat(outputs, dim=0)
                    else:
                        h_in = self.INITIAL_SIGMA+h_in/self.SCALE_FACTOR
                        tot_weights = feature(h_in)
                        weights = torch.reshape(
                            tot_weights[:230400],
                            (96, 96, 5, 5))
                        bias = tot_weights[230400:]
                        x = F.conv2d(x, weights, bias, padding=2)
                else:
                    x = feature(x)
        else:
            x = self.features(x)
        return x, None

class SharedNet(nn.Module):

    def __init__(self, hyper=False, multi_hyper=False, subcls=SubHyperDecisioNet, node_code: Tuple[int, ...] = ()):
        super().__init__()
        self.node_code = node_code
        self.hyper = hyper
        self.multi_hyper = multi_hyper
        self.features = nn.Sequential(nn.Conv2d(3, 192, 5, padding=2), nn.ReLU(inplace=True),
                                      nn.Conv2d(192, 160, 1, padding=0), nn.ReLU(inplace=True),
                                      nn.Conv2d(160, 96, 1, padding=0), nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                      nn.Dropout(0.5))
        if self.hyper:
            self.left = subcls(hyper=self.hyper, multi_hyper=self.multi_hyper, initial_hin=0,
                               node_code=self.node_code + (0,))
            self.right = subcls(hyper=self.hyper, multi_hyper=self.multi_hyper, initial_hin=1,
                                node_code=self.node_code + (1,))
        else:
            self.left = subcls(hyper=self.hyper, multi_hyper=self.multi_hyper,
                               node_code=self.node_code + (0,))
            self.right = subcls(hyper=self.hyper, multi_hyper=self.multi_hyper,
                                node_code=self.node_code + (1,))
        self.binary_selection_layer = BinarySelectionLayer(96)

    def forward(self, x, h_in=None, **kwargs):
        x = self.features(x)
        sigma = self.binary_selection_layer(x, **kwargs)
        if self.multi_hyper:
            x0, s0 = self.left(x, 1-sigma, **kwargs)
            x1, s1 = self.right(x, sigma, **kwargs)
        else:
            x0, s0 = self.left(x, 0, **kwargs)
            x1, s1 = self.right(x, 1, **kwargs)
        # sigma_broadcasted = sigma[..., None, None] if x0.ndim == 4 else sigma
        # x = (1 - sigma_broadcasted) * x0 + sigma_broadcasted * x1
        x = x0 + x1
        return x, sigma

class FixedBasicHyperDecisioNet(nn.Module):
    def __init__(self, classifier_flag=True, hyper=False, multi_hyper=False):
        super().__init__()
        hyperdecisionet_cls = SharedNet
        subhyperdecisionet_cls = SubHyperDecisioNet
        self.hyperdecisionet = hyperdecisionet_cls(hyper=hyper, multi_hyper=multi_hyper, subcls=subhyperdecisionet_cls)
        if classifier_flag:
            # self.classifier = nn.Linear(96, 10)
            self.classifier = nn.AdaptiveAvgPool2d((1, 1)) # original option
        print("NetworkInNetworkDecisioNet init - Using the following config:")
        print(self)

    def forward(self, x, **kwargs):
        features_out, sigmas = self.hyperdecisionet(x, **kwargs)
        out = self.classifier(features_out)
        out = torch.flatten(out, 1)
        return out, sigmas

class SharedNet_1(nn.Module):

    def __init__(self, hyper=False, multi_hyper=False, subcls=SubHyperDecisioNet, node_code: Tuple[int, ...] = ()):
        super().__init__()
        self.node_code = node_code
        self.hyper = hyper
        self.multi_hyper = multi_hyper
        self.features = nn.Sequential(nn.Conv2d(3, 192, 5, padding=2), nn.ReLU(inplace=True),
                                      nn.Conv2d(192, 160, 1, padding=0), nn.ReLU(inplace=True),
                                      nn.Conv2d(160, 96, 1, padding=0), nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                      nn.Dropout(0.5))

        self.sub = subcls(hyper=self.hyper, multi_hyper=self.multi_hyper,
                           node_code=self.node_code + (0,))

        if self.hyper:
            self.binary_selection_layer = BinarySelectionLayer(96)

    def forward(self, x, h_in=None, **kwargs):
        x = self.features(x)

        if self.hyper:
            sigma = self.binary_selection_layer(x, **kwargs)
            if self.multi_hyper:
                x0, s0 = self.sub(x, 1 - sigma, **kwargs)
                x1, s1 = self.sub(x, sigma, **kwargs)
            else:
                x0, s0 = self.sub(x, 0, **kwargs)
                x1, s1 = self.sub(x, 1,  **kwargs)
            # sigma_broadcasted = sigma[..., None, None] if x1.ndim == 4 else sigma
            # x = (1 - sigma_broadcasted) * x0 + sigma_broadcasted * x1
            x = x0 + x1
        else:
            x, sigma = self.sub(x, **kwargs)
        # if s0 is not None and s1 is not None:
        #     deeper_level_decisions = torch.stack([s0, s1], dim=-1)
        #     bs = sigma.size(0)
        #     sigma_idx = sigma.detach().ge(0.5).long().flatten()
        #     filtered_decisions = deeper_level_decisions[torch.arange(bs), :, sigma_idx]
        #     sigma = torch.column_stack([sigma, filtered_decisions])
        return x, sigma

class FixedBasicHyperDecisioNet_1(nn.Module):
    def __init__(self, hyperdecisionet_cls=None, classifier_flag=True, hyper=False, multi_hyper=False):
        super().__init__()
        if hyperdecisionet_cls is None:
            hyperdecisionet_cls = SharedNet_1
            subhyperdecisionet_cls = SubHyperDecisioNet
        print("NetworkInNetworkDecisioNet init - Using the following config:")
        self.hyperdecisionet = hyperdecisionet_cls(hyper=hyper, multi_hyper=multi_hyper, subcls=subhyperdecisionet_cls)
        if classifier_flag:
            self.classifier = nn.AdaptiveAvgPool2d((1, 1))  # original option
        print("NetworkInNetworkDecisioNet init - Using the following config:")
        print(self)

    def forward(self, x, **kwargs):
        features_out, sigmas = self.hyperdecisionet(x, **kwargs)
        out = self.classifier(features_out)
        out = torch.flatten(out, 1)
        return out, sigmas

if __name__ == '__main__':
    from torchinfo import summary

    model = FixedBasicHyperDecisioNet_1()
    # out, sigmas = model(images)
    summary(model, (1, 3, 32, 32))