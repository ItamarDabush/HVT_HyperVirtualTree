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
BATCH_SIZE = 100
INITIAL_SIGMA=0.5

ConfigTuple = Tuple[Union[int, str, Tuple[int, int]], ...]
ConfigList = List[Any]

class SubHyperDecisioNet(nn.Module):

    def __init__(self, hyper, initial_hin, multi_hyper, node_code: Tuple[int, ...] = ()):
        super().__init__()
        self.node_code = node_code
        self.hyper = hyper
        self.multi_hyper = multi_hyper
        self.flag = False
        if self.hyper:
            self.features = nn.Sequential(nn.Linear(1, 2320), nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=2, stride=None, padding=0),
                                          nn.Flatten(1, -1), nn.Linear(1, 65600), nn.ReLU())
        else:
            self.features = nn.Sequential((nn.Conv2d(16, 16, 3, stride=(1, 1), padding=(1, 1))), nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=2, stride=None, padding=0),
                                          nn.Flatten(1, -1), nn.Linear(1024, 64), nn.ReLU())
        self.initial_hin = torch.tensor([initial_hin]).to(next(self.parameters()).device)

    def forward(self, x, h_in=None, **kwargs):
        if self.hyper:
            for feature in self.features:
                if 'Linear' in feature.__str__():
                    if self.flag:
                        h_in = self.initial_hin.to(next(self.parameters()).device)
                        tot_weights = feature(h_in)
                        weights = torch.reshape(
                            tot_weights[:65536],
                            (64, 1024))
                        bias = tot_weights[65536:]
                        x = F.linear(x, weights, bias)
                        self.flag = False
                    else:
                        if self.multi_hyper:
                            h_in = h_in.to(next(self.parameters()).device)
                            outputs = []
                            tot_weights = feature(h_in)
                            for i, sigma in enumerate(h_in):
                                curr_tot_weights = tot_weights[i]
                                current_image = x[i:i+1]
                                current_weights = torch.reshape(
                                    curr_tot_weights[:2304],
                                    (16, 16, 3, 3))
                                current_bias = curr_tot_weights[2304:]
                                output = F.conv2d(current_image, current_weights, current_bias, stride=(1, 1), padding=(1, 1))
                                outputs.append(output)
                            x = torch.cat(outputs, dim=0)
                        else:
                            h_in = self.initial_hin.to(next(self.parameters()).device)
                            tot_weights = feature(h_in)
                            weights = torch.reshape(
                                tot_weights[:2304],
                                (16, 16, 3, 3))
                            bias = tot_weights[2304:]
                            x = F.conv2d(x, weights, bias, stride=(1, 1), padding=(1, 1))
                        self.flag = True
                else:
                    x = feature(x)
        else:
            x = self.features(x)
        return x, None

class MainHyperDecisioNet(nn.Module):

    def __init__(self, hyper=False, multi_hyper=False, subcls=SubHyperDecisioNet, node_code: Tuple[int, ...] = ()):
        super().__init__()
        self.node_code = node_code
        self.hyper = hyper
        self.multi_hyper = multi_hyper
        if self.hyper:
            self.features = nn.Sequential(nn.Linear(1, 448), nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=2, stride=None, padding=0))
        else:
            self.features = nn.Sequential((nn.Conv2d(3, 16, 3, stride=(1, 1), padding=(1, 1))), nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=2, stride=None, padding=0))
        self.initial_hin = torch.tensor([INITIAL_SIGMA]).to(next(self.parameters()).device)

        self.left = subcls(hyper=self.hyper, initial_hin=0.5, multi_hyper=self.multi_hyper,
                           node_code=self.node_code + (0,))
        self.right = subcls(hyper=self.hyper, initial_hin=0.5+1/64, multi_hyper=self.multi_hyper,
                            node_code=self.node_code + (1,))
        self.binary_selection_layer = BinarySelectionLayer(16)

    def forward(self, x, h_in=None, **kwargs):
        if self.hyper:
            if h_in is None:
                h_in = self.initial_hin.to(next(self.parameters()).device)
            for feature in self.features:
                if 'Linear' in feature.__str__():
                    tot_weights = feature(h_in)
                    weights = torch.reshape(
                        tot_weights[:432],
                        (16, 3, 3, 3))
                    bias = tot_weights[432:]
                    x = F.conv2d(x, weights, bias, stride=(1, 1), padding=(1, 1))
                else:
                    x = feature(x)
        else:
            x = self.features(x)
        sigma = self.binary_selection_layer(x, **kwargs)
        if self.multi_hyper:
            x0, s0 = self.left(x, 1-sigma, **kwargs)
            x1, s1 = self.right(x, sigma, **kwargs)
        else:
            x0, s0 = self.left(x, **kwargs)
            x1, s1 = self.right(x, **kwargs)
        sigma_broadcasted = sigma[..., None, None] if x0.ndim == 4 else sigma
        x = (1 - sigma_broadcasted) * x0 + sigma_broadcasted * x1
        if s0 is not None and s1 is not None:
            deeper_level_decisions = torch.stack([s0, s1], dim=-1)
            bs = sigma.size(0)
            sigma_idx = sigma.detach().ge(0.5).long().flatten()
            filtered_decisions = deeper_level_decisions[torch.arange(bs), :, sigma_idx]
            sigma = torch.column_stack([sigma, filtered_decisions])
        return x, sigma


class FixedNetworkInNetworkHyperDecisioNet(nn.Module):
    def __init__(self, hyperdecisionet_cls=None, classifier_flag=True, hyper=False, multi_hyper=False):
        super().__init__()
        if hyperdecisionet_cls is None:
            hyperdecisionet_cls = MainHyperDecisioNet
            subhyperdecisionet_cls = SubHyperDecisioNet
        print("NetworkInNetworkDecisioNet init - Using the following config:")
        self.hyperdecisionet = hyperdecisionet_cls(hyper=hyper, multi_hyper=multi_hyper, subcls=subhyperdecisionet_cls)
        if classifier_flag:
            self.classifier = nn.Linear(64, 10)

    def forward(self, x, **kwargs):
        features_out, sigmas = self.hyperdecisionet(x, **kwargs)
        out = self.classifier(features_out)
        return out, sigmas

if __name__ == '__main__':
    from torchinfo import summary

    model = FixedNetworkInNetworkHyperDecisioNet()
    # out, sigmas = model(images)
    summary(model, (1, 3, 32, 32))