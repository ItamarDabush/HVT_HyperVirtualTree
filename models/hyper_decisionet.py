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


NIN_CFG = {'cifar10_2_basic': [((16, 3, 1, 1), ('M', 2, None, 0)),
                               ((16, 3, 1, 1), ('M', 2, None, 0), ('V', int(16 * (8 ** 2))),
                                ('fc', 64, True))],
           'cifar10_4_basic': [((16, 3, 1, 1), ('M', 2, None, 0)),
                               ((16, 3, 1, 1), ('M', 2, None, 0)),
                               (('V', int(16 * (8 ** 2))), ('fc', 32, True))],
           'cifar10_2_hyper_basic': [(('conv2d', 3, 16, 3, 32), ('M', 2, None, 0)),
                               (('conv2d_2', 16, 16, 3, 16), ('M', 2, None, 0), ('V',),
                                ('fc', int(16 * (8 ** 2)), 10, True))]
           }


class HyperDecisioNet(nn.Module):

    def __init__(self,
                 config: ConfigList,
                 num_in_channels: int,
                 make_layers_func: Callable[[ConfigTuple, int], Tuple[nn.Module, int]],
                 classes_division: Optional[Node] = None,
                 node_code: Tuple[int, ...] = ()):
        super().__init__()
        num_levels = len(config)
        assert num_levels >= 1
        self.is_leaf = (num_levels == 1)
        self.curr_level_config = config[0]
        self.features, num_out_channels, size_out = make_layers_func(self.curr_level_config, num_in_channels, self.curr_level_config[0][1])
        self.node_code = node_code
        if classes_division is not None:
            self.node_classes = classes_division.value
        self.num_out_channels = num_out_channels
        self.initial_hin = torch.tensor([INITIAL_SIGMA]).to(next(self.parameters()).device)
        # self.initial_hin = torch.randn(100).to(next(self.parameters()).device)

        if not self.is_leaf:
            left_cd = classes_division.left if classes_division is not None else None
            right_cd = classes_division.right if classes_division is not None else None
            cls = self.__class__
            self.left = cls(config[1:], num_out_channels, make_layers_func, left_cd, self.node_code + (0,))
            self.right = cls(config[1:], num_out_channels, make_layers_func, right_cd, self.node_code + (1,))
            self.binary_selection_layer = BinarySelectionLayer(num_out_channels)

    def forward(self, x, h_in=None, **kwargs):
        if h_in is None:
            h_in = self.initial_hin.to(next(self.parameters()).device)
        for feature, layer_conf in zip(self.features, self.curr_level_config):
            if layer_conf[0] == 'fc':
                h_in = self.initial_hin.to(next(self.parameters()).device)
                tot_weights = feature(h_in)
                weights = torch.reshape(
                    tot_weights[:layer_conf[1] * layer_conf[2]],
                    (layer_conf[2], layer_conf[1]))
                bias = tot_weights[layer_conf[1] * layer_conf[2]:]
                x = F.relu(F.linear(x, weights, bias))
            elif layer_conf[0] in ['conv2d', 'conv2d_2']:
                h_in = self.initial_hin.to(next(self.parameters()).device)
                tot_weights = feature(h_in)
                weights = torch.reshape(
                    tot_weights[:layer_conf[2] * layer_conf[1] * layer_conf[3] * layer_conf[3]],
                    (layer_conf[2], layer_conf[1], layer_conf[3], layer_conf[3]))
                bias = tot_weights[layer_conf[2] * layer_conf[1] * layer_conf[3] * layer_conf[3]:]
                x = F.relu(F.conv2d(x, weights, bias, stride=(1, 1), padding=(1, 1)))
            # elif layer_conf[0] == 'conv2d_2':
            #     outputs = []
            #     tot_weights = feature(h_in)
            #     for i, sigma in enumerate(h_in):
            #         curr_tot_weights=tot_weights[i]
            #         current_image = x[i:i+1]
            #         current_weights = torch.reshape(
            #             curr_tot_weights[:layer_conf[2] * layer_conf[1] * layer_conf[3] * layer_conf[3]],
            #             (layer_conf[2], layer_conf[1], layer_conf[3], layer_conf[3]))
            #         current_bias = curr_tot_weights[layer_conf[2] * layer_conf[1] * layer_conf[3] * layer_conf[3]:]
            #         output = F.conv2d(current_image, current_weights, current_bias, stride=(1, 1), padding=(1, 1))
            #         outputs.append(output)
            #     x = torch.cat(outputs, dim=0)
            else:
                x = feature(x)
        if self.is_leaf:
            return x, None
        sigma = torch.zeros(100, 1).to(next(self.parameters()).device) #self.binary_selection_layer(x, **kwargs)
        x0, s0 = self.left(x, self.initial_hin, **kwargs)
        x1, s1 = self.right(x, self.initial_hin, **kwargs)
        sigma_broadcasted = sigma[..., None, None] if x0.ndim == 4 else sigma
        x = (1 - sigma_broadcasted) * x0 + sigma_broadcasted * x1
        if s0 is not None and s1 is not None:
            deeper_level_decisions = torch.stack([s0, s1], dim=-1)
            bs = sigma.size(0)
            sigma_idx = sigma.detach().ge(0.5).long().flatten()
            filtered_decisions = deeper_level_decisions[torch.arange(bs), :, sigma_idx]
            sigma = torch.column_stack([sigma, filtered_decisions])
        return x, sigma


class NetworkInNetworkHyperDecisioNet(nn.Module):
    def __init__(self, cfg_name='10_baseline', dropout=True, config=None,
                 classes_division: Optional[Node] = None, hyperdecisionet_cls=None,
                 num_in_channels=3):
        super().__init__()
        if config is None:
            config = NIN_CFG[cfg_name]
        if hyperdecisionet_cls is None:
            hyperdecisionet_cls = HyperDecisioNet
        if not dropout:
            config = [x for x in config if x != 'D']
        # config[-1][-1] = (num_classes, 1)
        print("NetworkInNetworkDecisioNet init - Using the following config:")
        print(config)
        self.hyperdecisionet = hyperdecisionet_cls(config, num_in_channels, NetworkInNetwork.make_hyper_layers_by_config,
                                         classes_division)
        if 'basic' in cfg_name:
            self.classifier = nn.Linear(config[-1][-1][-2], 10)
        else:
            self.classifier = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, **kwargs):
        features_out, sigmas = self.hyperdecisionet(x, **kwargs)
        out = self.classifier(features_out)
        # out = torch.flatten(out, 1)
        return out, sigmas

if __name__ == '__main__':
    from torchinfo import summary

    model = NetworkInNetworkHyperDecisioNet(cfg_name='100_baseline_single_early')
    # out, sigmas = model(images)
    summary(model, (1, 3, 32, 32))