import netron
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.constants import INPUT_SIZE, NUM_CLASSES, INPUT_SIZE
from utils.parallel_utils import ModuleParallel, convParallel


class HyperNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(HyperNetwork, self).__init__()

        # Hypernetwork layers
        # self.fc1 = nn.Linear(1, 64)
        # self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(1, output_size)

    def forward(self, x):
        # x = self.fc1(x)
        # x = self.fc2(x)
        x = self.fc3(x)
        return x

class MetaBlock(nn.Module):
    CONFIG_CIFAR10 = [(32, 32), 'relu', 'M', (16, 16), 'relu', 'M', (8, 8), 'V', 'FC', 'FC']
    CONFIG_MNIST = [(28, 28), 'relu', 'M', (14, 14), 'relu', 'M', (7, 7), 'V', 'FC', 'FC']

    def __init__(self, dataset_name, num_branches, out_size, scale_factor, parallel, device):
        super(MetaBlock, self).__init__()
        # meta network
        self.width = 128
        self.ker_size = 3
        self.filters = 16
        self.filters2 = 2 * self.filters
        self.out_size = out_size
        self.scale_factor = scale_factor
        self.parallel = parallel
        self.dataset_name = dataset_name
        self.num_branches = num_branches
        self.device = device

        if dataset_name == 'MNIST':
            self.config = MetaBlock.CONFIG_MNIST
        if dataset_name == 'CIFAR10':
            self.config = MetaBlock.CONFIG_CIFAR10

        self.hyper_0 = HyperNetwork(input_size=1,
                                    hidden_size=10,
                                    output_size=(int(self.filters) + self.filters * INPUT_SIZE[self.dataset_name][
                                        0] * self.ker_size * self.ker_size))
        self.hyper_1 = HyperNetwork(input_size=1, hidden_size=10,
                                    output_size=(int(self.filters2) + self.filters2 * self.filters *
                                                 self.ker_size * self.ker_size))
        self.hyper_2 = HyperNetwork(input_size=1, hidden_size=10,
                                    output_size=(int(self.width) + self.width *
                                                 self.filters2 * self.config[6][0] * self.config[6][1]))
        self.hyper_3 = HyperNetwork(input_size=1, hidden_size=10,
                                    output_size=(int(self.out_size) + self.width *
                                                 self.out_size))

        if self.parallel:
            self.scale_replicated = [self.scale.to(f'cuda:{device}') for device in [0, 1]]
        else:
            self.scale = torch.stack(
                [torch.tensor(0.1 + i / self.scale_factor, device=self.device, dtype=torch.float32).unsqueeze(0) for i
                 in
                 range(1, self.num_branches + 1)])

    def forward(self, x):
        if self.parallel:
            if x.device.index == 0:
                self.scale = self.scale_replicated[0]
            else:
                self.scale = self.scale_replicated[1]

        out = [None for _ in range(self.num_branches)]

        for i in range(self.num_branches):
            tot_weights = self.hyper_0(self.scale[i])
            weights = torch.reshape(
                tot_weights[:self.filters * INPUT_SIZE[self.dataset_name][0] * self.ker_size * self.ker_size],
                (self.filters, INPUT_SIZE[self.dataset_name][0], self.ker_size, self.ker_size))
            bias = tot_weights[self.filters * INPUT_SIZE[self.dataset_name][0] * self.ker_size * self.ker_size:]
            out[i] = F.max_pool2d(F.relu(F.conv2d(x, weights, bias, stride=(1, 1), padding=(1, 1))), 2)

        for i in range(self.num_branches):
            tot_weights = self.hyper_1(self.scale[i])
            weights = torch.reshape(tot_weights[:self.filters2 * self.filters * self.ker_size * self.ker_size],
                                    (self.filters2, self.filters, self.ker_size, self.ker_size))
            bias = tot_weights[self.filters2 * self.filters * self.ker_size * self.ker_size:]
            out[i] = F.max_pool2d(F.relu(F.conv2d(out[i], weights, bias, stride=(1, 1), padding=(1, 1))), 2)
            out[i] = out[i].view(-1, self.filters2 * self.config[6][0] * self.config[6][1])

        for i in range(self.num_branches):
            tot_weights = self.hyper_2(self.scale[i])
            weights = torch.reshape(tot_weights[:self.width * self.filters2 * self.config[6][0] * self.config[6][1]],
                                    (self.width, self.filters2 * self.config[6][0] * self.config[6][1]))
            bias = tot_weights[self.width * self.filters2 * self.config[6][0] * self.config[6][1]:]
            out[i] = F.relu(F.linear(out[i], weights, bias))

        for i in range(self.num_branches):
            tot_weights = self.hyper_3(self.scale[i])
            weights = torch.reshape(tot_weights[:self.width * self.out_size],
                                    (self.out_size, self.width))
            bias = tot_weights[self.width * self.out_size:]
            out[i] = F.linear(out[i], weights, bias)

        return out

class MetaBlockCls(nn.Module):
    CONFIG_CIFAR10 = [(32, 32), 'relu', 'M', (16, 16), 'relu', 'M', (8, 8), 'V', 'FC', 'FC']
    CONFIG_MNIST = [(28, 28), 'relu', 'M', (14, 14), 'relu', 'M', (7, 7), 'V', 'FC', 'FC']

    def __init__(self, dataset_name, num_branches, out_size, scale_factor, parallel, device):
        super(MetaBlockCls, self).__init__()
        # meta network
        self.width = 128
        self.ker_size = 3
        self.filters = 16
        self.filters2 = 2 * self.filters
        self.out_size = out_size
        self.scale_factor = scale_factor
        self.parallel = parallel
        self.dataset_name = dataset_name
        self.num_branches = num_branches
        self.device = device
        self.hyper_0 = HyperNetwork(input_size=1,
                                    hidden_size=(int(self.filters) + self.filters * INPUT_SIZE[self.dataset_name][
                                        0] * self.ker_size * self.ker_size),
                                    output_size=(int(self.filters) + self.filters * INPUT_SIZE[self.dataset_name][
                                        0] * self.ker_size * self.ker_size))
        self.hyper_1 = HyperNetwork(input_size=1, hidden_size=(int(self.filters2) + self.filters2 * self.filters *
                                 self.ker_size * self.ker_size), output_size=(int(self.filters2) + self.filters2 * self.filters *
                                 self.ker_size * self.ker_size))
        self.hyper_2 = HyperNetwork(input_size=1, hidden_size=(int(self.width) + self.width *
                                 self.filters2 * self.config[6][0] * self.config[6][1]), output_size=(int(self.width) + self.width *
                                 self.filters2 * self.config[6][0] * self.config[6][1]))
        self.hyper_3 = HyperNetwork(input_size=1, hidden_size=(int(self.out_size) + self.width *
                                 self.out_size), output_size=(int(self.out_size) + self.width *
                                 self.out_size))

        if dataset_name == 'MNIST':
            self.config = MetaBlock.CONFIG_MNIST
        if dataset_name == 'CIFAR10':
            self.config = MetaBlock.CONFIG_CIFAR10

        self.linear1 = nn.Linear(self.filters2 * self.config[6][0] * self.config[6][1], self.width)
        self.linear2 = nn.Linear(self.width, self.out_size)

        if self.parallel:
            self.scale_replicated = [self.scale.to(f'cuda:{device}') for device in [0, 1]]
        else:
            self.scale = torch.stack(
                [torch.tensor(0.1 + i / self.scale_factor, device=self.device, dtype=torch.float32).unsqueeze(0) for i
                 in
                 range(1, self.num_branches + 1)])

    def forward(self, x):
        if self.parallel:
            if x.device.index == 0:
                self.scale = self.scale_replicated[0]
            else:
                self.scale = self.scale_replicated[1]

        out = [None for _ in range(self.num_branches)]

        for i in range(self.num_branches):
            tot_weights = self.hyper_0(self.scale[i])
            weights = torch.reshape(
                tot_weights[:self.filters * INPUT_SIZE[self.dataset_name][0] * self.ker_size * self.ker_size],
                (self.filters, INPUT_SIZE[self.dataset_name][0], self.ker_size, self.ker_size))
            bias = tot_weights[self.filters * INPUT_SIZE[self.dataset_name][0] * self.ker_size * self.ker_size:]
            out[i] = F.max_pool2d(F.relu(F.conv2d(x[i], weights, bias, stride=(1, 1), padding=(1, 1))), 2)

        for i in range(self.num_branches):
            tot_weights = self.hyper_1(self.scale[i])
            weights = torch.reshape(tot_weights[:self.filters2 * self.filters * self.ker_size * self.ker_size],
                                    (self.filters2, self.filters, self.ker_size, self.ker_size))
            bias = tot_weights[self.filters2 * self.filters * self.ker_size * self.ker_size:]
            out[i] = F.max_pool2d(F.relu(F.conv2d(out[i], weights, bias, stride=(1, 1), padding=(1, 1))), 2)
            out[i] = out[i].view(-1, self.filters2 * self.config[6][0] * self.config[6][1])

        for i in range(self.num_branches):
            tot_weights = self.hyper_2(self.scale[i])
            weights = torch.reshape(tot_weights[:self.width * self.filters2 * self.config[6][0] * self.config[6][1]],
                                    (self.width, self.filters2 * self.config[6][0] * self.config[6][1]))
            bias = tot_weights[self.width * self.filters2 * self.config[6][0] * self.config[6][1]:]
            self.linear1.weight.data = weights
            self.linear1.bias.data = bias
            out[i] = F.relu(self.linear1(out[i]))

        for i in range(self.num_branches):
            tot_weights = self.hyper_3(self.scale[i])
            weights = torch.reshape(tot_weights[:self.width * self.out_size],
                                    (self.out_size, self.width))
            bias = tot_weights[self.width * self.out_size:]
            self.linear2.weight.data = weights
            self.linear2.bias.data = bias
            out[i] = self.linear2(out[i])

        return out

class MetaLearner(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MetaLearner, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class HyperBasicClassifier(nn.Module):

    def __init__(self, dataset_name, num_branches=1, meta_block_num=1, out_size=2, scale_factor=128, parallel=False,
                 meta_learn=False, device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')):
        super(HyperBasicClassifier, self).__init__()

        self.meta_block_num = meta_block_num
        self.parallel = parallel
        self.out_size = out_size
        self.dataset_name = dataset_name
        self.num_branches = num_branches
        self.meta_learn = meta_learn
        self.device = device

        if self.meta_learn:
            self.meta_learner = MetaLearner(input_dim=self.num_branches * self.out_size, num_classes=self.out_size)

        # self.meta_block = []
        # for idx in range(self.meta_block_num):
        #     self.meta_block.append(
        #         MetaBlock(self.ker_size, self.filters, self.width, self.out_size, self.parallel))
        # self.meta_block = nn.Sequential(*self.meta_block)
        self.meta_block = MetaBlock(self.dataset_name, self.num_branches, self.out_size, scale_factor, self.parallel,
                                    device=self.device)

    def forward(self, x):
        out = self.meta_block(x)
        if self.meta_learn:
            stacked_out = torch.cat(out, dim=1)
            out = self.meta_learner(stacked_out)
        return out


class HyperClsBasicClassifier(nn.Module):

    def __init__(self, dataset_name, num_branches=1, meta_block_num=1, out_size=2, scale_factor=128, parallel=False,
                 meta_learn=False, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(HyperClsBasicClassifier, self).__init__()

        self.meta_block_num = meta_block_num
        self.parallel = parallel
        self.out_size = out_size
        self.dataset_name = dataset_name
        self.num_branches = num_branches
        self.meta_learn = meta_learn
        self.device = device

        if self.meta_learn:
            self.meta_learner = MetaLearner(input_dim=self.num_branches * self.out_size, num_classes=self.out_size)

        # self.meta_block = []
        # for idx in range(self.meta_block_num):
        #     self.meta_block.append(
        #         MetaBlock(self.ker_size, self.filters, self.width, self.out_size, self.parallel))
        # self.meta_block = nn.Sequential(*self.meta_block)
        self.meta_block = MetaBlockCls(self.dataset_name, self.num_branches, self.out_size, scale_factor, self.parallel,
                                    device=self.device)

    def forward(self, x):
        out = self.meta_block(x)
        if self.meta_learn:
            stacked_out = torch.cat(out, dim=1)
            out = self.meta_learner(stacked_out)
        return out


class EnsembleBasicClassifier(nn.Module):

    def __init__(self, num_classes=10, image_size=[3, 32, 32], num_classifiers=10):
        super(EnsembleBasicClassifier, self).__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.num_classifiers = num_classifiers
        self.basic_classifiers = [BasicClassifier(self.num_classes, self.image_size) for i in
                                  range(self.num_classifiers)]

    def forward(self, x):
        out = [None for _ in range(self.num_classifiers)]
        for i in range(self.num_classifiers):
            out[i] = self.basic_classifiers[i](x)

        return out


class BasicClassifier(nn.Module):
    def __init__(self, num_classes=10, image_size=[3, 32, 32]):
        super(BasicClassifier, self).__init__()
        self.num_classes = num_classes
        self.num_in_channels = image_size[0]
        self.image_dim = (image_size[1] + image_size[2]) // 2
        self.width = 128
        self.filters = 16
        self.filters2 = self.filters * 2
        self.image_dim_after_maxpool = self.image_dim // 4
        self.conv1 = nn.Conv2d(self.num_in_channels, self.filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.filters, self.filters2, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(self.image_dim_after_maxpool * self.image_dim_after_maxpool * self.filters2, self.width)
        self.fc2 = nn.Linear(self.width, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, self.image_dim_after_maxpool * self.image_dim_after_maxpool * self.filters2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Create an instance of the MNISTClassifier
if __name__ == '__main__':
    from torchinfo import summary

    ds_name = 'MNIST'
    c, h, w = INPUT_SIZE[ds_name]
    num_classes = NUM_CLASSES[ds_name]
    # mnist_baseline_summary = summary(mnist_baseline, (1, c, h, w))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_tensor = torch.randn(128, 1, 28, 28).to(device)
    # torch.save(mnist_baseline.state_dict(), '../checkpoints/model_illustration/mnist_classifier.pt')
    # netron.start('../checkpoints/model_illustration/mnist_classifier.pt')
