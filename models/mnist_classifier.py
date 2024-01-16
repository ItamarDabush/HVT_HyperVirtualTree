# import netron
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.constants import INPUT_SIZE, NUM_CLASSES


class MnistClassifier(nn.Module):

    def __init__(self, num_classes=10, num_in_channels=1):
        super(MnistClassifier, self).__init__()
        self.conv1 = nn.Conv2d(num_in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(7 * 7 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 7 * 7 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Create an instance of the MNISTClassifier
if __name__ == '__main__':
    from torchinfo import summary

    ds_name = 'MNIST'
    c, h, w = INPUT_SIZE[ds_name]
    num_classes = NUM_CLASSES[ds_name]
    mnist_baseline = MnistClassifier(num_classes=num_classes, num_in_channels=c)
    mnist_baseline_summary = summary(mnist_baseline, (1, c, h, w))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_tensor = torch.randn(128, 1, 28, 28).to(device)
    mnist_baseline(input_tensor)
    # torch.save(mnist_baseline.state_dict(), '../checkpoints/model_illustration/mnist_classifier.pt')
    # netron.start('../checkpoints/model_illustration/mnist_classifier.pt')