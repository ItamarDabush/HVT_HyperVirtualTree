from typing import List
import torch
import numpy as np
from utils.constants import NUM_CLASSES
from scipy.stats import entropy


def topk_correct_predictions(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[torch.FloatTensor]:
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    Args:
        output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
        target: target is the truth
        topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    Returns:
        list of topk accuracy [top1st, top2nd, ...] depending on your topk input

    """

    with torch.no_grad():
        max_k = max(topk)
        _, y_pred = output.topk(k=max_k, dim=1)

        correct = (y_pred == target.view(-1, 1))

        topk_correct_count = []
        for k in topk:
            num_correct_samples_k = correct[:, :k].sum()
            topk_correct_count.append(num_correct_samples_k)
        return topk_correct_count

def calculate_binary_class_accuracy(output: torch.Tensor, target: torch.Tensor) -> List:
    accuracy = []
    correct_pred = [0] * 2      # binary
    total_pred = [0] * 2
    target = target.int()
    predictions = output
    # collect the correct predictions for each class
    for label, prediction in zip(target, predictions):
        if label == prediction:
            correct_pred[label] += 1
        total_pred[label] += 1

    for class_count, correct_count in zip(total_pred, correct_pred):
        if class_count == 0:
            accuracy.append(0.0)  # Handle the case when there are no predictions for a class
        else:
            accuracy.append(100 * float(correct_count) / class_count)

    return accuracy

def calculate_class_accuracy(output: torch.Tensor, target: torch.Tensor) -> List:
    accuracy = []
    correct_pred = [0] * (len(output[1]))
    total_pred = [0] * (len(output[1]))

    predictions = torch.argmax(output, 1)
    # collect the correct predictions for each class
    for label, prediction in zip(target, predictions):
        if label == prediction:
            correct_pred[label] += 1
        total_pred[label] += 1

    for class_count, correct_count in zip(total_pred, correct_pred):
        if class_count == 0:
            accuracy.append(0.0)  # Handle the case when there are no predictions for a class
        else:
            accuracy.append(100 * float(correct_count) / class_count)

    return accuracy


def topk_correct_predictions_not(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[torch.FloatTensor]:
    """

    """

    with torch.no_grad():
        max_k = max(topk)
        _, y_pred = output.topk(k=max_k, dim=1)

        correct = (y_pred != target.view(-1, 1))

        topk_correct_count = []
        for k in topk:
            num_correct_samples_k = correct[:, :k].sum()
            topk_correct_count.append(num_correct_samples_k)
        return topk_correct_count


class MetricsTracker:
    def __init__(self, num_classes, include_top5=False) -> None:
        self.total_loss = 0
        self.total_correct_top1 = 0
        self.total_correct_top5 = 0
        self.total_samples = 0
        self.num_batches = 0
        self.num_classes = num_classes
        self.total_class_accuracy = np.zeros(self.num_classes) # NUM_CLASSES
        self.include_top5 = include_top5

    def reset(self, num_batches):
        self.total_loss = 0
        self.total_correct_top1 = 0
        self.total_correct_top5 = 0
        self.total_samples = 0
        self.total_class_accuracy[:] = 1
        self.num_batches = num_batches

    def update(self, loss, outputs, targets):
        if type(targets) is list:
            targets = targets[-1]
        self.total_loss += loss.item()
        self.total_samples += targets.size(0)
        if outputs.shape[1] < 5:
            top1_correct = topk_correct_predictions(outputs, targets)
            self.total_correct_top1 += top1_correct[0]
        else:
            top1_correct, top5_correct = topk_correct_predictions(outputs, targets, (1, 5))
            self.total_correct_top1 += top1_correct
            self.total_correct_top5 += top5_correct
        class_accuracy = calculate_class_accuracy(outputs, targets)
        self.total_class_accuracy += class_accuracy

    def get_message_to_display(self, batch_idx):
        msg_to_display = f'Loss: {self.total_loss / (batch_idx + 1):.3f} | ' \
                         f'Top-1 Acc: {100. * self.total_correct_top1 / self.total_samples:.3f}%% ' \
                         f'({self.total_correct_top1}/{self.total_samples})'
        if self.include_top5:
            msg_to_display += f'| Top-5 Acc: {100. * self.total_correct_top5 / self.total_samples:.3f}%% ' \
                              f'({self.total_correct_top5}/{self.total_samples})'
        return msg_to_display

    def get_class_accuracy(self, batch_idx):
        return f'Class_accuracy: {[round(x / (batch_idx + 1), 3) for x in self.total_class_accuracy]}'

    def get_norm_loss(self):
        norm_loss = self.total_loss / self.num_batches
        return norm_loss

    def get_norm_top1_acc(self):
        norm_acc = self.total_correct_top1 / self.total_samples
        return norm_acc

    def get_norm_top5_acc(self):
        norm_acc = self.total_correct_top5_not / self.total_samples
        return norm_acc


class NotMetricsTracker:
    def __init__(self, include_top5=False) -> None:
        self.total_loss = 0
        self.total_correct_top1 = 0
        self.total_correct_top5 = 0
        self.total_correct_top1_not = 0
        self.total_correct_top5_not = 0
        self.total_samples = 0
        self.num_batches = 0
        self.include_top5 = include_top5

    def reset(self, num_batches):
        self.total_loss = 0
        self.total_correct_top1 = 0
        self.total_correct_top5 = 0
        self.total_correct_top1_not = 0
        self.total_correct_top5_not = 0
        self.total_samples = 0
        self.num_batches = num_batches

    def update(self, loss, not_outputs, yes_outputs, targets):
        if type(targets) is list:
            targets = targets[-1]
        self.total_loss += loss.item()
        self.total_samples += targets.size(0)

        top1_correct_not, top5_correct_not = topk_correct_predictions_not(not_outputs, targets, (1, 5))
        self.total_correct_top1_not += top1_correct_not
        self.total_correct_top5_not += top5_correct_not

        top1_correct, top5_correct = topk_correct_predictions(yes_outputs, targets, (1, 5))
        self.total_correct_top1 += top1_correct
        self.total_correct_top5 += top5_correct

    def get_message_to_display(self, batch_idx):
        msg_to_display = f'not Loss: {self.total_loss / (batch_idx + 1):.3f} | ' \
                         f'not Top-1 Acc: {100. * self.total_correct_top1_not / self.total_samples:.3f}%% ' \
                         f'yes Loss: {self.total_loss / (batch_idx + 1):.3f} | ' \
                         f'yes Top-1 Acc: {100. * self.total_correct_top1 / self.total_samples:.3f}%% ' \
                         f'({self.total_correct_top1}/{self.total_samples})'

        if self.include_top5:
            msg_to_display += f'| Top-5 Acc: {100. * self.total_correct_top5_not / self.total_samples:.3f}%% ' \
                              f'({self.total_correct_top5_not}/{self.total_samples})' \
                              f'({self.total_correct_top5}/{self.total_samples})'

        return msg_to_display

    def get_norm_loss(self):
        norm_loss = self.total_loss / self.num_batches
        return norm_loss

    def get_norm_top1_acc(self):
        norm_acc = self.total_correct_top1 / self.total_samples
        return norm_acc

    def get_norm_top5_acc(self):
        norm_acc = self.total_correct_top5 / self.total_samples
        return norm_acc


class BinaryMetricsTracker(MetricsTracker):

    def __init__(self, num_classes, include_top5=False) -> None:
        super().__init__(num_classes, include_top5=False)
        self.total_class_accuracy = np.zeros(2)     # binary

    def update(self, loss, outputs, targets):
        if type(targets) is list:
            targets = targets[-1]
        self.total_loss += loss.item()
        self.total_samples += targets.size(0)
        with torch.no_grad():
            top1_correct = torch.round(torch.sigmoid(outputs)).eq(targets).sum().item()
        self.total_correct_top1 += top1_correct
        class_accuracy = calculate_binary_class_accuracy(torch.round(torch.sigmoid(outputs)).int(), targets)
        self.total_class_accuracy += class_accuracy

    def get_norm_top5_acc(self):
        raise NotImplementedError("Should not be used!")


class SigmaLossMetricsTracker:
    def __init__(self, include_top5=False, include_entropy=False) -> None:
        self.total_loss = 0
        self.total_cls_loss = 0
        self.total_sigma_loss = 0
        self.total_cls_correct_top1 = 0
        self.total_cls_correct_top5 = 0
        self.total_sigma_correct = 0
        self.total_samples = 0
        self.num_batches = 0
        self.include_top5 = include_top5
        self.total_leaf_entropy = 0
        self.include_entropy=include_entropy

    def update(self, cls_loss, sigma_loss, loss, cls_outputs, cls_targets, sigma_outputs, sigma_targets):
        cls_loss = cls_loss.detach()
        sigma_loss = sigma_loss.detach()
        loss = loss.detach()
        cls_outputs = cls_outputs.detach()
        cls_targets = cls_targets.detach()
        sigma_outputs = sigma_outputs.detach()
        sigma_targets = sigma_targets.detach()

        self.total_samples += cls_targets.size(0)

        self.total_loss += loss.item()
        self.total_cls_loss += cls_loss.item()
        self.total_sigma_loss += sigma_loss.item()
        # _, predicted = cls_outputs.max(1)
        # self.total_cls_correct_top1 += predicted.eq(cls_targets).sum().item()
        top1_correct, top5_correct = topk_correct_predictions(cls_outputs, cls_targets, (1, 5))
        self.total_cls_correct_top1 += top1_correct
        self.total_cls_correct_top5 += top5_correct

        # self.total_sigma_correct += sigma_outputs.round().eq(sigma_targets).all(dim=1).sum().item()
        eq_sigma = sigma_outputs.float().round().eq(sigma_targets)
        eq_sigma[sigma_targets == 0.5] = True
        self.total_sigma_correct += eq_sigma.all(dim=1).sum().item()
        if self.include_entropy:
            avg_leaf_entropy = self.calculate_branchwise_entropy(sigma_outputs, sigma_targets)
            self.total_leaf_entropy += avg_leaf_entropy

    def calculate_branchwise_entropy(self, sigma_outputs, sigma_targets):
        entropies = []
        unique_sigma_targets = torch.unique(sigma_targets, dim=0)

        for branch_encoding in unique_sigma_targets:

            branch_idx = torch.where(sigma_targets == branch_encoding)[0]
            true_routing_mask = torch.all(sigma_outputs[branch_idx] == branch_encoding, dim=1)
            true_routing_count = torch.sum(true_routing_mask)
            target_routing_count = branch_idx.size(0)

            # Avoid division by zero
            if target_routing_count > 0:
                true_routing_prob = true_routing_count / target_routing_count
                branch_entropy = self.mean_entropy(true_routing_prob, base=2)
                entropies.append(branch_entropy)

        avg_entropy = torch.mean(torch.tensor(entropies)) if entropies else torch.tensor(0.0)
        return avg_entropy

    def mean_entropy(self, prob, base=None, epsilon=1e-4):
        # Ensure probabilities are valid
        prob = prob.clamp(min=epsilon, max=1 - epsilon)
        entropy = -prob * torch.log(prob) - (1 - prob) * torch.log(1 - prob)
        if base is not None:
            entropy /= torch.log(torch.tensor(base, dtype=entropy.dtype))
        return entropy

    def get_message_to_display(self, batch_idx):
        total_loss_msg = f'Total Loss: {self.total_loss / (batch_idx + 1):.3f}'
        cls_msg = f'Cls Loss: {self.total_cls_loss / (batch_idx + 1):.3f} | ' \
                  f'Cls Top-1 Acc: {100. * self.total_cls_correct_top1 / self.total_samples:.3f}%% ' \
                  f'({self.total_cls_correct_top1}/{self.total_samples})'
        if self.include_top5:
            cls_msg += f'| Cls Top-5 Acc: {100. * self.total_cls_correct_top5 / self.total_samples:.3f}%% ' \
                       f'({self.total_cls_correct_top5}/{self.total_samples})'
        sigma_msg = f'Sigma Loss: {self.total_sigma_loss / (batch_idx + 1):.3f} | ' \
                    f'Sigma Acc: {100. * self.total_sigma_correct / self.total_samples:.3f}%% ' \
                    f'({self.total_sigma_correct}/{self.total_samples})'
        msg_to_display = '; '.join([total_loss_msg, cls_msg, sigma_msg])
        return msg_to_display

    def get_leaf_entrophy(self):
        leaf_entrophy = self.total_leaf_entropy / self.num_batches
        return leaf_entrophy

    def get_norm_loss(self):
        norm_loss = self.total_loss / self.num_batches
        return norm_loss

    def get_norm_cls_loss(self):
        norm_cls_loss = self.total_cls_loss / self.num_batches
        return norm_cls_loss

    def get_norm_sigma_loss(self):
        norm_sigma_loss = self.total_sigma_loss / self.num_batches
        return norm_sigma_loss

    def get_norm_top1_acc(self):
        return self.get_norm_cls_top1_acc()

    def get_norm_top5_acc(self):
        return self.get_norm_cls_top5_acc()

    def get_norm_cls_top1_acc(self):
        norm_acc = self.total_cls_correct_top1 / self.total_samples
        return norm_acc

    def get_norm_cls_top5_acc(self):
        norm_acc = self.total_cls_correct_top5 / self.total_samples
        return norm_acc

    def get_norm_sigma_acc(self):
        norm_sigma_acc = self.total_sigma_correct / self.total_samples
        return norm_sigma_acc

    def reset(self, num_batches):
        self.total_loss = 0
        self.total_cls_loss = 0
        self.total_sigma_loss = 0
        self.total_cls_correct_top1 = 0
        self.total_cls_correct_top5 = 0
        self.total_sigma_correct = 0
        self.total_samples = 0
        self.num_batches = num_batches
        self.total_leaf_entropy = 0