import random
import functools
import torch
import torch.nn as nn
import wandb
from matplotlib import pyplot as plt
from fvcore.nn import FlopCountAnalysis

# from custom_layers.losses import WeightedMSELoss
from data.datasets import FilteredRelabeledDatasets
from models.fixed_hyper_decisionet import FixedBasicHyperDecisioNet, FixedBasicHyperDecisioNet_1
from models.small_fixed_hyper_decisionet import SmallFixedBasicHyperDecisioNet_1
from models.new_fixed_hyper_decisionet import NewFixedBasicHyperDecisioNet
from trainers.basic_trainer import BasicTrainer
from utils.constants import LABELS_MAP, CLASSES_NAMES, INPUT_SIZE, NUM_CLASSES
from utils.metrics_tracker import SigmaLossMetricsTracker
from utils.common_tools import set_random_seed, weights_init_kaiming
import torch.nn.init as init


class FixedHyperDecisioNetTrainer(BasicTrainer):

    def __init__(self):
        super().__init__()
        # sigma_weights = self._init_sigma_weights()
        if self.hyper:
            self.sigma_criterion = nn.MSELoss()  # WeightedMSELoss(sigma_weights)
            self.metrics_tracker = SigmaLossMetricsTracker(self.include_top5)

    def _init_model(self):
        raise NotImplementedError

    def _init_config_attributes(self):
        super()._init_config_attributes()
        self.beta = self.config['beta']
        self.always_binarize = self.config['always_binarize']

    def init_data_sets(self):
        labels_map = dict(LABELS_MAP[self.dataset_name])
        return FilteredRelabeledDatasets(self.transforms, use_validation=self.use_validation,
                                         classes_indices=self.classes_indices,
                                         labels_map=labels_map,
                                         dataset_name=self.dataset_name)

    def _feed_forward(self, inputs, targets):
        cls_targets, *sigma_targets = targets
        sigma_targets = torch.column_stack(sigma_targets)
        binarize = self.always_binarize or random.random() > 0.5
        outputs, sigmas = self.model(inputs, binarize=binarize)
        # print(f'Cls correct: {sum(torch.argmax(outputs, 1)==cls_targets)}')
        # print(f'Sigma correct: {sum(sigmas==sigma_targets)}\n')
        cls_loss = self.cls_criterion(outputs, cls_targets.long())
        sigma_loss = self.sigma_criterion(sigmas, sigma_targets.float())
        combined_loss = cls_loss + self.beta * sigma_loss
        self.metrics_tracker.update(cls_loss, sigma_loss, combined_loss, outputs, cls_targets, sigmas, sigma_targets)
        return outputs, combined_loss

    def _single_epoch(self, epoch: int, train_test_val: str):
        norm_acc, norm_loss = super()._single_epoch(epoch, train_test_val)
        if self.use_wandb:
            log_dict = {f"{train_test_val}_cls_loss": self.metrics_tracker.get_norm_cls_loss(),
                        f"{train_test_val}_sigma_loss": self.metrics_tracker.get_norm_sigma_loss(),
                        f"{train_test_val}_sigma_accuracy": 100. * self.metrics_tracker.get_norm_sigma_acc()}
            wandb.log(log_dict, step=epoch + 1)
        return norm_acc, norm_loss

    def input_and_targets_to_device(self, inputs, targets):
        inputs = inputs.to(self.device)
        for i in range(len(targets)):
            targets[i] = targets[i].to(self.device)
        return inputs, targets

    def init_parser(self):
        parser = super().init_parser()
        parser.add_argument('--beta', type=float, help='weight for the sigma loss', default=0.0)
        parser.add_argument('--always_binarize', action='store_true',
                            help='do not use non-binary values in the binarization layer (i.e., perform only hard routing)')
        return parser

    def register_hooks(self, activation_dict):
        def get_activation(activations_dict):
            def hook(model, input, output):
                predictions = output[0].detach()
                predictions = predictions.argmax(dim=1)
                sigma = output[1].detach()
                activations_dict['predictions'] = torch.cat([activations_dict['predictions'], predictions])
                activations_dict['sigma'] = torch.cat([activations_dict['sigma'], sigma])

            return hook

        hook_handles = []
        for name, layer in self.model.named_modules():
            if name == '':
                activation_dict['predictions'] = torch.Tensor([])
                activation_dict['sigma'] = torch.Tensor([])
                handle = layer.register_forward_hook(get_activation(activation_dict))
                hook_handles.append(handle)
        return hook_handles

    # noinspection PyTypeChecker
    def evaluate(self):
        activations_dict = {}
        self.register_hooks(activations_dict)
        super().evaluate()
        # norm_acc, _ = self._test_single_epoch(0)
        targets = torch.tensor(self.datasets.test_set.targets)
        predictions = activations_dict['predictions']
        sigmas = activations_dict['sigma']
        cls_targets = targets[:, 0]
        sigma_targets = targets[:, 1:]

        cls_acc = torch.sum(predictions == cls_targets) / targets.size(0)
        print(f"Class accuracy: {cls_acc * 100.}")
        sigma_diffs = (sigmas == sigma_targets)
        encoding = {0: 'both wrong', 1: 'first correct', 2: 'second correct', 3: 'both correct'}
        encoded_results = torch.sum(sigma_diffs * torch.tensor([1., 2.]), dim=1)
        for code, s in encoding.items():
            print(f'{s}: {torch.sum(encoded_results == code).item()}')
        num_images = 0
        for cls in self.classes_indices:
            if num_images % 10 == 0:
                plt.figure()
            plt.subplot(5, 2, num_images % 10 + 1)

            print("***********************************")
            class_name = CLASSES_NAMES[self.dataset_name][cls]
            plt.title(class_name)
            print(f"Class: {class_name}")
            cls_idx = torch.where(cls_targets == cls)[0]
            cls_acc = torch.sum(predictions[cls_idx] == cls) / cls_idx.size(0)
            print(f"Accuracy: {cls_acc * 100.}")
            sigma_diffs = (sigmas[cls_idx] == sigma_targets[cls_idx])
            encoded_results = torch.sum(sigma_diffs * torch.tensor([1., 2.]), dim=1)
            results = []
            for code, s in encoding.items():
                correct = torch.sum(encoded_results == code).item()
                results.append(correct)
                print(f'{s}: {correct}')
            plt.bar(list(encoding.values()), results)
            num_images += 1
        plt.show()
        plt.tight_layout()

    # noinspection PyTypeChecker
    def sigma_analysis(self):
        activations_dict = {}
        self.register_hooks(activations_dict)
        super().evaluate()
        targets = torch.tensor(self.datasets.test_set.targets)
        sigmas = activations_dict['sigma'].round()
        cls_targets = targets[:, 0]
        sigma_targets = targets[:, 1:]

        removed = []
        new_labels_map = {k: v[1:] for k, v in LABELS_MAP[self.dataset_name].items()}
        for s in range(2):
            soft_clustering_cands = []
            for cls in self.classes_indices:
                if cls in removed:
                    continue
                class_name = CLASSES_NAMES[self.dataset_name][cls]
                cls_idx = torch.where(cls_targets == cls)[0]
                results = []
                for i in range(2):
                    correct_until_split_idx = \
                        torch.where(torch.all(sigma_targets[cls_idx, :s] == sigmas[cls_idx, :s], dim=1))[0]
                    num_i = torch.sum(sigmas[cls_idx[correct_until_split_idx], s] == i).item()
                    results.append(num_i)
                if min(results) > 0.2 * sum(results):
                    soft_clustering_cands.append((class_name, cls, results))
                    plt.figure()
                    plt.bar(['0', '1'], results)
                    plt.title(f'{class_name} - split {s}')
            plt.show()
            print(f"SOFT CLUSTERING CANDS - SPLIT {s}:")
            for cls_name, cls, res in soft_clustering_cands:
                print(cls_name, f'({cls})', '- results:', res)
                orig_labels = new_labels_map[cls]
                new_labels = orig_labels[:s] + (0.5,) * (len(orig_labels) - s)
                new_labels_map[cls] = new_labels
                removed.append(cls)
        print(new_labels_map)

    def _init_sigma_weights(self):
        num_sigma_labels_per_samples = len(self.datasets.train_set.targets[0]) - 1
        sigma_weights = torch.tensor([0.5 ** i for i in range(num_sigma_labels_per_samples)]).to(self.device)
        return sigma_weights


class FixedNetworkInNetworkHyperDecisioNetTrainer(FixedHyperDecisioNetTrainer):

    def _init_model(self):
        set_random_seed(0)
        self.hyper = True
        # model = SmallFixedBasicHyperDecisioNet_1(hyper=self.hyper, multi_hyper=self.hyper)
        model = NewFixedBasicHyperDecisioNet()
        # model.apply(functools.partial(weights_init_kaiming, scale=0.01))
        # model.apply(self.weights_init_xavier)
        return model

    def init_parser(self):
        parser = super().init_parser()
        return parser

    def _init_config_attributes(self):
        super()._init_config_attributes()

    def init_data_sets(self):
        labels_map = dict(LABELS_MAP[self.dataset_name])
        num_blocks = 2
        for k, v in labels_map.items():
            labels_map[k] = v[:num_blocks]
        return FilteredRelabeledDatasets(self.transforms, use_validation=self.use_validation,
                                         classes_indices=self.classes_indices,
                                         labels_map=labels_map,
                                         dataset_name=self.dataset_name)

    def weights_init_xavier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0.0)

if __name__ == '__main__':
    trainer = FixedNetworkInNetworkHyperDecisioNetTrainer()
    # trainer = WideResNetDecisioNetTrainer()
    input_tensor = torch.randn(1, 3, 32, 32).to(trainer.device)
    flops = FlopCountAnalysis(trainer.model, input_tensor)
    print(f'Number Of Flops: {flops.total()}')
    print(f'Number Of Parameters: {sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)}')
    trainer.train_model()
    # trainer.evaluate()
