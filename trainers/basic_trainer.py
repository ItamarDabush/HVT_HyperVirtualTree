import argparse
import ast
import collections
import json
import os
import pprint
import time
from typing import TextIO, Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import wandb
from munch import munchify
from torchinfo import summary

from data.dataloaders import BasicDataLoaders
from data.datasets import BasicDatasets
from data.transforms import BasicTransforms, DATA_SETS_MEAN, DATA_SETS_STD, ImageNetTransforms
from utils.common import unnormalize_image
from utils.constants import NUM_CLASSES, INPUT_SIZE, CLASSES_NAMES, DATASET_NAMES, ENCODING_CLASS_HYPER_DIC
from utils.early_stopping import EarlyStopping
from utils.metrics_tracker import MetricsTracker, NotMetricsTracker, BinaryMetricsTracker
from utils.progress_bar import progress_bar

print('starting basic trainer')


class BasicTrainer:
    BASE_CHECKPOINTS_DIR = f'../checkpoints'  # change to your own directory
    MODEL_CLS = None

    def __init__(self):
        parser = self.init_parser()
        args = parser.parse_args()
        assert args.dataset_name in DATASET_NAMES
        self.dataset_name = args.dataset_name
        experiment_name = f"{self.dataset_name}_{args.exp_name}"
        config = vars(args)
        if args.use_wandb:
            assert args.exp_name, "Must choose experiment name when running in wandb!"
            resume_run_id = config["resume_run_id"]
            if resume_run_id is not None:
                wandb.init(project=experiment_name, config=config, id=resume_run_id, resume='must')
            else:
                wandb.init(project=experiment_name, config=config)
            self.config = wandb.config
        else:
            self.config = config
        print("Running with the following configuration:")
        pprint.pprint(self.config, width=1)
        self.scale_factor = self.config["scale_factor"]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_gpus = torch.cuda.device_count()
        self.num_gpus = 1  # comment if parallel is desired
        self.parallel = False
        if self.num_gpus > 1:
            self.parallel = True
        self.image_size = INPUT_SIZE[self.dataset_name]
        self.num_in_channels = INPUT_SIZE[self.dataset_name][0]
        self.num_classes = NUM_CLASSES[self.dataset_name]
        self.classes_indices = np.arange(self.num_classes)
        self._init_config_attributes()

        self.best_top1_acc = 0
        self.model: nn.Module = self.init_model()
        self.transforms = self.init_transforms()
        self.datasets = self.init_data_sets()
        self.data_loaders = self.init_data_loaders()
        if self.config["network_type"] == "hyper-cls-new":
            self.cls_criterion = nn.BCEWithLogitsLoss()
        elif self.config["network_type"] == "hyper-cls-new1":
            self.cls_criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9], device=self.device))
            self.model_cls_criterion = nn.CrossEntropyLoss()
        elif self.config["network_type"] == "hyper-cls":
            self.cls_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([9], device=self.device))
            self.model_cls_criterion = nn.CrossEntropyLoss()
        elif self.config["network_type"] == "hyper-ensemble-voting":
            self.cls_criterion = [nn.CrossEntropyLoss(weight=torch.tensor([3.0 if i in [j-2, j-1, j, j+1, j+2] else 1.0 for i in range(len(NUM_CLASSES) + 1)], device=self.device)) for j in range(len(NUM_CLASSES) + 1)]
            self.special_cls_criterion = nn.CrossEntropyLoss()
        elif self.config["network_type"] == "hyper-ensemble-stacking":
            self.cls_criterion = [nn.CrossEntropyLoss(weight=torch.tensor([3.0 if i in [j-2, j-1, j, j+1, j+2] else 1.0 for i in range(len(NUM_CLASSES) + 1)], device=self.device)) for j in range(len(NUM_CLASSES) + 1)]
            self.special_cls_criterion = nn.CrossEntropyLoss()
        else:
            self.cls_criterion = nn.CrossEntropyLoss()
        self.optimizer = self.init_optimizer()
        self.lr_scheduler = self.init_lr_scheduler()
        if self.config["network_type"] == "hyper-cls-new":
            self.branch_metrics_tracker = [BinaryMetricsTracker(NUM_CLASSES[self.dataset_name], self.include_top5) for _ in range(self.model.num_branches)]
            self.metrics_tracker = MetricsTracker(NUM_CLASSES[self.dataset_name], self.include_top5)
        elif self.config["network_type"] in ["hyper-ensemble-voting", "hyper-ensemble-stacking"]:
            self.branch_metrics_tracker = [MetricsTracker(NUM_CLASSES[self.dataset_name], self.include_top5) for _ in range(self.model.num_branches)]
            self.metrics_tracker = MetricsTracker(NUM_CLASSES[self.dataset_name], self.include_top5)
        elif self.config["network_type"] == "hyper-cls-new1":
            self.branch_metrics_tracker = [MetricsTracker(self.model.out_size, self.include_top5) for _
                                           in range(self.model.num_branches)]
            self.metrics_tracker = MetricsTracker(NUM_CLASSES[self.dataset_name], self.include_top5)
        elif self.config["network_type"] == "hyper-cls":
            self.branch_metrics_tracker = [BinaryMetricsTracker(self.model.out_size, self.include_top5) for _ in range(self.model.num_branches)]
            self.metrics_tracker = MetricsTracker(NUM_CLASSES[self.dataset_name], self.include_top5)
        else:
            self.metrics_tracker = MetricsTracker(NUM_CLASSES[self.dataset_name], self.include_top5)
        if self.do_early_stopping:
            self.early_stopping = self.init_early_stopping()

        ckpt_fn = 'best' if not self.use_wandb else wandb.run.name
        checkpoints_dir = os.path.join(self.BASE_CHECKPOINTS_DIR, experiment_name)
        if not os.path.isdir(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        self.best_ckpt_path = os.path.join(checkpoints_dir, f'{ckpt_fn}_ckpt.pth')
        self.epoch = 0

    def _init_config_attributes(self):
        self.use_wandb = self.config["use_wandb"]
        self.log_images = self.use_wandb and self.config["log_images"]
        self.augment = self.config["augment"]
        self.do_early_stopping = self.config["do_early_stopping"]
        self.early_stopping_params = self.config["early_stopping_params"]
        self.batch_size = self.config["batch_size"]
        self.test_batch_size = self.config["test_batch_size"]
        self.num_epochs = self.config["num_epochs"]
        self.learning_rate = self.config["learning_rate"]
        self.save_checkpoint = self.config["save_checkpoint"]
        self.include_top5 = self.config["include_top5"]
        self.use_validation = self.config["use_validation"]

    def summary(self, batch_size: int = 1) -> None:
        input_shape = INPUT_SIZE[self.dataset_name]
        summary(self.model, (batch_size,) + input_shape)

    def init_early_stopping(self):
        early_stopping_params = self.early_stopping_params
        if early_stopping_params is None:
            early_stopping_params = {'mode': 'min', 'patience': 30, 'verbose': True}
        return EarlyStopping(**early_stopping_params)

    @staticmethod
    def init_config_from_json(config_path: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"File does not exist ({config_path}")
        elif not os.path.isfile(config_path):
            raise Exception(f"Invalid file ({config_path})")
        with open(config_path, 'r') as fp:
            params_dict = json.load(fp)
        config = munchify(params_dict)
        return config

    def init_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset_name', '-d', dest='dataset_name', type=str, help='which dataset to train',
                            required=True, choices=['CIFAR10', 'CIFAR100', 'FashionMNIST', 'MNIST'])
        parser.add_argument('--exp_name', '-e', type=str, help='Name of the experiment (positional when using W&B)',
                            default='')
        parser.add_argument('--weights_path', '-w', type=str, help="Path to the model's weights file", default=None)
        parser.add_argument('--batch_size', type=int, help="Train batch size", default=128)
        parser.add_argument('--test_batch_size', type=int, help="Test batch size", default=128)
        parser.add_argument('--num_epochs', type=int, help="Number of training epochs", default=300)
        parser.add_argument('--num_workers', type=int, help="Number of dataloader workers", default=-1)
        parser.add_argument('--include_top5', action='store_true', help="Whether to log top5 accuracy data")
        parser.add_argument('--use_wandb', action='store_true', help="Track run with Weights and Biases")
        parser.add_argument('--log_images', action='store_true',
                            help="Log images to wandb (only works if use_wandb=True")
        parser.add_argument('--no_save', dest='save_checkpoint', action='store_false',
                            help="Do not save checkpoints")
        parser.add_argument('--learning_rate', type=float, help="Optimizer initial learning rate", default=0.1)
        parser.add_argument('--do_early_stopping', action='store_true', help="Enable early stopping")
        parser.add_argument('--augment', action='store_true', help="Perform data augmentation")
        parser.add_argument('--use_validation', action='store_true', help="Use validation set")
        parser.add_argument('--early_stopping_params', type=str,
                            help="JSON string with the EarlyStopping config dict")
        parser.add_argument('--lr_change_factor', type=float, default=0.1,
                            help='LR change factor (for the LR scheduler)')
        parser.add_argument('--num_lr_changes', type=int, default=2,
                            help="The number of LR changes allowed for the LR scheduler")
        parser.add_argument('--resume_run_id', type=str, default=None,
                            help="wandb run-id for resuming crashed runs (warning: this was not used thoroughly; "
                                 "use with caution)")
        parser.add_argument('--network_type', type=str, default=None,
                            help="not specified for basic network, "
                                 "hyper for basic with hyper network, "
                                 "hyper-cls for hyper with classification of one class at each branch, "
                                 "hyper-cls-new for new approach of hyper-cls"
                                 "ensemble-voting, "
                                 "ensemble-stacking, "
                                 "hyper-ensemble-voting, "
                                 "hyper-ensemble-stacking")
        parser.add_argument('--scale_factor', type=int, default=64,
                            help="when using multi branch network use scale factor")

        return parser

    def init_lr_scheduler(self):
        num_lr_changes = self.config["num_lr_changes"]
        lr_change_factor = self.config["lr_change_factor"]
        min_lr_ratio = lr_change_factor ** num_lr_changes
        return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, verbose=True, mode='max',
                                                          factor=lr_change_factor,
                                                          min_lr=self.optimizer.defaults['lr'] * min_lr_ratio)

    def _load_weights(self, model: nn.Module, weights_path: str):
        state = torch.load(weights_path, map_location=torch.device('cpu'))
        state_dict = state['net']
        # backwards comp.
        state_dict = collections.OrderedDict(
            {k.replace('branch0', 'left').replace('branch1', 'right'): v for k, v in state_dict.items()})
        if 'acc' in state:
            self.best_top1_acc = state['acc']
        model.load_state_dict(state_dict)

    def _init_model(self):
        assert self.MODEL_CLS
        model = self.MODEL_CLS(num_classes=self.num_classes)
        return model

    @staticmethod
    def _parse_weights_path(weights_path: str) -> Union[str, TextIO]:
        """
        we assume weights path is either an actual path, or a comma separated string with wandb checkpoint file name
        and run-path in the form of (run_name, run_path)
        Args:
            weights_path:

        Returns:

        """
        if os.path.exists(weights_path):
            return weights_path
        elif ',' in weights_path:
            ckpt_name, wandb_run_path = weights_path.split(',')
            best_model = wandb.restore(ckpt_name, run_path=wandb_run_path)
            # return best_model
            weights_path = best_model.name
            best_model.close()
            return weights_path

        else:
            raise Exception("Invalid weights path given")

    def create_batches(self, inputs, targets):
        target_classes = list(range(NUM_CLASSES[self.dataset_name]))  # Target classes (0 to 9)

        batches_inputs = []  # A list to store the created batches
        batches_targets = []

        for target_class in target_classes:
            # Identify indices of samples belonging to the target class
            target_indices = (targets == target_class).nonzero().squeeze()

            # Identify indices of samples belonging to the other classes
            other_indices = (targets != target_class).nonzero().squeeze()
            if (not (target_indices.shape)):
                print("bug!")
            # Determine the number of samples for the target class
            num_samples_target = len(target_indices)
            num_samples_other = num_samples_target

            # Create a batch by selecting an equal number of target and other class samples
            batch_indices = torch.cat((target_indices[:num_samples_target], other_indices[:num_samples_other]))

            # Shuffle the batch indices
            batch_indices = batch_indices[torch.randperm(batch_indices.size(0))]

            # Create the batch from the selected indices
            batch_inputs = inputs[batch_indices]
            batch_targets = targets[batch_indices]

            batches_inputs.append(batch_inputs)
            batches_targets.append(batch_targets)
        return batches_inputs, batches_targets

    def init_model(self):
        model = self._init_model()
        weights_path = self.config["weights_path"]
        if weights_path is not None:
            weights_path = self._parse_weights_path(weights_path)
            self._load_weights(model, weights_path)
        if self.parallel:
            model = torch.nn.DataParallel(model, device_ids=[0, 1], dim=0).to(self.device)
        else:
            self.num_gpus = 1
            model.to(self.device)
        cudnn.benchmark = True
        print(f"Using {self.num_gpus} GPUs!")
        if self.use_wandb:
            wandb.watch(model, log_freq=100)
            try:
                onnx_model_path = os.path.join(wandb.run.dir, 'model.onnx')
                model_to_save = model.module if (self.num_gpus > 1) else model
                torch.onnx.export(model_to_save,
                                  torch.randn((1,) + INPUT_SIZE[self.dataset_name], device=self.device),
                                  onnx_model_path)
                wandb.save(onnx_model_path, wandb.run.dir)
            except Exception as e:
                print("Failed to save onnx model:")
                print(str(e))
        return model

    def init_data_sets(self):
        return BasicDatasets(self.transforms, use_validation=self.use_validation, dataset_name=self.dataset_name)

    def init_data_loaders(self):
        num_workers = 0 if self.device == 'cpu' else self.config.get("num_workers")
        if num_workers < 0:
            num_workers = len(os.sched_getaffinity(0))
        print(f'num of workers: {num_workers}')
        return BasicDataLoaders(self.datasets, use_validation=self.use_validation, num_workers=num_workers,
                                train_bs=self.batch_size, test_bs=self.test_batch_size)

    def init_transforms(self, padding_mode='constant'):
        if self.dataset_name == 'ImageNet':
            return ImageNetTransforms(self.augment)
        return BasicTransforms(self.dataset_name, self.augment, padding_mode=padding_mode)

    def lr_scheduler_step(self, epoch=-1, train_acc=None, train_loss=None, test_acc=None, test_loss=None):
        assert train_acc
        self.lr_scheduler.step(train_acc)

    def early_stopping_step(self, train_acc=None, train_loss=None, test_acc=None, test_loss=None):
        if self.optimizer.param_groups[0]['lr'] <= self.lr_scheduler.min_lrs[0]:
            self.early_stopping.step(test_loss)

    def _train_model(self):
        if self.use_wandb and wandb.run.resumed:
            self._resume_update()

        while self.epoch < self.num_epochs:
            epoch = self.epoch
            print(f'\nEpoch: {epoch + 1}')
            train_acc, train_loss = self._train_single_epoch(epoch)
            test_acc, test_loss = self._test_single_epoch(epoch)
            self.lr_scheduler_step(epoch, train_acc, train_loss, test_acc, test_loss)
            if self.do_early_stopping:
                self.early_stopping_step(train_acc, train_loss, test_acc, test_loss)
                if self.early_stopping.early_stop:
                    print("Applied early stopping!")
                    return
            self.epoch += 1

    def train_model(self):
        try:
            self._train_model()
        finally:
            if self.use_wandb:
                wandb.save(self.best_ckpt_path, os.path.dirname(self.best_ckpt_path))
                if self.log_images:
                    wandb.log({'Test sample predictions': self.get_sample_predictions()})

    def sanity_check_train(self, train_single_image=True):
        """
        Training a single image/batch, until overfitting
        Returns:

        """
        self.model.train()
        data_loader = self.data_loaders.train_loader
        num_batches = 1
        inputs, targets = next(iter(data_loader))
        if train_single_image:
            inputs = inputs[:1]
            targets = targets[:1]
        inputs, targets = self.input_and_targets_to_device(inputs, targets)
        for epoch in range(self.num_epochs):
            self.metrics_tracker.reset(num_batches)
            self.optimizer.zero_grad()
            if self.config["network_type"] == "hyper-ensemble":
                outputs, loss = self._hyper_ensemble_feed_forward(inputs, targets)
            elif self.config["network_type"] == "hyper":
                outputs, loss = self._hyper_feed_forward(inputs, targets)
            else:
                outputs, loss = self._feed_forward(inputs, targets)
            loss.backward()
            self.optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}:")
                msg_to_display = self.metrics_tracker.get_message_to_display(0)
                print(msg_to_display)
            train_acc = self.metrics_tracker.get_norm_top1_acc()
            self.lr_scheduler_step(train_acc=train_acc)

    def _hyper_ensemble_stacking_feed_forward(self, *args):
        outputs = self.model(args[0])
        tar = args[1]
        avg_branches_loss = 0
        for i, out, metrics_t in zip(range(len(outputs)), outputs, self.branch_metrics_tracker):
            predicted_classes = torch.argmax(out, dim=1)
            branches_loss = self.cls_criterion[i](out, tar.long()) / len(outputs)
            metrics_t.update(branches_loss, out, tar)
            avg_branches_loss += branches_loss

        avg_output = torch.sum(torch.stack(outputs), dim=0) / len(outputs)
        avg_loss = self.special_cls_criterion(avg_output, tar)
        total_loss = 0 * avg_loss + 1 * avg_branches_loss
        self.metrics_tracker.update(total_loss, avg_output, tar)
        return outputs, total_loss

    def _hyper_ensemble_voting_feed_forward(self, *args):
        outputs = self.model(args[0])
        tar = args[1]
        batch_votes = torch.zeros((outputs[0].shape[0], NUM_CLASSES[self.dataset_name])).to(self.device)
        avg_branches_loss = 0
        # special_tensor = torch.tensor([]).to(self.device)
        for i, out, metrics_t in zip(range(len(outputs)), outputs, self.branch_metrics_tracker):
            predicted_classes = torch.argmax(out, dim=1)
            for j, class_idx in enumerate(predicted_classes):
                if class_idx in [i - 2, i - 1, i, i + 1, i + 2]:
                    batch_votes[j, class_idx] += (2 / len(outputs))
                else:
                    batch_votes[j, class_idx] += (1 / len(outputs))

            branches_loss = self.cls_criterion[i](out, tar.long()) / len(outputs)
            metrics_t.update(branches_loss, out, tar)
            avg_branches_loss += branches_loss
            # special_tensor = torch.cat((special_tensor, out), dim=1)

        # special_accuracy_tensor = torch.cat([torch.unsqueeze(item[:, i], dim=1) for i, item in enumerate(outputs)], dim=1)
        # special_tar = torch.stack([i * 10 + i for i in tar]).to(self.device)
        # special_loss = self.special_cls_criterion(special_tensor, special_tar)
        # print(special_loss)
        total_loss = avg_branches_loss   # + 1 * special_loss
        self.metrics_tracker.update(total_loss, batch_votes, tar)
        return outputs, total_loss

    def _hyper_cls_feed_forward(self, *args):
        outputs = self.model(args[0])
        tar = args[1]
        # avg_branches_loss = 0
        # for i in range(len(outputs)):
        #     one_hot_tensor = (tar == i).float().to(self.device)
        #     branches_loss = self.cls_criterion(outputs[i].squeeze(), one_hot_tensor) / len(outputs)
        #     self.branch_metrics_tracker[i].update(branches_loss, outputs[i].squeeze(), one_hot_tensor)
        #     avg_branches_loss += branches_loss
        output_tensor = torch.cat([tensor for tensor in outputs], dim=1)
        model_loss = self.model_cls_criterion(output_tensor, tar.long())
        # total_loss = 0.95 * model_loss + 0.05 * avg_branches_loss
        self.metrics_tracker.update(model_loss, output_tensor, args[1])
        return outputs, model_loss

    def _hyper_cls_new_feed_forward(self, *args):
        outputs = self.model(args[0])
        loss = 0
        tar = args[1]
        for i in range(len(outputs)):
            loss += self.cls_criterion(outputs[i].squeeze(), (tar[i] == i).float()) / len(outputs)
            self.branch_metrics_tracker[i].update(loss, outputs[i].squeeze(), (tar[i] == i).float())
        # self.metrics_tracker.update(loss, torch.tensor([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]), torch.tensor([1, 1]))
        return outputs, loss

    def _hyper_cls_new1_feed_forward(self, *args):
        outputs = self.model(args[0])
        output_tensor = torch.stack([tensor[:, 1] for tensor in outputs], dim=1)
        tar = args[1]
        branches_loss = 0
        for i in range(len(outputs)):
            one_hot_tensor = (tar == i).long().to(self.device)
            branches_loss += self.cls_criterion(outputs[i], one_hot_tensor) / len(outputs)
            self.branch_metrics_tracker[i].update(branches_loss, outputs[i], one_hot_tensor)
        model_loss = self.model_cls_criterion(output_tensor, tar)
        total_loss = 0.5 * model_loss + 0.5 * branches_loss
        self.metrics_tracker.update(total_loss, output_tensor, tar)
        return outputs, total_loss

    def _hyper_feed_forward(self, *args):
        outputs = self.model(args[0])
        tar = args[1]
        loss = self.cls_criterion(outputs[0], tar.long())
        # loss = [self.cls_criterion(out, tar.long()) for out in outputs]
        # loss_mean = sum(loss)/len(loss)
        self.metrics_tracker.update(loss, outputs[0], args[1])
        return outputs, loss

    def _feed_forward(self, *args):  # inputs: args[0] targets: args[1]
        outputs = self.model(args[0])
        loss = self.cls_criterion(outputs, args[1].long())
        self.metrics_tracker.update(loss, outputs, args[1])
        return outputs, loss

    def _single_epoch(self, epoch: int, train_test_val: str):
        assert train_test_val in ['train', 'test', 'validation']
        training = train_test_val == 'train'
        self.model.train(training)
        data_loader = getattr(self.data_loaders, f'{train_test_val}_loader')
        num_batches = len(data_loader)
        self.metrics_tracker.reset(num_batches)
        if self.config["network_type"] in ["hyper-cls-new", "hyper-cls-new1", "hyper-ensemble-voting", "hyper-ensemble-stacking", "hyper-cls"]:
            [tracker.reset(num_batches) for tracker in self.branch_metrics_tracker]
        with torch.set_grad_enabled(training):
            start_time = time.time()
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                self._single_iteration(inputs, targets, training, epoch)
                # msg_to_display = self.metrics_tracker.get_message_to_display(batch_idx)
                # print(msg_to_display)
                # if not self.use_wandb:  # or self.device == 'cpu':
                #     progress_bar(batch_idx, num_batches, msg_to_display)
            end_time = time.time()
        if self.config["network_type"] == "hyper-cls-new":
            msg_to_display = ""
            norm_top1_acc = self.branch_metrics_tracker[0].get_norm_top1_acc()
            norm_loss = self.branch_metrics_tracker[0].get_norm_loss()
        else:
            msg_to_display = self.metrics_tracker.get_message_to_display(num_batches)
            norm_top1_acc = self.metrics_tracker.get_norm_top1_acc()
            norm_loss = self.metrics_tracker.get_norm_loss()
        if self.use_wandb:
            if self.device != torch.device('cpu'):
                msg_to_display += f' | Avg. Batch Processing Time: {int(1000 * (end_time - start_time) / num_batches)} ms'
                print(msg_to_display)
            log_dict = {f"{train_test_val}_loss": norm_loss, f"{train_test_val}_top1_acc": 100. * norm_top1_acc}
            if self.include_top5:
                log_dict[f"{train_test_val}_top5_acc"] = 100. * self.metrics_tracker.get_norm_top5_acc()
            if training:
                for i, pg in enumerate(self.optimizer.param_groups):
                    log_dict[f"learning_rate_g{i}"] = pg['lr']
            wandb.log(log_dict, step=epoch + 1)
        else:
            msg_to_display += f' | Avg. Batch Processing Time: {int(1000 * (end_time - start_time) / num_batches)} ms'
            print(msg_to_display)

        if epoch % 10 == 0 and f'decisio' not in self.config["network_type"]:
            if self.config["network_type"] in ["hyper-ensemble-voting", "hyper-ensemble-stacking", "hyper-cls-new1"]:
                acc_msg_to_display = f'Tot_class_acc: {self.metrics_tracker.get_class_accuracy(num_batches)}\n'
                acc_msg_to_display += ''.join(
                    [f'{i}_branch_acc: {self.branch_metrics_tracker[i].get_class_accuracy(num_batches)}\n' for i in
                     range(len(self.branch_metrics_tracker))])
            elif self.config["network_type"] in ["hyper-cls-new", "hyper-cls"]:
                acc_msg_to_display = f'Tot_class_acc: {self.metrics_tracker.get_class_accuracy(num_batches)}\n'
                acc_msg_to_display += ''.join(
                    [f'{i}_branch_acc: {self.branch_metrics_tracker[i].get_class_accuracy(num_batches)}\n' for i in
                     range(len(self.branch_metrics_tracker))])
            else:
                acc_msg_to_display = f'Tot_class_acc: {self.metrics_tracker.get_class_accuracy(num_batches)}'
            print(acc_msg_to_display)

        if train_test_val == 'test':
            top1_acc = 100. * norm_top1_acc
            if top1_acc > self.best_top1_acc:
                print(f'Test accuracy improved! ({self.best_top1_acc:.3f} ==> {top1_acc:.3f}).')
                if self.save_checkpoint:
                    self._save_checkpoint(top1_acc, epoch)
                self.best_top1_acc = top1_acc
                if self.use_wandb:
                    wandb.log({"test_best_accuracy": top1_acc}, step=epoch + 1)
        return norm_top1_acc, norm_loss

    def _single_iteration(self, inputs, targets, training: bool, epoch: int):
        inputs, targets = self.input_and_targets_to_device(inputs, targets)
        if training:
            self.optimizer.zero_grad()
        if self.config["network_type"] == "hyper-ensemble-stacking":
            outputs, loss = self._hyper_ensemble_stacking_feed_forward(inputs, targets)
        elif self.config["network_type"] == "hyper-ensemble-voting":
            outputs, loss = self._hyper_ensemble_voting_feed_forward(inputs, targets)
        elif self.config["network_type"] == "ensemble-voting":
            outputs, loss = self._hyper_ensemble_voting_feed_forward(inputs, targets)
        elif self.config["network_type"] == "hyper":
            _, loss = self._hyper_feed_forward(inputs, targets)
        elif self.config["network_type"] == "hyper-cls":
            _, loss = self._hyper_cls_feed_forward(inputs, targets)
        elif self.config["network_type"] == "hyper-cls-new":
            _, loss = self._hyper_cls_new_feed_forward(*self.create_batches(inputs, targets))
        elif self.config["network_type"] == "hyper-cls-new1":
            outputs, loss = self._hyper_cls_new1_feed_forward(inputs, targets)
        else:
            _, loss = self._feed_forward(inputs, targets)
        if training:
            loss.backward()
            self.optimizer.step()

    def input_and_targets_to_device(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        return inputs, targets

    def _resume_update(self):
        self.config.update({'weights_path': self.best_ckpt_path}, allow_val_change=True)
        # self.config['weights_path'] = self.best_ckpt_path
        self.do_early_stopping = self.config['do_early_stopping']
        self.model = self.init_model()
        best_ckpt = torch.load(self.best_ckpt_path, torch.device('cpu'))
        self.best_top1_acc = best_ckpt['acc']
        self.epoch = best_ckpt['epoch']
        self.optimizer.load_state_dict(best_ckpt['optimizer'])
        self.lr_scheduler.load_state_dict(best_ckpt['lr_scheduler'])
        if self.do_early_stopping:
            self.early_stopping.load_state_dict(best_ckpt['early_stopping'])

    def _save_checkpoint(self, acc, epoch):
        print('Saving checkpoint...')
        net_state_dict = self.model.state_dict() if self.num_gpus <= 1 else self.model.module.state_dict()
        state = {
            'net': net_state_dict,
            'acc': acc,
            'epoch': epoch,
            'config': dict(self.config),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict()
        }
        if self.do_early_stopping:
            state['early_stopping'] = self.early_stopping.state_dict()
        torch.save(state, self.best_ckpt_path)

    def _train_single_epoch(self, epoch):
        return self._single_epoch(epoch, 'train')

    def _test_single_epoch(self, epoch):
        return self._single_epoch(epoch, 'test')

    def get_sample_predictions(self, num_images=24):
        output_predictions = []
        output_images = []
        test_set = self.datasets.test_set
        num_images = min(num_images, len(test_set))
        data_loader = torch.utils.data.DataLoader(test_set, batch_size=num_images, shuffle=True, num_workers=0)
        data_iter = iter(data_loader)
        ds_mean = DATA_SETS_MEAN[self.dataset_name]
        ds_std = DATA_SETS_STD[self.dataset_name]
        self.model.eval()
        with torch.no_grad():
            images, labels = next(data_iter)
            images, labels = self.input_and_targets_to_device(images, labels)
            predictions = self.model(images)
            if isinstance(predictions, tuple) and len(predictions) >= 2:
                predictions = predictions[0]
            _, predictions = predictions.max(1)
            for i in range(num_images):
                img = images[i]
                pred = predictions[i]
                pred = CLASSES_NAMES[self.dataset_name][int(pred)]
                np_img = unnormalize_image(img, ds_mean, ds_std)
                output_images.append(np_img)
                output_predictions.append(pred)
        sample_predictions = [wandb.Image(image, caption=prediction)
                              for (image, prediction) in zip(output_images, output_predictions)]
        return sample_predictions

    def init_optimizer(self):
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        return optimizer

    def evaluate(self):
        self.save_checkpoint = False
        self._test_single_epoch(0)
        print(self.metrics_tracker.get_message_to_display(len(self.data_loaders.test_loader) - 1))
