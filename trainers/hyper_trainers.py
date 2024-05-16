from utils.common_tools import set_random_seed, weights_init_kaiming
import functools
import wandb
import time

import torch.nn as nn
import torch.optim.lr_scheduler
from utils.constants import NUM_CLASSES, BRANCH_4_TO_CLASS, BRANCH_10_TO_CLASS
from utils.metrics_tracker import MetricsTracker, BinaryMetricsTracker

from models.basic_classifier import BasicClassifier, HyperBasicClassifier, EnsembleBasicClassifier
from trainers.basic_trainer import BasicTrainer
from utils.early_stopping import EarlyStopping

class BasicClassifierTrainer(BasicTrainer):

    def __init__(self):
        super(BasicClassifierTrainer, self).__init__()

        if self.config["network_type"] == "hyper-cls":
            self.cls_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([9], device=self.device))
            self.model_cls_criterion = nn.CrossEntropyLoss()
        elif self.config["network_type"] in ["hyper-ensemble-voting", "hyper-ensemble-stacking"]:
            self.cls_criterion = [nn.CrossEntropyLoss(weight=torch.tensor([2.0 if i in BRANCH_4_TO_CLASS[j] else 1.0 for i in range(self.num_classes)],device=self.device)) for j in range(self.model.num_branches)]
            # self.cls_criterion = [nn.CrossEntropyLoss(weight=torch.tensor([2.0 if i in BRANCH_10_TO_CLASS[j] else 1.0 for i in range(self.num_classes)], device=self.device)) for j in range(self.model.num_branches)]
            self.special_cls_criterion = nn.CrossEntropyLoss()
            if self.config["network_type"] == "hyper-ensemble-stacking":
                self.special_cls_criterion = nn.CrossEntropyLoss()

        if self.config["network_type"] in ["hyper-ensemble-voting", "hyper-ensemble-stacking"]:
            self.branch_metrics_tracker = [MetricsTracker(NUM_CLASSES[self.dataset_name], self.include_top5) for _ in range(self.model.num_branches)]
            self.metrics_tracker = MetricsTracker(NUM_CLASSES[self.dataset_name], self.include_top5)
        elif self.config["network_type"] == "hyper-cls":
            self.branch_metrics_tracker = [BinaryMetricsTracker(self.model.out_size, self.include_top5) for _ in range(self.model.num_branches)]
            self.metrics_tracker = MetricsTracker(NUM_CLASSES[self.dataset_name], self.include_top5)

    def init_early_stopping(self):
        early_stopping_params = self.early_stopping_params
        if early_stopping_params is None:
            early_stopping_params = {'mode': 'min', 'patience': 50, 'verbose': True}
        return EarlyStopping(**early_stopping_params)

    def _init_model(self):

        set_random_seed(0)
        if self.config["network_type"] == "hyper-ensemble-stacking":
            model = HyperBasicClassifier(num_branches=self.num_classes, out_size=self.num_classes,
                                         scale_factor=self.scale_factor, parallel=self.parallel,
                                         dataset_name=self.dataset_name, meta_learn=False, device=self.device)
        elif self.config["network_type"] == "hyper-ensemble-voting":
            model = HyperBasicClassifier(num_branches=4, out_size=self.num_classes,
                                         scale_factor=self.scale_factor, parallel=self.parallel,
                                         dataset_name=self.dataset_name, device=self.device)
        elif self.config["network_type"] == "hyper-ensemble-meta":
            model = HyperBasicClassifier(num_branches=4, out_size=self.num_classes,
                                         scale_factor=self.scale_factor, parallel=self.parallel,
                                         meta_learn=True, dataset_name=self.dataset_name, device=self.device)
        elif self.config["network_type"] == "ensemble-voting":
            model = EnsembleBasicClassifier(num_classes=self.num_classes, image_size=self.image_size,
                                            num_classifiers=self.num_classes)
        elif self.config["network_type"] == "hyper":
            model = HyperBasicClassifier(num_branches=1, out_size=self.num_classes, scale_factor=self.scale_factor,
                                         parallel=self.parallel, dataset_name=self.dataset_name, device=self.device)
        elif self.config["network_type"] == "hyper-cls":
            model = HyperBasicClassifier(num_branches=self.num_classes, out_size=1, scale_factor=self.scale_factor,
                                         parallel=self.parallel, dataset_name=self.dataset_name, device=self.device)
        else:
            model = BasicClassifier(num_classes=self.num_classes, image_size=self.image_size)
        model.apply(functools.partial(weights_init_kaiming, scale=0.1))
        return model

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
                if class_idx in BRANCH_4_TO_CLASS[i]:
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


    def _hyper_feed_forward(self, *args):
        outputs = self.model(args[0])
        tar = args[1]
        loss = self.cls_criterion(outputs[0], tar.long())
        # loss = [self.cls_criterion(out, tar.long()) for out in outputs]
        # loss_mean = sum(loss)/len(loss)
        self.metrics_tracker.update(loss, outputs[0], args[1])
        return outputs, loss

    def _single_epoch(self, epoch: int, train_test_val: str):
        assert train_test_val in ['train', 'test', 'validation']
        training = train_test_val == 'train'
        self.model.train(training)
        data_loader = getattr(self.data_loaders, f'{train_test_val}_loader')
        num_batches = len(data_loader)
        self.metrics_tracker.reset(num_batches)
        if self.config["network_type"] in ["hyper-ensemble-voting", "hyper-ensemble-stacking", "hyper-cls", "hyper-ensemble4-voting"]:
            [tracker.reset(num_batches) for tracker in self.branch_metrics_tracker]
        with torch.set_grad_enabled(training):
            start_time = time.time()
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                self._single_iteration(inputs, targets, training, epoch)
            end_time = time.time()
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
            if self.config["network_type"] in ["hyper-ensemble-voting", "hyper-ensemble-stacking"]:
                acc_msg_to_display = f'Tot_class_acc: {self.metrics_tracker.get_class_accuracy(num_batches)}\n'
                acc_msg_to_display += ''.join(
                    [f'{i}_branch_acc: {self.branch_metrics_tracker[i].get_class_accuracy(num_batches)}\n' for i in
                     range(len(self.branch_metrics_tracker))])
            elif self.config["network_type"] == "hyper-cls":
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
        else:
            _, loss = self._feed_forward(inputs, targets)
        if training:
            loss.backward()
            # for name, param in self.model.named_parameters():
            #     print(f'Parameter: {name}, Gradient norm: {param.grad.norm()}')
            self.optimizer.step()


if __name__ == '__main__':
    # trainer = NetworkInNetworkTrainer()
    trainer = BasicClassifierTrainer()
    # trainer = WideResNetTrainer()
    trainer.train_model()
    # trainer.evaluate()
