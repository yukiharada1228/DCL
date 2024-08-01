# coding: utf-8

# In[1]:


import logging
import pdb

import torch
from timm.data import Mixup

from lib import utils

logger = logging.getLogger(__name__)


def one_hot(x, num_classes, on_value=1.0, off_value=0.0):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=x.device).scatter_(
        1, x, on_value
    )


# In[48]:


class ClassificationTrainer:
    def __init__(self, args, sampler):
        self.args = args
        self.sampler = sampler
        pass

    def to_cuda(self, input_, target):
        input_ = input_.cuda(self.args.gpu.id)
        target = target.cuda(self.args.gpu.id)
        return input_, target

    def compute_outputs(self, models, input_, target):
        outputs = [model(input_) for model in models]
        return outputs

    def post_forward(self, models, outputs):
        for id_, model in enumerate(models):
            if hasattr(model, "post_forward"):
                outputs[id_] = model.post_forward(outputs)
        return outputs

    def calc_accuracy(self, output, target):
        return utils.accuracy(output, target, topk=(1,))

    def measure(self, output, target):
        return calc_accuracy(output, target)

    def update_meter(self, metric_dict, values, batch_size):
        for key, value in values.items():
            if key in metric_dict:
                metric_dict[key].update(value, batch_size)
        return

    def write_log(self, logs, metrics, epoch, mode="train"):
        for log, metric_dict in zip(logs.net, metrics):
            for metric_name, meter in metric_dict.items():
                log["epoch_log"][epoch][f"{mode}_{metric_name}"] = meter.avg
        return

    def train_on_batch(
        self, input_, target, models, criterions, optimizers, logs, metrics, **kwargs
    ):
        outputs = self.compute_outputs(models, input_, target)
        outputs = self.post_forward(models, outputs)

        losses = []
        for model_id, (criterion, optimizer, log, metric) in enumerate(
            zip(criterions, optimizers, logs.net, metrics)
        ):
            loss = criterion(model_id, outputs, target, log, **kwargs)
            losses += [loss]

            output = outputs[model_id]
            if len(output) == 2:
                output = output[0]
            # FIXME: Mixupの影響で正解率を通常の計算では出すことができない
            # acc = self.calc_accuracy(output, target)
            acc = torch.Tensor([0.0])
            # 現在の学習率を取得
            current_lr = optimizer.param_groups[0]["lr"]
            self.update_meter(
                metric,
                {
                    "loss": loss.item(),
                    "accuracy": acc[0].item(),
                    "learning_rate": current_lr,  # 学習率を記録
                },
                batch_size=input_.size(0),
            )

        # initialize gradient
        for optimizer in optimizers:
            optimizer.zero_grad()
        # exclude loss if it equal 0
        update_idxs = [id_ for id_, loss in enumerate(losses) if loss != 0]
        # compute gradient
        for id_ in update_idxs:
            losses[id_].backward(retain_graph=True)
        # update parameters
        for id_ in update_idxs:
            optimizers[id_].step()

        return

    def train_on_dataset(
        self, data_loader, models, criterions, optimizers, epoch, logs, **kwargs
    ):
        """
        train on dataset for one epoch
        """
        metrics = [
            {
                "loss": utils.AverageMeter(),
                "accuracy": utils.AverageMeter(),
                "learning_rate": utils.AverageMeter(),  # 学習率用のメーターを追加
            }
            for _ in range(len(models))
        ]

        for i, model in enumerate(models):
            is_cutoff = all(
                [loss.args.gate.name == "CutoffGate" for loss in self.args.losses[i]]
            )
            if is_cutoff:
                model.eval()
            else:
                model.train()
        # FIXME: num_classesがハードコーディングになっている．
        mixup_fn = Mixup(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            cutmix_minmax=None,
            prob=1.0,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=0.1,
            num_classes=100,
        )

        for i, (input_, target) in enumerate(data_loader):
            input_, target = self.to_cuda(input_, target)
            input_, target = mixup_fn(input_, target)
            self.train_on_batch(
                input_, target, models, criterions, optimizers, logs, metrics, **kwargs
            )

        self.write_log(logs, metrics, epoch, mode="train")

        return logs

    def validate_on_batch(
        self, input_, target, models, criterions, logs, metrics, **kwargs
    ):
        outputs = self.compute_outputs(models, input_, target)
        outputs = self.post_forward(models, outputs)
        for model_id, (criterion, log, metric) in enumerate(
            zip(criterions, logs.net, metrics)
        ):
            # FIXME: num_classesがハードコーディング
            _target = one_hot(target, num_classes=100)
            loss = criterion(model_id, outputs, _target, log=None, **kwargs)
            acc = self.calc_accuracy(outputs[model_id], target)

            self.update_meter(
                metric,
                {"loss": loss.item(), "accuracy": acc[0].item()},
                batch_size=input_.size(0),
            )

        return

    def validate_on_dataset(
        self, data_loader, models, criterions, epoch, logs, **kwargs
    ):
        """
        validate on dataset
        """

        metrics = [
            {"loss": utils.AverageMeter(), "accuracy": utils.AverageMeter()}
            for _ in range(len(models))
        ]

        for model in models:
            model.eval()

        for i, (input_, target) in enumerate(data_loader):
            input_, target = self.to_cuda(input_, target)
            self.validate_on_batch(
                input_, target, models, criterions, logs, metrics, **kwargs
            )

        self.write_log(logs, metrics, epoch, mode="test")

        return logs

    def train(
        self,
        nets,
        criterions,
        optimizers,
        train_loader,
        test_loader,
        logs=None,
        **kwargs,
    ):
        import os
        import time

        print("manual seed : %d" % self.args.manualSeed)

        for epoch in range(self.args.trainer.start_epoch, self.args.trainer.epochs + 1):
            print("epoch %d" % epoch)
            start_time = time.time()
            
            self.sampler.set_epoch(epoch)

            for optimizer, model_args in zip(optimizers, self.args.models):
                utils.adjust_learning_rate(optimizer, epoch, model_args.optim)
            kwargs = {} if kwargs is None else kwargs
            kwargs.update(
                {
                    "_trainer": self,
                    "_train_loader": train_loader,
                    "_test_loader": test_loader,
                    "_nets": nets,
                    "_criterions": criterions,
                    "_optimizers": optimizers,
                    "_epoch": epoch,
                    "_logs": logs,
                    "_args": self.args,
                }
            )

            # train for one epoch
            self.train_on_dataset(
                train_loader, nets, criterions, optimizers, epoch, logs, **kwargs
            )
            # evaluate on validation set
            self.validate_on_dataset(
                test_loader, nets, criterions, epoch, logs, **kwargs
            )

            # print log
            for i, log in enumerate(logs.net):
                print(
                    "  net{0}    loss :train={1:.3f}, test={2:.3f}    acc :train={3:.3f}, test ={4:.3f}".format(
                        i,
                        log["epoch_log"][epoch]["train_loss"],
                        log["epoch_log"][epoch]["test_loss"],
                        log["epoch_log"][epoch]["train_accuracy"],
                        log["epoch_log"][epoch]["test_accuracy"],
                    )
                )

            if epoch % self.args.trainer.saving_interval == 0:
                ckpt_dir = os.path.join(self.args.trainer.base_dir, "checkpoint")
                utils.save_checkpoint(nets, optimizers, epoch, ckpt_dir)

            logs.save(self.args.trainer.base_dir + r"log/")

            elapsed_time = time.time() - start_time
            print("  elapsed_time:{0:.3f}[sec]".format(elapsed_time))

            if "_callback" in kwargs:
                kwargs["_callback"](**kwargs)

        return
