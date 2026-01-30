import torch
import numpy as np


class WarmupCosineSchedulerWithRestarts(torch.optim.lr_scheduler._LRScheduler):
    """Warmup and cosine annealing learning rate scheduler with restarts.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
        warmup_epochs (int): Number of epochs for the warmup phase.
        total_epochs (int): Total number of epochs for the cosine annealing schedule.
        min_lr (float, optional): Minimum learning rate after decay. Default is 0.0.
        red_fac (float, optional): Factor by which to reduce the learning rate at each restart. Default is 0.1.
        last_epoch (int, optional): The index of the last epoch. Default is -1.
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs,
        total_epochs,
        min_lr=0.0,
        red_fac=0.1,
        last_epoch=-1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.red_fac = red_fac
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        super().__init__(optimizer)
        self.last_epoch = last_epoch

    def get_lr(self):
        epoch = self.last_epoch % self.total_epochs
        red_fac_pow = self.last_epoch // self.total_epochs
        if epoch < self.warmup_epochs:
            factor = self.red_fac**red_fac_pow * (epoch) / self.warmup_epochs
            return [max(factor * base_lr, self.min_lr) for base_lr in self.base_lrs]
        else:
            cos_decay = (
                self.red_fac**red_fac_pow
                * 0.5
                * (
                    1
                    + np.cos(
                        np.pi
                        * (epoch - self.warmup_epochs)
                        / (self.total_epochs - self.warmup_epochs)
                    )
                )
            )
            return [max(base_lr * cos_decay, self.min_lr) for base_lr in self.base_lrs]
