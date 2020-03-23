# CREDIT: JULIUS RUSECKAS
# http://web.vu.lt/tfai/j.ruseckas/files/presentations/cifar10-tricks.ipynb

# imports from other tools
from .utils import wer_eval

import numpy as np
from itertools import islice
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

import time

# torch.__version__ == 1.3: get_lr
# torch.__version__ == 1.3: get_last_lr


class Accuracy:
    def __init__(self, counter=None):
        self.num_samples = 0
        self.num_correct = 0
        self.counter = counter if counter else self

    def count_correct(self, pred, y):
        return (pred == y).float().sum()

    def process_prediction(self, Y_pred, Y):
        batch_size = Y.size(0)
        labels_pred = torch.argmax(Y_pred, -1)
        self.num_correct += self.counter.count_correct(labels_pred, Y)
        self.num_samples += batch_size

    def get_average(self):
        return self.num_correct / self.num_samples


class CharErrRate:
    def __init__(self, counter=None):
        self.num_samples = 0
        self.err_count = 0
        # interesting...
        self.counter = counter if counter else self

    def count_err_rate(self, y_hat, y_gt):
        '''
        mean error count per example in batch
        '''
        err_count = 0
        for batch_idx in range(y_hat.shape[0]):
            # TODO: detach() can cause backprop problems here?
            err_count += wer_eval(
                            y_hat[batch_idx, :, :],
                            y_gt[batch_idx, :].detach().cpu().numpy().tolist()
                         )
        return err_count # mean error count inside example, summ over batch

    def process_prediction(self, Y_pred, Y):
        batch_size = Y.size(0)
        self.err_count += self.counter.count_err_rate(Y_pred, Y)
        self.num_samples += batch_size

    def get_average(self):
        return self.err_count / self.num_samples

class AverageLoss:
    def __init__(self):
        self.total_loss = 0.0
        self.num_samples = 0

    def process_loss(self, batch_size, batch_loss):
        self.total_loss += batch_size * batch_loss
        self.num_samples += batch_size

    def get_average(self):
        return self.total_loss / self.num_samples

class AccumulateSmoothLoss:
    def __init__(self, smooth_f=0.05, diverge_th=5):
        self.diverge_th = diverge_th
        self.smooth_f = smooth_f
        self.losses = []

    def process_loss(self, batch_size, loss):
        if not self.losses:
            self.best_loss = loss
        else:
            if self.smooth_f > 0:
                loss = self.smooth_f * loss + (1. - self.smooth_f) * self.losses[-1]
            if loss < self.best_loss:
                self.best_loss = loss

        self.losses.append(loss)
        if loss > self.diverge_th * self.best_loss:
            raise StopIteration

    def get_losses(self):
        return self.losses

class Learner:
    def __init__(self, model, loss, optimizer, train_loader, val_loader,
                 epoch_scheduler=None, batch_scheduler=None, mixup=None,
                 time_limit_hr = 1
                 ):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epoch_scheduler = epoch_scheduler
        self.batch_scheduler = batch_scheduler
        self.mixup = mixup
        self.time_limit_hr = time_limit_hr
        self.start_time = time.time()

        self.reset_history()

        # read device from the model
        self.DEVICE = next(model.parameters()).device


    def reset_history(self):
        self.train_losses = []
        self.val_losses = []
        self.lrs = []


    def plot_losses(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        epochs = np.arange(1, len(self.train_losses)+1)
        ax.plot(epochs, self.train_losses, '.-', label='train')
        ax.plot(epochs, self.val_losses, '.-', label='val')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.set_yscale('log')
        ax.legend()
        plt.show()


    def iterate(self, loader, cbs=[], backward_pass=False):
        for X, Y in loader:

            if time.time() - self.start_time > self.time_limit_hr*60*60:
                print('Stopping due to my time limit')
                break;

            X, Y = X.to(self.DEVICE), Y.to(self.DEVICE)
            for cb in cbs:
                if hasattr(cb, 'process_batch'): X = cb.process_batch(X)

            Y_pred = self.model(X)
            batch_loss = self.loss(Y_pred, Y)
            if backward_pass: self.backward_pass(batch_loss)

            for cb in cbs:
                if hasattr(cb, 'process_loss'): cb.process_loss(X.size(0), batch_loss.item())
                if hasattr(cb, 'process_prediction'): cb.process_prediction(Y_pred, Y)


    def backward_pass(self, batch_loss):
        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        if self.batch_scheduler:
            self.lrs.append(self.batch_scheduler.get_lr()[0])
            self.batch_scheduler.step()


    def train_for_epoch(self):
        self.model.train()

        if self.mixup: self.loss = self.mixup.mixup_loss
        cbs = [AverageLoss(), CharErrRate(self.mixup), self.mixup]
        self.iterate(self.train_loader, cbs, backward_pass=True)
        if self.mixup: self.loss = self.mixup.loss

        train_loss, train_err_rate = [cb.get_average() for cb in cbs[:2]]
        self.train_losses.append(train_loss)
        print(f'train loss {train_loss:.3f}, train CER {train_err_rate:.3f},', end=' ')


    def eval_on_validation(self):
        self.model.eval()

        cbs = [AverageLoss(), CharErrRate()]
        with torch.no_grad():
            self.iterate(self.val_loader, cbs)

        val_loss, val_err_rate = [cb.get_average() for cb in cbs]
        self.val_losses.append(val_loss)
        print(f'val loss {val_loss:.3f}, val CER: {val_err_rate:.3f}')


    def plot_lr_find(self, skip_start=10, skip_end=5):
        def split_list(vals, skip_start, skip_end):
            return vals[skip_start:-skip_end] if skip_end > 0 else vals[skip_start:]

        lrs = split_list(self.lrs, skip_start, skip_end)
        losses = split_list(self.train_losses, skip_start, skip_end)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(lrs, losses, '.-')
        ax.set_xscale('log')
        ax.set_xlabel('Learning rate')
        ax.set_ylabel('Loss')
        plt.show()


    def set_learning_rate(self, lr):
        new_lrs = [lr] * len(self.optimizer.param_groups)
        for param_group, new_lr in zip(self.optimizer.param_groups, new_lrs):
            param_group["lr"] = new_lr


    def lr_find(self, start_lr, end_lr, num_iter):
        model_state = copy.deepcopy(self.model.state_dict())
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())

        self.model.train()
        self.set_learning_rate(start_lr)

        gamma = (end_lr / start_lr)**(1 / num_iter)
        self.batch_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma)

        if self.mixup: self.loss = self.mixup.mixup_loss
        cbs = [AccumulateSmoothLoss(), self.mixup]
        try:
            self.iterate(islice(self.train_loader, num_iter), cbs, backward_pass=True)
        except StopIteration:
            print("Stopping early, the loss has diverged")
        if self.mixup: self.loss = self.mixup.loss
        self.batch_scheduler = None

        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)

        self.train_losses = cbs[0].get_losses()
        self.plot_lr_find()
        self.reset_history()


    def fit(self, epochs):
        for i in range(epochs):
            print(f'{i+1}/{epochs}:', end=' ')
            self.train_for_epoch()
            self.eval_on_validation()

            if self.epoch_scheduler:
                self.lrs.append(self.epoch_scheduler.get_lr()[0])
                self.epoch_scheduler.step()


def plot_lrs(lrs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    batches = np.arange(1, len(lrs)+1)
    ax.plot(batches, lrs)
    ax.set_xlabel('batch')
    ax.set_ylabel('lr')
    plt.show()
