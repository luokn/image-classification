#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : trainer.py
# @Date    : 2021/05/23
# @Time    : 13:50:53

import json

import torch
from torch.nn import Module

from lib.metrics import Metrics
from lib.progressbar import ProgressBar


class Trainer:
    def __init__(self, net: Module, optimizer, criterion, checkpoints):
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.checkpoints = checkpoints
        self.history = [[], [], []]
        self.epoch, self.best_epoch, self.min_loss = 1, -1, float('inf')

    def fit(self, data_loader_t, data_loader_v, epochs=100, device=None):
        while self.epoch <= epochs:
            print(f"Epoch {self.epoch} training...")
            stats_t = self._run_epoch(data_loader_t, device, train=True)  # train epoch
            print(f"Epoch {self.epoch} validating...")
            stats_v = self._run_epoch(data_loader_v, device, train=False)  # validate epoch
            if self.epoch > epochs // 10 and stats_v['loss'] < self.min_loss:
                self.best_epoch, self.min_loss = self.epoch, stats_v['loss']
                self.save_checkpoint(self.best_epoch, self.min_loss)  # save checkpoint
            self.epoch += 1
            self.history[0].append(stats_t)
            self.history[1].append(stats_v)

    @torch.no_grad()
    def evaluate(self, data_loader_e, device=None):
        print('Evaluating...')
        stats_e = self._run_epoch(data_loader_e, device, train=False)  # evaluate
        self.history[-1].append(stats_e)

    def _run_epoch(self, data_loader, device, train=True):
        metrics, total_loss = Metrics(), .0  # initialize
        with torch.set_grad_enabled(train):  # enable or disable autograd
            self.net.train(train)
            with ProgressBar(total=len(data_loader)) as bar:
                for i, batch in enumerate(data_loader):
                    batch = [tensor.cuda(device) for tensor in batch]  # move tensors to device
                    inputs, target = batch[:-1], batch[-1]
                    pred = self.net(*inputs)
                    loss = self.criterion(pred, target)
                    if train:
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    total_loss += loss.item()  # update total loss
                    metrics.update(pred, target)  # update metrics
                    stats = {'loss': total_loss / (i + 1), **metrics.stats()}  # update stats
                    bar.update(' '.join([f'{k}:{v:7.4f}' for k, v in stats.items()]))  # update progress bar
        return stats

    def save_history(self):
        with open(f'{self.checkpoints}/history.json', 'w') as f:
            f.write(json.dumps(self.history))

    def save_checkpoint(self, epoch, loss):
        checkpoint = f'epoch={epoch}_loss={loss:.2f}.pkl'
        print(f'Save checkpoint {checkpoint}')
        torch.save({
            'epoch': epoch, 'loss': loss, 'history': self.history,
            'net': self.net.state_dict(), 'optimizer': self.optimizer.state_dict()
        }, f'{self.checkpoints}/{checkpoint}')

    def load_checkpoint(self, checkpoint):
        print(f'Load checkpoint {checkpoint}')
        states = torch.load(f'{self.checkpoints}/{checkpoint}')
        self.net.load_state_dict(states['net'])
        self.optimizer.load_state_dict(states['optimizer'])
        self.epoch = states['epoch'] + 1
        self.history = states['history']
        self.best_epoch, self.min_loss = states['epoch'], states['loss']
