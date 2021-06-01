#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : metrics.py
# @Date    : 2021/05/23
# @Time    : 13:50:41

import torch
from torch import FloatTensor


class Metrics:
    def __init__(self):
        self.n, self.correct, self.accuracy = 0, 0, .0

    def update(self, prediction: FloatTensor, target: FloatTensor):
        self.n += target.nelement()
        self.correct += torch.sum(prediction.argmax(-1) == target).item()
        self.accuracy = self.correct / self.n

    def stats(self):
        return {'acc': self.accuracy}
