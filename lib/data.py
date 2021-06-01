#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : data.py
# @Date    : 2021/05/31
# @Time    : 21:05:55


import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ImagesDataset(Dataset):
    def __init__(self, images, labels):
        self.images, self.labels = images, labels

    def __getitem__(self, index):
        image = Image.open(self.images[index]).resize([224, 224])
        image = torch.from_numpy(np.array(image)).float() / 256 - .5
        if image.ndim == 2:
            image = image.repeat(3, 1, 1)
        else:
            image = image.permute(2, 0, 1)
        label = self.labels[index]
        return image, label

    def __len__(self):
        return len(self.images)


def load_data(path, batch_size, num_workers=0):
    def split_array(array: list):
        split1, split2 = int(.6 * len(array)), int(.8 * len(array))
        return [array[:split1], array[split1:split2], array[split2:]]

    images, labels = [[], [], []], [[], [], []]
    for c, d in enumerate(os.listdir(path)):
        # print(f'{d} => {c}')
        for i, imgs in enumerate(split_array([f'{path}/{d}/{f}' for f in os.listdir(f'{path}/{d}')])):
            images[i] += imgs
            labels[i] += [c] * len(imgs)
    return [
        DataLoader(dataset=ImagesDataset(*args),
                   batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
        for args in zip(images, labels)
    ]
