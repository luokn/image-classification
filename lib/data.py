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
        split0, split1 = int(.6 * len(array)), int(.8 * len(array))
        for s, e in [(0, split0), (split0, split1), (split1, len(array))]:
            yield array[s:e]

    images, labels = [[], [], []], [[], [], []]
    for category, directory in enumerate(os.listdir(path)):
        # print(f'{directory} => {category}')
        files = [f'{path}/{directory}/{file}' for file in os.listdir(f'{path}/{directory}')]
        for i, imgs in enumerate(split_array(files)):
            images[i] += imgs
            labels[i] += [category] * len(imgs)
    return [
        DataLoader(dataset=ImagesDataset(*args),
                   batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
        for args in zip(images, labels)
    ]
