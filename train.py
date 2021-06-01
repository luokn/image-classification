#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : train.py
# @Date    : 2021/05/23
# @Time    : 13:51:16

import os
from argparse import ArgumentParser

from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import Adam

from lib import Trainer, load_data
from nn import resnet18, resnet34, resnet50, resnet101, resnet152

parser = ArgumentParser()

parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--checkpoints', type=str, default='./checkpoints')
parser.add_argument('--data', type=str, default='data/101_ObjectCategories')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--gpus', type=str, default=None)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--workers', type=int, default=0)

parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--classes', type=int, default=102)

models = {
    'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50,
    'resnet101': resnet101, 'resnet152': resnet152
}
args = parser.parse_args()
print(args)

if __name__ == '__main__':
    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    # data loaders
    data_loaders = load_data(args.data, batch_size=args.batch, num_workers=args.workers)
    # network
    net = models[args.model](args.classes)
    net = DataParallel(net).cuda() if args.gpus else net.cuda(args.gpu)
    # optimizer
    optimizer = Adam(net.parameters(), lr=args.lr)
    # loss function
    criterion = CrossEntropyLoss()
    # trainer
    trainer = Trainer(net, optimizer, criterion, args.checkpoints)
    if args.checkpoint:
        # load checkpoint
        trainer.load_checkpoint(args.checkpoint)
    # train and validate
    trainer.fit(data_loaders[0], data_loaders[1], epochs=args.epochs, device=args.gpu)
    # load best checkpoint
    trainer.load_checkpoint(f'epoch={trainer.best_epoch}_loss={trainer.min_loss:.2f}.pkl')
    # evaluate
    trainer.evaluate(data_loaders[2], device=args.gpu)
    # save history
    trainer.save_history()
